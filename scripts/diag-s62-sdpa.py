#!/usr/bin/env python3
"""m4: direct SDPA A/B at layer 3 (the m3 root cause), no full model runs.

The m3 per-layer diff localized the first diverging COMPUTATION to the
layer-3 attention (SDPA) output for token 3545: row 1 of a 2-row causal
SDPA (variant A, T=2 verify) vs a 1-row unmasked SDPA (variant B, T=1 AR),
over bit-identical q/k/v. This script isolates WHERE in the SDPA path the
T-dependence lives (mask handling vs row-count tiling vs kernel dispatch)
and records the exact kernel name + parameter delta (the fix pointer).

Two modes:

  MODE=harvest
    Re-run the m3 flow (deterministic; re-arms at the S62 arm:
    prev=279 draft=3545 n_out=61). Snapshot the whole cache, then hook
    mx.fast.scaled_dot_product_attention and run variant A (the T=2 verify
    forward). The FIRST hooked SDPA call is layer 3 (the first
    full-attention layer; layers 0-2 are GDN). Capture its exact kernel
    inputs (q_2row, k_89, v_89), the scale, the mask object, and the
    SDPA output. Verify the captured k/v is bit-exact against the m3
    pkl's layer-3 cache state. Save to cap-m4-sdpa.pkl.

  MODE=ab
    Load cap-m4-sdpa.pkl. Run the direct SDPA A/B calls (no model):
      A  : SDPA(q_2row, k_89, v_89, scale, "causal") -> row 1
           (must be bit-exact vs the captured SDPA output - sanity that
           the re-run is on the same kernel path)
      B1 : SDPA(q279, k_88, v_88, scale, None)  -> control (== A row 0)
      B2 : SDPA(q3545, k_89, v_89, scale, None) -> vs A row 1 (m3 diverge)
    If B2 != A row 1, run single-variable bisection (a)-(e) to find the
    minimal trigger. Then run the same pairs on fresh synthetic q/k/v
    (random, same shapes) to confirm the trigger is structural (not
    data-dependent). Print the A/B table + minimal trigger + verdict.

The exact SDPA call the model makes (from qwen3_next.Qwen3NextAttention +
mlx_lm.models.base.scaled_dot_product_attention + cache.py
create_attention_mask):
  - q layout [B, n_h=16, S, h_dim=256]; k/v layout [B, n_kv=4, S, h_dim=256]
  - scale = head_dim**-0.5 = 256**-0.5 = 0.0625
  - mask = the string "causal" for N>1 (T=2), None for N==1 (T=1)
  - GQA 16 q heads / 4 kv heads (4:1) is handled natively by
    mx.fast.scaled_dot_product_attention
  - the non-quantized path (KVCache has no `bits`) is taken, so the call
    is exactly mx.fast.scaled_dot_product_attention(q, k, v, scale=...,
    mask=...)

Exit codes: 0 = done; 2 = arm mismatch (harvest); 3 = unexpected failure.
"""

import os
import pickle
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
MAX_TOKENS = int(os.environ.get("DIAG_MAX", "160"))
EOS = 248044
MODE = os.environ.get("MODE", "ab")
D = os.environ.get("CAP_DIR", "/mnt/models/tmp/an earlier run")
M3_PKL = os.path.join(D, "cap-m3-s62.pkl")
M4_PKL = os.path.join(D, "cap-m4-sdpa.pkl")
S62_PREV = 279
S62_DRAFT = 3545
EXPECT_N_OUT = 61
LAYER3 = 3


def npf(a):
    """mx array -> f32 numpy (avoids the bf16 PEP 3118 buffer issue)."""
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def of(npv, dt):
    """(f32 numpy, dtype) -> fresh mx array (f32 round-trip is exact)."""
    if npv is None:
        return None
    return mx.array(npv).astype(dt)


def cmp_report(name, a, b):
    """Bit-exact compare of two f32 arrays; prints MATCH / DIFF n maxabs."""
    if a is None or b is None:
        print(f"  {name:<40} SKIP (None)", flush=True)
        return True
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        print(f"  {name:<40} SHAPE {a.shape} vs {b.shape}", flush=True)
        return False
    diff = int(np.count_nonzero(a != b))
    if diff == 0:
        print(f"  {name:<40} MATCH  (n={a.size})", flush=True)
        return True
    ad = np.abs(a - b)
    fi = int(np.argmax(a != b))
    print(
        f"  {name:<40} DIFF   n={diff}/{a.size} maxabs={float(ad.max()):.3e} first={fi}",
        flush=True,
    )
    return False


def cmp_status(a, b):
    """Bit-exact compare; returns (status, n_diff, total, maxabs)."""
    if a is None and b is None:
        return ("NONE", 0, 0, 0.0)
    if a is None or b is None:
        return ("NONE-vs-ARR", 0, 0, 0.0)
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return (f"SHAPE {a.shape} vs {b.shape}", 0, 0, 0.0)
    d = a != b
    n = int(np.count_nonzero(d))
    if n == 0:
        return ("MATCH", 0, a.size, 0.0)
    return ("DIFF", n, a.size, float(np.abs(a - b).max()))


# ---------------------------------------------------------------------------
# SDPA call hook (harvest)
# ---------------------------------------------------------------------------


class SDPAHook:
    """Wraps mx.fast.scaled_dot_product_attention to capture kernel I/O."""

    def __init__(self):
        self.orig = mx.fast.scaled_dot_product_attention
        self.capturing = False
        self.calls = []

    def __call__(self, queries, keys, values, scale=1.0, mask=None, sinks=None):
        out = self.orig(queries, keys, values, scale=scale, mask=mask, sinks=sinks)
        if self.capturing:
            self.calls.append(
                {
                    "q": npf(queries),
                    "k": npf(keys),
                    "v": npf(values),
                    "out": npf(out),
                    "scale": float(scale),
                    "mask": mask,
                    "qshape": tuple(queries.shape),
                    "kshape": tuple(keys.shape),
                    "vshape": tuple(values.shape),
                    "qdtype": str(queries.dtype),
                    "kdtype": str(keys.dtype),
                    "vdtype": str(values.dtype),
                    "outdtype": str(out.dtype),
                }
            )
        return out

    def __enter__(self):
        self.orig = mx.fast.scaled_dot_product_attention
        mx.fast.scaled_dot_product_attention = self
        self.capturing = False
        self.calls = []
        return self

    def __exit__(self, *a):
        self.capturing = False
        mx.fast.scaled_dot_product_attention = self.orig
        return False


# ---------------------------------------------------------------------------
# harvest: re-run the m3 flow, capture the layer-3 SDPA call
# ---------------------------------------------------------------------------


def snap_arr(a):
    if a is None:
        return (None, None)
    return (npf(a), a.dtype)


def snap_cache(cache):
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append({"kind": "kv", "keys": snap_arr(c.keys), "values": snap_arr(c.values), "offset": c.offset})
        elif isinstance(c, ArraysCache):
            snap.append(
                {
                    "kind": "ac",
                    "cache": [snap_arr(x) for x in c.cache],
                    "left_padding": snap_arr(c.left_padding),
                    "lengths": snap_arr(c.lengths),
                }
            )
        else:
            raise ValueError(f"unknown cache entry type {type(c)!r}")
    return snap


def run_harvest():
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    lmap = ["attn" if isinstance(c, KVCache) else "gdn" for c in cache]
    print(f"layer map ({len(lmap)} layers): {lmap}", flush=True)

    logits, hidden = lm(
        mx.array([prompt_ids]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    lg = logits[0, -1]
    hidd = hidden[0, -1]
    draft = None
    prev_token = None
    out = []
    armed = False

    for step in range(MAX_TOKENS):
        if draft is None:
            t = int(mx.argmax(lg))
            if t == EOS:
                break
            out.append(t)
            emb = lm.model.embed_tokens(mx.array([[t]]))
            dh = lm.mtp(hidd, emb, mtp_cache)
            draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
            prev_token = t
            if prev_token == S62_PREV and not armed:
                armed = True
                do_harvest(lm, cache, out, prev_token, draft, step)
        else:
            d = draft
            draft = None
            saved = [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if isinstance(c, ArraysCache)]
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, _h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d:
                out.append(t2)
                lg, hidd = l1, h1
            else:
                for i, conv, ssm, lengths in saved:
                    cache[i][0] = conv
                    cache[i][1] = ssm
                    cache[i].lengths = lengths
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                out.append(t2)
                logits, hidden = lm(
                    mx.array([[prev_token]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                logits, hidden = lm(
                    mx.array([[t2]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                lg, hidd = logits[0, -1], hidden[0, -1]

    if not armed:
        print(
            f"ARM NOT REACHED: n_out={len(out)} last_out={out[-8:]} "
            f"(flow did not reach prev_token={S62_PREV}; not forcing the arm)",
            flush=True,
        )
        sys.exit(2)


def do_harvest(lm, cache, out, prev_token, draft, step):
    n_out = len(out)
    print(f"ARM harvest: step={step} prev={prev_token} draft={draft} n_out={n_out} out[-8:]={out[-8:]}", flush=True)
    if n_out != EXPECT_N_OUT:
        print(f"ARM MISMATCH: expected n_out=={EXPECT_N_OUT}, got {n_out}", flush=True)
        sys.exit(2)
    if draft != S62_DRAFT:
        print(f"ARM MISMATCH: expected draft=={S62_DRAFT}, got {draft}", flush=True)
        sys.exit(2)
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            print(f"ARM attn offset layer={i}: {c.offset}", flush=True)

    for c in cache:
        if isinstance(c, KVCache):
            mx.eval([c.keys, c.values])
        else:
            mx.eval([x for x in c.cache if x is not None])
    snap = snap_cache(cache)
    print(f"snapshot taken: {len(snap)} entries", flush=True)

    # ---- variant A: one T=2 verify forward, hooking the SDPA call --------
    print(f"== variant A: T=2 verify forward [[{S62_PREV}, {S62_DRAFT}]] (SDPA hooked) ==", flush=True)
    with SDPAHook() as hook:
        hook.capturing = True
        logits_a, hidden_a = lm(
            mx.array([[S62_PREV, S62_DRAFT]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval([logits_a, hidden_a])
    t2 = int(mx.argmax(logits_a[0, 0]))
    act = "ACC" if t2 == S62_DRAFT else "REJ"
    print(f"variant A act={act} (t2={t2} draft={S62_DRAFT})", flush=True)
    print(f"SDPA calls captured during variant A: {len(hook.calls)}", flush=True)
    if len(hook.calls) < 1:
        print("NO SDPA CALL CAPTURED - unexpected", flush=True)
        sys.exit(3)
    l3 = hook.calls[0]  # the first attention layer is layer 3
    print(f"layer-3 SDPA: qshape={l3['qshape']} kshape={l3['kshape']} vshape={l3['vshape']} scale={l3['scale']} mask={l3['mask']!r}", flush=True)
    print(f"layer-3 SDPA dtypes: q={l3['qdtype']} k={l3['kdtype']} v={l3['vdtype']} out={l3['outdtype']}", flush=True)
    if l3["qshape"][2] != 2:
        print(f"UNEXPECTED: layer-3 SDPA query S={l3['qshape'][2]} (expected 2)", flush=True)
        sys.exit(3)

    # ---- verify the captured k/v against the m3 pkl layer-3 state --------
    m3_k = m3_v = None
    if os.path.exists(M3_PKL):
        with open(M3_PKL, "rb") as f:
            m3 = pickle.load(f)
        for kind, off, k, v in m3["ha"]:
            if kind == "attn" and off == 89:
                m3_k, m3_v = k, v
                break
    if m3_k is not None:
        st, n, tot, mx_ = cmp_status(l3["k"], m3_k)
        print(f"captured k vs m3 pkl layer-3 k: {st} n={n}/{tot}", flush=True)
        st, n, tot, mx_ = cmp_status(l3["v"], m3_v)
        print(f"captured v vs m3 pkl layer-3 v: {st} n={n}/{tot}", flush=True)
    else:
        print("m3 pkl not found or no layer-3 entry; skipping k/v cross-check", flush=True)

    with open(M4_PKL, "wb") as f:
        pickle.dump(
            {
                "q": l3["q"],
                "k": l3["k"],
                "v": l3["v"],
                "out": l3["out"],
                "scale": l3["scale"],
                "mask": l3["mask"],
                "qshape": l3["qshape"],
                "kshape": l3["kshape"],
                "vshape": l3["vshape"],
                "qdtype": l3["qdtype"],
                "kdtype": l3["kdtype"],
                "vdtype": l3["vdtype"],
                "outdtype": l3["outdtype"],
                "act": act,
                "n_out": n_out,
                "out_tail": out[-8:],
                "n_sdpa_calls": len(hook.calls),
            },
            f,
        )
    print(f"m4 captures saved -> {M4_PKL}", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# ab: direct SDPA A/B on the captured tensors (no model)
# ---------------------------------------------------------------------------


def sdpa(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def run_ab():
    if not os.path.exists(M4_PKL):
        print(f"missing {M4_PKL}; run MODE=harvest first", flush=True)
        sys.exit(3)
    with open(M4_PKL, "rb") as f:
        c = pickle.load(f)
    q = c["q"]  # [1, 16, 2, 256]
    k = c["k"]  # [1, 4, 89, 256]
    v = c["v"]  # [1, 4, 89, 256]
    out_cap = c["out"]  # [1, 16, 2, 256]
    scale = c["scale"]
    mask_cap = c["mask"]
    qdt = getattr(mx, c.get("qdtype", "bfloat16").replace("mlx.core.", "").replace(".", "_"))
    kdt = getattr(mx, c.get("kdtype", "bfloat16").replace("mlx.core.", "").replace(".", "_"))
    vdt = getattr(mx, c.get("vdtype", "bfloat16").replace("mlx.core.", "").replace(".", "_"))
    print(f"loaded: qshape={c['qshape']} kshape={c['kshape']} vshape={c['vshape']} scale={scale} mask={mask_cap!r} act={c['act']}", flush=True)
    print(f"dtypes: q={c.get('qdtype')} k={c.get('kdtype')} v={c.get('vdtype')} out={c.get('outdtype')}", flush=True)

    qm = of(q, qdt)
    km = of(k, kdt)
    vm = of(v, vdt)

    # ---- (1) A: single call, S=2, causal mask ----------------------------
    A = sdpa(qm, km, vm, scale, "causal")
    A_np = npf(A)
    print("== (1) A: S=2 causal re-run vs captured SDPA output (sanity) ==", flush=True)
    a_ok = cmp_report("A re-run == captured out", A_np, out_cap)
    A_row0 = A_np[0, :, 0, :]
    A_row1 = A_np[0, :, 1, :]

    # ---- (2) B1: single call, S=1, row [279] (control) -------------------
    q279 = qm[:, :, 0:1, :]  # (1,16,1,256)
    k88 = km[:, :, :88, :]  # (1,4,88,256)
    v88 = vm[:, :, :88, :]  # (1,4,88,256)
    B1 = sdpa(q279, k88, v88, scale, None)
    B1_np = npf(B1)[0, :, 0, :]
    print("== (2) B1: S=1 unmasked row[279] (control) vs A row 0 ==", flush=True)
    b1_ok = cmp_report("B1 == A row 0", B1_np, A_row0)

    # ---- (3) B2: single call, S=1, row [3545], 89 keys -------------------
    q3545 = qm[:, :, 1:2, :]  # (1,16,1,256)
    B2 = sdpa(q3545, km, vm, scale, None)
    B2_np = npf(B2)[0, :, 0, :]
    print("== (3) B2: S=1 unmasked row[3545] (89 keys) vs A row 1 ==", flush=True)
    b2_match = cmp_report("B2 == A row 1", B2_np, A_row1)

    print("== core A/B verdict ==", flush=True)
    print(f"A re-run reproduces capture: {'YES' if a_ok else 'NO'}", flush=True)
    print(f"B1 (control) == A row 0:    {'YES' if b1_ok else 'NO'}", flush=True)
    print(f"B2 == A row 1:              {'YES' if b2_match else 'NO  <-- the m3 divergence is reproduced by direct SDPA'}", flush=True)

    # ---- (4) bisection: single-variable A/Bs -----------------------------
    print("== (4) bisection (single-variable A/Bs) ==", flush=True)
    results = {}

    # (a) mask=None on both: unmasked S=2 vs unmasked S=1 (2 keys)
    unmasked_S2 = npf(sdpa(qm, km, vm, scale, None))[0, :, 1, :]
    results["a_unmasked_S2_row1_vs_B2"] = cmp_status(unmasked_S2, B2_np)
    print(f"(a) unmasked S=2 row1 vs B2 (unmasked S=1): {results['a_unmasked_S2_row1_vs_B2'][0]}", flush=True)

    # (b) explicit full-True mask vs causal (both S=2, row 1 unmasked in both)
    full_true = mx.ones((1, 1, 2, k.shape[2]), dtype=mx.bool_)
    full_S2 = npf(sdpa(qm, km, vm, scale, full_true))[0, :, 1, :]
    results["b_fullTrue_S2_row1_vs_causal_S2_row1"] = cmp_status(full_S2, A_row1)
    print(f"(b) full-True S=2 row1 vs causal S=2 row1: {results['b_fullTrue_S2_row1_vs_causal_S2_row1'][0]}", flush=True)

    # (c) S=2 with row 0 zero-padded q (attend set identical)
    qz = mx.concatenate([mx.zeros_like(qm[:, :, 0:1, :]), qm[:, :, 1:2, :]], axis=2)  # (1,16,2,256)
    zero_row0 = npf(sdpa(qz, km, vm, scale, "causal"))[0, :, 1, :]
    results["c_zeroRow0_causal_S2_row1_vs_A_row1"] = cmp_status(zero_row0, A_row1)
    print(f"(c) zero-row0 causal S=2 row1 vs A row1: {results['c_zeroRow0_causal_S2_row1_vs_A_row1'][0]}", flush=True)

    # (d) different seq-len padding: S=4 with 2 valid (rows 2,3 zero + causal)
    z1 = mx.zeros_like(qm[:, :, 0:1, :])
    pad_q = mx.concatenate([qm[:, :, 0:1, :], qm[:, :, 1:2, :], z1, z1], axis=2)  # (1,16,4,256)
    pad_S4 = npf(sdpa(pad_q, km, vm, scale, "causal"))[0, :, 1, :]
    results["d_S4pad_causal_row1_vs_A_row1"] = cmp_status(pad_S4, A_row1)
    print(f"(d) S=4 (2 valid) causal row1 vs A row1: {results['d_S4pad_causal_row1_vs_A_row1'][0]}", flush=True)

    # (e) different batch: B=2, S=1 each (batch the two rows, 89 keys, unmasked)
    q_batch = qm.reshape(2, 16, 1, 256)  # batch 0 = row0 (279), batch 1 = row1 (3545)
    k_b = mx.repeat(km, 2, axis=0)  # (2,4,89,256)
    v_b = mx.repeat(vm, 2, axis=0)  # (2,4,89,256)
    batched = npf(sdpa(q_batch, k_b, v_b, scale, None))  # (2,16,1,256)
    results["e_batched_row1_vs_B2"] = cmp_status(batched[1, :, 0, :], B2_np)
    print(f"(e) batched (B=2,S=1) row1 vs B2: {results['e_batched_row1_vs_B2'][0]}", flush=True)

    # ---- (6) synthetic q/k/v (random, same shapes) -----------------------
    print("== (6) synthetic q/k/v (random, same shapes) ==", flush=True)
    rng = np.random.default_rng(0)
    sq = mx.array(rng.standard_normal(c["qshape"]).astype(np.float32)).astype(qdt)
    sk = mx.array(rng.standard_normal(c["kshape"]).astype(np.float32)).astype(kdt)
    sv = mx.array(rng.standard_normal(c["vshape"]).astype(np.float32)).astype(vdt)
    sA = npf(sdpa(sq, sk, sv, scale, "causal"))[0, :, 1, :]
    sB2 = npf(sdpa(sq[:, :, 1:2, :], sk, sv, scale, None))[0, :, 0, :]
    s_match = cmp_status(sA, sB2)
    print(f"synthetic causal S=2 row1 vs unmasked S=1: {s_match[0]}", flush=True)
    print(f"synthetic verdict: {'STRUCTURAL (reproduces on random data)' if s_match[0] == 'DIFF' else 'NOT reproduced on random data (data-dependent)'}", flush=True)

    # ---- summary --------------------------------------------------------
    print("== m4 A/B summary ==", flush=True)
    print(f"capture sanity (A re-run == capture): {'PASS' if a_ok else 'FAIL'}", flush=True)
    print(f"control (B1 == A row 0):              {'PASS' if b1_ok else 'FAIL'}", flush=True)
    print(f"divergence (B2 vs A row 1):          {'reproduced' if not b2_match else 'NOT reproduced'}", flush=True)
    print(f"synthetic (structural):              {'YES' if s_match[0] == 'DIFF' else 'NO'}", flush=True)
    return {
        "a_ok": a_ok,
        "b1_ok": b1_ok,
        "b2_match": b2_match,
        "results": results,
        "synthetic": s_match,
    }


def main():
    if MODE == "harvest":
        run_harvest()
    elif MODE == "ab":
        run_ab()
    else:
        print(f"unknown MODE={MODE} (use harvest or ab)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
