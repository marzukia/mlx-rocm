#!/usr/bin/env python3
"""PLANE-EXACTNESS PROBE (the ONE ticket per op, 2026-11-05).

Question under test, bit-for-bit: is the GDN recurrent+conv state at row k
(k=1,2,3; the state after k of the 4 verify rows) of a BATCHED S=4 verify
bit-identical to the state after k SEQUENTIAL S=1 forwards over the same
tokens [t0, t1, t2, t3]?

Method (ONE GPU window, code-first, measure second):
  - Both arms run in ONE process, ONE model load, under the 13-var
    config-of-record GDN env (MLX_ROCM_GDN_TAPE=1 + MLX_ROCM_GDN_FUSE=1:
    S<=4 forwards route to gdn_fused_step_tape; the batched arm is the
    TMAX=4 instantiation, the sequential arm the TMAX=1 instantiation).
  - Base state: the canonical 322-token anchor prompt (greedy-extended to
    exactly 322 tokens) is PREFILLED (one S=322 forward, the non-tape GDN
    path) into a FRESH cache per arm. The prefill is identical across arms
    (same weights, same tokens, same kernel) -> identical base state. A
    built-in determinism self-check re-runs the prefill into a second fresh
    cache and requires the base GDN state to be bit-identical.
  - BATCHED arm: prefill(322) -> ONE S=4 forward over [t0..t3] -> the
    per-row state tape (linear_attn.last_tape = (state_tape,
    conv_state_tape)) is the batched plane. Capture rows 0..3.
  - SEQUENTIAL arm: prefill(322) -> 4x S=1 forwards over [t0..t3] -> after
    each forward, last_tape (a 1-row tape) is the sequential state. Capture
    row 0 of each.
  - Compare, ALL IN MLX (no numpy; native-dtype `!=` is exact bitwise
    equality; the tape arrays are fresh kernel outputs kept alive by
    reference, so no bf16 byte-extraction is needed):
        identical  = sum(a != b) == 0
        maxdiff    = max(|a.astype(f32) - b.astype(f32)|)
        first-idx  = argmax((a != b).flatten())  (over the full tensor)

The four boundaries (labeled by the token content they cover) map to the
ticket's k=1,2,3 as:  b0 = after [t0], b1 = after [t0,t1], b2 = after
[t0,t1,t2]  (b3 = after [t0,t1,t2,t3] is the full-accept boundary). Three
rounds.

Decision rule (applied in REPORT-planes.md, not here):
  BIT-IDENTICAL all 3 rounds  -> PLANES VIABLE
  mismatch at row 1 all 3     -> RECURRENCE-ORDER BUG
  any other pattern           -> PATTERN NON-CANONICAL
"""

import os
import time

import mlx.core as mx
from mlx_lm import load

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = os.environ.get(
    "MTP6_PROMPT",
    "Explain in great detail how a modern smartphone works, from the hardware "
    "components to the operating system, covering every subsystem. Be exhaustive.",
)
# 322-token canonical anchor context (greedy-extended to exactly this length).
BASE_LEN = int(os.environ.get("PLANE_BASE_LEN", "322"))
# 4 fixed verify rows (deterministic; bit-exactness is token-identity
# independent). All in-vocab, none == EOS.
VTOK = [int(x) for x in os.environ.get("PLANE_VTOK", "100 200 300 400").split()]
assert len(VTOK) == 4, VTOK
NROUNDS = int(os.environ.get("PLANE_ROUNDS", "3"))
EOS = 248044

# --- 13-var config-of-record GDN env (batched + sequential arms) ------------
# GDN tape + fuse on: S<=4 forwards -> gdn_fused_step_tape (TMAX=4 batched /
# TMAX=1 sequential). The 3 shape levers are presence-gated -> UNSET for the
# per-row path (NOT =0); the 2 value-gated opt-outs are =0.
for _v in ("MLX_ROCM_QMV_ROWS", "MLX_ROCM_QMV_ROWS_MIN_N", "MLX_ROCM_QMM_M3"):
    os.environ.pop(_v, None)
os.environ["MLX_QMV_NO_MR"] = "1"
os.environ["MLX_QMV_MR_LX"] = "0"
os.environ["MLX_ROCM_SDPA_S4VA"] = "0"
os.environ["MLX_ROCM_GDN_TAPE"] = "1"
os.environ["MLX_ROCM_GDN_FUSE"] = "1"
os.environ["MLX_ROCM_GDN_T3"] = "1"
# No draft head / no MTP module is exercised here (pure trunk GDN probe), so
# the head probes are irrelevant and left unset.


def gdn_layer_ids(lm):
    return [i for i, layer in enumerate(lm.model.layers) if hasattr(layer, "linear_attn")]


def norm_shape(a):
    """Squeeze a leading batch dim of 1 so both arms' tensors share a shape.

    Note: mx.eval(single_array) returns None in this mlx-rocm build, so we
    use the array directly (it computes on demand; a held reference is a
    stable fresh allocation, verified by a separate stability probe).
    """
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]
    return a


def cmp_pair(a, b):
    """Exact element compare in the native dtype + f32 maxdiff + first idx."""
    a = norm_shape(a)
    b = norm_shape(b)
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    ne = a != b
    n_diff = int(mx.sum(ne))
    if n_diff == 0:
        return True, 0, 0.0, -1
    maxdiff = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
    first = int(mx.argmax(ne.flatten()))
    return False, n_diff, maxdiff, first


def build_base_context(lm, tok, prompt, base_len):
    """Canonical anchor context: prompt tokens, greedy-extended to base_len."""
    ids = list(tok.encode(prompt))
    scratch = lm.make_cache()
    lm(mx.array([ids]), cache=scratch)
    while len(ids) < base_len:
        logits, _ = lm(
            mx.array([[ids[-1]]]),
            cache=scratch,
            return_hidden=True,
            return_prenorm=True,
        )
        t = int(mx.argmax(logits[0, -1]))
        if t == EOS:
            t = 42  # degenerate stop: pad (bit-exactness is token-independent)
        ids.append(t)
    return ids[:base_len]


def prefill(lm, cache, ctx_ids):
    lm(mx.array([ctx_ids]), cache=cache)


def run_batched(lm, gids, ctx_ids, vtok):
    """One S=4 forward over the 4 verify rows; return per-layer per-row tapes."""
    cache = lm.make_cache()
    prefill(lm, cache, ctx_ids)
    lm(mx.array([[vtok[0], vtok[1], vtok[2], vtok[3]]]), cache=cache)
    rec, cstab = {}, {}
    for i in gids:
        la = lm.model.layers[i].linear_attn
        st, cst = getattr(la, "last_tape", (None, None))
        if st is None:
            raise RuntimeError(f"GDN tape not materialized (batched) layer {i}")
        # hold references directly (stable fresh allocations; no mx.eval - it
        # returns None in this build and would drop the array).
        rec[i] = [st[k] for k in range(4)]
        cstab[i] = [cst[k] for k in range(4)]
    return rec, cstab


def run_sequential(lm, gids, ctx_ids, vtok):
    """4x S=1 forwards over the 4 verify rows; per-forward 1-row tape."""
    cache = lm.make_cache()
    prefill(lm, cache, ctx_ids)
    rec, cstab, leaf_rec, leaf_conv = {}, {}, {}, {}
    for j in range(4):
        lm(mx.array([[vtok[j]]]), cache=cache)
        for i in gids:
            la = lm.model.layers[i].linear_attn
            st, cst = getattr(la, "last_tape", (None, None))
            if st is None:
                raise RuntimeError(f"GDN tape not materialized (seq) layer {i}")
            # hold references directly (stable; no mx.eval in this build).
            rec.setdefault(i, []).append(st[0])
            cstab.setdefault(i, []).append(cst[0])
            # cache leaves (ArraysCache: [0]=conv, [1]=recurrent) - the
            # secondary cross-check target for the per-row tape slice.
            leaf_rec.setdefault(i, []).append(cache[i][1])
            leaf_conv.setdefault(i, []).append(cache[i][0])
    return rec, cstab, leaf_rec, leaf_conv


def fold(gids, b, Brec, Srec, Bcst, Scst):
    """Aggregate (identical, n_diff, maxdiff, first_idx) over GDN layers."""
    ids, nds, mds, fis = True, 0, 0.0, -1
    for i in gids:
        idn, nd, md, fi = cmp_pair(Brec[i][b], Srec[i][b])
        ids = ids and idn
        nds += nd
        mds = max(mds, md)
        fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
    for i in gids:
        idn, nd, md, fi = cmp_pair(Bcst[i][b], Scst[i][b])
        ids = ids and idn
        nds += nd
        mds = max(mds, md)
        fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
    return ids, nds, mds, fis


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    gids = gdn_layer_ids(lm)
    t_load = time.time() - t0
    print(f"LOAD={t_load:.2f} GDN_LAYERS={len(gids)} IDS={gids}", flush=True)

    print(f"PROMPT={PROMPT!r}", flush=True)
    ctx = build_base_context(lm, tok, PROMPT, BASE_LEN)
    prompt_n = len(list(tok.encode(PROMPT)))
    print(
        f"BASE_CTX_LEN={len(ctx)} PROMPT_TOK={prompt_n} EXT={len(ctx) - prompt_n}",
        flush=True,
    )
    print(f"VTOK={VTOK}", flush=True)

    # ---- Determinism self-check: prefill into two fresh caches ------------
    c1, c2 = lm.make_cache(), lm.make_cache()
    prefill(lm, c1, ctx)
    prefill(lm, c2, ctx)
    det_ok, det_n = True, 0
    for i in gids:
        for slot in (0, 1):
            a, b = c1[i][slot], c2[i][slot]
            if a is not None and b is not None:
                idn, nd, _md, _fi = cmp_pair(a, b)
                det_n += nd
                det_ok = det_ok and idn
    print(f"DETERMINISM_SELF_CHECK={det_ok} NDIFF={det_n}", flush=True)
    if not det_ok:
        print(
            "WARNING: prefill not deterministic across fresh caches; base-state identity across arms is unproven",
            flush=True,
        )

    results = []
    for r in range(1, NROUNDS + 1):
        print(f"=== ROUND {r}/{NROUNDS} ===", flush=True)
        b_rec, b_cst = run_batched(lm, gids, ctx, VTOK)
        s_rec, s_cst, s_leaf_rec, s_leaf_conv = run_sequential(lm, gids, ctx, VTOK)
        for b in range(4):
            # combined per-boundary summary (both planes, all GDN layers)
            results.append((r, b, "both") + fold(gids, b, b_rec, s_rec, b_cst, s_cst))
            # recurrent plane
            ids, nds, mds, fis = True, 0, 0.0, -1
            for i in gids:
                idn, nd, md, fi = cmp_pair(b_rec[i][b], s_rec[i][b])
                ids = ids and idn
                nds += nd
                mds = max(mds, md)
                fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
            results.append((r, b, "recurrent", ids, nds, mds, fis))
            # conv plane
            ids, nds, mds, fis = True, 0, 0.0, -1
            for i in gids:
                idn, nd, md, fi = cmp_pair(b_cst[i][b], s_cst[i][b])
                ids = ids and idn
                nds += nd
                mds = max(mds, md)
                fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
            results.append((r, b, "conv", ids, nds, mds, fis))
            # cross-checks against the sequential cache leaves (secondary)
            ids, nds, mds, fis = True, 0, 0.0, -1
            for i in gids:
                idn, nd, md, fi = cmp_pair(b_rec[i][b], s_leaf_rec[i][b])
                ids = ids and idn
                nds += nd
                mds = max(mds, md)
                fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
            results.append((r, b, "rec_vs_leaf", ids, nds, mds, fis))
            ids, nds, mds, fis = True, 0, 0.0, -1
            for i in gids:
                idn, nd, md, fi = cmp_pair(b_cst[i][b], s_leaf_conv[i][b])
                ids = ids and idn
                nds += nd
                mds = max(mds, md)
                fis = fi if (fi >= 0 and (fis < 0 or fi < fis)) else fis
            results.append((r, b, "conv_vs_leaf", ids, nds, mds, fis))
        print(f"ROUND {r} done", flush=True)

    # ---- Report ----------------------------------------------------------
    print("\n=== PLANE EXACTNESS RESULTS ===", flush=True)
    print(
        f"{'round':-6s} {'bndry':-7s} {'plane':-13s} "
        f"{'identical':-10s} {'n_diff':-9s} {'maxdiff':-12s} {'first_idx':-9s}",
        flush=True,
    )
    for r, b, plane, ids, nds, mds, fis in results:
        print(
            f"{r:-6d} b{b:-6d} {plane:-13s} {'YES' if ids else 'NO':-10s} {nds:-9d} {mds:-12.3e} {fis:-9d}",
            flush=True,
        )
    # verdict helper (the DECISION RULE is applied in REPORT-planes.md)
    primary = [x for x in results if x[2] in ("recurrent", "conv")]
    all_ident = all(x[3] for x in primary)
    row1 = [x for x in primary if x[1] == 0]
    row1_all_mismatch = all((not x[3]) for x in row1)
    row1_any_ident = any(x[3] for x in row1)
    print(
        f"\nVERDICT_HELPER all_primary_identical={all_ident} "
        f"row1_all_mismatch={row1_all_mismatch} "
        f"row1_any_identical={row1_any_ident}",
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
