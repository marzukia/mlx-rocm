#!/usr/bin/env python3
"""m8 round 2: locate the EXACT diverging structure in the flow leak.

The round-1 bisection (m8-mtp148-bisect.py) had a probe bug: instance-level
``mod.__call__ = w`` hooks are silently dead for mlx nn.Modules (C-level
tp_call bypasses the instance dict), so k_proj/k_norm/rope "clean" results
were never measured. This probe replaces those with wrapper modules
(a fresh nn.Module whose class-level __call__ always fires), so every
captured component is real.

Reproduces the mtp-bench3 mode-2 flow exactly and forks at the FIRST
state-trace leak step (verify appending token index 47, d=24051, reject):

  A  = S=2 [prev, d]
  B1 = S=1 [prev]   (from the pre-verify state)
  B2 = S=1 [d]      (from the state after B1)

Per-component comparison (numpy forced-eval):
  A row 0 vs B1 row 0 (prev row), A row 1 vs B2 row 0 (draft row)

Captured per forward (wrapper-based, in+out):
  - every decoder layer: input x + output (class-level __call__ hook)
  - every decoder layer: input_layernorm + post_attention_layernorm
  - FA layers: k_proj, k_norm, rope (and q_proj/q_norm/v_proj at FA15)
  - mx.fast GDN function hooks (gdn_prep / gated_delta_update / norm_gated)

Run ON the build VM with the venv-d python + LD_LIBRARY_PATH, under flock.
"""

import os

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
FORK_TOKEN_IDX = 47  # first state-trace leak (d=24051, reject)
EOS = 248044
MAX_TOKENS = 60


def npf(a):
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def snap_all(cache, mtp_cache):
    gdn, kv, mtp = [], [], None
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            kv.append((i, c.keys.dtype, npf(c.keys), c.values.dtype, npf(c.values), c.offset))
        else:
            ln = c.lengths
            gdn.append((i, c[0].dtype, npf(c[0]), c[1].dtype, npf(c[1]), npf(ln) if ln is not None else None))
    if mtp_cache:
        c = mtp_cache[0]
        if c.keys is not None:
            mtp = (c.keys.dtype, npf(c.keys), c.values.dtype, npf(c.values), c.offset)
    return gdn, kv, mtp


def restore_all(cache, mtp_cache, snap):
    gdn, kv, mtp = snap
    for i, dt0, cc, dt1, ss, ln in gdn:
        cache[i][0] = mx.array(cc, dtype=dt0)
        cache[i][1] = mx.array(ss, dtype=dt1)
        cache[i].lengths = mx.array(ln, dtype=cache[i].lengths.dtype) if ln is not None else None
    for i, dtk, kk, dtv, vv, off in kv:
        cache[i].keys = mx.array(kk, dtype=dtk)
        cache[i].values = mx.array(vv, dtype=dtv)
        cache[i].offset = off
    if mtp is not None and mtp_cache:
        dtk, kk, dtv, vv, off = mtp
        mtp_cache[0].keys = mx.array(kk, dtype=dtk)
        mtp_cache[0].values = mx.array(vv, dtype=dtv)
        mtp_cache[0].offset = off


class Capture:
    def __init__(self):
        self.mode = None
        self.seq = {}
        self.idx = {}
        self.raw = {}

    def set_mode(self, mode):
        self.mode = mode
        self.idx = {}
        self.gdn_call = 0

    def put(self, key, arr):
        if self.mode is None:
            return
        e = self.seq.setdefault(key, {})
        if self.mode not in e:
            e[self.mode] = npf(arr)


CAP = Capture()


class W(mnn.Module):
    """Wrapper module: class-level __call__ (always fires), captures in+out."""

    def __init__(self, inner, key):
        super().__init__()
        self.inner = inner
        self.key = key

    def __call__(self, *args, **kwargs):
        mode = CAP.mode
        idx = 0
        if mode is not None:
            idx = CAP.idx.get(self.key, 0)
            CAP.idx[self.key] = idx + 1
            if args:
                CAP.put(f"{self.key}_{idx}_in", args[0])
        out = self.inner(*args, **kwargs)
        if mode is not None:
            CAP.put(f"{self.key}_{idx}_out", out)
            if args:
                CAP.raw[f"{self.key}_{idx}_{mode}"] = args[0]
                if kwargs:
                    CAP.raw[f"{self.key}_{idx}_{mode}_kw"] = dict(kwargs)
        return out


def main():
    import mlx_lm.models.qwen3_next as qn

    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    lg = logits[0, -1]
    h = hidden[0, -1]

    # ---------- capture machinery ----------
    CAP.gdn_call = 0
    orig_gdn_prep = mx.fast.gdn_prep

    def gdn_prep_wrap(*a, **k):
        out = orig_gdn_prep(*a, **k)
        if CAP.mode is not None:
            i = CAP.gdn_call
            CAP.gdn_call += 1
            CAP.put(f"gp{i}_q", a[0])
            CAP.put(f"gp{i}_k", a[1])
            CAP.put(f"gp{i}_v", a[2])
            qn_, kn_, vc_, beta_, g_ = out
            CAP.put(f"gp{i}_qn", qn_)
            CAP.put(f"gp{i}_kn", kn_)
            CAP.put(f"gp{i}_vc", vc_)
            CAP.put(f"gp{i}_beta", beta_)
            CAP.put(f"gp{i}_g", g_)
        return out

    orig_gdu = mx.fast.gated_delta_update

    def gdu_wrap(*a, **k):
        out = orig_gdu(*a, **k)
        if CAP.mode is not None:
            i = CAP.gdn_call - 1
            CAP.put(f"gdu{i}_out", out[0])
            CAP.put(f"gdu{i}_state", out[1])
        return out

    orig_ng = mx.fast.gdn_norm_gated

    def ng_wrap(*a, **k):
        out = orig_ng(*a, **k)
        if CAP.mode is not None:
            i = CAP.gdn_call - 1
            CAP.put(f"ng{i}_out", out)
        return out

    mx.fast.gdn_prep = gdn_prep_wrap
    mx.fast.gated_delta_update = gdu_wrap
    mx.fast.gdn_norm_gated = ng_wrap
    CAP.gdn_call = 0

    layers = lm.model.layers
    id_to_idx = {id(layers[i]): i for i in range(len(layers))}
    gdn_layer_ids = [i for i, ly in enumerate(layers) if hasattr(ly, "linear_attn")]
    fa_ids = [i for i, ly in enumerate(layers) if hasattr(ly, "self_attn")]
    layer_cls = type(layers[0])
    orig_layer = layer_cls.__call__

    def hooked_layer(self, *args, **kwargs):
        out = orig_layer(self, *args, **kwargs)
        i = id_to_idx.get(id(self))
        if i is not None and CAP.mode is not None:
            o = out[0] if isinstance(out, tuple) else out
            CAP.put(f"L{i}_out", o)
            x0 = args[0]
            CAP.put(f"L{i}_xin", x0 if isinstance(x0, mx.array) else mx.array(x0))
        return out

    layer_cls.__call__ = hooked_layer

    # wrapper-based component hooks (in+out) — all layers' norms
    for i, ly in enumerate(layers):
        ly.input_layernorm = W(ly.input_layernorm, f"L{i}_input_ln")
        ly.post_attention_layernorm = W(ly.post_attention_layernorm, f"L{i}_post_ln")

    # FA layers: k_proj / k_norm / rope wrappers
    fa_id_to_idx = {id(layers[i].self_attn): i for i in fa_ids}
    for i in fa_ids:
        at = layers[i].self_attn
        at.k_proj = W(at.k_proj, f"FA{i}_k_proj")
        at.k_norm = W(at.k_norm, f"FA{i}_k_norm")
        at.rope = W(at.rope, f"FA{i}_rope")
        if i == 15:
            at.q_proj = W(at.q_proj, "FA15_q_proj")
            at.q_norm = W(at.q_norm, "FA15_q_norm")
            at.v_proj = W(at.v_proj, "FA15_v_proj")

    # per-layer sdpa capture (class-level attention __call__ sets current idx)
    CUR = {"fa": None}
    attn_cls = qn.Qwen3NextAttention
    orig_attn = attn_cls.__call__

    def attn_hook(self, *a, **k):
        CUR["fa"] = fa_id_to_idx.get(id(self))
        return orig_attn(self, *a, **k)

    attn_cls.__call__ = attn_hook

    orig_sdpa = qn.scaled_dot_product_attention

    def sdpa_wrap(queries, keys, values, cache=None, scale=1.0, mask=None):
        out = orig_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask)
        if CAP.mode is not None and CUR["fa"] is not None:
            fi = CUR["fa"]
            CAP.put(f"FA{fi}_sdpa_q", queries)
            CAP.put(f"FA{fi}_sdpa_k", keys)
            CAP.put(f"FA{fi}_sdpa_v", values)
            CAP.put(f"FA{fi}_scale", float(scale))
            CAP.put(f"FA{fi}_sdpa_out", out)
        return out

    qn.scaled_dot_product_attention = sdpa_wrap

    # ---------- mode-2 flow (exact mtp-bench3) ----------
    def save_gdn(cache):
        return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]

    def restore_gdn(cache, saved):
        for i, conv, ssm, lengths in saved:
            cache[i][0] = conv
            cache[i][1] = ssm
            cache[i].lengths = lengths

    def save_mtp(mtp_cache):
        c = mtp_cache[0]
        if c.keys is None:
            return (None, None, c.offset)
        return (npf(c.keys), npf(c.values), c.offset)

    def restore_mtp(mtp_cache, saved):
        keys, values, offset = saved
        c = mtp_cache[0]
        if keys is not None:
            c.keys = mx.array(keys)
            c.values = mx.array(values)
        c.offset = offset

    out = []
    accepted = 0
    verifies = 0
    draft = None
    prev_token = None
    mtp_saved = None
    fork_done = False

    def fork_probe(d, prev_token):
        nonlocal fork_done
        snap_pre = snap_all(cache, mtp_cache)
        # --- Arm A: S=2 [prev, d] ---
        CAP.set_mode("A")
        logitsA, hiddenA = lm(mx.array([[prev_token, d]]), cache=cache, return_hidden=True, return_prenorm=True)
        l0A, _ = logitsA[0, 0], hiddenA[0, 0]
        l1A, h1A = logitsA[0, 1], hiddenA[0, 1]
        t2 = int(mx.argmax(l0A))
        CAP.put("LOGITS_r1", l1A)
        # ---- early direct (right after A forward) ----
        xA_early = CAP.raw.get("FA15_rope_1_A")
        kwA = CAP.raw.get("FA15_rope_1_A_kw")
        if xA_early is not None:
            rAe = mx.fast.rope(xA_early, 64, traditional=False, base=10000000.0, scale=1.0, offset=73)
            mx.eval(rAe)
            flowA = np.asarray(CAP.seq["FA15_rope_1_out"]["A"])
            de = flowA[:, :, 1:2, :] - np.asarray(rAe.astype(mx.float32))[:, :, 1:2, :]
            print(
                f"  EARLY-direct vs FLOW-out A-r1: "
                f"n={int(np.count_nonzero(de != 0))} "
                f"maxabs={float(np.abs(de).max()):.3e}",
                flush=True,
            )
            CAP.early_xA = npf(xA_early)
        if kwA:
            print(f"  FLOW offset kw: A={kwA.get('offset')}", flush=True)
        snap_Apost = snap_all(cache, mtp_cache)
        # --- restore pre-verify; Arm B1: S=1 [prev] ---
        restore_all(cache, mtp_cache, snap_pre)
        CAP.set_mode("B1")
        logitsB1, hiddenB1 = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
        # --- Arm B2: S=1 [d] ---
        CAP.set_mode("B2")
        logitsB2, hiddenB2 = lm(mx.array([[d]]), cache=cache, return_hidden=True, return_prenorm=True)
        snap_B2post = snap_all(cache, mtp_cache)
        CAP.mode = None
        report_fork(t2, l1A, logitsB2, snap_Apost, snap_B2post)
        rope_decompose(CAP.seq.get("FA15_k_norm_0_out", {}), "k-path")
        rope_decompose(CAP.seq.get("FA15_q_norm_0_out", {}), "q-path")
        # ---- direct T-dependence test on the FLOW's exact rope inputs ----
        if hasattr(CAP, "early_xA"):
            nowA = npf(CAP.raw.get("FA15_rope_1_A"))
            dd = nowA - CAP.early_xA
            print(
                f"  xA CONTENT drift A-time->late: "
                f"n={int(np.count_nonzero(dd != 0))} "
                f"maxabs={float(np.abs(dd).max()):.3e}",
                flush=True,
            )
        kwB = CAP.raw.get("FA15_rope_1_B2_kw")
        if kwB:
            print(f"  FLOW offset kw: B2={kwB.get('offset')}", flush=True)
        xA = CAP.raw.get("FA15_rope_1_A")
        xB = CAP.raw.get("FA15_rope_1_B2")
        if xA is not None and xB is not None:

            def fro(x, off):
                return mx.fast.rope(x, 64, traditional=False, base=10000000.0, scale=1.0, offset=off)

            rA = fro(xA, 73)
            rB = fro(xB, 74)
            rA2 = fro(xA, 73)
            rB2 = fro(xB, 74)
            xAc = mx.array(xA)  # C-contiguous copy (bf16 preserved)
            xBc = mx.array(xB)
            rAc = fro(xAc, 73)
            rBc = fro(xBc, 74)
            mx.eval(rA, rB, rA2, rB2, rAc, rBc)
            fA = np.asarray(rA.astype(mx.float32))
            fB = np.asarray(rB.astype(mx.float32))
            fA2 = np.asarray(rA2.astype(mx.float32))
            fB2 = np.asarray(rB2.astype(mx.float32))
            fAc = np.asarray(rAc.astype(mx.float32))
            fBc = np.asarray(rBc.astype(mx.float32))
            print(
                f"  DIRECT strided: A-vs-A2 n={int(np.count_nonzero(fA - fA2))} "
                f"B-vs-B2 n={int(np.count_nonzero(fB - fB2))}",
                flush=True,
            )
            d = fA[:, :, 1:2, :] - fB[:, :, 0:1, :]
            print(
                f"  DIRECT strided: A-r1 vs B-r0 n={int(np.count_nonzero(d != 0))} maxabs={float(np.abs(d).max()):.3e}",
                flush=True,
            )
            if np.count_nonzero(d != 0):
                for i in np.argwhere(d != 0)[:6]:
                    h, dd_ = int(i[0]), int(i[1])
                    print(f"    head={h} dim={dd_}: A={fA[0, h, 1, dd_]:.6f} B={fB[0, h, 0, dd_]:.6f}")
            dc = fAc[:, :, 1:2, :] - fBc[:, :, 0:1, :]
            print(
                f"  DIRECT contiguous: A-r1 vs B-r0 n={int(np.count_nonzero(dc != 0))} "
                f"maxabs={float(np.abs(dc).max()):.3e}",
                flush=True,
            )
            da = fA - fAc
            db = fB - fBc
            print(
                f"  DIRECT strided-vs-contiguous: A n={int(np.count_nonzero(da != 0))} "
                f"maxabs={float(np.abs(da).max()):.3e} "
                f"B n={int(np.count_nonzero(db != 0))} "
                f"maxabs={float(np.abs(db).max()):.3e}",
                flush=True,
            )
            # flow-captured out vs direct (determinism of the flow call)
            flowA = np.asarray(CAP.seq["FA15_rope_1_out"]["A"])
            n_flow_b = int(
                np.count_nonzero(np.asarray(CAP.seq["FA15_rope_1_out"]["B2"])[:, :, 0:1, :] - fB[:, :, 0:1, :])
            )
            print(
                f"  FLOW-out vs DIRECT: A-r1 n={int(np.count_nonzero(flowA[:, :, 1:2, :] - fA[:, :, 1:2, :]))} "
                f"B-r0 n={n_flow_b}",
                flush=True,
            )
        # continue the flow from the A post-state (reject path as usual)
        restore_all(cache, mtp_cache, snap_Apost)
        fork_done = True
        return t2, l1A, h1A

    def rowx(a, r, key):
        a = np.asarray(a, dtype=np.float32)
        if "_state" in key:
            return a[0].reshape(-1) if a.ndim >= 1 else a.reshape(-1)
        if a.ndim == 4 and ("_k_norm" in key or "_q_norm" in key):
            # (B, S, H, D): sequence is axis 1
            return a[0, r, :, :].reshape(-1) if a.shape[1] > r else None
        if a.ndim >= 3:
            # (B, S, ...) or (B, H, S, ...) — find the sequence axis
            if a.ndim == 4:
                # (B, H, S, D): S is axis 2
                return a[0, :, r, :].reshape(-1) if a.shape[2] > r else None
            return a[0, r].reshape(-1) if a.shape[1] > r else None
        if a.ndim == 2:
            return a[r].reshape(-1) if a.shape[0] > r else None
        return a.reshape(-1)

    def cmp_pair(name, ea, eb):
        if ea is None or eb is None:
            return None
        d = ea - eb
        n = int(np.count_nonzero(d != 0))
        ma = float(np.abs(d).max()) if d.size else 0.0
        return n, ma

    def rope_decompose(norm_entry, tag):
        # A row 1 (d @ pos 48, L=2 -> batched rope kernel) vs
        # B2 row 0 (d @ pos 48, L=1 -> single-1D rope kernel)
        if "A" not in norm_entry or "B2" not in norm_entry:
            print(f"  {tag}: no norm data", flush=True)
            return
        a = np.asarray(norm_entry["A"], dtype=np.float32)  # (1, 2, H, 256)
        b = np.asarray(norm_entry["B2"], dtype=np.float32)  # (1, 1, H, 256)
        if a.ndim != 4 or b.ndim != 4:
            print(f"  {tag}: unexpected shapes {a.shape} {b.shape}", flush=True)
            return
        at_ = a.transpose(0, 2, 1, 3)  # (1, H, 2, 256)
        bt_ = b.transpose(0, 2, 1, 3)  # (1, H, 1, 256)
        # ---- fused kernel direct A/B (dims=64, base=1e7, partial rotary) ----
        xA = mx.array(at_).astype(mx.bfloat16)
        xB = mx.array(bt_).astype(mx.bfloat16)
        rA = mx.fast.rope(xA, 64, traditional=False, base=10000000.0, scale=1.0, offset=75)
        rB = mx.fast.rope(xB, 64, traditional=False, base=10000000.0, scale=1.0, offset=76)
        mx.eval(rA, rB)
        rAf = np.asarray(rA.astype(mx.float32)).reshape(at_.shape)
        rBf = np.asarray(rB.astype(mx.float32)).reshape(bt_.shape)
        diff = rAf[:, :, 1:2, :] - rBf[:, :, 0:1, :]
        n = int(np.count_nonzero(diff != 0))
        ma = float(np.abs(diff).max()) if diff.size else 0.0
        print(f"  {tag} FUSED rope (A r1 vs B2 r0): n={n}/{diff.size} maxabs={ma:.3e}", flush=True)
        if n:
            for i in np.argwhere(diff != 0)[:4]:
                print(
                    f"    at {tuple(int(v) for v in i)}: "
                    f"A={rAf[tuple(int(v) for v in i + (1, 0))]:.6f} "
                    f"B={rBf[tuple(int(v) for v in i + (0, 0))]:.6f}"
                )
        # ---- explicit recompute (dims=64, half=32, base=1e7) ----
        half_ = 32
        lf = np.float32(np.log2(10000000.0))
        invf = np.array([np.exp2(-np.float32(i / half_) * lf) for i in range(half_)], dtype=np.float32)
        posA = (mx.arange(2, dtype=mx.int32) + 75).astype(mx.float32)
        posB = (mx.arange(1, dtype=mx.int32) + 76).astype(mx.float32)
        thA = posA[:, None] * mx.array(invf)[None, :]  # (2, 32)
        thB = posB[:, None] * mx.array(invf)[None, :]  # (1, 32)
        mx.eval(thA, thB)
        print(
            f"  {tag} explicit theta: "
            f"n={int(np.count_nonzero(np.asarray(thA)[1:2] - np.asarray(thB)[0:1]))} "
            f"maxabs={float(np.abs(np.asarray(thA)[1:2] - np.asarray(thB)[0:1]).max()):.3e}",
            flush=True,
        )
        cA = mx.cos(thA)
        sA = mx.sin(thA)
        cB = mx.cos(thB)
        sB = mx.sin(thB)
        mx.eval(cA, sA, cB, sB)
        print(
            f"  {tag} explicit cos: "
            f"n={int(np.count_nonzero(np.asarray(cA)[1:2] - np.asarray(cB)[0:1]))} "
            f"maxabs={float(np.abs(np.asarray(cA)[1:2] - np.asarray(cB)[0:1]).max()):.3e}",
            flush=True,
        )
        print(
            f"  {tag} explicit sin: "
            f"n={int(np.count_nonzero(np.asarray(sA)[1:2] - np.asarray(sB)[0:1]))} "
            f"maxabs={float(np.abs(np.asarray(sA)[1:2] - np.asarray(sB)[0:1]).max()):.3e}",
            flush=True,
        )
        x1a = mx.array(at_[:, :, 1:2, :half_]).astype(mx.float32)
        x2a = mx.array(at_[:, :, 1:2, half_ : 2 * half_]).astype(mx.float32)
        x1b = mx.array(bt_[:, :, 0:1, :half_]).astype(mx.float32)
        x2b = mx.array(bt_[:, :, 0:1, half_ : 2 * half_]).astype(mx.float32)
        m1a = x1a * cA[None, None, 1:2, :]
        m1b = x1b * cB[None, None, 0:1, :]
        m2a = x2a * sA[None, None, 1:2, :]
        m2b = x2b * sB[None, None, 0:1, :]
        mx.eval(m1a, m2a, m1b, m2b)
        print(
            f"  {tag} x1*cos: "
            f"n={int(np.count_nonzero(np.asarray(m1a) - np.asarray(m1b)))} "
            f"maxabs={float(np.abs(np.asarray(m1a) - np.asarray(m1b)).max()):.3e}",
            flush=True,
        )
        print(
            f"  {tag} x2*sin: "
            f"n={int(np.count_nonzero(np.asarray(m2a) - np.asarray(m2b)))} "
            f"maxabs={float(np.abs(np.asarray(m2a) - np.asarray(m2b)).max()):.3e}",
            flush=True,
        )
        r1a = m1a - m2a
        r1b = m1b - m2b
        r2a = x1a * sA[None, None, 1:2, :] + x2a * cA[None, None, 1:2, :]
        r2b = x1b * sB[None, None, 0:1, :] + x2b * cB[None, None, 0:1, :]
        mx.eval(r1a, r2a, r1b, r2b)
        print(
            f"  {tag} r1(x1c-x2s): "
            f"n={int(np.count_nonzero(np.asarray(r1a) - np.asarray(r1b)))} "
            f"maxabs={float(np.abs(np.asarray(r1a) - np.asarray(r1b)).max()):.3e}",
            flush=True,
        )
        print(
            f"  {tag} r2(x1s+x2c): "
            f"n={int(np.count_nonzero(np.asarray(r2a) - np.asarray(r2b)))} "
            f"maxabs={float(np.abs(np.asarray(r2a) - np.asarray(r2b)).max()):.3e}",
            flush=True,
        )
        rotA = mx.concatenate([r1a, r2a], axis=-1)
        rotB = mx.concatenate([r1b, r2b], axis=-1)
        pasA = mx.array(at_[:, :, 1:2, 2 * half_ :])
        pasB = mx.array(bt_[:, :, 0:1, 2 * half_ :])
        fullA = mx.concatenate([rotA, pasA], axis=-1)
        fullB = mx.concatenate([rotB, pasB], axis=-1)
        bfA = fullA.astype(mx.bfloat16)
        bfB = fullB.astype(mx.bfloat16)
        mx.eval(bfA, bfB)
        d = np.asarray(bfA.astype(mx.float32)) - np.asarray(bfB.astype(mx.float32))
        print(
            f"  {tag} explicit full+bf16: n={int(np.count_nonzero(d != 0))} maxabs={float(np.abs(d).max()):.3e}",
            flush=True,
        )

    def report_fork(t2, l1A, logitsB2, snap_Apost, snap_B2post):
        print(f"=== FORK at verify appending token index {FORK_TOKEN_IDX} (t2={t2}, reject={t2 != d_ref}) ===")
        S = CAP.seq
        # ---- row-1: A row1 vs B2 row0 (draft row) ----
        first1 = None
        order = []
        n_layers = len(layers)
        for i in range(n_layers):
            order.append(f"L{i}_xin")
            if i in gdn_layer_ids:
                gi = gdn_ids(i)
                order += [
                    f"gp{gi}_q",
                    f"gp{gi}_k",
                    f"gp{gi}_v",
                    f"gp{gi}_qn",
                    f"gp{gi}_kn",
                    f"gp{gi}_vc",
                    f"gp{gi}_beta",
                    f"gp{gi}_g",
                    f"gdu{gi}_out",
                    f"ng{gi}_out",
                ]
            else:
                order += [f"L{i}_input_ln_0_out"]
                if i == 15:
                    order += ["FA15_q_proj_0_out"]
                order += [f"FA{i}_k_proj_0_in", f"FA{i}_k_proj_0_out"]
                if i == 15:
                    order += ["FA15_q_norm_0_out"]
                order += [
                    f"FA{i}_k_norm_0_out",
                    f"FA{i}_rope_0_out",
                    f"FA{i}_rope_0_in",
                    f"FA{i}_rope_1_out",
                    f"FA{i}_rope_1_in",
                ]
                if i == 15:
                    order += ["FA15_v_proj_0_out"]
                order += [f"FA{i}_sdpa_q", f"FA{i}_sdpa_out"]
            order.append(f"L{i}_out")
        order += ["LOGITS_r1"]
        for key in order:
            e = S.get(key)
            if e is None or "A" not in e or "B2" not in e:
                if "k_norm" in key:
                    print(
                        f"  DEBUG skip {key}: e={0 if e is None else 'y'} modes={sorted(e.keys()) if e else None}",
                        flush=True,
                    )
                continue
            a1 = rowx(e["A"], 1, key)
            b0 = rowx(e["B2"], 0, key)
            if a1 is None or b0 is None or a1.shape != b0.shape:
                if "k_norm" in key or "q_norm" in key:
                    sa = None if a1 is None else a1.shape
                    sb = None if b0 is None else b0.shape
                    print(f"  DEBUG skip {key}: a1={sa} b2={sb} rawA={e['A'].shape} rawB2={e['B2'].shape}", flush=True)
                continue
            n, ma = cmp_pair(key, a1, b0)
            tag = ""
            if n > 0:
                if first1 is None:
                    first1 = key
                    tag = "  <-- FIRST ROW-1 DIVERGENCE"
                else:
                    tag = "  (div)"
            print(f"  {key}: n={n}/{a1.size} maxabs={ma:.3e}{tag}", flush=True)
        print(f"ROW-1 FIRST DIVERGING COMPONENT: {first1}")
        # ---- row-0: A row0 vs B1 row0 (prev row) ----
        first0 = None
        for key in order:
            e = S.get(key)
            if e is None or "A" not in e or "B1" not in e:
                continue
            a0 = rowx(e["A"], 0, key)
            b0 = rowx(e["B1"], 0, key)
            if a0 is None or b0 is None or a0.shape != b0.shape:
                continue
            n, ma = cmp_pair(key, a0, b0)
            if n > 0 and first0 is None:
                first0 = key
        print(f"ROW-0 FIRST DIVERGING COMPONENT: {first0}")
        # ---- dynamic proof: A post-state vs B1+B2 post-state ----
        print("=== DYNAMIC PROOF: A post-state vs B1+B2 post-state ===")
        leak = False
        for gA, gB in zip(snap_Apost[0], snap_B2post[0], strict=True):
            for x, y, nm in [(gA[2], gB[2], "conv"), (gA[4], gB[4], "ssm")]:
                if x is None or y is None:
                    continue
                dd = x - y
                n = int(np.count_nonzero(dd != 0))
                if n > 0:
                    leak = True
                    print(f"  GDN layer {gA[0]} {nm}: n={n}/{dd.size} maxabs={np.abs(dd).max():.3e} (LEAK)")
        for kA, kB in zip(snap_Apost[1], snap_B2post[1], strict=True):
            for x, y, nm in [(kA[2], kB[2], "keys"), (kA[4], kB[4], "values")]:
                dd = x - y
                n = int(np.count_nonzero(dd != 0))
                if n > 0:
                    leak = True
                    print(f"  KV layer {kA[0]} {nm}: n={n}/{dd.size} maxabs={np.abs(dd).max():.3e} (LEAK)")
        print(f"DYNAMIC PROOF: {'LEAK FOUND' if leak else 'NO LEAK (state exact)'}")
        # ---- KV key element-level detail for the first KV leak layer ----
        for kA, kB in zip(snap_Apost[1], snap_B2post[1], strict=True):
            x, y = kA[2], kB[2]
            if x is None or y is None:
                continue
            dd = x - y
            flat_idx = np.flatnonzero(dd != 0)
            if 0 < flat_idx.size <= 64:
                for fi in flat_idx[:8]:
                    t0 = np.unravel_index(int(fi), dd.shape)
                    print(f"  KV layer {kA[0]} keys diff at {t0}: a={x[t0]:.6f} b={y[t0]:.6f}")

    gdn_id_map = {}
    gdn_i = 0
    for i in range(len(layers)):
        if i in gdn_layer_ids:
            gdn_id_map[i] = gdn_i
            gdn_i += 1

    def gdn_ids(i):
        return gdn_id_map[i]

    d_ref = None
    for _ in range(MAX_TOKENS):
        if draft is None:
            t = int(mx.argmax(lg))
            if t == EOS:
                break
            out.append(t)
            mtp_saved = save_mtp(mtp_cache)
            emb = lm.model.embed_tokens(mx.array([[t]]))
            dh = lm.mtp(h, emb, mtp_cache)
            draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
            prev_token = t
        else:
            d_ref = draft
            d = draft
            draft = None
            verifies += 1
            saved = save_gdn(cache)
            if len(out) == FORK_TOKEN_IDX and not fork_done:
                t2, l1, h1 = fork_probe(d, prev_token)
            else:
                logits, hidden = lm(mx.array([[prev_token, d]]), cache=cache, return_hidden=True, return_prenorm=True)
                l0, _ = logits[0, 0], hidden[0, 0]
                l1, h1 = logits[0, 1], hidden[0, 1]
                t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d:
                accepted += 1
                out.append(t2)
                lg, h = l1, h1
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                restore_mtp(mtp_cache, mtp_saved)
                out.append(t2)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                lg, h = logits[0, -1], hidden[0, -1]
            if len(out) > FORK_TOKEN_IDX and fork_done:
                break
            draft = None
    print(f"done: n_out={len(out)} accepted={accepted}/{verifies} fork_done={fork_done}")


if __name__ == "__main__":
    main()
