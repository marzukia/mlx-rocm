#!/usr/bin/env python3
"""m7: GDN-path row-invariance A/B at the S62 inputs (the m2-vs-m0 divergence).

m6 localized the remaining T-dependence to the GDN (gated delta net) path:
the first diverging layer is 4 (gdn conv 2059/24576, ssm 515560/524288),
bit-identical to m3. The layer-4 conv state is a shift-register copy of the
layer-3 input vectors, so the divergence proves the layer-3 input row 0
(the GDN-2 / layer-2 output row 0) is T-dependent. This script localizes
WHICH component of the GDN path computes row 0 differently at T=2 vs T=1.

Setup: the m3 flow (deterministic; re-arms at the S62 arm: step=60 prev=279
draft=3545 n_out=61 act=ACC). At the arm the ENTIRE main-model cache is
snapshotted, then the two variants:

  variant A: one T=2 verify forward   lm([[279, 3545]])  (row 0 = 279, row 1 = 3545)
  restore:   the snapshot is written back (verified bit-exact)
  variant B: two T=1 forwards        lm([[279]]) then lm([[3545]])

The GDN layer (qwen3_5.GatedDeltaNet) has TWO structurally different
paths (the prime row-0 suspect):
  - S == 1 (AR / the B variant): the FUSED tail path
      qkv = in_proj_qkv(x); z,b,a = in_proj_z/b/a(x)
      conv_input = [conv_state, qkv]; cache[0] = shift
      qn,kn,vc,beta,g = mx.fast.gdn_prep(q,k,v,b,a,A_log,dt_bias,conv_state,conv1d.w,inv,eps)
      out,state = mx.fast.gated_delta_update(qn,kn,vc,g,beta,state,mask)
      out = mx.fast.gdn_norm_gated(out,z,norm.w,eps)[0]; out = out_proj(out)
  - S > 1 (verify / the A variant): the UNFUSED path
      qkv = in_proj_qkv(x); z,b,a = in_proj_z/b/a(x)
      conv_input = [conv_state, qkv]; cache[0] = shift
      conv_out = silu(conv1d(conv_input)); q,k,v = split(conv_out)
      q,k = normalize_qk(q,k,inv,eps)
      out,state = gated_delta_update(q,k,v,a,b,A_log,dt_bias,state,mask,use_kernel=True)
      out = norm(out,z); out = out_proj(out)

So even if every individual kernel were row-invariant, the two paths use
different kernel compositions (gdn_prep vs conv1d+normalize_qk; gdn_norm_gated
vs RMSNormGated) and can differ by 1 ULP on the same logical row.

Modes:

  MODE=harvest
    Re-run the m3 flow (deterministic; re-arms at the S62 arm). Snapshot the
    whole cache, run variant A (T=2, hooks on every GDN layer 0,1,2,4 + the
    layer-3 attn block), restore + verify bit-exact, run variant B (two T=1
    forwards, hooks on the GDN layers + the layer-3 attn block for the
    second forward). Save cap-m7-s62.pkl (per-component intermediates for
    both rows of A and the single row of each B step, + the GDN weights).

  MODE=ab
    Load cap-m7-s62.pkl. For every GDN layer and the layer-3 attn block,
    compare A row 0 vs B1 and A row 1 vs B2, per component, in dataflow
    order. The first diverging component (row 0) is the row-0 T-dependence
    source; the row-1 components must be clean (m6 verified the attn block
    row 1; this re-confirms and extends to the GDN path).

Exit codes: 0 = done; 2 = arm mismatch (harvest); 3 = unexpected failure.
"""

import os
import pickle
import sys

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import qwen3_5 as q5
from mlx_lm.models.cache import ArraysCache, KVCache
MODELS = os.environ.get("MODELS", "/models")

MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
MAX_TOKENS = int(os.environ.get("DIAG_MAX", "160"))
EOS = 248044
MODE = os.environ.get("MODE", "ab")
D = os.environ.get("CAP_DIR", f"{MODELS}/tmp/an earlier run")
M3_PKL = os.environ.get("M3_PKL", f"{MODELS}/tmp/an earlier run/cap-m3-s62.pkl")
M7_PKL = os.path.join(D, "cap-m7-s62.pkl")
S62_PREV = 279
S62_DRAFT = 3545
EXPECT_N_OUT = 61
GDN_LAYERS = [0, 1, 2, 4]
ATTN_LAYER = 3
N_H = 16
N_KV = 4


def npf(a):
    """mx array -> f32 numpy (avoids the bf16 PEP 3118 buffer issue)."""
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def of(npv, dt):
    """f32 numpy -> mx array of dtype dt (bit-exact round-trip)."""
    if npv is None:
        return None
    return mx.array(np.asarray(npv, dtype=np.float32)).astype(dt)


def cmp_status(a, b):
    """Bit-exact comparison of two f32 arrays -> status string."""
    if a is None and b is None:
        return "NONE"
    if a is None or b is None:
        return "NONE-vs-ARR"
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return f"SHAPE {a.shape} vs {b.shape}"
    d = a != b
    n = int(np.count_nonzero(d))
    return "MATCH" if n == 0 else "DIFF"


def cmp_report(name, a, b, first_idx=True):
    st = cmp_status(a, b)
    if st == "MATCH":
        print(f"  {name:<46} MATCH  (n={np.asarray(a).size})", flush=True)
    elif st in ("NONE", "NONE-vs-ARR"):
        print(f"  {name:<46} {st}", flush=True)
    elif st.startswith("SHAPE"):
        print(f"  {name:<46} {st}", flush=True)
    else:
        d = np.asarray(a, dtype=np.float32).ravel() - np.asarray(b, dtype=np.float32).ravel()
        n = int((d != 0).sum())
        idx = int(np.argmax(d != 0))
        print(
            f"  {name:<46} DIFF   n={n}/{d.size} maxabs={np.abs(d).max():.3e} first={idx}",
            flush=True,
        )
    return st


def row_of(a, i):
    """Row i of an (1, T, ...) array -> (1, ...) collapsed to (T==1 ? ... : ...)."""
    if a is None:
        return None
    return a[0, i, ...] if a.ndim >= 2 and a.shape[1] > 1 else a[0, ...]


# ---------------------------------------------------------------------------
# GDN layer hook (harvest)
# ---------------------------------------------------------------------------


class GDNHook:
    """Capture the I/O of every component of one GatedDeltaNet layer.

    Wraps the layer's submodules (in_proj_*, conv1d, norm, out_proj) and the
    path-specific functions (mx.fast.gdn_prep / gdn_norm_gated, the Python
    gated_delta_update + normalize_qk, mx.fast.gated_delta_update) and records
    I/O. The originals are restored on exit. The cache[0]/cache[1] pre/post
    (conv + ssm state) are captured by wrapping the layer __call__.
    """

    def __init__(self, layer, cache):
        self.layer = layer
        self.cache = cache
        self.cap = {}
        self._hooks = []
        self._fnhooks = []

    def _wrap_mod(self, parent, attr, name, cache_tag=None):
        obj = parent[attr]
        cap = self.cap
        cache = self.cache

        class W:
            def __call__(self, *a, **kw):
                out = obj(*a, **kw)
                rec = {
                    "in": [npf(x) if isinstance(x, mx.array) else x for x in a],
                    "out": npf(out) if isinstance(out, mx.array) else out,
                }
                if cache_tag is not None and cache is not None:
                    c0 = cache[0]
                    c1 = cache[1]
                    rec["conv_state"] = npf(c0) if c0 is not None else None
                    rec["ssm_state"] = npf(c1) if c1 is not None else None
                cap.setdefault(name, []).append(rec)
                return out

            def __getattr__(self, name):
                return getattr(obj, name)

        self._hooks.append((parent, attr, obj))
        setattr(parent, attr, W())

    def _wrap_fn(self, mod, attr, name):
        obj = getattr(mod, attr)

        def W(*a, **kw):
            out = obj(*a, **kw)
            rec = {
                "in": [npf(x) if isinstance(x, mx.array) else x for x in a],
                "kw": {k: (npf(v) if isinstance(v, mx.array) else v) for k, v in kw.items()},
            }
            if isinstance(out, (tuple, list)):
                rec["out"] = [npf(x) if isinstance(x, mx.array) else x for x in out]
            else:
                rec["out"] = npf(out) if isinstance(out, mx.array) else out
            self.cap.setdefault(name, []).append(rec)
            return out

        self._fnhooks.append((mod, attr, obj))
        setattr(mod, attr, W)

    def __enter__(self):
        L = self.layer
        # in_proj_qkv is the first submodule called; capture the cache state
        # here as the PRE-state (the conv/ssm update happens later in __call__).
        self._wrap_mod(L, "in_proj_qkv", "in_proj_qkv", cache_tag="pre")
        self._wrap_mod(L, "in_proj_z", "in_proj_z")
        self._wrap_mod(L, "in_proj_b", "in_proj_b")
        self._wrap_mod(L, "in_proj_a", "in_proj_a")
        self._wrap_mod(L, "conv1d", "conv1d")
        self._wrap_mod(L, "norm", "norm")
        # out_proj is the last submodule called; capture the cache state here
        # as the POST-state (after the conv/ssm update).
        self._wrap_mod(L, "out_proj", "out_proj", cache_tag="post")
        # path-specific functions
        self._wrap_fn(mx.fast, "gdn_prep", "gdn_prep")
        self._wrap_fn(mx.fast, "gdn_norm_gated", "gdn_norm_gated")
        self._wrap_fn(mx.fast, "gated_delta_update", "mx_gdn_update")
        self._wrap_fn(q5, "normalize_qk", "normalize_qk")
        self._wrap_fn(q5, "gated_delta_update", "py_gdn_update")
        return self

    def __exit__(self, *a):
        for parent, attr, obj in reversed(self._hooks):
            setattr(parent, attr, obj)
        for mod, attr, obj in reversed(self._fnhooks):
            setattr(mod, attr, obj)
        return False


# ---------------------------------------------------------------------------
# layer-3 attn block hook (the m5 BlockHook pattern, for the row-0 check)
# ---------------------------------------------------------------------------


class AttnHook:
    """Capture the I/O of every component of the layer-3 full-attention block."""

    def __init__(self, layers, layer_idx):
        self.layers = layers
        self.layer_idx = layer_idx
        self.cap = {}
        self._hooks = []
        self.sdpa_calls = []

    def _wrap(self, parent, attr, name):
        obj = parent[attr] if not isinstance(parent, list) else parent[int(attr)]
        cap = self.cap

        class W:
            def __call__(self, *a, **kw):
                out = obj(*a, **kw)
                rec = {
                    "in": [
                        npf(x) if isinstance(x, mx.array) else (x if isinstance(x, (str, int, type(None))) else "<obj>")
                        for x in a
                    ],
                    "kw": {
                        k: (npf(v) if isinstance(v, mx.array) else (v if isinstance(v, (str, int, type(None))) else "<obj>"))
                        for k, v in kw.items()
                    },
                    "out": npf(out),
                }
                cap.setdefault(name, []).append(rec)
                return out

            def __getattr__(self, name):
                return getattr(obj, name)

        self._hooks.append((parent, attr, obj))
        if isinstance(parent, list):
            parent[int(attr)] = W()
        else:
            setattr(parent, attr, W())

    def __enter__(self):
        L3 = self.layers[self.layer_idx]
        attn = L3.self_attn
        mlp = L3.mlp
        self._wrap(self.layers, self.layer_idx, "block")
        self._wrap(L3, "input_layernorm", "input_ln")
        self._wrap(attn, "q_proj", "q_proj")
        self._wrap(attn, "k_proj", "k_proj")
        self._wrap(attn, "v_proj", "v_proj")
        self._wrap(attn, "o_proj", "o_proj")
        self._wrap(attn, "q_norm", "q_norm")
        self._wrap(attn, "k_norm", "k_norm")
        self._wrap(attn, "rope", "rope")
        self._wrap(L3, "post_attention_layernorm", "post_ln")
        self._wrap(L3, "mlp", "mlp")
        self._wrap(mlp, "gate_proj", "gate_proj")
        self._wrap(mlp, "up_proj", "up_proj")
        self._wrap(mlp, "down_proj", "down_proj")
        self.orig_sdpa = mx.fast.scaled_dot_product_attention

        def sdpa_hook(q, k, v, scale=1.0, mask=None, sinks=None):
            out = self.orig_sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks)
            self.sdpa_calls.append(
                {
                    "q": npf(q), "k": npf(k), "v": npf(v), "out": npf(out), "scale": float(scale), "mask": mask,
                    "qshape": tuple(q.shape), "kshape": tuple(k.shape), "vshape": tuple(v.shape),
                }
            )
            return out

        mx.fast.scaled_dot_product_attention = sdpa_hook
        return self

    def __exit__(self, *a):
        for parent, attr, obj in reversed(self._hooks):
            if isinstance(parent, list):
                parent[int(attr)] = obj
            else:
                setattr(parent, attr, obj)
        mx.fast.scaled_dot_product_attention = self.orig_sdpa
        return False


# ---------------------------------------------------------------------------
# GDN weights (for a direct re-run if needed)
# ---------------------------------------------------------------------------


def snap_w(a):
    if a is None:
        return (None, None)
    if a.dtype == mx.bfloat16:
        return ((npf(a).view(np.uint32) >> 16).astype(np.uint16), "bf16")
    if a.dtype == mx.uint32:
        return (np.asarray(a).copy(), "u32")
    return (npf(a), "f32")


def of_w(v, dt):
    if v is None:
        return None
    npv, tag = v
    if tag == "bf16":
        f = (npv.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f).astype(mx.bfloat16)
    if tag == "u32":
        return mx.array(npv)
    return mx.array(np.asarray(npv, dtype=np.float32))


def cap_gdn_weights(L):
    w = {}
    for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        mod = getattr(L, nm)
        if isinstance(mod, mnn.QuantizedLinear):
            w[nm] = {
                "quantized": True,
                "bits": int(mod.bits), "group": int(mod.group_size), "mode": mod.mode,
                "in": int(mod.weight.shape[1] * 32 // mod.bits), "out": int(mod.weight.shape[0]),
                "weight": snap_w(mod.weight), "scales": snap_w(mod.scales), "biases": snap_w(mod.biases),
            }
        else:
            w[nm] = {"quantized": False, "weight": snap_w(mod.weight), "bias": snap_w(getattr(mod, "bias", None))}
    w["conv1d"] = {"weight": snap_w(L.conv1d.weight)}
    w["norm"] = {"weight": snap_w(L.norm.weight), "eps": float(L.norm.eps)}
    w["A_log"] = snap_w(L.A_log)
    w["dt_bias"] = snap_w(L.dt_bias)
    return w


# ---------------------------------------------------------------------------
# cache snapshot / restore (the m3 pattern)
# ---------------------------------------------------------------------------


def snap_cache(cache):
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append(
                {
                    "kind": "kv",
                    "keys": npf(c.keys),
                    "values": npf(c.values),
                    "offset": c.offset,
                    "keys_mx": c.keys,
                    "values_mx": c.values,
                }
            )
        elif isinstance(c, ArraysCache):
            snap.append(
                {
                    "kind": "ac",
                    "cache": [npf(x) for x in c.cache],
                    "cache_mx": list(c.cache),
                    "left_padding": npf(c.left_padding),
                    "lengths": npf(c.lengths),
                    "left_padding_mx": c.left_padding,
                    "lengths_mx": c.lengths,
                }
            )
        else:
            raise ValueError(f"unknown cache entry type {type(c)!r}")
    return snap


def restore_cache(cache, snap):
    for c, s in zip(cache, snap, strict=True):
        if s["kind"] == "kv":
            # the verify forward appends 2 entries in-place; restore the offset
            # (the m3 flow's approach). The arrays retain the appended entries,
            # but the offset points to the pre-forward end.
            c.trim(2)
        else:
            c.cache = list(s["cache_mx"])
            c.left_padding = s["left_padding_mx"]
            c.lengths = s["lengths_mx"]


def verify_restore(cache, snap):
    nbad = 0
    for i, (c, s) in enumerate(zip(cache, snap, strict=True)):
        if s["kind"] == "kv":
            # check the keys/values up to the restored offset (the effective state)
            for nm in ("keys", "values"):
                live = c.keys if nm == "keys" else c.values
                off = s["offset"]
                snap_arr = s[nm]
                # keys/values are (B, T, H, D); the T axis is axis 1. The snapshot
                # is the full pre-allocated array (>= off entries); compare the
                # first off entries of both the live and the snapshot.
                if live is not None and live.ndim == 4 and snap_arr is not None and snap_arr.ndim == 4:
                    # keys/values are (B, H, T, D); the T axis is axis 2 (pre-allocated).
                    live_eff = live[:, :, :off, :]
                    snap_eff = snap_arr[:, :, :off, :]
                else:
                    live_eff, snap_eff = live, snap_arr
                if cmp_status(npf(live_eff), snap_eff) not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} {nm}", flush=True)
            if c.offset != s["offset"]:
                nbad += 1
                print(f"  RESTORE-DIFF entry={i} offset (live={c.offset} snap={s['offset']})", flush=True)
        else:
            for j, live in enumerate(c.cache):
                if cmp_status(npf(live), s["cache"][j]) not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} arr{j}", flush=True)
    return nbad


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------


def run_m7(lm, cache, out, prev_token, draft, step):
    n_out = len(out)
    print(f"ARM m7: step={step} prev={prev_token} draft={draft} n_out={n_out} out[-8:]={out[-8:]}", flush=True)
    if n_out != EXPECT_N_OUT:
        print(f"ARM MISMATCH: expected n_out=={EXPECT_N_OUT}, got {n_out}", flush=True)
        sys.exit(2)
    if draft != S62_DRAFT:
        print(f"ARM MISMATCH: expected draft=={S62_DRAFT}, got {draft}", flush=True)
        sys.exit(2)

    layers = lm.model.layers
    for i in GDN_LAYERS:
        print(f"GDN layer {i}: {type(layers[i].linear_attn).__name__}", flush=True)
    print(f"attn layer {ATTN_LAYER}: {type(layers[ATTN_LAYER].self_attn).__name__}", flush=True)

    for c in cache:
        if isinstance(c, KVCache):
            mx.eval([c.keys, c.values])
        else:
            mx.eval([x for x in c.cache if x is not None])
    snap = snap_cache(cache)
    print(f"snapshot taken: {len(snap)} entries", flush=True)

    # ---- variant A: one T=2 verify forward (hooks on) -------------------
    print(f"== variant A: T=2 verify forward [[{S62_PREV}, {S62_DRAFT}]] ==", flush=True)
    gdnA = {}
    attnA = {}
    sdpaA = []
    # hook all GDN layers + the attn layer for the whole forward
    hooks = [GDNHook(layers[i].linear_attn, cache[i]) for i in GDN_LAYERS]
    ah = AttnHook(layers, ATTN_LAYER)
    for h in hooks:
        h.__enter__()
    ah.__enter__()
    try:
        logits_a, hidden_a = lm(
            mx.array([[S62_PREV, S62_DRAFT]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval([logits_a, hidden_a])
    finally:
        ah.__exit__()
        for h in hooks:
            h.__exit__()
    for i, h in zip(GDN_LAYERS, hooks):
        gdnA[i] = h.cap
    attnA = ah.cap
    sdpaA = ah.sdpa_calls
    t2 = int(mx.argmax(logits_a[0, 0]))
    act = "ACC" if t2 == S62_DRAFT else "REJ"
    print(f"variant A act={act} (t2={t2} draft={S62_DRAFT})", flush=True)
    print(f"variant A: {len(sdpaA)} SDPA calls; layer-3 rope_offset={attnA['rope'][0]['kw']['offset']}", flush=True)
    weights = {i: cap_gdn_weights(layers[i].linear_attn) for i in GDN_LAYERS}

    # ---- restore + verify the fork ---------------------------------------
    restore_cache(cache, snap)
    nbad = verify_restore(cache, snap)
    print(f"restore verified: {len(snap) - nbad}/{len(snap)} entries bit-exact", flush=True)
    if nbad:
        print("RESTORE NOT BIT-EXACT - the A/B diff is suspect", flush=True)

    # ---- variant B: two T=1 forwards (hooks on each) --------------------
    gdnB = {}
    attnB = {}
    sdpaB = []
    print(f"== variant B: T=1 forward [[{S62_PREV}]] ==", flush=True)
    hooks = [GDNHook(layers[i].linear_attn, cache[i]) for i in GDN_LAYERS]
    ah = AttnHook(layers, ATTN_LAYER)
    for h in hooks:
        h.__enter__()
    ah.__enter__()
    try:
        logits_b1, hidden_b1 = lm(
            mx.array([[S62_PREV]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval([logits_b1, hidden_b1])
    finally:
        ah.__exit__()
        for h in hooks:
            h.__exit__()
    for i, h in zip(GDN_LAYERS, hooks):
        gdnB[f"{i}_1"] = h.cap
    attnB["1"] = ah.cap
    sdpaB.extend(ah.sdpa_calls)

    print(f"== variant B: T=1 forward [[{S62_DRAFT}]] ==", flush=True)
    hooks = [GDNHook(layers[i].linear_attn, cache[i]) for i in GDN_LAYERS]
    ah = AttnHook(layers, ATTN_LAYER)
    for h in hooks:
        h.__enter__()
    ah.__enter__()
    try:
        logits_b2, hidden_b2 = lm(
            mx.array([[S62_DRAFT]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval([logits_b2, hidden_b2])
    finally:
        ah.__exit__()
        for h in hooks:
            h.__exit__()
    for i, h in zip(GDN_LAYERS, hooks):
        gdnB[f"{i}_2"] = h.cap
    attnB["2"] = ah.cap
    sdpaB.extend(ah.sdpa_calls)

    # ---- cross-check vs m3 (determinism) --------------------------------
    if os.path.exists(M3_PKL):
        with open(M3_PKL, "rb") as f:
            m3 = pickle.load(f)
        for i, (kind, off, k, v) in enumerate(m3["ha"]):
            if kind == "gdn" and i == 4:
                st = cmp_status(npf(cache[4][0]), k)
                print(f"post-A layer-4 conv state vs m3 pkl: {st}", flush=True)
                break
    else:
        print(f"{M3_PKL} not found; skipping the m3 cross-check", flush=True)
    cmp_report("logits after 279  (A[0,0] vs B1[0,0])", npf(logits_a[0, 0]), npf(logits_b1[0, 0]))
    cmp_report("logits after 3545 (A[0,1] vs B2[0,0])", npf(logits_a[0, 1]), npf(logits_b2[0, 0]))

    with open(M7_PKL, "wb") as f:
        pickle.dump(
            {
                "gdnA": gdnA,
                "gdnB": gdnB,
                "attnA": attnA,
                "attnB": attnB,
                "sdpaA": sdpaA,
                "sdpaB": sdpaB,
                "weights": weights,
                "act": act,
                "n_out": n_out,
                "out_tail": out[-8:],
                "restore_nbad": nbad,
            },
            f,
        )
    print(f"m7 captures saved -> {M7_PKL}", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# ab: per-component row-0 and row-1 comparison
# ---------------------------------------------------------------------------


def gdn_components(cap):
    """Extract the LOGICAL per-component intermediates from a GDNHook cap, as a
    dict of logical-name -> (array, is_per_token). The logical names are the
    common quantities present in both the fused (S=1) and unfused (S>1) paths,
    so the A (unfused) and B (fused) captures can be compared by name.

    is_per_token=True means the array has a T axis (use row_of to slice row 0/1);
    False means it is a state array with no T axis (compare as-is).

    Logical names:
      inputs, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a  (common)
      conv_state_pre, ssm_state_pre  (common, state)
      norm_q, norm_k  (A: normalize_qk; B: gdn_prep)
      ssm_out  (common, via mx_gdn_update)
      norm_out  (A: RMSNormGated; B: gdn_norm_gated)
      out_proj  (common)
    """
    c = cap
    qkv_rec = c.get("in_proj_qkv", [{}])[0]
    op_rec = c.get("out_proj", [{}])[0]
    out = {}
    out["inputs"] = (qkv_rec.get("in", [None])[0] if qkv_rec.get("in") else None, True)
    out["in_proj_qkv"] = (qkv_rec.get("out"), True)
    out["in_proj_z"] = (c.get("in_proj_z", [{}])[0].get("out"), True)
    out["in_proj_b"] = (c.get("in_proj_b", [{}])[0].get("out"), True)
    out["in_proj_a"] = (c.get("in_proj_a", [{}])[0].get("out"), True)
    out["conv_state_pre"] = (qkv_rec.get("conv_state"), False)
    out["ssm_state_pre"] = (qkv_rec.get("ssm_state"), False)
    # the normalized q/k (path-specific names -> logical norm_q / norm_k)
    if "normalize_qk" in c and c["normalize_qk"]:
        nq = c["normalize_qk"][0].get("out")
        if nq is not None:
            out["norm_q"] = (nq[0], True)
            out["norm_k"] = (nq[1], True)
    if "gdn_prep" in c and c["gdn_prep"]:
        gp = c["gdn_prep"][0].get("out")
        if gp is not None:
            # gdn_prep returns (qn, kn, vc, beta, g)
            out["norm_q"] = (gp[0], True)
            out["norm_k"] = (gp[1], True)
    # the ssm_out (both, via mx_gdn_update)
    if "mx_gdn_update" in c and c["mx_gdn_update"]:
        out_ssm = c["mx_gdn_update"][0].get("out")
        if out_ssm is not None:
            out["ssm_out"] = (out_ssm[0], True)
    # the norm output (path-specific names -> logical norm_out)
    if "norm" in c and c["norm"]:
        out["norm_out"] = (c["norm"][0].get("out"), True)
    if "gdn_norm_gated" in c and c["gdn_norm_gated"]:
        gng = c["gdn_norm_gated"][0].get("out")
        # gdn_norm_gated returns a tuple; the model takes [0]
        if isinstance(gng, (list, tuple)):
            gng = gng[0]
        out["norm_out"] = (gng, True)
    out["out_proj"] = (op_rec.get("out"), True)
    return out


GDN_LOGICAL_ORDER = [
    "inputs", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
    "conv_state_pre", "ssm_state_pre", "norm_q", "norm_k", "ssm_out",
    "norm_out", "out_proj",
]


def slice_row(a, i, pt):
    """Slice row i of a per-token array (pt=True); state arrays (pt=False) are
    compared as-is (no T axis)."""
    if a is None:
        return None
    if pt:
        return row_of(a, i)
    return a


def run_ab():
    if not os.path.exists(M7_PKL):
        print(f"missing {M7_PKL}; run MODE=harvest first", flush=True)
        sys.exit(3)
    with open(M7_PKL, "rb") as f:
        c = pickle.load(f)
    gdnA, gdnB = c["gdnA"], c["gdnB"]
    attnA, attnB = c["attnA"], c["attnB"]
    sdpaA, sdpaB = c["sdpaA"], c["sdpaB"]
    print(f"loaded: act={c['act']} n_out={c['n_out']} restore_nbad={c['restore_nbad']} out_tail={c['out_tail']}", flush=True)

    first_diff_row0 = {}
    first_diff_row1 = {}

    # ---- GDN layers: A row 0 vs B1, A row 1 vs B2 ------------------------
    for i in GDN_LAYERS:
        print(f"== GDN layer {i}: A row 0 vs B1 (token {S62_PREV}) ==", flush=True)
        compA = gdn_components(gdnA[i])
        compB = gdn_components(gdnB[f"{i}_1"])
        fd0 = None
        for nm in GDN_LOGICAL_ORDER:
            a, pt = compA.get(nm, (None, False))
            b, pt2 = compB.get(nm, (None, False))
            a_r = slice_row(a, 0, pt)
            b_r = slice_row(b, 0, pt2)
            st = cmp_status(a_r, b_r)
            if st == "MATCH":
                print(f"  {nm:<40} MATCH", flush=True)
            elif st in ("NONE", "NONE-vs-ARR"):
                print(f"  {nm:<40} {st}", flush=True)
            else:
                d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
                print(f"  {nm:<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
                if fd0 is None:
                    fd0 = nm
        first_diff_row0[i] = fd0
        print(f"  first diverging (row 0): {fd0 or 'NONE'}", flush=True)

        print(f"== GDN layer {i}: A row 1 vs B2 (token {S62_DRAFT}) ==", flush=True)
        compB2 = gdn_components(gdnB[f"{i}_2"])
        fd1 = None
        for nm in GDN_LOGICAL_ORDER:
            a, pt = compA.get(nm, (None, False))
            b, pt2 = compB2.get(nm, (None, False))
            a_r = slice_row(a, 1, pt)
            b_r = slice_row(b, 0, pt2)
            st = cmp_status(a_r, b_r)
            if st == "MATCH":
                print(f"  {nm:<40} MATCH", flush=True)
            elif st in ("NONE", "NONE-vs-ARR"):
                print(f"  {nm:<40} {st}", flush=True)
            else:
                d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
                print(f"  {nm:<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
                if fd1 is None:
                    fd1 = nm
        first_diff_row1[i] = fd1
        print(f"  first diverging (row 1): {fd1 or 'NONE'}", flush=True)

    # ---- layer-3 attn block: A row 0 vs B1, A row 1 vs B2 ----------------
    print(f"== layer-3 attn block: A row 0 vs B1 (token {S62_PREV}) ==", flush=True)
    attn_comps = [
        ("block_in", "block", 0, "in", 0),
        ("input_ln", "input_ln", 0, "out", None),
        ("q_proj", "q_proj", 0, "out", None),
        ("k_proj", "k_proj", 0, "out", None),
        ("v_proj", "v_proj", 0, "out", None),
        ("q_norm", "q_norm", 0, "out", None),
        ("k_norm", "k_norm", 0, "out", None),
        ("rope_q", "rope", 0, "out", None),
        ("rope_k", "rope", 1, "out", None),
        ("o_proj_in", "o_proj", 0, "in", 0),
        ("o_proj", "o_proj", 0, "out", None),
        ("post_ln", "post_ln", 0, "out", None),
        ("gate_proj", "gate_proj", 0, "out", None),
        ("up_proj", "up_proj", 0, "out", None),
        ("down_proj", "down_proj", 0, "out", None),
        ("block_out", "block", 0, "out", None),
    ]
    fd0 = None
    def row_of_comp(a, i, layout):
        # layout 'qkv' = (B, H, T, D) or (B, T, H, D); layout 'bt' = (B, T, ...)
        if a is None:
            return None
        if layout == "qkv":
            # try (B, H, T, D) first (T axis 2), then (B, T, H, D) (T axis 1)
            cand_axis2 = a[0, :, i, :] if a.ndim == 4 else a
            cand_axis1 = a[0, i, :, :] if a.ndim == 4 else a
            # pick the one with a consistent shape (H, D) — the T axis is the small one
            # for A (T=2) vs B (T=1), the T axis differs; pick the layout where
            # the result shape is (H, D) (2D, not 3D)
            if a.ndim == 4:
                # if shape[1] (H or T) is small (<=8) and shape[2] (T or H) is large, T is axis 1
                if a.shape[1] <= 8 and a.shape[2] > a.shape[1]:
                    return cand_axis1
                return cand_axis2
            return a
        return row_of(a, i)
    for nm, comp, idx, io, sub in attn_comps:
        a = attnA.get(comp, [{}])[idx].get(io)
        if sub is not None:
            a = a[sub] if a is not None else None
        b = attnB["1"].get(comp, [{}])[idx].get(io)
        if sub is not None:
            b = b[sub] if b is not None else None
        layout = "qkv" if nm in ("q", "k", "v") else "bt"
        a_r = row_of_comp(a, 0, layout)
        b_r = row_of_comp(b, 0, layout)
        if a_r is not None and b_r is not None and a_r.shape != b_r.shape:
            # fall back: ravel and take the first min(len) elements
            a_flat = np.asarray(a_r, dtype=np.float32).ravel()
            b_flat = np.asarray(b_r, dtype=np.float32).ravel()
            n = min(a_flat.size, b_flat.size)
            a_r, b_r = a_flat[:n], b_flat[:n]
        st = cmp_status(a_r, b_r)
        if st == "MATCH":
            print(f"  {nm:<40} MATCH", flush=True)
        elif st in ("NONE", "NONE-vs-ARR"):
            print(f"  {nm:<40} {st}", flush=True)
        else:
            d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
            print(f"  {nm:<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
            if fd0 is None:
                fd0 = nm
    # SDPA row 0 — SDPA output layout varies; take the first token row robustly
    if sdpaA:
        a_out = sdpaA[0]["out"]
        b_out = sdpaB[0]["out"]
        # try (B, T, H, D) first (T axis 1), then (B, H, T, D) (T axis 2)
        a_r = a_out[0, 0, :, :] if a_out.ndim == 4 else a_out
        b_r = b_out[0, 0, :, :] if b_out.ndim == 4 else b_out
        if a_r.shape != b_r.shape:
            # fall back to (B, H, T, D): T axis 2
            a_r = a_out[0, :, 0, :] if a_out.ndim == 4 else a_out
            b_r = b_out[0, :, 0, :] if b_out.ndim == 4 else b_out
        if a_r.shape != b_r.shape:
            # last resort: ravel and take the first min(len) elements
            a_flat = np.asarray(a_r, dtype=np.float32).ravel()
            b_flat = np.asarray(b_r, dtype=np.float32).ravel()
            n = min(a_flat.size, b_flat.size)
            a_r, b_r = a_flat[:n], b_flat[:n]
        st = cmp_status(a_r, b_r)
        if st == "MATCH":
            print(f"  {'sdpa_out':<40} MATCH", flush=True)
        else:
            d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
            print(f"  {'sdpa_out':<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
            if fd0 is None:
                fd0 = "sdpa_out"
    first_diff_row0["attn3"] = fd0
    print(f"  first diverging (row 0): {fd0 or 'NONE'}", flush=True)

    print(f"== layer-3 attn block: A row 1 vs B2 (token {S62_DRAFT}) ==", flush=True)
    fd1 = None
    for nm, comp, idx, io, sub in attn_comps:
        a = attnA.get(comp, [{}])[idx].get(io)
        if sub is not None:
            a = a[sub] if a is not None else None
        b = attnB["2"].get(comp, [{}])[idx].get(io)
        if sub is not None:
            b = b[sub] if b is not None else None
        a_r = row_of(a, 1) if a is not None and a.ndim >= 2 and a.shape[1] > 1 else a
        b_r = row_of(b, 0) if b is not None and b.ndim >= 2 and b.shape[1] > 1 else b
        if a_r is not None and b_r is not None and a_r.shape != b_r.shape:
            a_flat = np.asarray(a_r, dtype=np.float32).ravel()
            b_flat = np.asarray(b_r, dtype=np.float32).ravel()
            n = min(a_flat.size, b_flat.size)
            a_r, b_r = a_flat[:n], b_flat[:n]
        st = cmp_status(a_r, b_r)
        if st == "MATCH":
            print(f"  {nm:<40} MATCH", flush=True)
        elif st in ("NONE", "NONE-vs-ARR"):
            print(f"  {nm:<40} {st}", flush=True)
        else:
            d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
            print(f"  {nm:<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
            if fd1 is None:
                fd1 = nm
    if sdpaA:
        a_r = row_of(sdpaA[0]["out"], 1)
        b_r = row_of(sdpaB[1]["out"], 0)
        if a_r is not None and b_r is not None and a_r.shape != b_r.shape:
            a_flat = np.asarray(a_r, dtype=np.float32).ravel()
            b_flat = np.asarray(b_r, dtype=np.float32).ravel()
            n = min(a_flat.size, b_flat.size)
            a_r, b_r = a_flat[:n], b_flat[:n]
        st = cmp_status(a_r, b_r)
        if st == "MATCH":
            print(f"  {'sdpa_out':<40} MATCH", flush=True)
        else:
            d = np.asarray(a_r, dtype=np.float32).ravel() - np.asarray(b_r, dtype=np.float32).ravel()
            print(f"  {'sdpa_out':<40} DIFF n={int((d!=0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}", flush=True)
            if fd1 is None:
                fd1 = "sdpa_out"
    first_diff_row1["attn3"] = fd1
    print(f"  first diverging (row 1): {fd1 or 'NONE'}", flush=True)

    # ---- summary ---------------------------------------------------------
    print("== m7 summary ==", flush=True)
    print("first diverging component (row 0):", flush=True)
    for k, v in first_diff_row0.items():
        print(f"  layer {k}: {v or 'NONE'}", flush=True)
    print("first diverging component (row 1):", flush=True)
    for k, v in first_diff_row1.items():
        print(f"  layer {k}: {v or 'NONE'}", flush=True)


def main():
    if MODE == "harvest":
        model, tok = load(MODEL)
        lm = model.language_model
        mx.eval(model.parameters())
        mtp_cache = [KVCache()]
        prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
        cache = lm.make_cache()
        lmap = ["attn" if isinstance(c, KVCache) else "gdn" for c in cache]
        print(f"layer map ({len(lmap)} layers): {lmap}", flush=True)
        logits, hidden = lm(
            mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True
        )
        lg = logits[0, -1]
        hidd = hidden[0, -1]
        draft = prev_token = None
        out, armed = [], False
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
                    run_m7(lm, cache, out, prev_token, draft, step)
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
                        mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    logits, hidden = lm(
                        mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    lg, hidd = logits[0, -1], hidden[0, -1]
        if not armed:
            print(
                f"ARM NOT REACHED: n_out={len(out)} last_out={out[-8:]} "
                f"(flow did not reach prev_token={S62_PREV}; not forcing the arm)",
                flush=True,
            )
            sys.exit(2)
    elif MODE == "ab":
        run_ab()
    else:
        print(f"unknown MODE={MODE} (use harvest or ab)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(3)
