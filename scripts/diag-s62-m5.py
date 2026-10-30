#!/usr/bin/env python3
"""m5: direct A/B of the layer-3 full-attention block (non-SDPA components).

m4 EXONERATED the SDPA: the layer-3 SDPA output for token 3545 is bit-exact
A row 1 (2-row causal) vs B (1-row unmasked), on captured + random data.
m3's chain (first diverging layer = 4 GDN conv/ssm) is solid, and the GDN
math is exonerated. So the m2-vs-m0 divergence must be in the layer-3
block's NON-SDPA parts, or in a flaw of m3's comparison chain. This script
re-localizes it with a direct A/B on captured tensors (the m4 method): ONE
more m3-flow run (deterministic; re-arms at the S62 arm: step=60 prev=279
draft=3545 n_out=61 act=ACC) hooks EVERY component of the layer-3 block for
BOTH variants:

  variant A: one T=2 verify forward  [[279, 3545]]  (row 1 = token 3545)
  variant B: two T=1 forwards        [[279]] then [[3545]]

The layer-3 block (qwen3_5.DecoderLayer, full_attention; the an earlier runc-patched
venv model, $MODELS/venv-d; linear layers are 4-bit quantized, group 64):

  x   = block input (the GDN-2 block output / residual stream)
  r   = self_attn(input_layernorm(x), mask, cache)
  h   = x + r
  out = h + mlp(post_attention_layernorm(h))

  self_attn (Qwen3NextAttention, attn_output_gate=true):
    qkv_raw  = q_proj(ln_x)        # (B, L, n_h*2*hd); split -> queries, gate
    keys, values = k_proj(ln_x), v_proj(ln_x)
    queries = q_norm(queries).transpose(0,2,1,3)
    keys    = k_norm(keys).transpose(0,2,1,3)
    values  = values.transpose(0,2,1,3)
    queries = rope(queries, offset=cache.offset)   # A: 87, B: 88
    keys    = rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)
    sdpa_out = sdpa(queries, keys, values, scale, mask)
        # A: "causal" S=2 (string), B: None S=1
    return o_proj(sdpa_out.transpose(0,2,1,3).reshape(B,L,-1) * sigmoid(gate))

  mlp (Qwen3NextMLP): down_proj(silu(gate_proj(x)) * up_proj(x))

Modes:

  MODE=harvest
    Re-run the m3 flow (deterministic; re-arms at the S62 arm). Snapshot the
    whole cache, run variant A (T=2, hooks on every layer-3 component + the
    SDPA call), restore + verify bit-exact, run variant B (two T=1 forwards,
    hooks on the second). Cross-check the captured layer-3 SDPA q/k/v/out
    against cap-m4-sdpa.pkl (determinism) and the layer-3 k/v against
    cap-m3-s62.pkl. Save cap-m5-s62.pkl (components + layer-3 weights).

  MODE=ab
    Load cap-m5-s62.pkl. Part 1: the in-model A/B chain (A row 1 vs B, every
    component in dataflow order; component 1 (block input) is the control and
    MUST match; the first diverging component is the suspect). Part 2: the
    direct re-run (M=2 row 1 vs M=1, captured inputs + captured weights, the
    same module code paths): a component whose in-model A/B is DIFF but whose
    re-run is MATCH is row-invariant (the in-model diff is input-driven);
    DIFF in both is a T-dependent kernel (the suspect; the kernel dispatch ID
    comes from MODE=prof).

  MODE=oneshot SPEC=<name>
    One direct call (for rocprofv3 --kernel-trace); no model load.

  MODE=prof
    Run every oneshot spec under rocprofv3 --kernel-trace, parse the kernel
    trace, print the dispatch table (kernel name + grid + workgroup + LDS +
    VGPR per spec, A vs B).

Exit codes: 0 = done; 2 = arm mismatch (harvest); 3 = unexpected failure.
"""

import csv
import os
import pickle
import subprocess
import sys

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm import load
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
SPEC = os.environ.get("SPEC", "")
D = os.environ.get("CAP_DIR", f"{MODELS}/tmp/an earlier run")
M3_PKL = os.path.join(D, "cap-m3-s62.pkl")
M4_PKL = os.path.join(D, "cap-m4-sdpa.pkl")
M5_PKL = os.path.join(D, "cap-m5-s62.pkl")
PY = os.environ.get("PY", f"{MODELS}/venv-d/bin/python")
ROCPROF = os.environ.get("ROCPROF", "/opt/rocm/core-10.1/bin/rocprofv3")
S62_PREV = 279
S62_DRAFT = 3545
EXPECT_N_OUT = 61
LAYER3 = 3
N_H = 16
N_KV = 4
HD = 256


def npf(a):
    """mx array -> f32 numpy (avoids the bf16 PEP 3118 buffer issue)."""
    return None if a is None else np.asarray(mx.array(a).astype(mx.float32)).copy()


def of(npv, dt):
    """(f32 numpy, dtype) -> fresh mx array (f32 round-trip is exact)."""
    return None if npv is None else mx.array(npv).astype(dt)


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


def cmp_report(name, a, b):
    st, n, tot, mx_ = cmp_status(a, b)
    if st == "MATCH":
        print(f"  {name:<44} MATCH  (n={tot})", flush=True)
    elif st == "NONE":
        print(f"  {name:<44} SKIP (None)", flush=True)
    else:
        print(f"  {name:<44} {st:<10} n={n}/{tot} maxabs={mx_:.3e}", flush=True)
    return st == "MATCH"


def row1(a):
    """A row 1: (1,2,...) -> [0,1,...]; (1,H,2,...) -> [0,:,1,...]."""
    return a[0, 1, :] if a.ndim == 3 else a[0, :, 1, :]


def row0(a):
    """B single row: (1,1,...) -> [0,0,...]; (1,H,1,...) -> [0,:,0,...]."""
    return a[0, 0, :] if a.ndim == 3 else a[0, :, 0, :]


def snap_arr(a):
    return (None, None) if a is None else (npf(a), a.dtype)


def snap_w(a):
    """mx array -> (numpy, tag) for the pkl: bf16 as uint16 bits (half size), f32/u32 as-is."""
    if a is None:
        return (None, None)
    if a.dtype == mx.bfloat16:
        return ((npf(a).view(np.uint32) >> 16).astype(np.uint16), "bf16")
    if a.dtype == mx.uint32:
        return (np.asarray(a).copy(), "u32")
    return (npf(a), "f32")


def of_w(v, dt):
    """(numpy, tag) -> mx array (exact)."""
    if v is None:
        return None
    npv, tag = v
    if tag == "bf16":
        f = (npv.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f).astype(mx.bfloat16)
    if tag == "u32":
        return mx.array(npv)
    return mx.array(npv)


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


def restore_cache(cache, snap):
    for c, s in zip(cache, snap, strict=True):
        if s["kind"] == "kv":
            c.keys = restore_arr(s["keys"])
            c.values = restore_arr(s["values"])
            c.offset = s["offset"]
        else:
            c.cache = [restore_arr(x) for x in s["cache"]]
            c.left_padding = restore_arr(s["left_padding"])
            c.lengths = restore_arr(s["lengths"])


def restore_arr(v):
    npv, dt = v
    if npv is None:
        return None
    return mx.array(npv).astype(dt)


def verify_restore(cache, snap):
    nbad = 0
    for i, (c, s) in enumerate(zip(cache, snap, strict=True)):
        if s["kind"] == "kv":
            for nm in ("keys", "values"):
                live = c.keys if nm == "keys" else c.values
                st, n, tot, _, = cmp_status(npf(live), s[nm][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} {nm}: {st} n={n}/{tot}", flush=True)
            if c.offset != s["offset"]:
                nbad += 1
                print(f"  RESTORE-DIFF entry={i} offset {c.offset} vs {s['offset']}", flush=True)
        else:
            for j, live in enumerate(c.cache):
                st, n, tot, _ = cmp_status(npf(live), s["cache"][j][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} arr{j}: {st} n={n}/{tot}", flush=True)
            for nm in ("left_padding", "lengths"):
                st, n, tot, _ = cmp_status(npf(getattr(c, nm)), s[nm][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} {nm}: {st} n={n}/{tot}", flush=True)
    return nbad


# --- layer-3 block hook (harvest) --------------------


class BlockHook:
    """Capture the I/O of every component of the layer-3 full-attention block.

    nn.Module's call path bypasses instance __call__ attributes (verified on
    venv-d), so each module is replaced by a plain-Python wrapper object
    that records I/O and delegates; the originals are restored on exit.
    """

    def __init__(self, layers, layer_idx):
        self.layers = layers
        self.layer_idx = layer_idx
        self.cap = {}
        self._hooks = []
        self.sdpa_calls = []
        self.orig_sdpa = None

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
        self._wrap(L3, "self_attn", "attn")
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
                    "qdtype": str(q.dtype), "kdtype": str(k.dtype), "vdtype": str(v.dtype), "outdtype": str(out.dtype),
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


def cap_weights(L3):
    """Layer-3 submodule weights (pkl-packed + dtype) for the ab re-run."""
    w = {}
    w["input_ln"] = {"weight": snap_w(L3.input_layernorm.weight), "eps": float(L3.input_layernorm.eps)}

    def cap_lin(nm, mod):
        if isinstance(mod, mnn.QuantizedLinear):
            return {
                "quantized": True,
                "bits": int(mod.bits), "group": int(mod.group_size), "mode": mod.mode,
                "in": int(mod.weight.shape[1] * 32 // mod.bits), "out": int(mod.weight.shape[0]),
                "weight": snap_w(mod.weight), "scales": snap_w(mod.scales), "biases": snap_w(mod.biases),
            }
        return {"quantized": False, "weight": snap_w(mod.weight), "bias": snap_w(getattr(mod, "bias", None))}

    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w[f"attn.{nm}"] = cap_lin(nm, getattr(L3.self_attn, nm))
    for nm in ("q_norm", "k_norm"):
        mod = getattr(L3.self_attn, nm)
        w[f"attn.{nm}"] = {"weight": snap_w(mod.weight), "eps": float(mod.eps)}
    r = L3.self_attn.rope
    w["attn.rope"] = {
        "dims": int(r.dims),
        "base": float(r.base),
        "scale": float(r.scale),
        "traditional": bool(r.traditional),
    }
    w["post_ln"] = {"weight": snap_w(L3.post_attention_layernorm.weight), "eps": float(L3.post_attention_layernorm.eps)}
    for nm in ("gate_proj", "up_proj", "down_proj"):
        w[f"mlp.{nm}"] = cap_lin(nm, getattr(L3.mlp, nm))
    return w


def mk_linear(w):
    if w.get("quantized"):
        lin = mnn.QuantizedLinear(w["in"], w["out"], bias=False, bits=w["bits"], group_size=w["group"], mode=w["mode"])
        lin.weight = of_w(w["weight"], w["weight"][1])
        lin.scales = of_w(w["scales"], w["scales"][1])
        if w["biases"][0] is not None:
            lin.biases = of_w(w["biases"], w["biases"][1])
        return lin
    npv, dt = w["weight"]
    lin = mnn.Linear(npv.shape[1], npv.shape[0], bias=w["bias"][0] is not None)
    lin.weight = of_w(w["weight"], dt)
    if w["bias"][0] is not None:
        lin.bias = of_w(w["bias"], w["bias"][1])
    return lin


def mk_rms(w):
    npv, dt = w["weight"]
    n = mnn.RMSNorm(npv.shape[0], eps=w["eps"])
    n.weight = of_w(w["weight"], dt)
    return n


# --- harvest: re-run the m3 flow, capture the layer-3 block for both variants --------------------


def run_m5(lm, cache, out, prev_token, draft, step):
    n_out = len(out)
    print(f"ARM m5: step={step} prev={prev_token} draft={draft} n_out={n_out} out[-8:]={out[-8:]}", flush=True)
    if n_out != EXPECT_N_OUT:
        print(f"ARM MISMATCH: expected n_out=={EXPECT_N_OUT}, got {n_out}", flush=True)
        sys.exit(2)
    if draft != S62_DRAFT:
        print(f"ARM MISMATCH: expected draft=={S62_DRAFT}, got {draft}", flush=True)
        sys.exit(2)
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            print(f"ARM attn offset layer={i}: {c.offset}", flush=True)

    L3 = lm.model.layers[LAYER3]
    print(
        f"layer-3 block: attn={type(L3.self_attn).__name__} mlp={type(L3.mlp).__name__} "
        f"q_proj_out={L3.self_attn.q_proj.weight.shape} rope_dims={L3.self_attn.rope.dims}",
        flush=True,
    )
    for c in cache:
        if isinstance(c, KVCache):
            mx.eval([c.keys, c.values])
        else:
            mx.eval([x for x in c.cache if x is not None])
    snap = snap_cache(cache)
    print(f"snapshot taken: {len(snap)} entries", flush=True)

    # ---- variant A: one T=2 verify forward (hooks on) -------------------
    print(f"== variant A: T=2 verify forward [[{S62_PREV}, {S62_DRAFT}]] (layer-3 block hooked) ==", flush=True)
    with BlockHook(lm.model.layers, LAYER3) as hk:
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
    capA = hk.cap
    sdpaA = hk.sdpa_calls[0]
    print(f"variant A: {len(hk.sdpa_calls)} SDPA calls; layer-3: qshape={sdpaA['qshape']} kshape={sdpaA['kshape']} mask={sdpaA['mask']!r} rope_offset={capA['rope'][0]['kw']['offset']}", flush=True)
    weights = cap_weights(L3)

    # ---- restore + verify the fork ---------------------------------------
    restore_cache(cache, snap)
    nbad = verify_restore(cache, snap)
    print(f"restore verified: {len(snap) - nbad}/{len(snap)} entries bit-exact", flush=True)
    if nbad:
        print("RESTORE NOT BIT-EXACT - the A/B diff is suspect", flush=True)

    # ---- variant B: two T=1 forwards (hooks on the second) ---------------
    print(f"== variant B: T=1 forward [[{S62_PREV}]] ==", flush=True)
    logits_b1, hidden_b1 = lm(
        mx.array([[S62_PREV]]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    mx.eval([logits_b1, hidden_b1])
    print(f"== variant B: T=1 forward [[{S62_DRAFT}]] (layer-3 block hooked) ==", flush=True)
    with BlockHook(lm.model.layers, LAYER3) as hk:
        logits_b2, hidden_b2 = lm(
            mx.array([[S62_DRAFT]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval([logits_b2, hidden_b2])
    capB = hk.cap
    sdpaB = hk.sdpa_calls[0]
    print(f"variant B: {len(hk.sdpa_calls)} SDPA calls; layer-3: qshape={sdpaB['qshape']} kshape={sdpaB['kshape']} mask={sdpaB['mask']!r} rope_offset={capB['rope'][0]['kw']['offset']}", flush=True)

    # ---- cross-checks (determinism vs m3/m4) -----------------------------
    if os.path.exists(M4_PKL):
        with open(M4_PKL, "rb") as f:
            m4 = pickle.load(f)
        for nm in ("q", "k", "v", "out"):
            st, n, tot, _ = cmp_status(sdpaA[nm], m4[nm])
            print(f"captured A sdpa {nm} vs m4 pkl: {st} n={n}/{tot}", flush=True)
        if sdpaA["mask"] != m4["mask"]:
            print(f"WARNING: A sdpa mask {sdpaA['mask']!r} != m4 {m4['mask']!r}", flush=True)
    else:
        print(f"{M4_PKL} not found; skipping the m4 cross-check", flush=True)
    if os.path.exists(M3_PKL):
        with open(M3_PKL, "rb") as f:
            m3 = pickle.load(f)
        for kind, off, k, v in m3["ha"]:
            if kind == "attn" and off == 89:
                st, n, tot, _ = cmp_status(npf(cache[3].keys[..., :89, :]), k)
                print(f"post-A layer-3 k vs m3 pkl: {st} n={n}/{tot}", flush=True)
                st, n, tot, _ = cmp_status(npf(cache[3].values[..., :89, :]), v)
                print(f"post-A layer-3 v vs m3 pkl: {st} n={n}/{tot}", flush=True)
                break
    else:
        print(f"{M3_PKL} not found; skipping the m3 cross-check", flush=True)
    cmp_report("logits after 279  (A[0,0] vs B1[0,0])", npf(logits_a[0, 0]), npf(logits_b1[0, 0]))
    cmp_report("logits after 3545 (A[0,1] vs B2[0,0])", npf(logits_a[0, 1]), npf(logits_b2[0, 0]))
    if capA["rope"][0]["kw"]["offset"] != 87 or capB["rope"][0]["kw"]["offset"] != 88:
        print(f"UNEXPECTED rope offsets: A={capA['rope'][0]['kw']['offset']} B={capB['rope'][0]['kw']['offset']} (expected 87/88)", flush=True)

    with open(M5_PKL, "wb") as f:
        pickle.dump(
            {
                "capA": capA,
                "capB": capB,
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
    print(f"m5 captures saved -> {M5_PKL}", flush=True)
    sys.exit(0)


# --- ab: in-model chain A/B + direct re-run A/B on the captured tensors --------------------


def run_ab():
    if not os.path.exists(M5_PKL):
        print(f"missing {M5_PKL}; run MODE=harvest first", flush=True)
        sys.exit(3)
    with open(M5_PKL, "rb") as f:
        c = pickle.load(f)
    capA, capB = c["capA"], c["capB"]
    sdpaA, sdpaB = c["sdpaA"], c["sdpaB"]
    w = c["weights"]
    print(
        f"loaded: act={c['act']} n_out={c['n_out']} restore_nbad={c['restore_nbad']} "
        f"out_tail={c['out_tail']}",
        flush=True,
    )
    print(
        f"A: qshape={sdpaA['qshape']} kshape={sdpaA['kshape']} mask={sdpaA['mask']!r} "
        f"| B: qshape={sdpaB['qshape']} kshape={sdpaB['kshape']} mask={sdpaB['mask']!r}",
        flush=True,
    )

    # ---- derived components (numpy, from the captured parts) -------------
    def gate_sig_of(cap):
        qkv = cap["q_proj"][0]["out"]
        B, L, _ = qkv.shape
        raw = mx.split(of(qkv, mx.bfloat16).reshape(B, L, N_H, -1), 2, axis=-1)[1].reshape(B, L, -1)
        return np.asarray(mx.sigmoid(raw).astype(mx.float32)).copy()

    def h_of(cap):
        x = cap["block"][0]["in"][0]
        r = cap["o_proj"][0]["out"]
        return npf(of(x, mx.bfloat16) + of(r, mx.bfloat16))

    def sw_of(cap):
        gp = cap["gate_proj"][0]["out"]
        up = cap["up_proj"][0]["out"]
        sig = np.asarray(mx.sigmoid(of(gp, mx.bfloat16)).astype(mx.float32)).copy()
        return npf(of(gp, mx.bfloat16) * of(sig, mx.bfloat16) * of(up, mx.bfloat16))

    gateA, gateB = gate_sig_of(capA), gate_sig_of(capB)
    hA, hB = h_of(capA), h_of(capB)
    swA, swB = sw_of(capA), sw_of(capB)

    # ---- Part 1: the in-model A/B chain (A row 1 vs B) ------------------
    print("== Part 1: in-model A/B chain (A row 1 vs B, dataflow order) ==", flush=True)
    comps = [
        ("1  block_in (GDN-2 out / control)", row1(capA["block"][0]["in"][0]), row0(capB["block"][0]["in"][0])),
        ("2  input_layernorm out", row1(capA["input_ln"][0]["out"]), row0(capB["input_ln"][0]["out"])),
        ("3a q_proj out (raw qkv)", row1(capA["q_proj"][0]["out"]), row0(capB["q_proj"][0]["out"])),
        ("3b k_proj out", row1(capA["k_proj"][0]["out"]), row0(capB["k_proj"][0]["out"])),
        ("3c v_proj out", row1(capA["v_proj"][0]["out"]), row0(capB["v_proj"][0]["out"])),
        ("4a q_norm out", row1(capA["q_norm"][0]["out"]), row0(capB["q_norm"][0]["out"])),
        ("4b k_norm out", row1(capA["k_norm"][0]["out"]), row0(capB["k_norm"][0]["out"])),
        ("5a rope_q out", row1(capA["rope"][0]["out"]), row0(capB["rope"][0]["out"])),
        ("5b rope_k out", row1(capA["rope"][1]["out"]), row0(capB["rope"][1]["out"])),
        ("6  sdpa out", row1(sdpaA["out"]), row0(sdpaB["out"])),
        ("7  gate sigmoid", row1(gateA), row0(gateB)),
        ("8  o_proj in (sdpa*gate)", row1(capA["o_proj"][0]["in"][0]), row0(capB["o_proj"][0]["in"][0])),
        ("9  o_proj out", row1(capA["o_proj"][0]["out"]), row0(capB["o_proj"][0]["out"])),
        ("10 h = x + r", row1(hA), row0(hB)),
        ("11 post_attention_layernorm out", row1(capA["post_ln"][0]["out"]), row0(capB["post_ln"][0]["out"])),
        ("12a gate_proj out", row1(capA["gate_proj"][0]["out"]), row0(capB["gate_proj"][0]["out"])),
        ("12b up_proj out", row1(capA["up_proj"][0]["out"]), row0(capB["up_proj"][0]["out"])),
        ("13 swiglu out (silu*up)", row1(swA), row0(swB)),
        ("14 down_proj out", row1(capA["down_proj"][0]["out"]), row0(capB["down_proj"][0]["out"])),
        ("15 block_out (= layer-4 input)", row1(capA["block"][0]["out"]), row0(capB["block"][0]["out"])),
    ]
    first_diff = None
    for nm, a, b in comps:
        st, n, tot, mx_ = cmp_status(a, b)
        tag = "MATCH" if st == "MATCH" else st
        extra = "" if st == "MATCH" else f" maxabs={mx_:.3e}"
        print(f"  {nm:<40} {tag:<10} n={n}/{tot}{extra}", flush=True)
        if st != "MATCH" and first_diff is None:
            first_diff = nm
    print(f"first diverging component (in-model): {first_diff or 'NONE (whole block bit-exact)'}", flush=True)

    # ---- consistency: captured vs recomputed (in-model) ------------------
    print("== consistency (captured vs recomputed, in-model) ==", flush=True)
    def oin_check(cap, tag):
        qkv = cap["q_proj"][0]["out"]
        B, L, _ = qkv.shape
        raw = mx.split(of(qkv, mx.bfloat16).reshape(B, L, N_H, -1), 2, axis=-1)[1].reshape(B, L, -1)
        oin_re = (of(sdpaA["out"] if tag == "A" else sdpaB["out"], mx.bfloat16).transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(raw))
        cmp_report(f"o_proj in {tag} (captured vs sdpa*gate)", npf(oin_re), cap["o_proj"][0]["in"][0])
    oin_check(capA, "A")
    oin_check(capB, "B")
    cmp_report("post_ln in A (captured vs x+r)", capA["post_ln"][0]["in"][0], hA)
    cmp_report("post_ln in B (captured vs x+r)", capB["post_ln"][0]["in"][0], hB)
    cmp_report("down in A (captured vs silu*up)", capA["down_proj"][0]["in"][0], swA)
    cmp_report("down in B (captured vs silu*up)", capB["down_proj"][0]["in"][0], swB)
    cmp_report("block out A (captured vs h+mlp)", capA["block"][0]["out"], npf(of(hA, mx.bfloat16) + of(capA["down_proj"][0]["out"], mx.bfloat16)))
    cmp_report("block out B (captured vs h+mlp)", capB["block"][0]["out"], npf(of(hB, mx.bfloat16) + of(capB["down_proj"][0]["out"], mx.bfloat16)))

    # ---- Part 2: the direct re-run (M=2 row 1 vs M=1) -------------------
    print("== Part 2: direct re-run (M=2 row 1 vs M=1, captured inputs + weights) ==", flush=True)
    xA = of(capA["block"][0]["in"][0], mx.bfloat16)
    xB = of(capB["block"][0]["in"][0], mx.bfloat16)
    ln = mk_rms(w["input_ln"])
    pln = mk_rms(w["post_ln"])
    qproj, kproj, vproj, oproj = mk_linear(w["attn.q_proj"]), mk_linear(w["attn.k_proj"]), mk_linear(w["attn.v_proj"]), mk_linear(w["attn.o_proj"])
    qnorm, knorm = mk_rms(w["attn.q_norm"]), mk_rms(w["attn.k_norm"])
    gp_l, up_l, down_l = mk_linear(w["mlp.gate_proj"]), mk_linear(w["mlp.up_proj"]), mk_linear(w["mlp.down_proj"])
    rd = w["attn.rope"]

    def chain(x, off, ksdpa, vsdpa, scale, mask):
        ln_x = ln(x)
        qkv = qproj(ln_x)
        B, L, _ = qkv.shape
        queries, gate = mx.split(qkv.reshape(B, L, N_H, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = kproj(ln_x), vproj(ln_x)
        qn = qnorm(queries)
        kn = knorm(keys.reshape(B, L, N_KV, -1))
        qt, kt = qn.transpose(0, 2, 1, 3), kn.transpose(0, 2, 1, 3)
        vt = values.reshape(B, L, N_KV, -1).transpose(0, 2, 1, 3)
        rq = mx.fast.rope(qt, rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=off)
        rk = mx.fast.rope(kt, rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=off)
        sdpa_out = mx.fast.scaled_dot_product_attention(rq, ksdpa, vsdpa, scale=scale, mask=mask)
        oin = sdpa_out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)
        r = oproj(oin)
        h = x + r
        p = pln(h)
        gp, up = gp_l(p), up_l(p)
        sw = mnn.silu(gp) * up
        dp = down_l(sw)
        out = h + dp
        return {
            "ln": ln_x, "qkv": qkv, "keys": keys, "values": values, "qn": qn, "kn": kn,
            "rq": rq, "rk": rk, "vt": vt, "sdpa": sdpa_out, "gate": mx.sigmoid(gate),
            "oin": oin, "r": r, "h": h, "pln": p, "gp": gp, "up": up, "sw": sw,
            "dp": dp, "out": out,
        }

    offA = capA["rope"][0]["kw"]["offset"]
    offB = capB["rope"][0]["kw"]["offset"]
    A2 = chain(xA, offA, of(sdpaA["k"], mx.bfloat16), of(sdpaA["v"], mx.bfloat16), sdpaA["scale"], "causal")
    B2 = chain(xB, offB, of(sdpaB["k"], mx.bfloat16), of(sdpaB["v"], mx.bfloat16), sdpaB["scale"], None)

    # sanity: the re-run reproduces the in-model capture (same kernel path)
    print("-- re-run vs in-model capture (sanity) --", flush=True)
    for nm, a2, b2, ca, cb in [
        ("ln out", A2["ln"], B2["ln"], capA["input_ln"][0]["out"], capB["input_ln"][0]["out"]),
        ("q_proj out", A2["qkv"], B2["qkv"], capA["q_proj"][0]["out"], capB["q_proj"][0]["out"]),
        ("q_norm out", A2["qn"], B2["qn"], capA["q_norm"][0]["out"], capB["q_norm"][0]["out"]),
        ("rope_q out", A2["rq"], B2["rq"], capA["rope"][0]["out"], capB["rope"][0]["out"]),
        ("sdpa out", A2["sdpa"], B2["sdpa"], sdpaA["out"], sdpaB["out"]),
        ("o_proj in", A2["oin"], B2["oin"], capA["o_proj"][0]["in"][0], capB["o_proj"][0]["in"][0]),
        ("o_proj out", A2["r"], B2["r"], capA["o_proj"][0]["out"], capB["o_proj"][0]["out"]),
        ("post_ln out", A2["pln"], B2["pln"], capA["post_ln"][0]["out"], capB["post_ln"][0]["out"]),
        ("gate_proj out", A2["gp"], B2["gp"], capA["gate_proj"][0]["out"], capB["gate_proj"][0]["out"]),
        ("down_proj out", A2["dp"], B2["dp"], capA["down_proj"][0]["out"], capB["down_proj"][0]["out"]),
        ("block out", A2["out"], B2["out"], capA["block"][0]["out"], capB["block"][0]["out"]),
    ]:
        cmp_report(f"{nm} A re-run vs capture", npf(a2), ca)
        cmp_report(f"{nm} B re-run vs capture", npf(b2), cb)

    # the row-invariance test: re-run A row 1 vs re-run B
    print("-- re-run A row 1 vs re-run B (row-invariance) --", flush=True)
    re = [
        ("2  input_layernorm", A2["ln"], B2["ln"]),
        ("3a q_proj", A2["qkv"], B2["qkv"]),
        ("3b k_proj", A2["keys"], B2["keys"]),
        ("3c v_proj", A2["values"], B2["values"]),
        ("4a q_norm", A2["qn"], B2["qn"]),
        ("4b k_norm", A2["kn"], B2["kn"]),
        ("5a rope_q", A2["rq"], B2["rq"]),
        ("5b rope_k", A2["rk"], B2["rk"]),
        ("6  sdpa", A2["sdpa"], B2["sdpa"]),
        ("7  gate sigmoid", A2["gate"], B2["gate"]),
        ("8  o_proj in", A2["oin"], B2["oin"]),
        ("9  o_proj", A2["r"], B2["r"]),
        ("10 h = x + r", A2["h"], B2["h"]),
        ("11 post_attention_layernorm", A2["pln"], B2["pln"]),
        ("12a gate_proj", A2["gp"], B2["gp"]),
        ("12b up_proj", A2["up"], B2["up"]),
        ("13 swiglu", A2["sw"], B2["sw"]),
        ("14 down_proj", A2["dp"], B2["dp"]),
        ("15 block_out", A2["out"], B2["out"]),
    ]
    re_diff = {}
    for nm, a, b in re:
        st, n, tot, mx_ = cmp_status(row1(npf(a)), row0(npf(b)))
        tag = "MATCH" if st == "MATCH" else st
        extra = "" if st == "MATCH" else f" maxabs={mx_:.3e}"
        print(f"  {nm:<40} {tag:<10} n={n}/{tot}{extra}", flush=True)
        re_diff[nm] = st != "MATCH"

    # ---- verdict ---------------------------------------------------------
    print("== m5 verdict ==", flush=True)
    st1, n1, tot1, _ = cmp_status(row1(capA["block"][0]["in"][0]), row0(capB["block"][0]["in"][0]))
    print(f"control (block_in, the GDN-2 out): {'MATCH' if st1 == 'MATCH' else 'DIFF (flaw in the GDN-2 output; probe the GDN-2 split next)'}", flush=True)
    if first_diff is None:
        print("first diverging component: NONE - the WHOLE layer-3 block is bit-exact (m3's chain has a measurement flaw)", flush=True)
    else:
        print(f"first diverging component: {first_diff}", flush=True)
    return {"first_diff": first_diff, "control": st1 == "MATCH", "re_diff": re_diff}


# --- oneshot: one direct call (for rocprofv3 --kernel-trace) --------------------

ONESHOT_SPECS = [
    "ln_A", "ln_B", "qproj_A", "qproj_B", "kproj_A", "kproj_B", "vproj_A", "vproj_B",
    "oproj_A", "oproj_B", "qnorm_A", "qnorm_B", "knorm_A", "knorm_B",
    "ropeq_A", "ropeq_B", "ropek_A", "ropek_B", "sdpa_A", "sdpa_B",
    "sig_A", "sig_B", "silu_A", "silu_B", "addh_A", "addh_B", "addout_A", "addout_B",
]


def run_oneshot():
    if not os.path.exists(M5_PKL):
        print(f"missing {M5_PKL}; run MODE=harvest first", flush=True)
        sys.exit(3)
    with open(M5_PKL, "rb") as f:
        c = pickle.load(f)
    capA, capB = c["capA"], c["capB"]
    sdpaA, sdpaB = c["sdpaA"], c["sdpaB"]
    w = c["weights"]
    bf = mx.bfloat16

    def gate_sig_mx(cap):
        qkv = cap["q_proj"][0]["out"]
        B, L, _ = qkv.shape
        raw = mx.split(of(qkv, bf).reshape(B, L, N_H, -1), 2, axis=-1)[1].reshape(B, L, -1)
        return raw

    xA = of(capA["block"][0]["in"][0], bf)
    xB = of(capB["block"][0]["in"][0], bf)
    ln = mk_rms(w["input_ln"])
    pln = mk_rms(w["post_ln"])
    qproj, kproj, vproj, oproj = mk_linear(w["attn.q_proj"]), mk_linear(w["attn.k_proj"]), mk_linear(w["attn.v_proj"]), mk_linear(w["attn.o_proj"])
    qnorm, knorm = mk_rms(w["attn.q_norm"]), mk_rms(w["attn.k_norm"])
    gp_l, up_l, down_l = mk_linear(w["mlp.gate_proj"]), mk_linear(w["mlp.up_proj"]), mk_linear(w["mlp.down_proj"])
    rd = w["attn.rope"]
    offA = capA["rope"][0]["kw"]["offset"]
    offB = capB["rope"][0]["kw"]["offset"]

    def one(fn):
        r = fn()
        mx.eval(r)
        return r

    fns = {
        "ln_A": lambda: ln(xA), "ln_B": lambda: ln(xB),
        "qproj_A": lambda: qproj(of(capA["input_ln"][0]["out"], bf)), "qproj_B": lambda: qproj(of(capB["input_ln"][0]["out"], bf)),
        "kproj_A": lambda: kproj(of(capA["input_ln"][0]["out"], bf)), "kproj_B": lambda: kproj(of(capB["input_ln"][0]["out"], bf)),
        "vproj_A": lambda: vproj(of(capA["input_ln"][0]["out"], bf)), "vproj_B": lambda: vproj(of(capB["input_ln"][0]["out"], bf)),
        "oproj_A": lambda: oproj(of(capA["o_proj"][0]["in"][0], bf)), "oproj_B": lambda: oproj(of(capB["o_proj"][0]["in"][0], bf)),
        "qnorm_A": lambda: qnorm(of(capA["q_norm"][0]["in"][0], bf)), "qnorm_B": lambda: qnorm(of(capB["q_norm"][0]["in"][0], bf)),
        "knorm_A": lambda: knorm(of(capA["k_norm"][0]["in"][0], bf)), "knorm_B": lambda: knorm(of(capB["k_norm"][0]["in"][0], bf)),
        "ropeq_A": lambda: mx.fast.rope(of(capA["rope"][0]["in"][0], bf), rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=offA),
        "ropeq_B": lambda: mx.fast.rope(of(capB["rope"][0]["in"][0], bf), rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=offB),
        "ropek_A": lambda: mx.fast.rope(of(capA["rope"][1]["in"][0], bf), rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=offA),
        "ropek_B": lambda: mx.fast.rope(of(capB["rope"][1]["in"][0], bf), rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"], offset=offB),
        "sdpa_A": lambda: mx.fast.scaled_dot_product_attention(of(sdpaA["q"], bf), of(sdpaA["k"], bf), of(sdpaA["v"], bf), scale=sdpaA["scale"], mask="causal"),
        "sdpa_B": lambda: mx.fast.scaled_dot_product_attention(of(sdpaB["q"], bf), of(sdpaB["k"], bf), of(sdpaB["v"], bf), scale=sdpaB["scale"], mask=None),
        "sig_A": lambda: mx.sigmoid(gate_sig_mx(capA)), "sig_B": lambda: mx.sigmoid(gate_sig_mx(capB)),
        "silu_A": lambda: mnn.silu(of(capA["gate_proj"][0]["out"], bf)) * of(capA["up_proj"][0]["out"], bf),
        "silu_B": lambda: mnn.silu(of(capB["gate_proj"][0]["out"], bf)) * of(capB["up_proj"][0]["out"], bf),
        "addh_A": lambda: xA + of(capA["o_proj"][0]["out"], bf), "addh_B": lambda: xB + of(capB["o_proj"][0]["out"], bf),
        "addout_A": lambda: (xA + of(capA["o_proj"][0]["out"], bf)) + of(capA["down_proj"][0]["out"], bf),
        "addout_B": lambda: (xB + of(capB["o_proj"][0]["out"], bf)) + of(capB["down_proj"][0]["out"], bf),
    }
    if SPEC not in fns:
        print(f"unknown SPEC={SPEC!r} (use one of {ONESHOT_SPECS})", flush=True)
        sys.exit(3)
    r = one(fns[SPEC]())
    print(f"spec={SPEC} done outshape={tuple(r.shape)}", flush=True)


# --- prof: run every oneshot spec under rocprofv3 --kernel-trace --------------------


def run_prof():
    env = dict(os.environ)
    env["MODE"] = "oneshot"
    print("== m5 kernel dispatch table (rocprofv3 --kernel-trace, gfx1201) ==", flush=True)
    print(f"{'spec':<10} {'kernel':<70} {'grid':<14} {'workgroup':<12} {'LDS':<8} VGPR", flush=True)
    for s in ONSHOT_SPECS:
        env["SPEC"] = s
        outdir = os.path.join(D, f"prof-m5-{s}")
        cmd = [ROCPROF, "--kernel-trace", "-o", outdir, PY, os.path.abspath(__file__)]
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        csvf = outdir + ".csv_kernel_trace.csv"
        seen = []
        if os.path.exists(csvf):
            with open(csvf) as f:
                rd = csv.reader(f)
                next(rd)
                for row in rd:
                    if len(row) < 22 or row[0] != "KERNEL_DISPATCH":
                        continue
                    key = (row[7], row[16], row[17], row[18], row[19], row[20], row[21], row[11], row[13])
                    if key not in seen:
                        seen.append(key)
        if not seen:
            print(f"{s:<10} (no kernel trace)", flush=True)
        for key in seen:
            name = key[0]
            short = name if len(name) <= 68 else name[:65] + "..."
            kind = "copy" if "copy_v" in name else "comp"
            grid = f"({key[4]},{key[5]},{key[6]})"
            wg = f"({key[1]},{key[2]},{key[3]})"
            print(f"{s:<10} {short:<70} {grid:<14} {wg:<12} {key[7]:<8} {key[8]}  [{kind}]", flush=True)


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
        logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
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
                    run_m5(lm, cache, out, prev_token, draft, step)
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
                        cache[i][0], cache[i][1], cache[i].lengths = conv, ssm, lengths
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(2)
                    out.append(t2)
                    for t1 in (prev_token, t2):
                        logits, hidden = lm(mx.array([[t1]]), cache=cache, return_hidden=True, return_prenorm=True)
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
    elif MODE == "oneshot":
        run_oneshot()
    elif MODE == "prof":
        run_prof()
    else:
        print(f"unknown MODE={MODE} (use harvest, ab, oneshot, or prof)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
