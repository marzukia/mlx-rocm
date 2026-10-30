#!/usr/bin/env python3
"""m8 P3: FA layer-3 component bisection (S=2 verify vs S=1 decode, row 0).

P2 localized the row-0 T-dependence to the layer-3 full-attention block
(state clean, row-0 output T-dependent). This probe bisects the FA
block's components at the same fork (step 100, single fork):

    A = one S=2 forward [prev, draft]
    B = first S=1 forward [prev]   (row 0 only; the second S=1 is not
                                    captured)

Captured per component (f32 numpy, forced eval; numpy copies only, never
reference saves):
  - layer 0-3 decoder-layer row-0 output (GDN 0-2 expected clean; FA 3
    the diverging layer, per P2)
  - FA3: input_layernorm, q_proj, q_norm, k_proj, k_norm, v_proj,
    rope (queries after rope), SDPA (output row 0 of head 0 + the mask
    + the last-2 key/value rows, to confirm inputs are bit-identical),
    o_proj, post_attention_layernorm

First component with a non-zero diff = the T-dependent quantity.
Class-level __call__ hooks (dunders resolve on the type), keyed by
id(self); the module-level scaled_dot_product_attention is wrapped in
the qwen3_next namespace (where Qwen3NextAttention looks it up).
"""
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache
import mlx_lm.models.qwen3_next as qn
MODELS = os.environ.get("MODELS", "/models")

MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = ("Explain in great detail how a modern smartphone works, from the "
          "hardware components to the operating system, covering every "
          "subsystem. Be exhaustive.")
FORK_STEP = 100
EOS = 248044


def npf(a):
    """mx array -> f32 numpy (forced eval). None -> None."""
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


class C:
    """capture state"""

    def __init__(self):
        self.mode = None
        self.cap = {}  # name -> {"A": np, "B": np, "meta": ...}

    def put(self, name, arr, meta=None):
        m = self.mode
        if m is None:
            return
        e = self.cap.setdefault(name, {"A": None, "B": None, "meta": {}})
        if e[m] is None:
            e[m] = npf(arr)
            if meta:
                e["meta"][m] = meta


CAP = C()


def hook_cls(cls, id_map):
    """Wrap cls.__call__; capture row-0 for the keyed instances."""
    orig = cls.__call__

    def hooked(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        name = id_map.get(id(self))
        if name is not None:
            CAP.put(name, out[0, 0] if hasattr(out, "shape") and out.ndim >= 2 else out)
        return out

    cls.__call__ = hooked
    return orig


def main():
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache,
                       return_hidden=True, return_prenorm=True)
    lg = logits[0, -1]
    hidd = hidden[0, -1]

    layers = lm.model.layers
    fa3 = layers[3].self_attn

    # --- decoder-layer row-0 output hooks (layer 0-3) ---
    id_to_idx = {id(layers[i]): i for i in range(4)}
    layer_cls = type(layers[0])
    orig_layer = layer_cls.__call__

    def hooked_layer(self, *args, **kwargs):
        out = orig_layer(self, *args, **kwargs)
        idx = id_to_idx.get(id(self))
        if idx is not None:
            CAP.put(f"layer{idx}_out", out[0, 0])
        return out

    layer_cls.__call__ = hooked_layer

    # --- FA3 component hooks ---
    id_map = {
        id(layers[3].input_layernorm): "fa3_input_layernorm",
        id(fa3.q_proj): "fa3_q_proj",
        id(fa3.q_norm): "fa3_q_norm",
        id(fa3.k_proj): "fa3_k_proj",
        id(fa3.k_norm): "fa3_k_norm",
        id(fa3.v_proj): "fa3_v_proj",
        id(fa3.rope): "fa3_rope",
        id(fa3.o_proj): "fa3_o_proj",
        id(layers[3].post_attention_layernorm): "fa3_post_attn_layernorm",
    }
    orig_rms = hook_cls(type(fa3.q_norm), id_map)
    orig_lin = hook_cls(type(fa3.q_proj), id_map)
    orig_rope = hook_cls(type(fa3.rope), id_map)

    # --- SDPA module-level wrap (qwen3_next namespace) ---
    orig_sdpa = qn.scaled_dot_product_attention

    def sdpa_wrap(queries, keys, values, cache=None, scale=1.0, mask=None,
                  sinks=None):
        out = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                       mask=mask, sinks=sinks)
        if CAP.mode is not None:
            CAP.put("fa3_sdpa_out", out[0, 0, 0])
            CAP.put("fa3_sdpa_q_row0", queries[0, 0, 0])
            # full keys/values (numpy copies); the report aligns the shared
            # prefix (rows 0..lenB-1) and flags A's extra last row (draft).
            CAP.put("fa3_sdpa_k", keys)
            CAP.put("fa3_sdpa_v", values)
            CAP.put("fa3_sdpa_mask", np.array([0]) if mask is None else
                   (np.array([1]) if mask == "causal" else np.asarray(mask)[0, 0, 0]))
        return out

    qn.scaled_dot_product_attention = sdpa_wrap

    out_tokens = []
    for step in range(130):
        t = int(mx.argmax(lg))
        out_tokens.append(t)
        emb = lm.model.embed_tokens(mx.array([[t]]))
        dh = lm.mtp(hidd, emb, mtp_cache)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        prev = t
        if step == FORK_STEP:
            # numpy-copy state snapshot before the fork
            gdn = [(i, npf(c[0]), npf(c[1])) for i, c in enumerate(cache)
                   if hasattr(c, "cache")]
            kv = [(i, npf(c.keys), npf(c.values), c.offset)
                 for i, c in enumerate(cache) if isinstance(c, KVCache)]
            # Fork A: S=2 [prev, draft]
            CAP.mode = "A"
            logitsA, _ = lm(mx.array([[prev, draft]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
            mx.eval([logitsA])
            # restore pre-fork
            for i, cc, ss in gdn:
                cache[i][0] = mx.array(cc)
                cache[i][1] = mx.array(ss)
            for i, kk, vv, off in kv:
                cache[i].keys = mx.array(kk)
                cache[i].values = mx.array(vv)
                cache[i].offset = off
            # Fork B: first S=1 [prev]
            CAP.mode = "B"
            logitsB, _ = lm(mx.array([[prev]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
            mx.eval([logitsB])
            CAP.mode = None

            d0 = npf(logitsA[0, 0]) - npf(logitsB[0, 0])
            print(f"step {step}: prev={prev} draft={draft}")
            print(f"logits row0: n={np.count_nonzero(d0 != 0)}/{d0.size} "
                  f"maxabs={np.abs(d0).max():.3e}")
            print("=== FA layer-3 component bisection (row 0, A=S=2 vs B=S=1) ===")
            order = ["fa3_input_layernorm", "fa3_q_proj", "fa3_q_norm",
                     "fa3_rope", "fa3_k_proj", "fa3_k_norm", "fa3_v_proj",
                     "fa3_sdpa_q_row0", "fa3_sdpa_k", "fa3_sdpa_v",
                     "fa3_sdpa_out", "fa3_o_proj",
                     "fa3_post_attn_layernorm", "layer3_out",
                     "layer0_out", "layer1_out", "layer2_out"]
            first = None
            for name in order:
                e = CAP.cap.get(name)
                if e is None or e["A"] is None or e["B"] is None:
                    print(f"  {name}: SKIP")
                    continue
                a = np.asarray(e["A"], dtype=np.float32)
                b = np.asarray(e["B"], dtype=np.float32)
                if name in ("fa3_sdpa_k", "fa3_sdpa_v"):
                    # align the shared prefix (A has one extra row: draft)
                    m = min(a.shape[2], b.shape[2])
                    d = a[:, :, :m, :] - b[:, :, :m, :]
                    n = int(np.count_nonzero(d != 0))
                    ma = float(np.abs(d).max()) if d.size else 0.0
                    extra = ("A has extra last row (draft)" if a.shape[2] > b.shape[2]
                             else "")
                    tag = "  <-- T-DEPENDENT" if n > 0 else ""
                    print(f"  {name}: prefix n={n}/{d.size} maxabs={ma:.3e} "
                          f"kL A={a.shape[2]} B={b.shape[2]} {extra}{tag}")
                    if n > 0 and first is None:
                        first = name
                    continue
                if a.shape != b.shape:
                    print(f"  {name}: SHAPE {a.shape} vs {b.shape}  <-- T-DEP?")
                    if first is None:
                        first = name
                    continue
                d = a - b
                n = int(np.count_nonzero(d != 0))
                ma = float(np.abs(d).max()) if d.size else 0.0
                tag = "  <-- T-DEPENDENT" if n > 0 else ""
                print(f"  {name}: n={n}/{d.size} maxabs={ma:.3e}{tag}")
                if n > 0 and first is None:
                    first = name
            # SDPA input metadata
            ma = CAP.cap.get("fa3_sdpa_mask", {}).get("A")
            mb = CAP.cap.get("fa3_sdpa_mask", {}).get("B")
            print(f"  sdpa mask: A={ma} B={mb}")
            print(f"FIRST DIVERGING COMPONENT: {first}")
            break
        # continue decode
        logits, hidden = lm(mx.array([[prev]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
        lg, hidd = logits[0, -1], hidden[0, -1]
    # restore
    layer_cls.__call__ = orig_layer
    type(fa3.q_norm).__call__ = orig_rms
    type(fa3.q_proj).__call__ = orig_lin
    type(fa3.rope).__call__ = orig_rope
    qn.scaled_dot_product_attention = orig_sdpa
    print(f"done: n_out={len(out_tokens)}")


if __name__ == "__main__":
    main()
