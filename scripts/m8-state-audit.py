#!/usr/bin/env python3
"""m8: FULL state-leak audit + compute-level row-0 T-dependence probe.

Two jobs in one forced-eval (NUMPY COPIES, never reference saves) probe:

1. DYNAMIC PROOF (item 3): across an actual reject/rollback event, every
   buffer bit-identical post-rollback. We fork at step 100:
     - Fork A: one S=2 forward [prev, draft]
     - Fork B: two S=1 forwards [prev] then [draft]
   and compare EVERY state buffer (GDN conv/ssm/lengths, attention
   keys/values/offset) between A and B. Any non-zero diff = a LEAK.

2. COMPUTE-LEVEL CHECK (item 4): if (1) proves all state is restored,
   the residual first-div must be compute-level. We hook every decoder
   layer's __call__ to capture the ROW-0 output (S=2 row 0 vs S=1 row 0)
   and report the exact first diverging layer + row-0 maxabs. This finds
   a T-dependent kernel (the class m6's rope_q row-invariance fix handled).

All state saves are numpy copies (forced eval) per the run-893 trap:
reference saves produce stale reads or GPU-memory-fault hangs.
"""
import os
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
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


def snap_all(cache):
    """Snapshot the FULL target state with numpy copies (forced eval).
    GDN: conv (c[0]), ssm (c[1]), lengths. Attention: keys, values, offset."""
    gdn = []
    kv = []
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            kv.append((i, npf(c.keys), npf(c.values), c.offset))
        else:
            gdn.append((i, npf(c[0]), npf(c[1]),
                       npf(c.lengths) if c.lengths is not None else None))
    return gdn, kv


def cmp_buf(name, a, b):
    if a is None or b is None:
        return f"  {name}: SKIP (None)"
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return f"  {name}: SHAPE {a.shape} vs {b.shape}"
    d = a - b
    n = int(np.count_nonzero(d != 0))
    ma = float(np.abs(d).max()) if d.size else 0.0
    return f"  {name}: n={n}/{d.size} maxabs={ma:.3e}"


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

    # --- Hook every decoder layer to capture row-0 output ---
    layers = lm.model.layers
    row0 = {  # layer_idx -> {"A": np, "B": np}
        i: {"A": None, "B": None} for i in range(len(layers))
    }
    # Hook the decoder-layer class __call__ (dunders are looked up on the
    # type, not the instance; the layers define __call__ directly, not
    # forward). Use id(self) to distinguish instances by index.
    _capture = {"mode": None}
    id_to_idx = {id(layers[i]): i for i in range(len(layers))}
    layer_cls = type(layers[0])
    orig_call = layer_cls.__call__

    def hooked(self, *args, **kwargs):
        out = orig_call(self, *args, **kwargs)
        idx = id_to_idx.get(id(self))
        if idx is not None:
            m = _capture["mode"]
            if m == "A" and row0[idx]["A"] is None:
                row0[idx]["A"] = npf(out[0, 0])
            elif m == "B" and row0[idx]["B"] is None:
                row0[idx]["B"] = npf(out[0, 0])
        return out

    layer_cls.__call__ = hooked

    out = []
    forked = False
    for step in range(130):
        t = int(mx.argmax(lg))
        out.append(t)
        emb = lm.model.embed_tokens(mx.array([[t]]))
        dh = lm.mtp(hidd, emb, mtp_cache)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        prev = t
        if step == FORK_STEP:
            # snapshot the FULL state (numpy copies) before the fork
            gdn_snap, kv_snap = snap_all(cache)
            # Fork A: S=2 [prev, draft]
            _capture["mode"] = "A"
            logitsA, _ = lm(mx.array([[prev, draft]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
            mx.eval([logitsA])
            stateA = snap_all(cache)
            # restore to pre-fork
            for i, conv, ssm, lengths in gdn_snap:
                cache[i][0] = mx.array(conv)
                cache[i][1] = mx.array(ssm)
                cache[i].lengths = mx.array(lengths) if lengths is not None else None
            for i, keys, values, offset in kv_snap:
                cache[i].keys = mx.array(keys)
                cache[i].values = mx.array(values)
                cache[i].offset = offset
            # Fork B: S=1 [prev] then [draft]; only the FIRST S=1 forward is
            # captured (row0[idx]["B"] is set on the first, so the second is
            # skipped by the hook's None check).
            _capture["mode"] = "B"
            logitsB1, _ = lm(mx.array([[prev]]), cache=cache,
                            return_hidden=True, return_prenorm=True)
            mx.eval([logitsB1])
            _capture["mode"] = None  # stop capturing before the second S=1
            logitsB2, _ = lm(mx.array([[draft]]), cache=cache,
                            return_hidden=True, return_prenorm=True)
            mx.eval([logitsB2])
            stateB = snap_all(cache)
            # --- DYNAMIC PROOF: compare every buffer ---
            print(f"step {step}: prev={prev} draft={draft}")
            print("=== DYNAMIC PROOF: state buffers S=2 vs two S=1 (numpy copies) ===")
            any_leak = False
            for (iA, convA, ssmA, lenA), (iB, convB, ssmB, lenB) in zip(stateA[0], stateB[0]):
                r = cmp_buf(f"layer {iA} GDN conv", convA, convB)
                print(r); any_leak |= ("n=0/" not in r and "SKIP" not in r)
                r = cmp_buf(f"layer {iA} GDN ssm", ssmA, ssmB)
                print(r); any_leak |= ("n=0/" not in r and "SKIP" not in r)
                if lenA is not None or lenB is not None:
                    r = cmp_buf(f"layer {iA} GDN lengths", lenA, lenB)
                    print(r); any_leak |= ("n=0/" not in r and "SKIP" not in r)
            for (iA, kA, vA, offA), (iB, kB, vB, offB) in zip(stateA[1], stateB[1]):
                r = cmp_buf(f"layer {iA} KV keys", kA, kB)
                print(r); any_leak |= ("n=0/" not in r and "SKIP" not in r)
                r = cmp_buf(f"layer {iA} KV values", vA, vB)
                print(r); any_leak |= ("n=0/" not in r and "SKIP" not in r)
                if offA != offB:
                    print(f"  layer {iA} KV offset: {offA} vs {offB} (LEAK)")
                    any_leak = True
            print(f"DYNAMIC PROOF: {'LEAK FOUND' if any_leak else 'ALL STATE RESTORED (no leak)'}")
            # --- COMPUTE-LEVEL: per-layer row-0 output T-dependence ---
            print("=== COMPUTE-LEVEL: per-layer row-0 output S=2 vs S=1 ===")
            first_div_layer = -1
            for i in range(len(layers)):
                a = row0[i]["A"]
                b = row0[i]["B"]
                if a is None or b is None:
                    print(f"  layer {i}: SKIP (no capture)")
                    continue
                a = np.asarray(a, dtype=np.float32)
                b = np.asarray(b, dtype=np.float32)
                if a.shape != b.shape:
                    print(f"  layer {i}: SHAPE {a.shape} vs {b.shape}")
                    continue
                d = a - b
                n = int(np.count_nonzero(d != 0))
                ma = float(np.abs(d).max()) if d.size else 0.0
                tag = "" if n == 0 else "  <-- T-DEPENDENT"
                print(f"  layer {i}: n={n}/{d.size} maxabs={ma:.3e}{tag}")
                if n > 0 and first_div_layer < 0:
                    first_div_layer = i
            print(f"COMPUTE-LEVEL: first diverging layer = {first_div_layer}")
            # logits row-0
            dA = npf(logitsA[0, 0]) - npf(logitsB1[0, 0])
            print(f"logits row-0 (A[0,0] vs B1[0,0]): n={np.count_nonzero(dA!=0)}/{dA.size} maxabs={np.abs(dA).max():.3e}")
            forked = True
            break
        # continue the MTP loop
        logits2, hidden2 = lm(mx.array([[prev, draft]]), cache=cache,
                             return_hidden=True, return_prenorm=True)
        t2 = int(mx.argmax(logits2[0, 0]))
        if t2 == draft:
            out.append(t2)
            lg, hidd = logits2[0, 1], hidden2[0, 1]
        else:
            saved = [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if not isinstance(c, KVCache)]
            for i, conv, ssm, lengths in saved:
                cache[i][0] = conv
                cache[i][1] = ssm
                cache[i].lengths = lengths
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(2)
            out.append(t2)
            logits2, hidden2 = lm(mx.array([[prev]]), cache=cache,
                                 return_hidden=True, return_prenorm=True)
            logits2, hidden2 = lm(mx.array([[t2]]), cache=cache,
                                 return_hidden=True, return_prenorm=True)
            lg, hidd = logits2[0, -1], hidden2[0, -1]
    print(f"done: n_out={len(out)} forked={forked} last={out[-8:]}")


if __name__ == "__main__":
    main()
