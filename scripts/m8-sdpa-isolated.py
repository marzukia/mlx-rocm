#!/usr/bin/env python3
"""m8 P3: SDPA isolation (in-model A/B calls re-run offline on the SAME arrays).

The FA bisection (m8-fa-bisect) found the SDPA output as the first
T-dependent component (1 bf16 ulp, all 256 dims of head 0) with bit-
identical inputs (q/k/v prefix + prev row all clean; mask A=causal,
B=None). The 1pass vector kernel is row-0 T-invariant by construction
(row 0's used keys are identical for (kL=T+2, causal, qL=2) and
(kL=T+1, none, qL=1)), so this probe re-runs the two in-model SDPA
calls OFFLINE on the captured numpy arrays:

  O1  = sdpa(Q_A[...0], K_A[0:128], V_A[0:128], mask=None)      <- B config
  O2  = sdpa(Q_A[...2rows], K_A[129], V_A[129], mask="causal")  <- A config
  O3  = re-run of O2 (determinism check)
  O4  = sdpa(Q_B, K_B, V_B, mask=None)                          <- B as captured
  compare row 0: O1 vs O2, O4 vs O2, O2 vs O3.

If O1==O2==O3==O4 (bit-exact) -> the in-model diff is NOT the SDPA
kernel on these inputs (re-examine the in-model path). If O2 != O1 ->
isolated kernel/dispatch T-dependence (then env-toggle runs).
"""
import os

import numpy as np
import mlx.core as mx

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = ("Explain in great detail how a modern smartphone works, from the "
          "hardware components to the operating system, covering every "
          "subsystem. Be exhaustive.")
FORK_STEP = 100
EOS = 248044


def npf(a):
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def main():
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    import mlx_lm.models.qwen3_next as qn

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

    cap = {}
    dbg = {}

    orig_sdpa = qn.scaled_dot_product_attention

    def sdpa_wrap(queries, keys, values, cache=None, scale=1.0, mask=None,
                  sinks=None):
        out = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                       mask=mask, sinks=sinks)
        if CAPMODE[0] is not None:
            m = CAPMODE[0]
            cap[m] = (npf(queries), npf(keys), npf(values),
                     npf(out)[0], mask, scale)
        return out

    qn.scaled_dot_product_attention = sdpa_wrap
    CAPMODE = [None]

    # decoder-layer hooks (layers 0-3) for row-0 outputs
    layers = lm.model.layers
    id_to_idx = {id(layers[i]): i for i in range(4)}
    layer_cls = type(layers[0])
    orig_layer = layer_cls.__call__

    def hooked_layer(self, *args, **kwargs):
        out = orig_layer(self, *args, **kwargs)
        idx = id_to_idx.get(id(self))
        if idx is not None and CAPMODE[0] is not None:
            e = cap.setdefault(CAPMODE[0] + f"_layer{idx}", {})
            if "out" not in e:
                o = out[0] if isinstance(out, tuple) else out
                e["out"] = npf(o[0, 0])
                x0 = args[0]
                e["x"] = npf(x0[0, 0] if isinstance(x0, mx.array)
                           else mx.array(x0)[0, 0])
        return out

    layer_cls.__call__ = hooked_layer

    for step in range(130):
        t = int(mx.argmax(lg))
        emb = lm.model.embed_tokens(mx.array([[t]]))
        dh = lm.mtp(hidd, emb, mtp_cache)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        prev = t
        if step == FORK_STEP:
            # numpy-copy state snapshot (record dtypes for restore check)
            gdn = [(i, c[0].dtype, npf(c[0]), c[1].dtype, npf(c[1]))
                  for i, c in enumerate(cache) if hasattr(c, "cache")]
            kv = [(i, c.keys.dtype, npf(c.keys), c.values.dtype,
                  npf(c.values), c.offset)
                 for i, c in enumerate(cache) if isinstance(c, KVCache)]
            dbg["pre_dtypes"] = [(i, str(dt0), str(dt1)) for i, dt0, _, dt1, _ in gdn]
            CAPMODE[0] = "A"
            lm(mx.array([[prev, draft]]), cache=cache,
               return_hidden=True, return_prenorm=True)
            # restore (M8_RESTORE_DTYPE=preserve keeps conv dtype; f32 = old)
            preserve = os.environ.get("M8_RESTORE_DTYPE", "f32") == "preserve"
            for i, dt0, cc, dt1, ss in gdn:
                cache[i][0] = mx.array(cc, dtype=dt0) if preserve else mx.array(cc)
                cache[i][1] = mx.array(ss, dtype=dt1) if preserve else mx.array(ss)
            for i, dtk, kk, dtv, vv, off in kv:
                cache[i].keys = mx.array(kk, dtype=dtk) if preserve else mx.array(kk)
                cache[i].values = mx.array(vv, dtype=dtv) if preserve else mx.array(vv)
                cache[i].offset = off
            dbg["post_restore_dtypes"] = [
                (str(cache[i][0].dtype), str(cache[i][1].dtype))
                for i in range(len(cache)) if hasattr(cache[i], "cache")]
            CAPMODE[0] = "B"
            lm(mx.array([[prev]]), cache=cache,
               return_hidden=True, return_prenorm=True)
            CAPMODE[0] = None
            qn.scaled_dot_product_attention = orig_sdpa
            layer_cls.__call__ = orig_layer
            break
        logits, hidden = lm(mx.array([[prev]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
        lg, hidd = logits[0, -1], hidden[0, -1]

    qA, kA, vA, oA, mA, sA = cap["A"]
    qB, kB, vB, oB, mB, sB = cap["B"]
    print(f"pre-restore dtypes (first 4): {dbg['pre_dtypes'][:4]}")
    print(f"post-restore dtypes (first 4): {dbg['post_restore_dtypes'][:4]}")
    for li in range(4):
        a = cap.get(f"A_layer{li}", {}).get("out")
        b = cap.get(f"B_layer{li}", {}).get("out")
        xa = cap.get(f"A_layer{li}", {}).get("x")
        xb = cap.get(f"B_layer{li}", {}).get("x")
        if a is None or b is None:
            print(f"  layer{li} out: SKIP")
            continue
        d = a - b
        n = int(np.count_nonzero(d != 0))
        print(f"  layer{li} out row0: n={n}/{d.size} "
              f"maxabs={np.abs(d).max():.3e}")
        if xa is not None and xb is not None:
            di = xa - xb
            ni = int(np.count_nonzero(di != 0))
            print(f"  layer{li} input row0: n={ni}/{di.size} "
                  f"maxabs={np.abs(di).max():.3e}")
    print(f"A: q{qA.shape} k{kA.shape} v{vA.shape} mask={mA!r} scale={sA}")
    print(f"B: q{qB.shape} k{kB.shape} v{vB.shape} mask={mB!r} scale={sB}")
    # input identity
    m = min(kA.shape[2], kB.shape[2])
    d = kA[:, :, :m] - kB[:, :, :m]
    print(f"k prefix n={np.count_nonzero(d != 0)}/{d.size} "
          f"maxabs={np.abs(d).max():.3e}")
    d = vA[:, :, :m] - vB[:, :, :m]
    print(f"v prefix n={np.count_nonzero(d != 0)}/{d.size} "
          f"maxabs={np.abs(d).max():.3e}")
    dq = qA[0, :, 0] - qB[0, :, 0]
    print(f"q row0 n={np.count_nonzero(dq != 0)}/{dq.size} "
          f"maxabs={np.abs(dq).max():.3e}")
    din = oA[0, :, 0] - oB[0, :, 0]
    print(f"in-model out row0 n={np.count_nonzero(din != 0)}/{din.size} "
          f"maxabs={np.abs(din).max():.3e}")

    # rebuild inputs: bf16 (A always bf16; B bf16 or f32) + f32-preserved
    # variants so B's f32 values are not downcast in the f32-restore run.
    qa = mx.array(np.asarray(qA, dtype=np.float32), dtype=mx.bfloat16)
    qb = mx.array(np.asarray(qB, dtype=np.float32), dtype=mx.bfloat16)
    ka = mx.array(np.asarray(kA, dtype=np.float32), dtype=mx.bfloat16)
    kb = mx.array(np.asarray(kB, dtype=np.float32), dtype=mx.bfloat16)
    va = mx.array(np.asarray(vA, dtype=np.float32), dtype=mx.bfloat16)
    vb = mx.array(np.asarray(vB, dtype=np.float32), dtype=mx.bfloat16)
    qaf = mx.array(np.asarray(qA, dtype=np.float32))
    qbf = mx.array(np.asarray(qB, dtype=np.float32))
    kaf = mx.array(np.asarray(kA, dtype=np.float32))
    kbf = mx.array(np.asarray(kB, dtype=np.float32))
    vaf = mx.array(np.asarray(vA, dtype=np.float32))
    vbf = mx.array(np.asarray(vB, dtype=np.float32))

    O1 = npf(mx.fast.scaled_dot_product_attention(
        qa[:, :, 0:1, :], ka[:, :, :m, :], va[:, :, :m, :], scale=sA,
        mask=None))[0, :, 0]
    O2 = npf(mx.fast.scaled_dot_product_attention(
        qa, ka, va, scale=sA, mask="causal"))[0, :, 0]
    O3 = npf(mx.fast.scaled_dot_product_attention(
        qa, ka, va, scale=sA, mask="causal"))[0, :, 0]
    O4 = npf(mx.fast.scaled_dot_product_attention(
        qb, kb, vb, scale=sB, mask=None))[0, :, 0]
    O4f = npf(mx.fast.scaled_dot_product_attention(
        qbf, kbf, vbf, scale=sB, mask=None))[0, :, 0]
    O2f = npf(mx.fast.scaled_dot_product_attention(
        qaf, kaf, vaf, scale=sA, mask="causal"))[0, :, 0]

    def cmp(name, a, b):
        d = a - b
        n = int(np.count_nonzero(d != 0))
        ma = float(np.abs(d).max()) if d.size else 0.0
        print(f"{name}: n={n}/{d.size} maxabs={ma:.3e}")

    cmp("O1(Bcfg) vs O2(Acfg)", O1, O2)
    cmp("O4(B-captured) vs O2(Acfg)", O4, O2)
    cmp("O2 vs O3 (determinism)", O2, O3)
    oA0 = oA[:, 0, :]  # in-model A out, all heads row 0
    oB0 = oB[:, 0, :]
    cmp("in-model A vs O2 (A-path determinism)", oA0, O2)
    cmp("in-model B vs O4 (bf16 downcast)", oB0, O4)
    cmp("in-model B vs O4f (f32 preserved)", oB0, O4f)
    cmp("O2 vs O2f (dtype effect, same values)", O2, O2f)
    cmp("in-model A vs in-model B", oA0, oB0)
    print("done")


if __name__ == "__main__":
    main()
