#!/usr/bin/env python3
"""R4c: draft-head traffic probe - int2 vs the shared head, M=1.

Times one draft-head call (M=1) three ways on the model's own lm_head:
  head   : lm.lm_head(x)               (the OFF arm's per-draft call)
  int2g  : mx.fast.lm_head_int2       (the coarse int2 pass alone)
  int2   : int2 pass + exact top-32 re-rank (the ON arm's full call;
           radiance's "473 us including the rerank" definition)
median over 32 repeated x. Then over 256 seeded random x rows: argmax
agreement (int2 rerank vs the shared head) and max abs logit diff of
the coarse int2 logits vs the shared head. Prints HEAD_PROBE lines.
GPU only.
"""

import os
import time

import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("MTP_MODEL", os.environ.get("MODELS", "/mnt/models") + "/qwen3-9b-mlx")
K = 4096
N_T = 32
N_TEST = 256


def quantize_int2(w, chunk=8192):
    """Int2 g128-group asymmetric quantization, quarter-split 2-bit
    packing (radiance port). Same algebra as bench/mtp-bench6.py."""
    N, K_ = w.shape
    NG, Q = K_ // 128, K_ // 4
    packed = mx.zeros((N, Q), dtype=mx.int32)
    scale = mx.zeros((N, NG), dtype=mx.float32)
    zs = mx.zeros((N, NG), dtype=mx.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        wf = w[i:j].astype(mx.float32).reshape(j - i, NG, 128)
        lo = wf.min(axis=2)
        hi = wf.max(axis=2)
        s = mx.maximum((hi - lo) / 3.0, 1e-8)
        z = mx.minimum(mx.maximum(mx.round(-lo / s), 0.0), 3.0)
        q = (
            mx.minimum(
                mx.maximum(mx.round(wf / s[..., None] + z[..., None]), 0.0),
                3.0,
            )
            .astype(mx.int32)
            .reshape(j - i, K_)
        )
        packed[i:j] = q[:, :Q] | (q[:, Q : 2 * Q] << 2) | (q[:, 2 * Q : 3 * Q] << 4) | (q[:, 3 * Q :] << 6)
        scale[i:j] = s
        zs[i:j] = z * s
    return packed.astype(mx.uint8), scale, zs


def main():
    model, _tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    _h = lm.lm_head
    is_q = type(_h).__name__ == "QuantizedLinear"
    print(f"HEAD type={type(_h).__name__}", flush=True)

    mx.random.seed(0)
    x = mx.random.uniform(0.0, 1.0, (N_TEST, K)).astype(mx.bfloat16)
    x1 = x[:1]

    def timeit(fn):
        for _ in range(4):
            mx.eval(fn())
        ts = []
        for _ in range(N_T):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e6

    t_head = timeit(lambda: lm.lm_head(x1))

    has_int2 = hasattr(mx.fast, "lm_head_int2")
    t_int2 = t_int2g = None
    packed = scale = zs = w_bf16 = None
    if has_int2:
        w_bf16 = (
            mx.dequantize(
                _h.weight,
                _h.scales,
                _h.biases,
                group_size=_h.group_size,
                bits=_h.bits,
            )
            if is_q
            else _h.weight
        )
        packed, scale, zs = quantize_int2(w_bf16)
        mx.eval(packed, scale, zs)
        NG = K // 128
        sx = mx.sum(x1.astype(mx.float32).reshape(NG, 128), axis=1)

        def int2g():
            return mx.fast.lm_head_int2(x1, packed, scale, zs, sx)

        t_int2g = timeit(int2g)

        def int2_full():
            _l, cval, ccol = mx.fast.lm_head_int2(x1, packed, scale, zs, sx)
            perm = mx.argsort(cval.reshape(-1))
            idx = mx.take(ccol.reshape(-1), perm[-32:])
            cols = (
                mx.dequantize(
                    mx.take(_h.weight, idx, 0),
                    mx.take(_h.scales, idx, 0),
                    mx.take(_h.biases, idx, 0),
                    group_size=_h.group_size,
                    bits=_h.bits,
                )
                if is_q
                else mx.take(w_bf16, idx, 0)
            )
            exact = mx.sum(x1.astype(mx.float32) * cols.astype(mx.float32), axis=1)
            return mx.take(idx, mx.argmax(exact))

        t_int2 = timeit(int2_full)

    # accuracy over 256 rows (M=1 op, one row per call, untimed)
    ref_l = lm.lm_head(x)
    ref_am = mx.argmax(ref_l, axis=1)
    if has_int2:
        NG = K // 128
        sxs = mx.sum(x.astype(mx.float32).reshape(N_TEST, NG, 128), axis=2)
        n_match = 0
        maxdiff = 0.0
        for i in range(N_TEST):
            xi = x[i : i + 1]
            l2, cval, ccol = mx.fast.lm_head_int2(xi, packed, scale, zs, sxs[i])
            perm = mx.argsort(cval.reshape(-1))
            idx = mx.take(ccol.reshape(-1), perm[-32:])
            cols = (
                mx.dequantize(
                    mx.take(_h.weight, idx, 0),
                    mx.take(_h.scales, idx, 0),
                    mx.take(_h.biases, idx, 0),
                    group_size=_h.group_size,
                    bits=_h.bits,
                )
                if is_q
                else mx.take(w_bf16, idx, 0)
            )
            exact = mx.sum(xi.astype(mx.float32) * cols.astype(mx.float32), axis=1)
            am = mx.take(idx, mx.argmax(exact))
            n_match += int(mx.all(am == ref_am[i : i + 1]).item())
            maxdiff = max(
                maxdiff,
                float(mx.max(mx.abs(l2.reshape(-1) - ref_l[i : i + 1].astype(mx.float32).reshape(-1)))),
            )
        print(
            f"HEAD_PROBE argmax_match={n_match}/{N_TEST} maxdiff={maxdiff:.4g}",
            flush=True,
        )
        print(
            f"HEAD_PROBE head={t_head:.0f} us int2g="
            f"{(f'{t_int2g:.0f}' if t_int2g is not None else 'off')} us int2="
            f"{(f'{t_int2:.0f}' if t_int2 is not None else 'off')} us"
            f" int2_op={'on' if has_int2 else 'off'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
