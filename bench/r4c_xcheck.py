#!/usr/bin/env python3
"""R4c xcheck: the int2 op's per-call cost in a tight warm loop,
bench6-style (per-row M=1 calls) vs standalone, plus the shared
QuantizedLinear head for scale. Prints XCHECK lines. GPU only."""

import os
import time

import mlx.core as mx
from mlx_lm import load
from r4c_head_probe import quantize_int2

MODEL = os.environ.get("MTP_MODEL", os.environ.get("MODELS", "<models-mount>") + "/qwen3-9b-mlx")


def main():
    model, _tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    _h = lm.lm_head
    assert type(_h).__name__ == "QuantizedLinear"
    w = mx.dequantize(_h.weight, _h.scales, _h.biases, group_size=_h.group_size, bits=_h.bits)
    N, K = w.shape
    NG = K // 128
    packed, scale, zs = quantize_int2(w)
    mx.eval(packed, scale, zs)

    mx.random.seed(0)
    x = mx.random.uniform(0.0, 1.0, (8, K)).astype(mx.bfloat16)
    rows = [x[i : i + 1] for i in range(8)]
    sxs = [mx.sum(xx.astype(mx.float32).reshape(NG, 128), axis=1) for xx in rows]

    has_wide = hasattr(mx.fast, "lm_head_int2_wide")

    # warmup (kernel load + L2 warm)
    for _ in range(3):
        for i in range(8):
            mx.eval(mx.fast.lm_head_int2(rows[i], packed, scale, zs, sxs[i]))
        for i in range(8):
            mx.eval(lm.lm_head(rows[i]))
        if has_wide:
            for i in range(8):
                mx.eval(mx.fast.lm_head_int2_wide(rows[i], packed, scale, zs, sxs[i]))

    def loop(fn, iters=10):
        t0 = time.perf_counter()
        for _ in range(iters):
            for i in range(8):
                mx.eval(fn(i))
        return (time.perf_counter() - t0) / iters / 8 * 1e6

    t_int2 = loop(lambda i: mx.fast.lm_head_int2(rows[i], packed, scale, zs, sxs[i]))
    t_head = loop(lambda i: lm.lm_head(rows[i]))
    t_wide = (
        loop(lambda i: mx.fast.lm_head_int2_wide(rows[i], packed, scale, zs, sxs[i]))
        if has_wide
        else None
    )
    # Losslessness probe: on 256 seeded rows, the wide op's re-rank selection
    # (top-32 candidate set + argmax token) vs the scalar int2 op.
    sel_match = None
    if has_wide:
        mx.random.seed(42)
        xw = mx.random.uniform(0.0, 1.0, (256, K)).astype(mx.bfloat16)
        nmatch = 0
        for i in range(256):
            xx = xw[i : i + 1]
            sxx = mx.sum(xx.astype(mx.float32).reshape(NG, 128), axis=1)
            _, cv1, cc1 = mx.fast.lm_head_int2(xx, packed, scale, zs, sxx)
            _, cv2, cc2 = mx.fast.lm_head_int2_wide(xx, packed, scale, zs, sxx)
            p1 = mx.argsort(cv1.reshape(-1)); i1 = mx.take(cc1.reshape(-1), p1[-32:])
            p2 = mx.argsort(cv2.reshape(-1)); i2 = mx.take(cc2.reshape(-1), p2[-32:])
            s1 = set(mx.eval(i1).tolist()); s2 = set(mx.eval(i2).tolist())
            if s1 == s2:
                nmatch += 1
        sel_match = nmatch
        print(f"XCHECK sel_match={nmatch}/256", flush=True)
    print(
        f"XCHECK int2_loop={t_int2:.0f}us head_loop={t_head:.0f}us wide_loop={t_wide if t_wide is not None else 'NA'}us ratio={t_int2 / t_head:.2f} wide_vs_head={t_wide / t_head:.2f if t_wide else 'NA'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
