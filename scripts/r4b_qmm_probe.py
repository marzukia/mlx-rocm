#!/usr/bin/env python3
"""R4b probe: which QMM kernel path serves bias-free 4-bit affine M=4 GEMMs?

Hypothesis under test (the R4b lever): in the standing wheel, bias-free
quantized matmul at M=2..4 falls to the per-row qmv_warp_shared path
(fast_grid Y = M -> each output row re-reads the weights), because the
wide/rows/mr dispatch section is gated on `has_bias`. The probe measures:

  1. bias-free M=1 vs M=4 wall (same K/N): ratio ~4x => per-row confirmed.
  2. biased M=1 vs M=4 wall (bias=True, same shapes): ratio ~1.1-1.5x =>
     the fused (rows/wide) section IS reachable when has_bias is true
     (i.e. the gate, not the kernel, is the blocker).
  3. bias-free M=4 with MLX_ROCM_QMV_ROWS set vs unset: identical timing
     => rows is dead for bias-free shapes in the standing wheel.
  4. correctness spot check: bias-free M=4 output == per-row
     quantized_matmul maxdiff (the row-exact contract).

Shapes are the qwen3-9b-mlx model GEMM shapes (4-bit affine, group 64).
GPU box only; run under the host GPU lock. Stdlib + mlx only.
"""

import os
import time

import mlx.core as mx
import mlx.nn as nn

SHAPES = [
    # (name, K, N) — K = in, N = out
    ("outproj/q/o", 4096, 4096),
    ("gate/up", 4096, 12288),
    ("down", 12288, 4096),
    ("lm_head", 4096, 248320),
    ("inproj_z", 4096, 1024),
]

ITERS = 20
WARM = 5


def make_lin(K: int, N: int, bias: bool) -> nn.QuantizedLinear:
    lin = nn.QuantizedLinear(K, N, bias=bias)
    w, scales, _ = mx.quantize(mx.random.normal((N, K)), bits=4, group_size=64)
    lin.weight = w
    lin.scales = scales
    return lin


def bench(f, x) -> float:
    for _ in range(WARM):
        mx.eval(f(x))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(f(x))
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms


def main() -> None:
    rows_on = os.environ.get("MLX_ROCM_QMV_ROWS") is not None
    print(
        f"probe env: QMV_ROWS={os.environ.get('MLX_ROCM_QMV_ROWS')} "
        f"ROWS_MIN_N={os.environ.get('MLX_ROCM_QMV_ROWS_MIN_N')} "
        f"(rows_live={rows_on})"
    )
    for name, K, N in SHAPES:
        lin_nb = make_lin(K, N, bias=False)
        x1 = mx.random.normal((1, K)).astype(mx.bfloat16)
        x4 = mx.random.normal((4, K)).astype(mx.bfloat16)
        t_m1 = bench(lin_nb, x1)
        t_m4 = bench(lin_nb, x4)
        row = f"bias-free {name:12s} K={K:6d} N={N:6d}: M1={t_m1:8.3f} ms  M4={t_m4:8.3f} ms  M4/M1={t_m4 / t_m1:5.2f}x"
        # per-row reference (one row at a time, same op). NOTE: mx.eval in this wheel returns None (side effect only);
        # use the lazy graph and force with float() on a reducing op.
        refs = [lin_nb(x4[i : i + 1]) for i in range(4)]
        out4 = lin_nb(x4)
        maxdiff = 0.0
        for i, r in enumerate(refs):
            maxdiff = max(maxdiff, float(mx.max(mx.abs(out4[i : i + 1] - r))))
        row += f"  maxdiff_vs_perrow={maxdiff:.3e}"
        print(row)
        if N <= 12288:  # biased arm only for the cheap shapes
            lin_b = make_lin(K, N, bias=True)
            tb_m1 = bench(lin_b, x1)
            tb_m4 = bench(lin_b, x4)
            print(
                f"biased     {name:12s} K={K:6d} N={N:6d}: M1={tb_m1:8.3f} ms  "
                f"M4={tb_m4:8.3f} ms  M4/M1={tb_m4 / tb_m1:5.2f}x"
            )
            del lin_b
        del lin_nb, x1, x4


if __name__ == "__main__":
    main()
