#!/usr/bin/env python3
"""Rows-engagement probe (t1307, statistical bar).

Proves the ws[4] multi-row QMV kernel engages when MLX_ROCM_QMV_ROWS=1
(MIN_N via MLX_ROCM_QMV_ROWS_MIN_N) and falls back to the one-row wide
path when unset. Evidence = wall timing on the model's real shapes at
M=4 (the K=3 verify M): rows engages when the N=248320 shape drops to
~1.5 ms (t1297: 1.550 ms rows vs 4.873 ms wide) instead of ~4.9 ms.

Env (set by the wrapper, NOT here): MLX_ROCM_QMV_ROWS / MLX_ROCM_QMV_ROWS_MIN_N
Output: ARM=<rows-off|rows-on> line per shape.
"""

import os

import mlx.core as mx

SHAPES = [(4096, 248320, "lm_head"), (4096, 12288, "gate_up")]
WARMUP = 5
REPS = 20

rows_env = os.environ.get("MLX_ROCM_QMV_ROWS")
min_n = os.environ.get("MLX_ROCM_QMV_ROWS_MIN_N")
print(
    f"CONFIG: mlx_wheel={mx.__version__} "
    f"rows_env={rows_env!r} rows_min_n={min_n!r}",
    flush=True,
)
assert mx.rocm.is_available(), "mx.rocm not available"

import time

for K, N, tag in SHAPES:
    mx.random.seed(42)
    w = mx.random.normal((N, K)) * 0.05
    wq, s, b = mx.quantize(w, 64, 4)
    s = s.astype(mx.bfloat16)
    b = b.astype(mx.bfloat16)
    x = mx.random.normal((4, K)).astype(mx.bfloat16)
    for _ in range(WARMUP):
        mx.eval(mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4))
    ts = []
    for _ in range(REPS):
        ta = time.perf_counter()
        mx.eval(mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4))
        ts.append(time.perf_counter() - ta)
    ts.sort()
    med = ts[len(ts) // 2]
    print(
        f"QMV: K={K} N={N} M=4 {tag}: wall_med_ms={1000 * med:.3f} "
        f"wall_min_ms={1000 * ts[0]:.3f}",
        flush=True,
    )
