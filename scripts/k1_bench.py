#!/usr/bin/env python3
"""K1 bench: per-op latency (ms) of K1 (MLX_QMV_LX) vs the wide/warp baseline.

Decode-relevant shape: T=1, batch=1. Settings as used by the QMV probes
(k1_thresh.py pattern): K=4096, gs=64, bits=4, f32 x (the affine
quantized_matmul op runs with f32 x / f32 scales), M=1.

Usage: k1_bench.py <baseline|k1-16|k1-32>
  baseline : no MLX_QMV_LX  -> wide/warp path (MLX_QMV_NO_WIDE unset)
  k1-16    : MLX_QMV_LX=16  -> K1, 16 cols/block
  k1-32    : MLX_QMV_LX=1   -> K1, 32 cols/block

The K1 gate env is static-initialized at first use, so the env is set
before the first op (separate process per config).

Expected accuracy (from submitter's fix report): K1 stages x as int8
(per-group scale) -> ~1% relative error vs the ~3e-6 baseline; the maxrel
column below is a path-confirmation sanity signal, not a pass/fail gate.

NOTE: process hangs at exit on HIP teardown (known); results are printed
with flush before the hang. Wrap in `timeout 900` and kill after the last
N line appears.
"""

import os
import sys
import time

CFG = sys.argv[1]
if CFG == "k1-16":
    os.environ["MLX_QMV_LX"] = "16"
elif CFG == "k1-32":
    os.environ["MLX_QMV_LX"] = "1"
else:
    os.environ.pop("MLX_QMV_LX", None)
os.environ.pop("MLX_QMV_NO_WIDE", None)  # baseline = wide path

import mlx.core as mx  # noqa: E402 (must import after env set)

mx.set_default_device(mx.gpu)
mx.random.seed(42)

K, gs = 4096, 64
WARMUP, REPS = 10, 150
print(f"CFG={CFG} K={K} gs={gs} bits=4 M=1 warmup={WARMUP} reps={REPS}", flush=True)

for N in [1024, 2048, 4096, 8192, 12288]:
    w = mx.random.normal((N, K)) * 0.05
    wq, s, b = mx.quantize(w, gs, 4)
    x = mx.random.normal((1, K))  # f32
    # one-shot accuracy vs f32 dequant reference (path confirmation)
    wref = mx.dequantize(wq, s, b, group_size=gs, bits=4).astype(mx.float32)
    yref = x.astype(mx.float32) @ wref.T
    y = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
    mx.eval(y, yref)
    diff = mx.abs(y.astype(mx.float32) - yref)
    maxabs = float(mx.max(diff).item())
    refmax = float(mx.max(mx.abs(yref)).item())
    for _ in range(WARMUP):
        yy = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
        mx.eval(yy)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        yy = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
        mx.eval(yy)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2] * 1e3
    p95 = ts[int(0.95 * len(ts)) - 1] * 1e3
    print(
        f"CFG={CFG} N={N}: med_ms={med:.4f} p95_ms={p95:.4f} maxrel={maxabs / refmax:.2e}",
        flush=True,
    )
