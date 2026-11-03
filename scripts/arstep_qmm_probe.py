#!/usr/bin/env python3
"""AR step change: per-shape M=1 QMM probe (bf16 x, production shapes).

Per-op latency (host op + mx.eval) medians across the production decode
shapes, for the dispatch variant selected by env:
  baseline : wide M=1 (standing config)
  m1lx     : + MLX_ROCM_M1_LX=1 (int8-dot4)
  m1rowsmsg: + MLX_ROCM_M1_ROWSMSG=1 (rows-msg morphology)
  wcols-N  : + MLX_QMV_WCOLS=N (wide block height)

K=4096 gs=64 bits=4 M=1 bf16 x (production dtype). N=248320 = lm_head.
The K=12288 shape = down_proj. Prints med_ms per shape; the model-weighted
per-token GEMM total is computed by the runner.

Process hangs at exit on HIP teardown (known); wrap in `timeout` and kill
after the last line.
"""

import os
import sys
import time

CFG = sys.argv[1]
if CFG == "m1lx":
    os.environ["MLX_ROCM_M1_LX"] = "1"
elif CFG == "m1rowsmsg":
    os.environ["MLX_ROCM_M1_ROWSMSG"] = "1"
elif CFG.startswith("wcols-"):
    os.environ["MLX_QMV_WCOLS"] = CFG.split("-")[1]
os.environ.pop("MLX_QMV_LX", None)
os.environ.pop("MLX_QMV_NO_WIDE", None)

import mlx.core as mx  # noqa: E402 (env must be set before import)

mx.set_default_device(mx.gpu)
mx.random.seed(42)

gs = 64
WARMUP, REPS = 10, 100


def probe(K, N, reps=REPS):
    w = mx.random.normal((N, K)) * 0.05
    wq, s, b = mx.quantize(w, gs, 4)
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    for _ in range(WARMUP):
        yy = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
        mx.eval(yy)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        yy = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
        mx.eval(yy)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2] * 1e3
    p95 = ts[int(0.95 * len(ts)) - 1] * 1e3
    print(f"CFG={CFG} K={K} N={N}: med_ms={med:.4f} p95_ms={p95:.4f}", flush=True)


# (K, N, count per token)
SHAPES = [
    (4096, 32, 48),      # in_proj a/b x 24 GDN layers
    (4096, 1024, 16),    # k/v proj x 8 attn layers
    (4096, 4096, 56),    # in_proj_z + out_proj (24) + o_proj (8)
    (4096, 8192, 32),    # in_proj_qkv (24) + q_proj (8)
    (4096, 12288, 48),   # gate/up (24x2)
    (12288, 4096, 24),   # down_proj
    (4096, 248320, 1),   # lm_head
]
for K, N, cnt in SHAPES:
    probe(K, N, reps=30 if N >= 200000 else REPS)
print("PROBE-DONE", flush=True)
