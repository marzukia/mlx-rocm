#!/usr/bin/env python3
"""Step 1 (method-correction doc 2026-10-28, section 4 step 1): decode QMV
bench — GB/s from rocprofv3 kernel time + bytes moved (doc section 2 rule 2:
NEVER per-op latency including host op creation / mx.eval).

Model's real shapes/dtypes (doc rule 3): bf16 x, 4-bit affine, gs 64,
has_bias, transposed packed weights. Shapes: 4096->12288 (gate/up),
12288->4096 (down), 4096->4096 (o_proj), 4096->248320 (lm_head); M=1..4.

Run under:  rocprofv3 -d <out> -f json --kernel-trace -- python qmv_mr_bench.py
Env: QMV_SHAPE=<K>x<N>  QMV_M=<1..4>  QMV_WARMUP (default 10)  QMV_REPS (default 50)

Bytes per call = weights (N*K/2) + scales (N*K/64*2) + biases (N*K/64*2)
+ x (M*K*2) + y (M*N*2).
"""

import os

import mlx.core as mx

K = int(os.environ.get("QMV_K", "4096"))
N = int(os.environ.get("QMV_N", "12288"))
M = int(os.environ.get("QMV_M", "1"))
WARMUP = int(os.environ.get("QMV_WARMUP", "10"))
REPS = int(os.environ.get("QMV_REPS", "50"))

assert K % 32 == 0, K
assert 1 <= M <= 4, M

# --- effective config: print + assert (doc rule 6) --------------------------
cfg_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MLX_")}
print(
    f"CONFIG: mlx_wheel={mx.__version__} shape={K}x{N} M={M} warmup={WARMUP} reps={REPS} env={cfg_env}",
    flush=True,
)
assert mx.rocm.is_available(), "mx.rocm not available"

mx.random.seed(42)
w = mx.random.normal((N, K)) * 0.05
wq, s, b = mx.quantize(w, 64, 4)
# uint32 packing: 8 nibbles/element -> (N, K//8). (K//2 was a uint8 assumption.)
assert wq.shape == (N, K // 8), wq.shape
assert s.shape == (N, K // 64), s.shape
assert b.shape == (N, K // 64), b.shape
# Cast scales/biases to bf16 (the model's dtype). The wide/MR gate checks
# x.dtype() but the kernel hard-codes ScaleT=bf16; f32 scales are misread
# (all-zero output on the current wheel) — see REPORT.md section 3.
s = s.astype(mx.bfloat16)
b = b.astype(mx.bfloat16)
x = mx.random.normal((M, K)).astype(mx.bfloat16)
mx.eval(wq, s, b, x)

for _ in range(WARMUP):
    y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
    mx.eval(y)
for _ in range(REPS):
    y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
    mx.eval(y)
assert y.shape == (M, N), y.shape
print("DONE", flush=True)
