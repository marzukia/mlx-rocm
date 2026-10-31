#!/usr/bin/env python3
"""run-889: decode QMV A/B bench — run-852 LX hoist (MLX_QMV_MR_LX=1) vs the
run-825 path (MLX_QMV_MR_LX=0), one process per env (fresh process per run,
the run-835 convention). GB/s from rocprofv3 kernel time + bytes moved
(method doc section 2 rule 2: NEVER per-op latency including host op
creation / mx.eval).

Model's real shapes/dtypes (doc rule 3): bf16 x, 4-bit affine, gs 64,
has_bias, transposed packed weights. Shapes: 4096->12288 (gate/up),
12288->4096 (down — the A/B drift control: K=12288 -> LDS 96 KB > 32 KB
gate, so it runs the per-row qmv_wide_kernel for BOTH env values),
4096->4096 (o_proj), 4096->248320 (lm_head); M=1..4.

Run under:  rocprofv3 -d <out> -f json --kernel-trace -- python
    qmv_mr2_lx_bench.py
Env: MLX_QMV_MR_LX (0 = run-825 path, 1 = run-852 hoist; the dispatch gate
    enables the hoist only when K % 1024 == 0 for 4-bit), QMV_WARMUP
    (default 10), QMV_REPS (default 50).

The combo order is fixed (SHAPES x M below); the parser segments the
dispatch stream by that order. Both env processes run the identical
sequence, so per-position pairing is drift-free except for the
between-process offset, which hits both equally.
"""

import os

import mlx.core as mx

WARMUP = int(os.environ.get("QMV_WARMUP", "10"))
REPS = int(os.environ.get("QMV_REPS", "50"))

SHAPES = [
    (4096, 12288, "gate_up"),
    (12288, 4096, "down"),
    (4096, 4096, "o_proj"),
    (4096, 248320, "lm_head"),
]

# --- effective config: print + assert (doc rule 6) --------------------------
cfg_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MLX_")}
print(
    f"CONFIG: mlx_wheel={mx.__version__} shapes={[f'{K}x{N}' for K, N, _ in SHAPES]} "
    f"M=1..4 warmup={WARMUP} reps={REPS} env={cfg_env}",
    flush=True,
)
assert mx.rocm.is_available(), "mx.rocm not available"

for K, N, tag in SHAPES:
    for M in (1, 2, 3, 4):
        assert K % 32 == 0, K
        mx.random.seed(42)
        w = mx.random.normal((N, K)) * 0.05
        wq, s, b = mx.quantize(w, 64, 4)
        # uint32 packing: 8 nibbles/element -> (N, K//8).
        assert wq.shape == (N, K // 8), wq.shape
        assert s.shape == (N, K // 64), s.shape
        # Cast scales/biases to bf16 (the model's dtype). The wide/MR gate
        # checks x.dtype() but the kernel hard-codes ScaleT=bf16; f32
        # scales are misread (all-zero output) — REPORT.md section 3.
        s = s.astype(mx.bfloat16)
        b = b.astype(mx.bfloat16)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(wq, s, b, x)
        print(f"COMBO: shape={K}x{N} tag={tag} M={M}", flush=True)
        for _ in range(WARMUP):
            y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
            mx.eval(y)
        for _ in range(REPS):
            y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
            mx.eval(y)
        assert y.shape == (M, N), y.shape

print("DONE", flush=True)
