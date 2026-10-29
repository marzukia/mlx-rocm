#!/usr/bin/env python3
"""Step 1 lossless gate (doc section 2 rule 4): bit-exact vs the current
qmv_wide_kernel — maxdiff 0.0 on all four shapes, M=1..4.

Modes:
  ref   : compute + save reference outputs (run with the CURRENT wheel)
  check : compute + compare vs saved references (run with the NEW wheel)

Usage: qmv_mr_bitexact.py <ref|check> <outdir>

Inputs are seed-deterministic (mx.random.seed(42) per shape), so both runs
see identical inputs. Comparison is bit-exact (raw uint16 view equality).
"""

import os
import sys

import mlx.core as mx
import numpy as np

MODE = sys.argv[1]
OUTDIR = sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)

SHAPES = [
    (4096, 12288, "gate_up"),
    (12288, 4096, "down"),
    (4096, 4096, "o_proj"),
    (4096, 248320, "lm_head"),
]
print(
    f"CONFIG: mode={MODE} mlx_wheel={mx.__version__} outdir={OUTDIR} "
    f"env={ {k: v for k, v in sorted(os.environ.items()) if k.startswith('MLX_')} }",
    flush=True,
)
assert mx.rocm.is_available(), "mx.rocm not available"

fails = 0
for K, N, tag in SHAPES:
    for M in (1, 2, 3, 4):
        mx.random.seed(42)
        w = mx.random.normal((N, K)) * 0.05
        wq, s, b = mx.quantize(w, 64, 4)
        # bf16 scales (the model's dtype); f32 scales are misread by the
        # wide/MR kernels (all-zero on the current wheel) — REPORT.md §3.
        s = s.astype(mx.bfloat16)
        b = b.astype(mx.bfloat16)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
        mx.eval(y)
        arr = np.array(y.view(mx.uint16))
        path = f"{OUTDIR}/{tag}_{K}x{N}_M{M}.npy"
        if MODE == "ref":
            np.save(path, arr.view(np.uint16))
            print(f"REF: {tag} {K}x{N} M={M} saved", flush=True)
        else:
            ref = np.load(path)
            a, r = arr.view(np.uint16), ref
            assert a.shape == r.shape, (a.shape, r.shape)
            maxdiff = float(np.max(np.abs(a.astype(np.int32) - r.astype(np.int32))))
            bitexact = bool(np.array_equal(a, r))
            status = "PASS" if (maxdiff == 0.0 and bitexact) else "FAIL"
            if status == "FAIL":
                fails += 1
            print(f"CHECK: {tag} {K}x{N} M={M} maxdiff={maxdiff} bitexact={bitexact} {status}", flush=True)

print(f"BITEXACT-GATE: {'FAIL' if fails else 'PASS'} ({fails} failing)", flush=True)
sys.exit(1 if fails else 0)
