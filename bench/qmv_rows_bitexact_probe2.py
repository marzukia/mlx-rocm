#!/usr/bin/env python3
"""GPU probe 2 (restructure wheel): bit-exact vs the per-row reference, both
required shape configs, M = 1..4.

  ref   : run with MLX_ROCM_QMV_ROWS=0 (wide kernel only) — save outputs
  check : run with MLX_ROCM_QMV_ROWS=1 + MLX_ROCM_QMV_ROWS_MIN_N (set by the
          runner, printed for the record) — compare, bit-exact

The rows dispatch band is N >= MLX_ROCM_QMV_ROWS_MIN_N (default 16384):
  shape A: N=248320, MIN_N=16384  (default band floor; the lm_head shape)
  shape B: N=12288,  MIN_N=8192   (lowered floor; the gate_up shape)
M=1 stays on the wide path by dispatch (sanity control); M=2..4 hit the
rows kernel. Per-shape timing makes rows activation visible (rows is ~3x
faster at M=4 in the lab data); the check run also prints a wide-timing
reference from the saved ref metadata.

Inputs are seed-deterministic (mx.random.seed(42) per shape/M), so both
processes see identical inputs. Comparison is bit-exact (raw view equality,
float16/bf16 -> uint16, f32 -> uint32).
"""

import os
import sys
import time

import mlx.core as mx
import numpy as np

MODE = sys.argv[1]
OUTDIR = sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    # (K, N, MIN_N, tag)
    (4096, 248320, 16384, "lmhead"),
    (4096, 12288, 8192, "gateup"),
]
print(
    f"CONFIG: mode={MODE} mlx_wheel={mx.__version__} outdir={OUTDIR} "
    f"env={ {k: v for k, v in sorted(os.environ.items()) if k.startswith('MLX_')} }",
    flush=True,
)
assert mx.rocm.is_available(), "mx.rocm not available"

fails = 0
for K, N, MIN_N, tag in CONFIGS:
    for M in (1, 2, 3, 4):
        mx.random.seed(42)
        w = mx.random.normal((N, K)) * 0.05
        wq, s, b = mx.quantize(w, 64, 4)
        s = s.astype(mx.bfloat16)
        b = b.astype(mx.bfloat16)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(wq, s, b, x)
        # warm 1 (allocator/first-launch), timed 5
        for _ in range(1):
            mx.eval(mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4))
        t0 = time.perf_counter()
        nrep = 5
        for _ in range(nrep):
            y = mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
        mx.eval(y)
        ms = (time.perf_counter() - t0) / nrep * 1e3
        mvt = mx.uint32 if y.dtype == mx.float32 else mx.uint16
        arr = np.array(y.view(mvt))
        path = f"{OUTDIR}/{tag}_{K}x{N}_M{M}.npy"
        if MODE == "ref":
            np.save(path, arr)
            np.save(path + ".ms.npy", np.array([ms]))
            print(f"REF: {tag} {K}x{N} M={M} saved {ms:.3f} ms", flush=True)
        else:
            ref = np.load(path)
            refms = float(np.load(path + ".ms.npy").item())
            a, r = arr, ref
            assert a.shape == r.shape, (a.shape, r.shape)
            maxdiff = float(np.max(np.abs(a.astype(np.int64) - r.astype(np.int64))))
            bitexact = bool(np.array_equal(a, r))
            status = "PASS" if (maxdiff == 0.0 and bitexact) else "FAIL"
            if status == "FAIL":
                fails += 1
            print(
                f"CHECK: {tag} {K}x{N} M={M} maxdiff={maxdiff} bitexact={bitexact} "
                f"{ms:.3f} ms (wide ref {refms:.3f} ms, "
                f"speedup={refms / ms:.2f}x) {status}",
                flush=True,
            )
print(f"BITEXACT-GATE: {'FAIL' if fails else 'PASS'} ({fails} failing)", flush=True)
sys.exit(1 if fails else 0)
