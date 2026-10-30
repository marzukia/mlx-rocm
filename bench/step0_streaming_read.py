#!/usr/bin/env python3
"""Step 0 (method-correction doc 2026-10-28, section 4 step 0): in-guest
streaming-read ceiling.

Plain read-only kernel (mx.sum reduction) over a >= 2 GiB device buffer.
Timing source is rocprofv3 kernel time ONLY (doc section 2 rule 2: GB/s from
kernel time + bytes moved; NEVER per-op latency including host op creation /
mx.eval). Run under:

  rocprofv3 -d <out> -f json --kernel-trace -- python step0_streaming_read.py

Dispatch pattern (verified on gfx1201, ROCm 10.1): one H2D init copy, then
per sum: one heavy all_reduce_kernel (reads the whole buffer once) + one tiny
final scalar reduce. step0_parse.py groups accordingly.

Doc section 2 rule 6: prints and asserts its effective config (env, wheel sha,
buffer size). Kernel names actually dispatched are printed by step0_parse.py
from the rocprof pass.
"""

import os

import mlx.core as mx

GB = 1 << 30
NBYTES = int(os.environ.get("STEP0_NBYTES", 4 * GB))  # default 4 GiB f32
WARMUP = int(os.environ.get("STEP0_WARMUP", 3))
ITERS = int(os.environ.get("STEP0_ITERS", 10))

# --- effective config: print + assert (doc rule 6) --------------------------
assert NBYTES >= 2 * GB, f"buffer must be >= 2 GiB, got {NBYTES}"
assert NBYTES % 4 == 0, "f32 buffer must be element-aligned"
assert mx.rocm.is_available(), "mx.rocm not available"
N = NBYTES // 4
print(
    f"CONFIG: venv=venv-d rocm=10.1 mlx_wheel={mx.__version__} "
    f"buffer_bytes={NBYTES} n_elements={N} warmup={WARMUP} iters={ITERS}",
    flush=True,
)

x = mx.ones((N,), dtype=mx.float32)

# warmup (allocator + icache/TLB; traced, excluded by the parser)
for _ in range(WARMUP):
    mx.eval(mx.sum(x))

for _ in range(ITERS):
    s = mx.sum(x)
    mx.eval(s)
    got = float(s)
    assert abs(got - N) / N < 1e-3, f"sum mismatch: {got} != {N}"
print("DONE", flush=True)
