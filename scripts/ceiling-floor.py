#!/usr/bin/env python3
"""Ceiling report, item 1: re-derive the HBM floor.

Read-only streaming microbench: one H2D init, then mx.sum over a >4 GiB f32
buffer (pure read of the whole buffer per iteration), median of timed iters.
Also a read+write copy arm (x*1.0 into a fresh buffer) for reference. Wall
clock is valid: one iteration moves ~6 GiB (~11 ms at 546 GB/s) so host
op-creation overhead is <1%.

Prints CONFIG / READ_GBPS / COPY_GBPS / DONE. Run under flock with 25 s idle
before the first iter and an SCLK sampler running.
"""

import os
import time

import mlx.core as mx

GB = 1 << 30
NBYTES = int(os.environ.get("FLOOR_NBYTES", 6 * GB))
WARMUP = 5
ITERS = 20

assert NBYTES >= 4 * GB, "buffer must be > 4 GiB"
assert NBYTES % 4 == 0
assert mx.rocm.is_available(), "rocm not available"
print(f"CONFIG n_bytes={NBYTES} warmup={WARMUP} iters={ITERS}", flush=True)

x = mx.ones((NBYTES // 4,), dtype=mx.float32)
y = mx.zeros((NBYTES // 4,), dtype=mx.float32)

for _ in range(WARMUP):
    mx.eval(mx.sum(x))
    mx.eval(y * 1.0)


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2]


t = [time.perf_counter() for _ in range(ITERS)]
reads = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    s = mx.sum(x)
    mx.eval(s)
    reads.append(time.perf_counter() - t0)
assert abs(float(s) - NBYTES / 4) / (NBYTES / 4) < 1e-3

cops = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    o = y * 1.0
    mx.eval(o)
    cops.append(time.perf_counter() - t0)

rg = NBYTES / median(reads) / 1e9
cg = 2 * NBYTES / median(cops) / 1e9
print(
    f"READ_GBPS={rg:.1f} READ_ALL={[f'{b / 1e9 / t:.1f}' for b, t in zip([NBYTES] * ITERS, reads)]}",
    flush=True,
)
print(f"COPY_GBPS={cg:.1f} (read+write, 2x bytes)", flush=True)
print("DONE", flush=True)
