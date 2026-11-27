#!/usr/bin/env python3
# S4(1) trace smoke: prove the memory-255 SDK-injected rocprof (1.1.0 build at
# /home/andryo/rcp/new724) yields per-kernel rows with durations for the
# kquant_file_gemm path (the brief's "two-call smoke"). Two uint8 (K, N/256, 84)
# weight views (N=512 and N=1024, K=256, KQ2). Expects >= 2 rows with duration > 0.
import os
import time

import mlx.core as mx

KTYPE, MODE, AXIS, GROUP, KE, PR = 10, 0, 0, 32, 1, 15
print(f"SMOKE-START pid={os.getpid()}", flush=True)
w1 = mx.zeros((256, 2, 84), dtype=mx.uint8)  # N=512 (2 blocks)
x1 = mx.zeros((1, 256))
mx.eval(x1, w1)
t0 = time.perf_counter()
y1 = mx.kquant_file.kquant_file_gemm(x1, w1, KTYPE, MODE, AXIS, GROUP, KE, PR)
mx.eval(y1)
t1 = time.perf_counter()
print(f"CALL1 wall={(t1 - t0):.3f}s t0={t0:.6f} t1={t1:.6f}", flush=True)
del y1
w2 = mx.zeros((256, 4, 84), dtype=mx.uint8)  # N=1024 (4 blocks)
y2 = mx.kquant_file.kquant_file_gemm(x1, w2, KTYPE, MODE, AXIS, GROUP, KE, PR)
mx.eval(y2)
t2 = time.perf_counter()
print(f"CALL2 wall={(t2 - t1):.3f}s t2={t2:.6f}", flush=True)
mx.eval(mx.zeros((1,)))
print("SMOKE-END", flush=True)
