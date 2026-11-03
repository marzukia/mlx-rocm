#!/usr/bin/env python3
# Row-exact contract check at M=1: wide vs rows-msg (MLX_ROCM_M1_ROWSMSG).
# One process per path (gate is static-init at first use).
import os, sys
import numpy as np
if sys.argv[1] == "rowsmsg":
    os.environ["MLX_ROCM_M1_ROWSMSG"] = "1"
import mlx.core as mx
mx.set_default_device(mx.gpu)
mx.random.seed(123)
gs = 64
for K in (4096, 12288):
    for N in (32, 1024, 4096, 8192, 12288, 248320):
        w = mx.random.normal((N, K)) * 0.05
        wq, s, b = mx.quantize(w, gs, 4)
        x = mx.random.normal((1, K)).astype(mx.bfloat16)
        y = mx.quantized_matmul(x, wq, s, b, group_size=gs, bits=4)
        a = np.array(y, dtype=np.float64)
        np.save(f"/tmp/rmcheck_{sys.argv[1]}_{K}_{N}.npy", a)
        print(f"saved K={K} N={N} max={float(np.max(np.abs(a))):.4f}", flush=True)
print("DONE", flush=True)
