import os

import mlx.core as mx
import numpy as np

mx.random.seed(0)
N, K = 4096, int(os.environ.get("D3_K", "4096"))
M = int(os.environ.get("D3_M", "4"))
w = mx.random.uniform(0, 1, (N, K))
qw, s, b = mx.quantize(w, 64, 4, mode="affine")
s = s.astype(mx.bfloat16)
b = b.astype(mx.bfloat16)
x = mx.random.uniform(-1, 1, (M, K)).astype(mx.bfloat16)
y = mx.quantized_matmul(x, qw, s, b, transpose=True)
mx.eval(y)
yn = np.array(y.astype(mx.float32))
s32 = s.astype(mx.float32)
b32 = b.astype(mx.float32)
yr = mx.quantized_matmul(x.astype(mx.float32), qw, s32, b32, transpose=True)
mx.eval(yr)
yrn = np.array(yr.astype(mx.float32))
d = np.abs(yn - yrn)
for m in range(M):
    nz = np.isnan(d[m]).sum()
    print(f"row {m}: maxdiff {np.nanmax(d[m]):.4g}  nans={nz}")
