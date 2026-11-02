import os

import mlx.core as mx
import numpy as np

mx.random.seed(0)
N, K = 4096, 4096
M = int(os.environ.get("D2_M", "2"))
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
xr = x.astype(mx.float32)
yr = mx.quantized_matmul(xr, qw, s32, b32, transpose=True)
mx.eval(yr)
yrn = np.array(yr.astype(mx.float32))
d = np.abs(yn - yrn)
print("shape", yn.shape, "maxdiff", d.max())
for m in range(M):
    print(f"row {m}: maxdiff {d[m].max():.6g}")
bad = np.argwhere(d[1] > 1e-2) if M > 1 else np.array([[]])
print("bad cells row1:", bad.shape[0])
if bad.shape[0]:
    print("first bad (row, col):", bad[:4].tolist())
    print("col range bad:", bad[:, 1].min(), bad[:, 1].max())
    print("col mod 16:", (bad[:, 1] % 16)[:8].tolist())
