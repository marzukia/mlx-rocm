import os
import pickle
import numpy as np
import mlx.core as mx
MODELS = os.environ.get("MODELS", "/models")

def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None

with open(f"{MODELS}/tmp/run-880/cap-m7-s62.pkl", "rb") as f:
    c = pickle.load(f)
a3 = c["attnA"]
b3 = c["attnB"]["1"]
ra = a3["rope"]
rb = b3["rope"]
print("n rope calls: A=%d B=%d" % (len(ra), len(rb)))
for i, r in enumerate(ra):
    print("A rope[%d] in: %s out: %s" % (
        i, [npf(x).shape for x in r["in"] if x is not None],
        npf(r["out"]).shape if r["out"] is not None else None))
for i, r in enumerate(rb):
    print("B rope[%d] in: %s out: %s" % (
        i, [npf(x).shape for x in r["in"] if x is not None],
        npf(r["out"]).shape if r["out"] is not None else None))

# rope input row 0: A vs B
for i in range(min(len(ra), len(rb))):
    ain = npf(ra[i]["in"][0])
    bin_ = npf(rb[i]["in"][0])
    d = ain[0, :, 0, :] - bin_[0, :, 0, :]
    print("rope[%d] input row0: n=%d/%d maxabs=%.3e" % (
        i, np.count_nonzero(d != 0), d.size, np.abs(d).max()))

# re-run rope on A row-0 input (reshaped to T=1), exact model params
# model: nn.RoPE(dims=64, traditional=False, base=100000, scale=1.0), offset=87
for i in range(min(len(ra), len(rb))):
    x = mx.array(npf(ra[i]["in"][0])[0, :, 0, :].reshape(1, -1, 1, 256).copy())
    y = mx.fast.rope(x, 64, traditional=False, base=100000.0, scale=1.0, offset=87)
    a_out = npf(ra[i]["out"])[0, :, 0, :]
    b_out = npf(rb[i]["out"])[0, :, 0, :]
    d1 = npf(y).reshape(-1, 256) - a_out
    d2 = npf(y).reshape(-1, 256) - b_out
    print("rope[%d] re-run(T=1, off=87) vs A out row0: n=%d/%d maxabs=%.3e" % (
        i, np.count_nonzero(d1 != 0), d1.size, np.abs(d1).max()))
    print("rope[%d] re-run(T=1, off=87) vs B out row0: n=%d/%d maxabs=%.3e" % (
        i, np.count_nonzero(d2 != 0), d2.size, np.abs(d2).max()))

# also: re-run rope on the FULL A input (T=2) and check row 0
for i in range(min(len(ra), len(rb))):
    x = mx.array(npf(ra[i]["in"][0]).copy())
    y = mx.fast.rope(x, 64, traditional=False, base=100000.0, scale=1.0, offset=87)
    a_out = npf(ra[i]["out"])
    d = npf(y) - a_out
    print("rope[%d] re-run(T=2, off=87) vs A out (full): n=%d/%d maxabs=%.3e" % (
        i, np.count_nonzero(d != 0), d.size, np.abs(d).max()))
