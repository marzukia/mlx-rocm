import pickle

import mlx.core as mx
import numpy as np


def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None


with open("/mnt/models/tmp/t880/cap-m7-s62.pkl", "rb") as f:
    c = pickle.load(f)
a3 = c["attnA"]
b3 = c["attnB"]["1"]
ra = a3["rope"]
rb = b3["rope"]

# re-run rope on A row-0 input as BF16 (the model's actual dtype)
for i in range(min(len(ra), len(rb))):
    x32 = npf(ra[i]["in"][0])[0, :, 0, :].reshape(1, -1, 1, 256)
    x = mx.array(x32).astype(mx.bfloat16)
    y = mx.fast.rope(x, 64, traditional=False, base=100000.0, scale=1.0, offset=87)
    a_out = npf(ra[i]["out"])[0, :, 0, :]
    b_out = npf(rb[i]["out"])[0, :, 0, :]
    d1 = npf(y).reshape(-1, 256) - a_out
    d2 = npf(y).reshape(-1, 256) - b_out
    print(
        f"rope[{i}] re-run(bf16,T=1,off=87) vs A out row0: "
        f"n={np.count_nonzero(d1 != 0)}/{d1.size} maxabs={np.abs(d1).max():.3e}"
    )
    print(
        f"rope[{i}] re-run(bf16,T=1,off=87) vs B out row0: "
        f"n={np.count_nonzero(d2 != 0)}/{d2.size} maxabs={np.abs(d2).max():.3e}"
    )
    # A out row0 vs B out row0 (the ab's comparison)
    d3 = a_out - b_out
    print(f"rope[{i}] A out row0 vs B out row0: n={np.count_nonzero(d3 != 0)}/{d3.size} maxabs={np.abs(d3).max():.3e}")

# full T=2 re-run (bf16)
for i in range(min(len(ra), len(rb))):
    x = mx.array(npf(ra[i]["in"][0])).astype(mx.bfloat16)
    y = mx.fast.rope(x, 64, traditional=False, base=100000.0, scale=1.0, offset=87)
    d = npf(y) - npf(ra[i]["out"])
    print(
        f"rope[{i}] re-run(bf16,T=2,off=87) vs A out (full): "
        f"n={np.count_nonzero(d != 0)}/{d.size} maxabs={np.abs(d).max():.3e}"
    )
