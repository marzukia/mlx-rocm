import os
import pickle
import numpy as np
import mlx.core as mx
MODELS = os.environ.get("MODELS", "/models")

def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None

with open(f"{MODELS}/tmp/an earlier run/cap-m7-s62.pkl", "rb") as f:
    c = pickle.load(f)
a3 = c["attnA"]
b3 = c["attnB"]["1"]
b32 = c["attnB"]["2"]
for nm in ["rope", "q_norm", "k_norm", "q_proj"]:
    if nm in a3:
        r = a3[nm][0]
        print("A %-8s in shapes: %s" % (nm, [npf(x).shape for x in r["in"] if x is not None]))
        print("A %-8s out shape:  %s" % (nm, npf(r["out"]).shape if r["out"] is not None else None))
    if nm in b3:
        r = b3[nm][0]
        print("B %-8s in shapes: %s" % (nm, [npf(x).shape for x in r["in"] if x is not None]))
        print("B %-8s out shape:  %s" % (nm, npf(r["out"]).shape if r["out"] is not None else None))

# direct row-1 comparison (offset+1): A row 1 (pos 88) vs B2 row 0 (pos 88)
def row01(a, i):
    a = npf(a)
    return a[0, i] if a.ndim >= 3 else a[i] if a.ndim == 2 else a
for nm in ["rope", "q_norm", "k_norm"]:
    if nm in a3 and nm in b32:
        ar = row01(a3[nm][0]["out"], 1)
        br = row01(b32[nm][0]["out"], 0)
        ar = ar.ravel(); br = br.ravel()
        if ar.size == br.size:
            d = ar - br
            print("row1: %-8s A-row1 vs B2-row0: n=%d/%d maxabs=%.3e" % (
                nm, np.count_nonzero(d != 0), d.size, np.abs(d).max()))
        else:
            print("row1: %-8s size %d vs %d" % (nm, ar.size, br.size))
