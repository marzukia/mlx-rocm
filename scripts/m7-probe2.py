import pickle

import mlx.core as mx
import numpy as np


def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None


with open("/mnt/models/tmp/t880/cap-m7-s62.pkl", "rb") as f:
    c = pickle.load(f)

for i in [0, 1, 2, 4]:
    ca = c["gdnA"][i]
    cb1 = c["gdnB"][f"{i}_1"]
    print(f"== GDN layer {i} ==")
    # shapes
    for nm in [
        "mx_gdn_update",
        "py_gdn_update",
        "norm",
        "gdn_norm_gated",
        "out_proj",
        "normalize_qk",
        "gdn_prep",
        "conv1d",
    ]:
        if nm in ca:
            r = ca[nm][0]
            so = r.get("out")
            sh = [npf(x).shape for x in so] if isinstance(so, list) else (npf(so).shape if so is not None else None)
            print(f"  A {nm:<16} out shapes: {sh}")
        if nm in cb1:
            r = cb1[nm][0]
            so = r.get("out")
            sh = [npf(x).shape for x in so] if isinstance(so, list) else (npf(so).shape if so is not None else None)
            print(f"  B {nm:<16} out shapes: {sh}")


# ---- direct row-0 comparisons (shape-correct) ----
def cmp(name, a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size:
        print(f"  {name:<12} SIZE MISMATCH {a.size} vs {b.size}")
        return
    d = a - b
    print(f"  {name:<12} n={np.count_nonzero(d != 0)}/{d.size} maxabs={np.abs(d).max() if d.size else 0.0:.3e}")


for i in [0, 1, 2, 4]:
    ca = c["gdnA"][i]
    cb1 = c["gdnB"][f"{i}_1"]
    print(f"== layer {i} row 0 direct ==")
    # ssm_out: A mx_gdn_update row 0 vs B mx_gdn_update
    a_ssm = npf(ca["mx_gdn_update"][0]["out"][0]) if "mx_gdn_update" in ca else None
    b_ssm = npf(cb1["mx_gdn_update"][0]["out"][0]) if "mx_gdn_update" in cb1 else None
    if a_ssm is not None and b_ssm is not None:
        a_ssm = a_ssm[0, 0] if a_ssm.ndim >= 3 else a_ssm[0] if a_ssm.ndim == 2 else a_ssm
        b_ssm = b_ssm[0, 0] if b_ssm.ndim >= 3 else b_ssm[0] if b_ssm.ndim == 2 else b_ssm
        cmp("ssm_out", a_ssm, b_ssm)
    # norm_out: A norm (RMSNormGated) row 0 vs B gdn_norm_gated
    a_no = npf(ca["norm"][0]["out"]) if "norm" in ca else None
    b_no = npf(cb1["gdn_norm_gated"][0]["out"][0]) if "gdn_norm_gated" in cb1 else None
    if a_no is not None and b_no is not None:
        a_no = a_no[0, 0] if a_no.ndim >= 3 else a_no[0] if a_no.ndim == 2 else a_no
        b_no = b_no[0, 0] if b_no.ndim >= 3 else b_no[0] if b_no.ndim == 2 else b_no
        cmp("norm_out", a_no, b_no)
    # out_proj row 0
    a_op = npf(ca["out_proj"][0]["out"])
    b_op = npf(cb1["out_proj"][0]["out"])
    a_op = a_op[0, 0] if a_op.ndim >= 3 else a_op[0] if a_op.ndim == 2 else a_op
    b_op = b_op[0, 0] if b_op.ndim >= 3 else b_op[0] if b_op.ndim == 2 else b_op
    cmp("out_proj", a_op, b_op)
    # inputs row 0
    a_in = npf(ca["in_proj_qkv"][0]["in"][0])
    b_in = npf(cb1["in_proj_qkv"][0]["in"][0])
    a_in = a_in[0, 0] if a_in.ndim >= 3 else a_in[0] if a_in.ndim == 2 else a_in
    b_in = b_in[0, 0] if b_in.ndim >= 3 else b_in[0] if b_in.ndim == 2 else b_in
    cmp("inputs", a_in, b_in)
