#!/usr/bin/env python3
"""Rewrite the bisect snapshot/restore with dtype-correct copies.""" import sys p = sys.argv[1] s = open(p).read() old =
Rewrite the bisect snapshot/restore with dtype-correct copies.""" import sys p = sys.argv[1] s = open(p).read() old ="""
    if a is None:
        return None
    return np.asarray(a.astype(mx.float32)).copy()


def gdn_snap(c, copy_ = False):
    recs = []
    for i, x in enumerate(c):
        if hasattr(x, "cache"):
            if copy_:
                recs.append((i, "g", _copy(x[0]), _copy(x[1]), None))
            else:
                recs.append((i, "g", x[0], x[1], x.lengths))
        else:
            if copy_:
                recs.append((i, "k", _copy(x.keys), _copy(x.values), None))
            else:
                recs.append((i, "k", x.keys, x.values, x.offset))
    return recs


def gdn_restore(c, snap):
    for rec in snap:
        i, kind = rec[0], rec[1]
        if kind == "g":
            if rec[2] is not None and not isinstance(rec[2], mx.array):
                c[i][0] = mx.array(rec[2])  # not used (snap0 is raw)
            c[i][0] = rec[2]
            c[i][1] = rec[3]
            if rec[4] is not None:
                c[i].lengths = rec[4]
        else:
            c[i].keys = rec[2]
            c[i].values = rec[3]
            if rec[4] is not None:
                c[i].offset = rec[4]"""

new = '''def _nb(a):
    if a is None:
        return (None, False)
    return (np.asarray(a.astype(mx.float32)).copy(), a.dtype == mx.bfloat16)


def _mk(np_a, is_bf16):
    if np_a is None:
        return None
    if is_bf16:
        return mx.array(np_a).astype(mx.bfloat16)
    return mx.array(np_a)


def gdn_snap(c):
    """Full copy-based snapshot (immune to in-place cache mutations).""" recs = [] for i, x in enumerate(c): if hasattr(x,
    "cache"): a, ta = _nb(x[0]) b, tb = _nb(x[1]) recs.append((i, "g", a, ta, b, tb, np.asarray(x.lengths).copy()))
    Else: a, ta = _nb(x.keys) b, tb = _nb(x.values) recs.append((i, "k", a, ta, b, tb, x.offset)) return recs def
    Gdn_restore(c, snap): for i, kind, a, ta, b, tb, d in snap: if kind == "g": c[i][0] = _mk(a, ta) c[i][1] = _mk(b,
    Tb) c[i].lengths = mx.array(d) else: c[i].keys = _mk(a, ta) c[i].values = _mk(b, tb) c[i].offset = d''' assert
    S.count(old) == 1 s = s.replace(old, new) old2 ="""
def _hn(a):
    return _hl.md5(np.asarray(a).tobytes()).hexdigest()[:10]
print("BISect postf on:", _hn(s_on[0][2]), _hn(s_on[0][3]), flush=True)
print("BISect postf off:", _hn(s_off[0][2]), _hn(s_off[0][3]), flush=True)"""
new2 = """import hashlib as _hl
def _hn(a):
    return _hl.md5(np.asarray(a).tobytes()).hexdigest()[:10]
print("BISect postf on:", _hn(s_on[0][2]), _hn(s_on[0][4]), flush=True)
print("BISect postf off:", _hn(s_off[0][2]), _hn(s_off[0][4]), flush=True)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)

old3 = "snap0 = gdn_snap(cache)"
new3 = "snap0 = gdn_snap(cache)  # full copy-based (safe across repeated restores)"
assert s.count(old3) == 1
s = s.replace(old3, new3)

old4 = "s_on = gdn_snap(cache, copy_=True)"
new4 = "s_on = gdn_snap(cache)"
assert s.count(old4) == 1
s = s.replace(old4, new4)

old5 = "s_off = gdn_snap(cache, copy_=True)"
new5 = "s_off = gdn_snap(cache)"
assert s.count(old5) == 1
s = s.replace(old5, new5)

old6 = """for (i1, k1, c1, s1v, l1), (i2, k2, c2, s2v, l2) in zip(s_on, s_off):
    if k1 != "g":
        continue
    c1, c2 = np.asarray(c1.astype(mx.float32)), np.asarray(c2.astype(mx.float32))
    s1v, s2v = np.asarray(s1v), np.asarray(s2v)
    cd = int((c1 != c2).sum()) if c1.shape == c2.shape else -1
    sd = int((s1v != s2v).sum()) if s1v.shape == s2v.shape else -1
    if cd or sd:
        print(f"STATE DIVERGENCE layer {i1}: conv {cd} ssm {sd} "
              f"convmax {float(np.abs(c1 - c2).max())} "
              f"ssmmax {float(np.abs(s1v - s2v).max())}", flush=True)"""
new6 = """ndiv = 0
for (i1, k1, a1, _, b1, _, _), (i2, k2, a2, _, b2, _, _) in zip(s_on, s_off):
    if k1 != "g" or a1 is None or a2 is None:
        continue
    cd = int((a1 != a2).sum())
    sd = int((b1 != b2).sum())
    if cd or sd:
        ndiv += 1
        print(f"STATE DIVERGENCE layer {i1}: conv {cd} ssm {sd} "
              f"convmax {float(np.abs(a1 - a2).max())} "
              f"ssmmax {float(np.abs(b1 - b2).max())}", flush=True)
print("state-divergent layers:", ndiv, flush=True)"""
assert s.count(old6) == 1
s = s.replace(old6, new6)

open(p, "w").write(s)
print("patched")
