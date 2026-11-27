#!/usr/bin/env python3
"""Decode the iu8 WMMA probe output -> C fragment (m,n) layout.

Patterns 1..32  (A = e_l, B = 1): C != 0 iff row m of the cell contains lane l.
Patterns 33..64 (A = 1, B = e_l): C != 0 iff col n of the cell contains lane l.
Pattern 0 (control) must be 16 everywhere. Patterns 65-67 are cross-checks.
"""
import re
import sys

lines = open(sys.argv[1]).read().splitlines()
cur_p = None
# C[p][lane][reg]
C = {}
for ln in lines:
    m = re.match(r"== pattern (\d+)", ln)
    if m:
        cur_p = int(m.group(1))
        continue
    m = re.match(r"lane\s*(\d+): A=\S+ \S+ B=\S+ \S+ C=(.*)", ln)
    if m and cur_p is not None:
        lane = int(m.group(1))
        vals = [int(x) for x in m.group(2).split()]
        C.setdefault(cur_p, {})[lane] = vals

# sanity: control
ctrl = set()
for lane in range(32):
    ctrl.update(C[0][lane])
print("control values:", sorted(ctrl))
assert ctrl == {16}, f"control not 16: {ctrl}"

# build per-cell masks
a_masks = {}  # (lane, reg) -> 32-bit mask
b_masks = {}
for lane in range(32):
    for reg in range(8):
        am = 0
        bm = 0
        for l in range(32):
            if C[1 + l][lane][reg] != 0:
                am |= 1 << l
            if C[33 + l][lane][reg] != 0:
                bm |= 1 << l
        a_masks[(lane, reg)] = am
        b_masks[(lane, reg)] = bm

distinct_a = sorted(set(a_masks.values()))
distinct_b = sorted(set(b_masks.values()))
print(f"distinct A masks: {len(distinct_a)}")
for i, m in enumerate(distinct_a):
    lanes = [j for j in range(32) if m >> j & 1]
    print(f"  m={i}: lanes {lanes}")
print(f"distinct B masks: {len(distinct_b)}")
for i, m in enumerate(distinct_b):
    lanes = [j for j in range(32) if m >> j & 1]
    print(f"  n={i}: lanes {lanes}")

if len(distinct_a) != 16 or len(distinct_b) != 16:
    print("DECODE-FAIL: expected 16 distinct masks in each")
    sys.exit(1)

ai = {m: i for i, m in enumerate(distinct_a)}
bi = {m: i for i, m in enumerate(distinct_b)}

seen = set()
for lane in range(32):
    for reg in range(8):
        seen.add((ai[a_masks[(lane, reg)]], bi[b_masks[(lane, reg)]]))
ok_grid = len(seen) == 256
if not ok_grid:
    print(f"GRID-FAIL: {len(seen)} distinct (m,n) cells, expected 256")

# compare vs 93 layout: m = 8*(lane>>4) + reg, n = lane % 16
mismatch93 = 0
for lane in range(32):
    for reg in range(8):
        m = ai[a_masks[(lane, reg)]]
        n = bi[b_masks[(lane, reg)]]
        m93 = 8 * (lane >> 4) + reg
        n93 = lane % 16
        if m != m93 or n != n93:
            mismatch93 += 1
print(f"grid complete: {ok_grid}, mismatches vs 93 layout: {mismatch93}/256")
if ok_grid and mismatch93 == 0:
    print("VERDICT: iu8 C layout == 93 layout (m = 8*(lane>>4) + l, n = lane%16)")
else:
    # print the decoded table
    print("decoded (m,n) per (lane, reg):")
    for lane in range(32):
        row = []
        for reg in range(8):
            mm = ai[a_masks[(lane, reg)]]
            nn = bi[b_masks[(lane, reg)]]
            row.append(f"{mm},{nn}")
        print(f"  lane {lane:2d}: " + " ".join(row))
    print("93 layout would be:")
    for lane in range(32):
        row = []
        for reg in range(8):
            mm = 8 * (lane >> 4) + reg
            nn = lane % 16
            row.append(f"{mm},{nn}")
        print(f"  lane {lane:2d}: " + " ".join(row))
