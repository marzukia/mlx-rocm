#!/usr/bin/env python3
"""an earlier run: parse an earlier run bench logs -> in-run bar table.

Input: the 16 bench-<K>x<N>-M<M>.log files (each ends with the
qmv_mr_parse.py QMV: line). Output: per-shape table with in-run M-ratios
(vs M=1), per-call kernel time (us), GB/s, pct of the 640 GB/s measured
peak (method doc: 9070 XT = 640 GB/s, NOT 960), and bar verdicts
(M=1 >= 541 on gate_up; M=2 <= 1.15x; M=3 <= 1.3x; M=4 <= 1.3x).
"""

import glob
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/tmp/qmv-mr/new"
reps = 50

rows = {}
for f in sorted(glob.glob(f"{d}/bench-*.log")):
    with open(f) as fh:
        m = re.search(
            r"QMV: shape=(\d+)x(\d+) M=(\d+) reps=(\d+) kernel_time=([\d.]+) ms "
            r"bytes=(\d+) gbps=([\d.]+)",
            fh.read(),
        )
    assert m, f
    K, N, M, r, ktime, byts, gbps = m.groups()
    rows[(K, N, int(M))] = (float(ktime), int(byts), float(gbps))

shapes = [
    ("4096", "12288", "gate_up"),
    ("12288", "4096", "down"),
    ("4096", "4096", "o_proj"),
    ("4096", "248320", "lm_head"),
]
bars = {2: 1.15, 3: 1.3, 4: 1.3}

print(
    f"{'shape':<22} {'M':>2} {'us/call':>9} {'ms/token':>9} {'GB/s':>7} {'pct640':>7} {'M-ratio':>8} {'bar':>9} verdict"
)
fails = 0
for K, N, tag in shapes:
    base = rows[(K, N, 1)][2]
    for M in (1, 2, 3, 4):
        ktime, byts, gbps = rows[(K, N, M)]
        us = ktime * 1000.0 / reps  # ms/call -> us/call
        # in-run time ratio: time(M)/time(1) — the only drift-free comparison
        ratio = ktime / rows[(K, N, 1)][0]
        if M == 1:
            bar = ">= 541" if tag == "gate_up" else "-"
            verdict = "PASS" if (tag != "gate_up" or gbps >= 541) else "FAIL"
        else:
            bar = f"<= {bars[M]}x"
            verdict = "PASS" if ratio <= bars[M] else "FAIL"
        if M != 1 and verdict == "FAIL":
            fails += 1
        print(
            f"{tag} {K}x{N:<10} {M:>2} {us:>9.1f} {us / 1000:>9.3f} "
            f"{gbps:>7.1f} {100 * gbps / 640:>6.1f}% {ratio:>8.3f} "
            f"{bar:>9} {verdict}"
        )
print(f"BARS-FAILS: {fails}")
