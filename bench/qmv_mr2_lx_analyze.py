#!/usr/bin/env python3
"""t889: gain gate (method doc section 2: gain >= 3% to count) — t852 LX
hoist (LX=1) vs the t825 path (LX=0), per-combo kernel-time gain from
the two A/B combo files (one per env, identical sequence, per-position
pairing).

The down shape (K=12288) runs the per-row qmv_wide_kernel for BOTH env
values at M >= 2 (LDS min(4,M)*K*2 > 32 KB) — those rows are the drift
control (any A/B difference on them is session drift, not the lever).
Down M=1 (LDS 24576 <= 32768) is an MR-kernel row like the other
shapes.

Usage: qmv_mr2_lx_analyze.py <combos_LX0.json> <combos_LX1.json>
"""

import json
import sys

with open(sys.argv[1]) as f:
    a = json.load(f)  # LX=0 (t825 path)
with open(sys.argv[2]) as f:
    b = json.load(f)  # LX=1 (t852 hoist)
assert len(a) == len(b) == 16, (len(a), len(b))

print(
    f"{'combo':<24} {'t_LX0 ms':>9} {'t_LX1 ms':>9} {'gain%':>7} {'gbps_LX0':>9} {'gbps_LX1':>9} {'verdict':>8}",
    flush=True,
)

gains = []
control = None
for x, y in zip(a, b, strict=False):
    assert (x["tag"], x["M"]) == (y["tag"], y["M"])
    gain = 100.0 * (x["kernel_ns"] - y["kernel_ns"]) / x["kernel_ns"]
    if x["tag"] == "down":
        control = gain
        verdict = "control"
    else:
        gains.append((x["M"], gain))
        verdict = "PASS" if gain >= 3.0 else "FAIL"
    print(
        f"{x['tag'] + ' M=' + str(x['M']):<24} {x['kernel_ns'] / 1e6:>9.3f} "
        f"{y['kernel_ns'] / 1e6:>9.3f} {gain:>7.2f} {x['gbps']:>9.1f} "
        f"{y['gbps']:>9.1f} {verdict:>8}",
        flush=True,
    )

print(flush=True)
gvals = [g for _, g in gains]
print(f"drift control (down, wide fallback, env-invariant): {control:+.2f}%", flush=True)
print(
    f"MR-shape gains: n={len(gvals)} min={min(gvals):.2f}% "
    f"mean={sum(gvals) / len(gvals):.2f}% "
    f"pass(>=3%)={sum(1 for g in gvals if g >= 3.0)}/{len(gvals)}",
    flush=True,
)
by_m = {}
for m, g in gains:
    by_m.setdefault(m, []).append(g)
for m in sorted(by_m):
    gs = by_m[m]
    print(f"  M={m}: min={min(gs):.2f}% mean={sum(gs) / len(gs):.2f}%", flush=True)

ok = min(gvals) >= 3.0
print(f"GAIN-GATE: {'PASS' if ok else 'FAIL'} (>= 3% on all MR decode combos)", flush=True)
sys.exit(0 if ok else 1)
