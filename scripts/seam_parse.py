#!/usr/bin/env python3
"""t2231: parse a rocprofv2 kernel-trace CSV into per-step kernel sums.

The driver (seam_step.py) brackets the profiled step with two marker kernel
launches (an elementwise sigmoid over a (17,23) buffer). This parser finds the
first and last marker rows in the trace and reports, for the kernels strictly
between them: count, duration sum (ns), span (first start -> last end),
occupancy = duration sum / span, and the top-10 kernels by duration.

Usage: python seam_parse.py trace.csv [--marker <exact kernel name substring>]
"""
import argparse
import csv
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--marker", default=None, help="substring identifying the marker kernel row")
    args = ap.parse_args()

    with open(args.trace, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        print(f"COLS: {cols}")

        def pick(*needles):
            for c in cols:
                cl = c.lower()
                if all(n in cl for n in needles):
                    return c
            return None

        name_c = pick("kernel", "name") or pick("name")
        start_c = pick("start") or pick("begin")
        dur_c = pick("duration")
        end_c = pick("end")
        grid_c = pick("grid")
        if not name_c:
            print("ERROR: no kernel-name column", file=sys.stderr)
            return 2
        rows = list(reader)

    # Build (start_ns, end_ns, name, grid) tuples.
    items = []
    for r in rows:
        try:
            start = int(float(r[start_c])) if start_c else None
            if dur_c:
                dur = float(r[dur_c])
                end = (start or 0) + dur
            elif end_c:
                end = float(r[end_c])
                dur = end - (start or 0)
            else:
                continue
            items.append((start, end, r[name_c], r.get(grid_c, "") if grid_c else ""))
        except (TypeError, ValueError):
            continue
    if not items:
        print("ERROR: no parseable rows", file=sys.stderr)
        return 2
    items.sort(key=lambda t: (t[0] if t[0] is not None else 0))

    if args.marker:
        marker_rows = [i for i, t in enumerate(items) if args.marker in t[2]]
        if len(marker_rows) < 2:
            print(f"ERROR: only {len(marker_rows)} marker rows (need >=2) for {args.marker!r}", file=sys.stderr)
            print("sample kernel names:", file=sys.stderr)
            for n, _ in sorted(set((t[2], t[3]) for t in items))[:40]:
                print(f"  {n} | {t[3]}", file=sys.stderr)
            return 2
        a, b = min(marker_rows), max(marker_rows)
    else:
        # Fallback: first/last occurrence of the most grid-distinctive row is not
        # safe; require --marker.
        print("ERROR: --marker is required", file=sys.stderr)
        return 2

    step = items[a + 1 : b]
    if not step:
        print("ERROR: no kernels between markers", file=sys.stderr)
        return 2
    durs = [e - s for s, e, _, _ in step]
    span = (step[-1][1] - step[0][0]) / 1e6
    dsum = sum(durs) / 1e6
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for s, e, n, _ in step:
        agg[n] += e - s
        cnt[n] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:10]
    print(f"ROWS_TOTAL {len(items)}")
    print(f"MARKER_A_ROW {a}")
    print(f"MARKER_B_ROW {b}")
    print(f"KERNEL_COUNT {len(step)}")
    print(f"KERNEL_SUM_MS {dsum:.3f}")
    print(f"SPAN_MS {span:.3f}")
    print(f"OCCUPANCY_PCT {100.0 * dsum / span if span > 0 else 0.0:.1f}")
    print("TOP_KERNELS:")
    for n, d in top:
        print(f"  {d / 1e6:9.3f} ms  x{cnt[n]:3d}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
