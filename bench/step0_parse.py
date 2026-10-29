#!/usr/bin/env python3
"""Parse the rocprofv3 JSON trace from step0_streaming_read.py.

Usage: step0_parse.py <trace.json> <iters> <warmup> <nbytes>

Dispatch pattern (verified on gfx1201): [init H2D copy]
  [warmup: (heavy all_reduce, tiny final reduce) * warmup]
  [measure: (heavy all_reduce, tiny final reduce) * iters]

GB/s = bytes moved / rocprofv3 kernel time (doc section 2 rule 2).
Per measured iteration: bytes = nbytes (read) + 8 (partial read + scalar write,
negligible). Prints kernel names actually dispatched (doc section 2 rule 6)
and asserts the effective config.
"""

import json
import statistics
import sys

path, iters, warmup, nbytes = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
assert nbytes >= 2 * (1 << 30), f"buffer must be >= 2 GiB, got {nbytes}"

with open(path) as f:
    d = json.load(f)
t = d["rocprofiler-sdk-tool"][0]
symb = t["kernel_symbols"]
disps = sorted(t["buffer_records"]["kernel_dispatch"], key=lambda r: r["start_timestamp"])


def name(r):
    kid = r["dispatch_info"]["kernel_id"]
    if kid < len(symb):
        return symb[kid]["truncated_kernel_name"]
    return f"sym{kid}"


expected = 1 + 2 * (warmup + iters)
assert len(disps) == expected, f"unexpected dispatch count {len(disps)} (expected {expected})"
meas = disps[1 + 2 * warmup :]
assert len(meas) == 2 * iters, f"measured dispatch count {len(meas)} != {2 * iters}"

names = {}
per_iter_ns = []
for i in range(iters):
    heavy, tiny = meas[2 * i], meas[2 * i + 1]
    dur = (heavy["end_timestamp"] - heavy["start_timestamp"]) + (tiny["end_timestamp"] - tiny["start_timestamp"])
    per_iter_ns.append(dur)
    for r in (heavy, tiny):
        n = name(r)
        names[n] = names.get(n, 0) + r["end_timestamp"] - r["start_timestamp"]

bytes_per_iter = nbytes + 8
gbps = [bytes_per_iter / (ns / 1e9) / 1e9 for ns in per_iter_ns]
total_ns = sum(per_iter_ns)
total_bytes = bytes_per_iter * iters
headline = total_bytes / (total_ns / 1e9) / 1e9

print(f"KERNELS: {json.dumps({k: round(v / 1e6, 3) for k, v in names.items()})} (ns total)")
print(
    f"RESULT: iters={iters} buffer_bytes={nbytes} "
    f"total_kernel_ms={total_ns / 1e6:.3f} "
    f"gbps_min={min(gbps):.1f} gbps_median={statistics.median(gbps):.1f} "
    f"gbps_max={max(gbps):.1f} headline_gbps={headline:.1f}"
)
assert headline > 0.0, "headline GB/s must be positive"
assert names, "no kernel names captured"
print("PARSE-OK")
