#!/usr/bin/env python3
"""Parse the rocprofv3 JSON trace from qmv_mr_bench.py (doc section 2 rule 2:
GB/s = bytes moved / rocprofv3 kernel time; rule 6: print the kernel names
actually dispatched).

Usage: qmv_mr_parse.py <trace.json> <K> <N> <M> <warmup> <reps>

Dispatch pattern: [H2D/init copies] + [QMV kernel * (warmup + reps)]. The
QMV kernel is the LAST kernel symbol by name family (qmv_wide* /
qmv_wide_mr*); init copies (copy_*) are excluded by name.
"""

import json
import sys

path, K, N, M, warmup, reps = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)

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


names = {}
qmv = []
for r in disps:
    n = name(r)
    names[n] = names.get(n, 0) + 1
    if "qmv" in n.lower():
        qmv.append(r)

expected = warmup + reps
assert len(qmv) == expected, f"QMV dispatch count {len(qmv)} != {expected} (names={names})"
meas = qmv[warmup:]  # exclude warmup dispatches (icache/TLB warm)

bytes_per_call = (
    N * K // 2  # weights (4-bit)
    + N * (K // 64) * 2  # scales
    + N * (K // 64) * 2  # biases
    + M * K * 2  # x (bf16)
    + M * N * 2  # y (bf16)
)
total_bytes = bytes_per_call * reps
qmv_ns = sum(r["end_timestamp"] - r["start_timestamp"] for r in meas)
gbps = total_bytes / (qmv_ns / 1e9) / 1e9
pct = 100.0 * gbps / 636.0
print(f"KERNELS: {names}", flush=True)
print(
    f"QMV: shape={K}x{N} M={M} reps={reps} kernel_time={qmv_ns / 1e6:.3f} ms "
    f"bytes={total_bytes} gbps={gbps:.1f} pct_of_636={pct:.1f}%",
    flush=True,
)
