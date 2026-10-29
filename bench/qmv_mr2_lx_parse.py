#!/usr/bin/env python3
"""t889: parse the rocprofv3 JSON trace from qmv_mr2_lx_bench.py (method
doc section 2 rule 2: GB/s = bytes moved / rocprofv3 kernel time; rule 6:
print the kernel names actually dispatched).

The bench runs 16 combos (4 shapes x M=1..4) in a fixed order, each
WARMUP + REPS quantized_matmul calls; each call dispatches exactly one
qmv* kernel (t835: 60/60 dispatches per shape). The parser segments the
qmv dispatch stream by that order.

Usage: qmv_mr2_lx_parse.py <trace.json> <warmup> <reps> <outdir>
Writes <outdir>/combos.json (per-combo kernel time / GB/s / kernel names)
and prints one line per combo.
"""

import json
import sys

path, warmup, reps, outdir = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    sys.argv[4],
)

SHAPES = [
    (4096, 12288, "gate_up"),
    (12288, 4096, "down"),
    (4096, 4096, "o_proj"),
    (4096, 248320, "lm_head"),
]

with open(path) as f:
    d = json.load(f)
t = d["rocprofiler-sdk-tool"][0]
symb = t["kernel_symbols"]
disps = sorted(t["buffer_records"]["kernel_dispatch"], key=lambda r: r["start_timestamp"])


def names(r):
    kid = r["dispatch_info"]["kernel_id"]
    if kid < len(symb):
        return (
            symb[kid].get("kernel_name", ""),
            symb[kid].get("truncated_kernel_name", f"sym{kid}"),
        )
    return ("", f"sym{kid}")


qmv = []
for r in disps:
    full, trunc = names(r)
    if "qmv" in trunc.lower() or "qmv" in full.lower():
        qmv.append((r, full, trunc))

expected = 16 * (warmup + reps)
assert len(qmv) == expected, (
    f"QMV dispatch count {len(qmv)} != {expected} (got {sum(1 for r, f, tr in qmv)} of {len(disps)} dispatches)"
)

combos = []
i = 0
for K, N, tag in SHAPES:
    for M in (1, 2, 3, 4):
        group = qmv[i : i + warmup + reps]
        i += warmup + reps
        meas = group[warmup:]
        full = {g[1] for g in meas}
        trunc = {g[2] for g in meas}
        bytes_per_call = (
            N * K // 2  # weights (4-bit)
            + N * (K // 64) * 2  # scales
            + N * (K // 64) * 2  # biases
            + M * K * 2  # x (bf16)
            + M * N * 2  # y (bf16)
        )
        total_bytes = bytes_per_call * reps
        qmv_ns = sum(r["end_timestamp"] - r["start_timestamp"] for r, _, _ in meas)
        gbps = total_bytes / (qmv_ns / 1e9) / 1e9
        combos.append(
            {
                "tag": tag,
                "K": K,
                "N": N,
                "M": M,
                "kernel_ns": qmv_ns,
                "bytes": total_bytes,
                "gbps": gbps,
                "kernel_names": sorted(trunc),
                "kernel_names_full": sorted(full),
            }
        )
        print(
            f"COMBO: {tag} {K}x{N} M={M} kernel_time={qmv_ns / 1e6:.3f} ms gbps={gbps:.1f} kernels={sorted(trunc)}",
            flush=True,
        )

with open(f"{outdir}/combos.json", "w") as f:
    json.dump(combos, f, indent=1)
print(f"SAVED: {outdir}/combos.json", flush=True)
