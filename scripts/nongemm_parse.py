#!/usr/bin/env python3
"""t2230 bucket 1 - per-kernel table for ONE AR decode step.

Input: two rocprof kernel traces of the SAME bench run, differing only in
MTP_MAX_TOKENS (A: 1, B: 322). Both carry the identical prompt-prefill block,
so the steady-state per-AR-step kernel table is the delta, normalized:

    per_step[k] = (B[k] - A[k]) / (322 - 1)

(AR = 1 token/step; MLX_MTP=0 plain greedy.)

Handles both rocprof v3 ("rocprofiler-sdk-tool" buffer_records) and rocprofv2
(trace list) JSON layouts. A kernel record needs a name + start/end
timestamps; the parser locates them schema-agnostically (recursive walk) so a
layout change cannot silently produce an empty table: zero kernel records is
a hard error (exit 2).

Classification: ordered first-match-wins regex table (CLASS_RULES below),
class vocabulary: quantized-GEMV, other-GEMM, attention, GDN, norm, copy,
elementwise, other. The table is applied to kernel names verbatim; any name
that matches no rule lands in "other" and is flagged for review.

Usage:
  nongemm_parse.py A.json B.json STEPS_B STEPS_A [clean_wall_ms_per_step ...]

Prints (stdout, machine + markdown):
  per-kernel table (name, count/step, ms/step, class, % of kernel-sum),
  class subtotals, kernel-sum vs step wall (seam), total dispatches/step.
"""

import json
import re
import sys

# Ordered: first match wins. Order matters (qmv before gemm; gdn before
# attention since gdn linear-attention kernels contain 'attn').
CLASS_RULES = [
    ("quantized-GEMV", re.compile(r"qmv|quant.*gmv", re.I)),
    ("other-GEMM", re.compile(r"qmm|gemm|matmul|moe|wmma|dequant", re.I)),
    ("attention", re.compile(r"sdpa|flash|attention|fmha|paged|rope", re.I)),
    ("GDN", re.compile(r"gdn|r4d|conv1d|causal_conv", re.I)),
    ("norm", re.compile(r"norm|rms", re.I)),
    ("copy", re.compile(r"copy|fill|cast|memcpy|memset", re.I)),
    ("elementwise", re.compile(r"add|mul|div|sigmoid|silu|gelu|tanh|erf|sqrt|exp|axpy|scalar|scale", re.I)),
]
OTHER = "other"


def classify(name):
    for cls, rx in CLASS_RULES:
        if rx.search(name):
            return cls
    return OTHER


def _rec_from_dispatch(r, symb):
    """rocprof v3 style: buffer_records.kernel_dispatch + kernel_symbols."""
    di = r.get("dispatch_info", {})
    kid = di.get("kernel_id")
    name = None
    if isinstance(kid, int) and 0 <= kid < len(symb):
        name = symb[kid].get("truncated_kernel_name") or symb[kid].get("kernel_name")
    name = name or di.get("name") or r.get("kernel_name")
    return name, r.get("start_timestamp"), r.get("end_timestamp")


def collect(obj):
    """Return list of (name, start_ns, end_ns); hard-fail if none found."""
    out = []
    # v3 style
    if "rocprofiler-sdk-tool" in obj:
        t = obj["rocprofiler-sdk-tool"][0]
        symb = t.get("kernel_symbols", [])
        for r in t.get("buffer_records", {}).get("kernel_dispatch", []):
            n, s, e = _rec_from_dispatch(r, symb)
            if n and s is not None and e is not None:
                out.append((n, s, e))
        return out

    # v2 / generic: recursive walk for kernel records
    def walk(o):
        if isinstance(o, dict):
            kind = str(o.get("kind", "")).lower()
            if kind in ("kernel", "kd", "kernel_dispatch"):
                name = (
                    o.get("kernel_name")
                    or o.get("name")
                    or (o.get("dispatch_info") or {}).get("name")
                    or (o.get("dispatch_info") or {}).get("kernel_name")
                )
                s = o.get("start_timestamp", o.get("start"))
                e = o.get("end_timestamp", o.get("end"))
                if name is None and isinstance(o.get("dispatch_info", {}).get("kernel_id"), int):
                    name = f"sym{o['dispatch_info']['kernel_id']}"
                if s is not None and e is not None and name:
                    out.append((name, s, e))
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def aggregate(recs):
    agg = {}
    for n, s, e in recs:
        c = agg.setdefault(n, [0, 0])
        c[0] += 1
        c[1] += e - s
    return agg


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    steps_b, steps_a = int(sys.argv[3]), int(sys.argv[4])
    n_steps = steps_b - steps_a
    clean_wall = [float(x) for x in sys.argv[5:]]  # ms per step, clean rounds
    assert n_steps > 0, "B must carry more steps than A"

    with open(a_path) as f:
        ra = collect(json.load(f))
    with open(b_path) as f:
        rb = collect(json.load(f))
    if not ra or not rb:
        print(f"NO-KERNEL-RECORDS A={len(ra)} B={len(rb)}")
        sys.exit(2)

    A, B = aggregate(ra), aggregate(rb)
    names = sorted(set(A) | set(B), key=lambda n: -(B.get(n, (0, 0))[1]))

    rows = []
    for n in names:
        ca, na = A.get(n, (0, 0))
        cb, nb = B.get(n, (0, 0))
        if cb <= ca and nb <= na:
            continue  # did not participate in the step delta
        cnt = (cb - ca) / n_steps
        ms = (nb - na) / n_steps / 1e6
        rows.append((n, cnt, ms, classify(n)))

    ksum_ms = sum(r[2] for r in rows)
    total_cnt = sum(r[1] for r in rows)

    print("== per-kernel table (per AR step; AR = 1 tok/step) ==")
    print(f"kernel-sum {ksum_ms:.3f} ms/step; dispatches/step {total_cnt:.1f}")
    print("name | count/step | ms/step | class | % of kernel-sum")
    for n, cnt, ms, cls in rows:
        pct = 100.0 * ms / ksum_ms if ksum_ms else 0.0
        print(f"{n} | {cnt:.2f} | {ms:.4f} | {cls} | {pct:.2f}")

    print("== class subtotals ==")
    classes = {}
    for n, cnt, ms, cls in rows:
        c = classes.setdefault(cls, [0.0, 0.0])
        c[0] += cnt
        c[1] += ms
    for cls in sorted(classes, key=lambda c: -classes[c][1]):
        cnt, ms = classes[cls]
        print(f"{cls} | {cnt:.1f} | {ms:.4f} | {100.0 * ms / ksum_ms if ksum_ms else 0:.2f}")

    if clean_wall:
        cw = sorted(clean_wall)[len(clean_wall) // 2]
        seam = cw - ksum_ms
        print(
            f"== seam == clean-wall median {cw:.3f} ms/step; kernel-sum (profiled) "
            f"{ksum_ms:.3f} ms/step; gap {seam:.3f} ms/step "
            f"({100.0 * seam / cw if cw else 0:.1f}% of wall)"
        )


if __name__ == "__main__":
    main()
