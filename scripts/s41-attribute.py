#!/usr/bin/env python3
# S4(1) host-side attributor: join the rocprof kernel-trace CSV rows to the
# wall-ns windows logged by s41-sweep.py / s41-ar.py (attribute by kernel
# start timestamp; mx.eval is synchronous, so every kernel of call i starts
# inside [t0_i, t1_i]). No GPU needed. Outputs per-bucket / per-step
# GPU-attributed stats + the per-kernel-name table (the kernel mix is the
# campaign input).
import argparse
import csv
import json
import sys
from collections import defaultdict


def load_kernels(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Kind") != "KERNEL_DISPATCH":
                continue
            s = int(r["Start_Timestamp"])
            e = int(r["End_Timestamp"])
            rows.append((s, e, r["Kernel_Name"]))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)  # sweep or ar JSON
    ap.add_argument("--trace", required=True)  # kernel trace CSV
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.windows) as f:
        w = json.load(f)
    krows = load_kernels(a.trace)
    starts = [r[0] for r in krows]
    import bisect

    kind = "sweep" if "calls" in w else "ar"
    res = {"tag": w["tag"], "n_trace_rows": len(krows), "kind": kind, "by_kernel_name": {}}
    name_sum = defaultdict(float)
    name_cnt = defaultdict(int)
    if kind == "sweep":
        per = defaultdict(lambda: defaultdict(lambda: []))
        for b, p, _n, t0, t1 in w["calls"]:
            i0 = bisect.bisect_left(starts, t0)
            i1 = bisect.bisect_left(starts, t1)
            tot = 0.0
            for i in range(i0, i1):
                s, e, kn = krows[i]
                tot += (e - s) / 1e6
                name_sum[kn] += (e - s) / 1e6
                name_cnt[kn] += 1
            per[b][f"p{p + 1}"].append(tot)
        import numpy as np

        stats = {}
        for b in sorted(per):
            stats[b] = {}
            for p in sorted(per[b]):
                ar = np.array(per[b][p])
                stats[b][p] = {
                    "gpu_p50_ms": float(np.percentile(ar, 50)),
                    "gpu_mean_ms": float(ar.mean()),
                    "gpu_sum_ms": float(ar.sum()),
                    "n_zero": int(np.sum(ar < 0.01)),
                }
        g2 = sum(stats[b]["p2"]["gpu_sum_ms"] for b in ("lin", "lmhead") if b in stats)
        g1 = sum(stats[b]["p1"]["gpu_sum_ms"] for b in ("lin", "lmhead") if b in stats)
        res["buckets"] = stats
        res["split"] = {
            "gemm_family_pass2_gpu_sum_ms": g2,
            "gemm_family_pass1_gpu_sum_ms": g1,
            "vs_wall_smoke_1566ms": g2 / 1566.0,
            "vs_wall_scc_p50_2639ms": g2 / 2639.0,
        }
    else:
        steps = []
        for i, (t0, ts, t1) in enumerate(w["steps"]):
            i0 = bisect.bisect_left(starts, t0)
            i1 = bisect.bisect_left(starts, t1)
            tot = 0.0
            for i in range(i0, i1):
                s, e, kn = krows[i]
                tot += (e - s) / 1e6
                name_sum[kn] += (e - s) / 1e6
                name_cnt[kn] += 1
            steps.append(
                {
                    "i": i,
                    "wall_ms": (t1 - t0) / 1e6,
                    "host_ms": (ts - t0) / 1e6,
                    "sync_ms": (t1 - ts) / 1e6,
                    "gpu_ms": tot,
                }
            )
        import numpy as np

        sw = np.array([s["wall_ms"] for s in steps])
        sg = np.array([s["gpu_ms"] for s in steps])
        sh = np.array([s["host_ms"] for s in steps])
        ss = np.array([s["sync_ms"] for s in steps])
        res["steps"] = steps
        res["stats"] = {
            "n": len(steps),
            "wall_p50_ms": float(np.percentile(sw, 50)),
            "wall_mean_ms": float(sw.mean()),
            "gpu_p50_ms": float(np.percentile(sg, 50)),
            "gpu_mean_ms": float(sg.mean()),
            "gpu_sum_ms": float(sg.sum()),
            "gpu_share_of_wall": float(sg.sum() / sw.sum()) if sw.sum() > 0 else 0.0,
            "host_mean_ms": float(sh.mean()),
            "sync_mean_ms": float(ss.mean()),
        }
    for kn in sorted(name_sum, key=lambda k: -name_sum[k])[:40]:
        res["by_kernel_name"][kn] = {
            "sum_ms": name_sum[kn],
            "n": name_cnt[kn],
            "mean_ms": name_sum[kn] / name_cnt[kn],
        }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"ATTR-DONE kind={kind} rows={len(krows)} out={a.out}", file=sys.stderr)
    if kind == "sweep":
        print(json.dumps(res["split"]), file=sys.stderr)
    else:
        print(json.dumps(res["stats"]), file=sys.stderr)


if __name__ == "__main__":
    main()
