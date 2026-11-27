#!/usr/bin/env python3
# S4(1) step-1: pre-loaded per-tensor kquant_file_gemm M=1 sweep with
# per-call GPU timing. GPU time = per-kernel durations from the
# memory-255 SDK-injected rocprof kernel trace (this wheel has NO mx.Event
# API; the 1.3.2 tool of record is NO-TRACE per fleet memory 255; the 1.1.0
# build at /home/andryo/rcp/new724 traced in the two-call smoke). The trace
# is written at process exit, so this script logs per-call wall-ns windows
# (t0 before submit, t1 after mx.eval) to JSON; the host-side attributor
# (s41-attribute.py) joins kernel rows to call windows by start timestamp.
# Weights are PRE-LOADED in HBM (one mx.load + one mx.eval = single file
# read + one H2D pass, resident), then each tensor is timed standalone at
# M=1 (x = zeros((1, K))). Two consecutive passes: pass 1 = first-touch,
# pass 2 = steady (the lever-finding method). Primary = pass 2.
#
# Arm C = wheel default (mr_gate=1 -> M=1 routes to mmq, the leg-4 wheel);
# arm B = MLX_KQUANT_FILE_MR=0 (kqfg single-row). Arms are separate process
# runs (the gate env is read at dispatch); "same session, same weights" =
# same file, same load path, same box state, back-to-back under one lock.
#
# Buckets (by name, verified against the header parse at runtime):
#   lin      in-model KQLinear gemm family (the per-step M=1 set)
#   lmhead   output.weight (in-model gemm)
#   u64attn  blk.64 attn q/k/v/output (in file, UNUSED by the MTP-OFF vehicle)
#   eh       blk.64.nextn.eh_proj (unused MTP)
#   embed    token_embd.weight (in-model via GATHER; the gemm-shape call here
#            is a bound figure for the embed leg, not the in-model op)
#   small    the N=48 class (ssm_alpha/ssm_beta; dequant-once + tiny bf16
#            matmul in-model; recorded, excluded from the gemm sweep)
# SPLIT (primary) = pass-2 trace-attributed GPU sum of (lin + lmhead) vs the
# in-context step wall (1566ms SMOKE / 2639ms SC-C p50).
import argparse
import hashlib
import json
import os
import re
import time

import mlx.core as mx
import numpy as np

GGUF = "/home/andryo/models/Qwen3.8-27B-Q2_K-plain.gguf"
KTYPE, MODE, AXIS, GROUP, KE, PR = 10, 0, 0, 32, 1, 15
LIB = "/home/andryo/venv-ar/lib/python3.14/site-packages/mlx/lib/libmlx.so"
WALL_SMOKE_MS = 1566.0  # C7 leg-4 wheel SMOKE step
WALL_SCC_P50_MS = 2639.0  # C7 SC-C (mmq) p50 @64tok

LIN_PAT = re.compile(
    r"^blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|"
    r"ffn_down|attn_qkv|attn_gate|ssm_out)\.weight$"
)
U64_PAT = re.compile(r"^blk\.64\.(attn_q|attn_k|attn_v|attn_output)\.weight$")


def log(m):
    print(m, flush=True)


def bucket(name):
    if name == "output.weight":
        return "lmhead"
    if name == "token_embd.weight":
        return "embed"
    if U64_PAT.match(name):
        return "u64attn"
    if "nextn.eh_proj" in name:
        return "eh"
    if LIN_PAT.match(name):
        return "lin"
    return "small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="/home/andryo/s41")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    mr = os.environ.get("MLX_KQUANT_FILE_MR")
    log(f"SWEEP-START t0={t0:.2f} tag={a.tag} env_mr={mr} pid={os.getpid()}")
    log(f"WHEEL {mx.__version__}")
    libmd5 = hashlib.md5()
    with open(LIB, "rb") as f:
        libmd5.update(f.read())
    libmd5s = libmd5.hexdigest()
    log(f"LIB-MD5 {libmd5s}")

    # pre-load: single file read + one H2D pass, weights HBM-resident
    weights = mx.load(GGUF)
    mx.eval([v for v in weights.values()])
    t_load = time.time()
    n_raw = sum(1 for v in weights.values() if v.dtype == mx.uint8)
    log(f"LOAD-OK n_tensors={len(weights)} n_raw_uint8={n_raw} load_wall={t_load - t0:.1f}s")

    # header-parse verification of the set (cheap: names + raw shapes)
    census = {}
    for name in sorted(weights):
        v = weights[name]
        if v.dtype != mx.uint8 or v.ndim != 2:
            continue
        b = bucket(name)
        c = census.setdefault(b, {"n": 0, "shapes": []})
        c["n"] += 1
        c["shapes"].append((name, list(v.shape)))
    log("CENSUS {}".format(json.dumps({b: census[b]["n"] for b in sorted(census)})))
    bad = [
        s[0]
        for b in ("lin", "lmhead", "u64attn", "eh", "embed")
        for s in census.get(b, {}).get("shapes", [])
        if s[1][1] % 84 != 0
    ]
    log(f"SHAPE-84-REMAINDER-EXC {json.dumps(bad)}")

    sweep_buckets = ("lin", "lmhead", "u64attn", "eh", "embed")
    tensors = {}
    for b in sweep_buckets:
        tensors[b] = []
        for name, v in weights.items():
            if v.dtype != mx.uint8 or v.ndim != 2:
                continue
            if bucket(name) != b:
                continue
            if v.shape[1] % 84 != 0:
                log(f"SKIP-REM {name} {list(v.shape)}")
                continue
            tensors[b].append((name, v))

    res = {
        "tag": a.tag,
        "wheel": mx.__version__,
        "lib_md5": libmd5s,
        "env_mr": mr,
        "pid": os.getpid(),
        "load_wall_s": t_load - t0,
        "n_raw": n_raw,
        "census": {b: census[b]["n"] for b in sorted(census)},
        "buckets": {},
        "calls": [],  # (bucket, pass, tensor, t0_ns, t1_ns) windows for the attributor
    }
    for b in sweep_buckets:
        per = tensors[b]
        stat = {"n": len(per), "passes": {}}
        for p in range(a.passes):
            evs = []
            t_p = time.time()
            for _n, v in per:
                if b == "embed":
                    # embed is stored transposed: raw (248320, 1680) ->
                    # (5120, 970, 84); the gemm-shape call uses K=5120
                    w3 = v.reshape(5120, 970, 84)
                    x = mx.zeros((1, 5120))
                else:
                    K = v.shape[0]
                    w3 = v.reshape(K, v.shape[1] // 84, 84)
                    x = mx.zeros((1, K))
                mx.eval(x)
                t0c = time.perf_counter_ns()
                y = mx.kquant_file.kquant_file_gemm(x, w3, KTYPE, MODE, AXIS, GROUP, KE, PR)
                mx.eval(y)
                t1c = time.perf_counter_ns()
                evs.append((t1c - t0c) / 1e6)
                res["calls"].append([b, p, _n, t0c, t1c])
                del y
            ar = np.array(evs)
            stat["passes"][f"p{p + 1}"] = {
                "wall_p50_ms": float(np.percentile(ar, 50)),
                "wall_mean_ms": float(ar.mean()),
                "wall_sum_ms": float(ar.sum()),
                "wall_s": time.time() - t_p,
            }
        res["buckets"][b] = stat
        log(f"BUCKET-WALL {b} n={len(per)} p2={json.dumps(stat['passes']['p2'])}")

    g2w = sum(res["buckets"][b]["passes"]["p2"]["wall_sum_ms"] for b in ("lin", "lmhead"))
    res["split_wall"] = {
        "gemm_family_pass2_wall_sum_ms": g2w,
        "vs_wall_smoke_1566ms": g2w / WALL_SMOKE_MS,
        "vs_wall_scc_p50_2639ms": g2w / WALL_SCC_P50_MS,
    }
    log("SPLIT-WALL {}".format(json.dumps(res["split_wall"])))
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, a.tag + ".sweep.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"SWEEP-DONE tag={a.tag} exit_wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
