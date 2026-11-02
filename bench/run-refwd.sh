#!/bin/bash
# run-refwd.sh — canonical GPU battery for the REFWD-M1 toggle
# (MLX_ROCM_REFWD_M1; the correction re-forward, option (c) in REPORT-refwd.md).
#
# Runs ON guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest userland). It is
# ENV-ONLY arms: one wheel, three env configs, 2 runs/arm, flock per arm,
# the md5-pinned canonical bench (mtp-bench-k3-tape.py, the t1292/t1307/t1325
# method). It does NOT build the wheel (see the build note below) and does NOT
# take the GPU lock here — the next GPU round drives it.
#
# BUILD PREREQUISITE (next GPU round): build + install the wheel from THIS
# branch's vendor tree (the toggle lives in the vendor qmm.hip):
#   MLX_VENDOR_DIR=<this branch's vendor tree> bash scripts/build-wheel.sh
# This branch's vendor tree has NO rows kernel (rows is on
# worker/rows-statistical). Consequence for the arms below:
#   * B0 and R2 are the meaningful A/B on this wheel (the pure M=1 refwd lever:
#     the M=1 scalar-FMA wide default vs the int8-dot4 LX re-route).
#   * R1 (the current rows config, the 75.00 reference) REQUIRES the rows
#     kernel in the wheel — i.e. a wheel built from the COMBINED tree
#     (worker/rows-statistical + this branch's toggle). On this branch's wheel
#     the MLX_ROCM_QMV_ROWS env is a no-op, so R1 degenerates to B0; the real
#     75.00 R1 is benched on the combined tree.
#
# Usage (next GPU round, under the gpu-bench.lock the round holds):
#   MODELS=<models-dir> BENCH=<md5-pinned canonical bench> PY=<guest venv python> \
#     bash bench/run-refwd.sh                # run all arms
#   ... ARMS=B0 R2 bash bench/run-refwd.sh   # run a subset
#
# The script is env-driven (no host paths are baked in): MODELS / BENCH / PY are
# set by the round. The canonical method is the md5-pinned mtp-bench-k3-tape.py
# (md5 2c2d6c90c2349ae087e53e5d6c60018e); BENCH points at it. LD_PRELOAD and
# HCC_AMDGPU_TARGET default to the guest ROCm 10.1 / gfx1201 values.
#
set -uo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx1201}"
MODELS="${MODELS:-}"
PY="${PY:-}"
BENCH="${BENCH:-}"
if [ -z "$MODELS" ] || [ -z "$BENCH" ] || [ -z "$PY" ]; then
  echo "set MODELS, BENCH (the md5-pinned canonical bench) and PY before running" >&2
  exit 1
fi
OUT="${REFWD_OUT:-$MODELS/tmp/refwd}"
LOCK="${GPU_LOCK:-$MODELS/gpu-bench.lock}"
mkdir -p "$OUT"

# --- canonical bench pin (the t1292/t1307/t1325 method) ---------------------
# the script must be the md5-pinned canonical file (do not run a stale copy).
WANT_MD5=2c2d6c90c2349ae087e53e5d6c60018e
GOT_MD5="$(md5sum "$BENCH" 2>/dev/null | cut -d' ' -f1)"
if [ "$GOT_MD5" != "$WANT_MD5" ]; then
  echo "BENCH MD5 MISMATCH: got=$GOT_MD5 want=$WANT_MD5 ($BENCH)" >&2
  echo "refuse to run on a non-canonical bench" >&2
  exit 1
fi
echo "bench md5 ok: $GOT_MD5"

# the sidecar pin: only model-mtp.safetensors live (the canonical A3 frame).
if [ -f "$MODELS/qwen3-9b-mlx/model-mtp.safetensors.disabled" ]; then
  echo "WARNING: model-mtp.safetensors is .disabled — the sidecar pin is not in
the canonical state; the MTP module may not load. Re-pin before benching." >&2
fi

# --- the three arms (env-only) ---------------------------------------------
# The A3 base config (the t1292 baseline / t1307 pre-rows frame): the GDN
# state-tape commit (the canonical MTP path) + the fused GDN decode step + the
# SDPA S4VA. B0 is the base; R1 adds the rows config; R2 adds the refwd toggle.
ENV_COMMON="MLX_MTP=4"
ENV_TAPE="MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1 MLX_ROCM_SDPA_S4VA=1"
ENV_ROWS="MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1"
ENV_REFWD="MLX_ROCM_REFWD_M1=1"

# run_arm <name> <env-string>
run_arm() {
  local name=$1 envstr=$2 i log
  for i in 1 2; do
    log="$OUT/${name}_r${i}.log"
    echo "=== $name run $i: env: $envstr ==="
    # shellcheck disable=SC2086
    env $envstr \
      "$PY" "$BENCH" >"$log" 2>&1
    # the key lines, one per run
    grep -E "TOKS_S|WALL_MS|PER_PAIR_MS|TAPE n_ok|FB_FAIL" "$log" | sed "s/^/${name} r${i}: /"
  done
}

ARMS="${ARMS:-B0 R1 R2}"
for a in $ARMS; do
  case $a in
    B0) run_arm "B0" "$ENV_COMMON $ENV_TAPE" ;;
    R1) run_arm "R1" "$ENV_COMMON $ENV_TAPE $ENV_ROWS" ;;
    R2) run_arm "R2" "$ENV_COMMON $ENV_TAPE $ENV_REFWD" ;;
    *) echo "unknown arm: $a" >&2 ;;
  esac
done

# --- gates -----------------------------------------------------------------
# Extract ONE field value from one log line (the first match).
extract() { # extract <log> <field> -> prints the value or nothing
  awk -v f="$2" '
    /TOKS_S/    { if (f=="TOKS_S" && match($0, /TOKS_S=[0-9.]+/)) { print substr($0, RSTART+7, RLENGTH-7); exit } }
    /WALL_MS/   {
      if (f=="refwd"  && match($0, /refwd=[0-9.]+/))   { print substr($0, RSTART+6, RLENGTH-6); exit }
      if (f=="verify" && match($0, /verify=[0-9.]+/)) { print substr($0, RSTART+7, RLENGTH-7); exit }
      if (f=="single" && match($0, /single=[0-9.]+/)) { print substr($0, RSTART+7, RLENGTH-7); exit }
      if (f=="n_refwd"&& match($0, /n_refwd=[0-9]+/)) { print substr($0, RSTART+8, RLENGTH-8); exit }
    }
    /PER_PAIR_MS/ { if (f=="pair" && match($0, /wall=[0-9.]+/)) { print substr($0, RSTART+5, RLENGTH-5); exit } }
  ' "$1" 2>/dev/null
}
# The 2-run median (mean of the two runs) of a field per arm.
metric() { # metric <arm> <field> -> the 2-run median, or NA
  local arm=$1 field=$2 v1 v2
  v1="$(extract "$OUT/${arm}_r1.log" "$field")"
  v2="$(extract "$OUT/${arm}_r2.log" "$field")"
  awk -v a="$v1" -v b="$v2" 'BEGIN {
    if (a=="" && b=="") print "NA"
    else if (a=="") print b
    else if (b=="") print a
    else printf "%.3f\n", (a+b)/2
  }'
}
gate() { # gate <label> <expr> ; prints PASS/FAIL
  local label=$1; shift
  if awk "BEGIN { exit !($*) }" 2>/dev/null; then echo "$label: PASS"; else echo "$label: FAIL"; fi
}

B0_TOKS="$(metric B0 TOKS_S)"; R1_TOKS="$(metric R1 TOKS_S)"; R2_TOKS="$(metric R2 TOKS_S)"
B0_REFWD="$(metric B0 refwd)"; R2_REFWD="$(metric R2 refwd)"; R2_VER="$(metric R2 verify)"
B0_VER="$(metric B0 verify)"; R2_PAIR="$(metric R2 pair)"; B0_PAIR="$(metric B0 pair)"
B0_NRFWD="$(metric B0 n_refwd)"; R2_NRFWD="$(metric R2 n_refwd)"

echo
echo "=== summary (2-run medians) ==="
printf "B0: tok/s=%s verify=%s refwd=%s pair=%s n_refwd=%s\n" "$B0_TOKS" "$B0_VER" "$B0_REFWD" "$B0_PAIR" "$B0_NRFWD"
printf "R1: tok/s=%s  (rows reference; on this branch's wheel the rows env is a no-op == B0)\n" "$R1_TOKS"
printf "R2: tok/s=%s verify=%s refwd=%s pair=%s n_refwd=%s\n" "$R2_TOKS" "$R2_VER" "$R2_REFWD" "$R2_PAIR" "$R2_NRFWD"

echo
echo "=== gates ==="
# B0: the M=1-wide baseline reproduces the 69.48 control (within ±3%).
gate "B0 baseline tok/s ~69.48 (±3%)" \
  "$B0_TOKS >= 69.48*0.97 && $B0_TOKS <= 69.48*1.03"
# R2: the lever must make the refwd CHEAPER than B0 (the whole point).
gate "R2 refwd < B0 refwd (the lever helps)" \
  "$R2_REFWD < $B0_REFWD"
# R2: no regression on the M=4 verify (the toggle is M==1-only, so verify is
# untouched) — R2 verify must equal B0 verify (within 2%).
gate "R2 verify == B0 verify (M==1-only, verify untouched; ±2%)" \
  "$R2_VER <= $B0_VER*1.02 && $R2_VER >= $B0_VER*0.98"
# R2: n_refwd sanity (the tape commit holds; ~100-104 per 128 verifies).
gate "R2 n_refwd in [95,105]" \
  "$R2_NRFWD >= 95 && $R2_NRFWD <= 105"
# R2: the int8-dot4 path is STATISTICAL (not bit-exact vs the wide default) —
# gate on a healthy run (tok/s not collapsed; the WMMA/int8 collapse class is a
# tok/s drop to <40), NOT on bit-exactness.
gate "R2 runs (tok/s >= 60, no collapse class)" \
  "$R2_TOKS >= 60"

echo
echo "NOTE: R2 (the int8-dot4 LX re-route) is a STATISTICAL A/B arm, not a
lossless-serve candidate — it is default-off and the serve path is the
unchanged M=1 wide default (B0) / the rows config (R1, combined tree). See
REPORT-refwd.md section 3 (risk to losslessness)."
