#!/bin/bash
# t2246 seam battery: per-step wall + HIP-event span for AR and spec arms.
# AR  = venv-verify, MLX_MTP=0
# SPEC = venv-m3k,    MLX_MTP=3 (K=2, S=3 verify)
# One lock for the whole battery. Each round = full generation (to EOS).
# Box-specific paths come from the environment (HOME + MODELS) — no operator
# home or guest mount is hardcoded (fleet #19 persona-leak gate).
set -uo pipefail
H="$HOME"
: "${MODELS:?set MODELS to the models mount before running}"
export MODELS
cd "$H"
LOCK="$H/gpu-bench.lock"
OUT="$H/t2246/battery"
mkdir -p "$OUT"
BENCH="$H/t2246/mtp-bench6-stepspan.py"
ENVB="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$H MODELS=$MODELS HCC_AMDGPU_TARGET=gfx1201"

flock "$LOCK" bash -c '
set -uo pipefail
H="$HOME"
cd "$H"
ENVB="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$H MODELS=$MODELS HCC_AMDGPU_TARGET=gfx1201"
OUT="$H/t2246/battery"
BENCH="$H/t2246/mtp-bench6-stepspan.py"
echo "BATTERY_START $(date -Is)"
for r in 1 2 3; do
  echo "AR_ROUND_$r START $(date -Is)"
  $ENVB MLX_MTP=0 "$H/venv-verify/bin/python" "$BENCH" > "$OUT/ar_r${r}.log" 2>&1
  echo "AR_ROUND_$r rc=$?"
done
for r in 1 2 3; do
  echo "SPEC_ROUND_$r START $(date -Is)"
  $ENVB MLX_MTP=3 "$H/venv-m3k/bin/python" "$BENCH" > "$OUT/spec_r${r}.log" 2>&1
  echo "SPEC_ROUND_$r rc=$?"
done
echo "BATTERY_END $(date -Is)"
' 2>&1 | tee "$OUT/battery.log"
