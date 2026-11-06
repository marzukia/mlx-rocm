#!/bin/bash
# t2246 seam battery: per-step wall + HIP-event span for AR and spec arms.
# AR  = venv-verify, MLX_MTP=0
# SPEC = venv-m3k,    MLX_MTP=3 (K=2, S=3 verify)
# One lock for the whole battery. Each round = full generation (to EOS).
set -uo pipefail
cd /home/andryo
LOCK=/home/andryo/gpu-bench.lock
OUT=/home/andryo/t2246/battery
mkdir -p "$OUT"
BENCH=/home/andryo/t2246/mtp-bench6-stepspan.py
ENVB="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo MODELS=/mnt/models HCC_AMDGPU_TARGET=gfx1201"

flock "$LOCK" bash -c '
set -uo pipefail
cd /home/andryo
ENVB="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo MODELS=/mnt/models HCC_AMDGPU_TARGET=gfx1201"
OUT=/home/andryo/t2246/battery
BENCH=/home/andryo/t2246/mtp-bench6-stepspan.py
echo "BATTERY_START $(date -Is)"
for r in 1 2 3; do
  echo "AR_ROUND_$r START $(date -Is)"
  $ENVB MLX_MTP=0 /home/andryo/venv-verify/bin/python "$BENCH" > "$OUT/ar_r${r}.log" 2>&1
  echo "AR_ROUND_$r rc=$?"
done
for r in 1 2 3; do
  echo "SPEC_ROUND_$r START $(date -Is)"
  $ENVB MLX_MTP=3 /home/andryo/venv-m3k/bin/python "$BENCH" > "$OUT/spec_r${r}.log" 2>&1
  echo "SPEC_ROUND_$r rc=$?"
done
echo "BATTERY_END $(date -Is)"
' 2>&1 | tee "$OUT/battery.log"
