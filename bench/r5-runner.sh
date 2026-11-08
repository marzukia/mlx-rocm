#!/bin/bash
# R5 measurement runner: arm B (MTP6_TAPE=hybrid, the 48.93 baseline) vs R5 (MTP6_TAPE=r5).
# Protocols: anchor MTP_MAX_TOKENS=322 (161-verify), phase 64v MTP_MAX_TOKENS=128.
# 3 repeats per arm per protocol, interleaved B/R5, 25 s idle between runs,
# SCLK sampled every 10 s during each run, all under flock on ~/gpu.lock.
set -u
MODELS_DIR="${MODELS_DIR:?set by caller}"
W=$HOME/r5-bench
LOCK=$HOME/gpu.lock
mkdir -p "$W"
cd "$W"
LOG="$W/r5-run.log"
exec 9>"$LOCK"
LOCKED=0
for i in $(seq 1 96); do
  if flock -x -w 1800 9; then
    LOCKED=1
    echo "LOCK-ACQ iter=$i $(date -u +%H:%M:%S)" >"$LOG"
    break
  fi
  echo "LOCK-BUSY iter=$i $(date -u +%H:%M:%S) holders=$(fuser $LOCK 2>/dev/null | tr '\n' ' ')" >>"$LOG"
  sleep 60
done
[ "$LOCKED" = "1" ] || { echo "LOCK-FAIL $(date -u +%H:%M:%S)" >>"$LOG"; exit 1; }
PY=$HOME/venv-r4v/bin/python
B6="$W/mtp-bench6.py"
B6P="$W/mtp-bench6-phase.py"
LM=$HOME/venv-r4v/lib/python3.14/site-packages/mlx/lib/libmlx.so
SB="$MODELS_DIR/qwen3-9b-mlx/model-mtp.safetensors"
echo "PRE $(date -u +%H:%M:%S) libmlx=$(md5sum "$LM" | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1)" >>"$LOG"

sampler() {
  while :; do
    s=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | head -1 | tr -d ' ')
    echo "  sclk $(date -u +%H:%M:%S) $s" >>"$LOG"
    sleep 10
  done
}

run() { # arm maxtok benchfile name
  local arm=$1 mt=$2 bf=$3 name=$4
  sleep 25
  echo "RUN $(date -u +%H:%M:%S) $name arm=$arm maxtok=$mt" >>"$LOG"
  sampler & local SP=$!
  env MODELS="$MODELS_DIR" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 \
    MLX_MTP=4 MTP6_TAPE="$arm" GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 \
    MTP_MAX_TOKENS="$mt" "$PY" "$bf" >"$W/$name.log" 2>&1
  local rc=$?
  kill "$SP" 2>/dev/null
  echo "DONE $(date -u +%H:%M:%S) $name rc=$rc" >>"$LOG"
}

for i in 1 2 3; do
  run hybrid 322 "$B6" B-an-$i
  run r5 322 "$B6" R5-an-$i
done
for i in 1 2 3; do
  run hybrid 128 "$B6P" B-ph-$i
  run r5 128 "$B6P" R5-ph-$i
done
echo "POST $(date -u +%H:%M:%S) libmlx=$(md5sum "$LM" | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1)" >>"$LOG"
echo "ALL-DONE $(date -u +%H:%M:%S)" >>"$LOG"
