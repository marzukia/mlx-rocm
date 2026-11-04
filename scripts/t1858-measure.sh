#!/bin/bash
# t1858 GPU measurement: BEFORE (venv-r4v, base wheel) / AFTER (venv-kfix,
# patched wheel). One S=4 verify per-op profile x3 (median) + full MTP step
# (r5 arm) x3, per wheel. 25 s idle before and between runs. Holds ~/gpu.lock.
set -uo pipefail
export MODELS=/mnt/models
export LD_PRELOAD=libhsa-runtime64.so.1
export HCC_AMDGPU_TARGET=gfx1201
cd ~/t1858
mkdir -p out
exec 9>~/gpu.lock
LOCKED=0
for i in $(seq 1 96); do
  if flock -w 1800 9; then
    LOCKED=1
    echo "LOCK_ACQUIRED iter=$i $(date -u)"
    break
  fi
  echo "LOCK_BUSY iter=$i $(date -u)"
  sleep 60
done
if [ "$LOCKED" != "1" ]; then echo "LOCK_FAIL"; exit 1; fi

run_wheel() {
  P="${1/#\~/$HOME}"
  B="$P/bin"
  sleep 25
  echo "BITEXACT $P $(date -u)"
  "$B/python" bench/qmm_bitexact_ab.py > "out/$P.bitexact" 2>&1
  for r in 1 2 3; do
    sleep 25
    echo "PEROP $P rep=$r $(date -u)"
    MTP_MODEL=/mnt/models/qwen3-9b-mlx "$B/python" bench/roofline-verify-perop.py \
      > "out/$P.perop.$r" 2>&1
    echo "PEROP_DONE $P rep=$r $(date -u)"
  done
  for r in 1 2 3; do
    sleep 25
    echo "STEP $P rep=$r $(date -u)"
    MLX_MTP=4 MTP6_TAPE=r5 MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 \
    MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1 MTP_MAX_TOKENS=512 \
    MTP_MODEL=/mnt/models/qwen3-9b-mlx "$B/python" bench/mtp-bench6-phase.py \
      > "out/$P.step.$r" 2>&1
    echo "STEP_DONE $P rep=$r $(date -u)"
  done
}

run_wheel $HOME/venv-r4v
run_wheel $HOME/venv-kfix
echo "ALL_DONE $(date -u)"
