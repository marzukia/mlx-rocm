#!/bin/bash
# R5b measurement runner: arm B (MTP6_TAPE=hybrid, the 48.93 baseline) vs R5 (the
# R5 control for the fix delta) vs R5b (contiguous materialization of the
# selected tape row at rebind) vs R5e (the S=4 GEMM forced in the verify
# window; the deferred-eval-order diagnostic/fix).
# Protocols: anchor MTP_MAX_TOKENS=322 (161-verify), phase 64v MTP_MAX_TOKENS=128.
# 3 repeats per arm per protocol, interleaved, 25 s idle between runs,
# SCLK sampled every 10 s during each run, all under flock on ~/gpu.lock.
# venv-r4v is the common base; a venv-kfix pass runs ONLY if the kfix wheel
# md5 differs from r4v by the end of the r4v pass (the parallel kernel-fix
# ticket's wheel landing mid-run).
set -u
W=${R5B_W:-$HOME/r5b-bench}
LOCK=${R5B_LOCK:-$HOME/gpu.lock}
MODELS_MNT=${R5B_MODELS_MNT:?set R5B_MODELS_MNT to the models mount on the bench box}
HSALIB=${R5B_HSALIB:?set R5B_HSALIB to the libhsa-runtime64.so.1 path on the bench box}
VENV4=${R5B_VENV4:-$HOME/venv-r4v}
VENVK=${R5B_VENVK:-$HOME/venv-kfix}
[ -x "$VENV4/bin/python" ] || { echo "venv4 missing"; exit 1; }
SP=lib/python3.14/site-packages
mkdir -p "$W"
cd "$W"
LOG="$W/r5b-run.log"
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
LM="$VENV4/$SP/mlx/lib/libmlx.so"
LMK="$VENVK/$SP/mlx/lib/libmlx.so"
SB="$MODELS_MNT/qwen3-9b-mlx/model-mtp.safetensors"
echo "PRE $(date -u +%H:%M:%S) libmlx_r4v=$(md5sum "$LM" | cut -d' ' -f1) libmlx_kfix=$(md5sum "$LMK" 2>/dev/null | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1)" >>"$LOG"

sampler() {
  while :; do
    s=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | head -1 | tr -d ' ')
    echo "  sclk $(date -u +%H:%M:%S) $s" >>"$LOG"
    sleep 10
  done
}

run() { # py arm maxtok benchfile name
  local py=$1 arm=$2 mt=$3 bf=$4 name=$5
  sleep 25
  echo "RUN $(date -u +%H:%M:%S) $name arm=$arm maxtok=$mt" >>"$LOG"
  sampler & local SMP=$!
  env MODELS="$MODELS_MNT" LD_PRELOAD="$HSALIB" HCC_AMDGPU_TARGET=gfx1201 \
    MLX_MTP=4 MTP6_TAPE="$arm" GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 \
    MTP_MAX_TOKENS="$mt" "$py" "$bf" >"$W/$name.log" 2>&1
  local rc=$?
  kill "$SMP" 2>/dev/null
  echo "DONE $(date -u +%H:%M:%S) $name rc=$rc" >>"$LOG"
}

R4V="$VENV4/bin/python"
B6="$W/mtp-bench6.py"
B6P="$W/mtp-bench6-phase.py"
for i in 1 2 3; do
  run "$R4V" hybrid 322 "$B6" B-an-$i
  run "$R4V" r5 322 "$B6" R5-an-$i
  run "$R4V" r5b 322 "$B6" R5B-an-$i
  run "$R4V" r5e 322 "$B6" R5E-an-$i
done
for i in 1 2 3; do
  run "$R4V" hybrid 128 "$B6P" B-ph-$i
  run "$R4V" r5 128 "$B6P" R5-ph-$i
  run "$R4V" r5b 128 "$B6P" R5B-ph-$i
  run "$R4V" r5e 128 "$B6P" R5E-ph-$i
done
echo "POST $(date -u +%H:%M:%S) libmlx_r4v=$(md5sum "$LM" | cut -d' ' -f1) libmlx_kfix=$(md5sum "$LMK" 2>/dev/null | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1)" >>"$LOG"
echo "R4V-PASS-DONE $(date -u +%H:%M:%S)" >>"$LOG"
# KFIX pass: only if the kfix wheel differs from r4v by now (the parallel fix landed).
KFIX="$VENVK/bin/python"
if [ "$(md5sum "$LMK" 2>/dev/null | cut -d' ' -f1)" != "$(md5sum "$LM" 2>/dev/null | cut -d' ' -f1)" ]; then
  for i in 1 2 3; do
    run "$KFIX" hybrid 322 "$B6" K-B-an-$i
    run "$KFIX" r5 322 "$B6" K-R5-an-$i
    run "$KFIX" r5b 322 "$B6" K-R5B-an-$i
    run "$KFIX" r5e 322 "$B6" K-R5E-an-$i
  done
  for i in 1 2 3; do
    run "$KFIX" hybrid 128 "$B6P" K-B-ph-$i
    run "$KFIX" r5 128 "$B6P" K-R5-ph-$i
    run "$KFIX" r5b 128 "$B6P" K-R5B-ph-$i
    run "$KFIX" r5e 128 "$B6P" K-R5E-ph-$i
  done
  echo "KFIX-PASS-DONE $(date -u +%H:%M:%S)" >>"$LOG"
else
  echo "KFIX-SKIP $(date -u +%H:%M:%S) kfix libmlx md5 == r4v (the parallel kernel fix had not landed by the end of the r4v pass)" >>"$LOG"
fi
echo "ALL-DONE $(date -u +%H:%M:%S)" >>"$LOG"
