#!/bin/bash
# R4c: int2 g128-asymmetric draft head A/B (mtp-r4c-int2-head).
#
# Lever: the MTP draft path's head. The draft arm calls the shared lm_head per draft token (3x per step, 248320x4096); the int2 arm calls mx.fast.lm_head_int2 (0.27 GiB packed, quarter-split) + an exact top-32 re-rank against the shared head's columns. The verify head is the shared head in EVERY arm, so greedy output is lossless; the draft head only proposes (acceptance may move).
#
# Config of record (all arms): MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 REFWD_M1=1 (the R4b rows env is standing since t1558).
# Arms:
#   OFF   : venv-canon (standing wheel, no int2 op; bf16-class head) + MTP6_DRAFT_HEAD=bf16
#   ON    : venv-r4c (base vendor tree wheel: int2 + R4b qmm) + int2
#   CTRL  : venv-r4c + MTP6_DRAFT_HEAD=bf16 (wheel-delta attribution)
#   CTRL2 : venv-r4c + int2, rows env OFF (draft-head-only attribution)
# 25 s idle before each run; the whole run is one flock -x hold on the GPU lock (fuser checked free first: wait, never bench contended).
# Host paths from the environment: OFF_PY / ON_PY / MODELS / LOCK
set -u
W=$(dirname "$0")
mkdir -p $W
cd $W

B6=$W/mtp-bench6.py
B6P=$W/mtp-bench6-phase.py
AR=$W/mtp-bench3.py
PROBE=$W/r4c_head_probe.py
BASE="env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 REFWD_M1=1"
NOROWS="env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 REFWD_M1=1"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $W/runner.log; }

# ---- pre-verification ----
log "PRE bench6: $(md5sum $B6 | cut -d' ' -f1)"
log "PRE bench6-phase: $(md5sum $B6P | cut -d' ' -f1)"
log "PRE probe: $(md5sum $PROBE | cut -d' ' -f1)"
log "PRE AR: $(md5sum $AR | cut -d' ' -f1)"
log "PRE SBCAR: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors | cut -d' ' -f1)"
[ -f $MODELS/qwen3-9b-mlx/model-mtp.safetensors ] || { log "FATAL: model-mtp.safetensors sidecar missing"; exit 1; }
log "PRE OFF libmlx: $(md5sum $(dirname $OFF_PY)/../lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -d' ' -f1)"
log "PRE ON  libmlx: $(md5sum $(dirname $ON_PY)/../lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -d' ' -f1)"

run(){ # $1 = tag, $2 = python, $3 = bench, $4 = iterations, $5 = env prefix, $6 = extra env
  sleep 25
  log "RUN $1 ($4-iter) starting"
  $5 $6 MTP_MAX_TOKENS=$4 $2 $3 > $W/$1.log 2>&1
  local rc=$?
  log "RUN $1 rc=$rc"
  grep -E "TOKS_S|TOKS=|PHASE ms|INFO|FIRST20|HEAD_PROBE|INT2_BUILD" $W/$1.log | tail -10 | tee -a $W/runner.log
}

# ---- head-traffic probe (ON wheel: has the int2 op; times both paths) ----
sleep 25
log "PROBE r4c starting"
$BASE MTP6_DRAFT_HEAD=int2 $ON_PY $PROBE > $W/probe_r4c.log 2>&1
log "PROBE r4c rc=$?"
grep -E "HEAD_PROBE|HEAD type|INT2_BUILD" $W/probe_r4c.log | tee -a $W/runner.log

# ---- AR spot-check (MTP=0: the int2 wheel must not perturb AR) ----
for P in OFF ON; do
  PY=$OFF_PY; [ $P = ON ] && PY=$ON_PY
  sleep 25
  log "AR $P starting"
  env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=0 $PY $AR > $W/ar_$P.log 2>&1
  log "AR $P rc=$?"
  grep -E "TOKS=|TOKS_S|FIRST16" $W/ar_$P.log | tail -4 | tee -a $W/runner.log
done

# ---- anchor protocol (322 iterations = 161 verifies / ~510 tokens) ----
for i in 1 2 3; do
  run O$i $OFF_PY $B6 322 "$BASE" "MTP6_DRAFT_HEAD=bf16"
  run N$i $ON_PY  $B6 322 "$BASE" "MTP6_DRAFT_HEAD=int2"
  run C$i $ON_PY  $B6 322 "$BASE" "MTP6_DRAFT_HEAD=bf16"
  run M$i $ON_PY  $B6 322 "$NOROWS" "MTP6_DRAFT_HEAD=int2"
done
# ---- phase decomposition (MTP6_MTPTRIM=1): 128 (64v) and 1024 (512v) ----
for i in 1 2 3; do
  run P64O$i $OFF_PY $B6P 128  "$BASE" "MTP6_MTPTRIM=1 MTP6_DRAFT_HEAD=bf16"
  run P64N$i $ON_PY  $B6P 128  "$BASE" "MTP6_MTPTRIM=1 MTP6_DRAFT_HEAD=int2"
  run P512O$i $OFF_PY $B6P 1024 "$BASE" "MTP6_MTPTRIM=1 MTP6_DRAFT_HEAD=bf16"
  run P512N$i $ON_PY  $B6P 1024 "$BASE" "MTP6_MTPTRIM=1 MTP6_DRAFT_HEAD=int2"
done

sleep 25
log "POST SBCAR: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors | cut -d' ' -f1)"
log "DONE"
