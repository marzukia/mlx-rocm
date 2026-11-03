#!/bin/bash
# R4b: fused S=4 verify lever A/B (mtp-r4b-fused-verify).
#
# Lever: multi-row QMV (one weight pass for all M=4 verify rows). Probe findings that shaped the matrix:
set -u
W=$(dirname "$0")
mkdir -p $W
cd $W

B6=$W/mtp-bench6.py
B6P=$W/mtp-bench6-phase.py
PROBE=$W/r4b_qmm_probe.py
BASE="env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 QMV_ROWS=1 QMV_ROWS_MIN_N=1 REFWD_M1=1"
ROWSON="MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $W/runner.log; }

# ---- pre-verification ----
log "PRE bench6: $(md5sum $B6 | cut -d' ' -f1)"
log "PRE bench6-phase: $(md5sum $B6P | cut -d' ' -f1)"
log "PRE probe: $(md5sum $PROBE | cut -d' ' -f1)"
log "PRE SBCAR: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors | cut -d' ' -f1)"
log "PRE OFF libmlx: $(md5sum $(dirname $OFF_PY)/../lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -d' ' -f1)"
log "PRE ON  libmlx: $(md5sum $(dirname $ON_PY)/../lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -d' ' -f1)"

run(){ # $1 = tag, $2 = python, $3 = bench, $4 = iterations, $5 = extra env
  sleep 25
  log "RUN $1 ($4-iter) starting"
  $BASE $5 MTP_MAX_TOKENS=$4 $2 $3 > $W/$1.log 2>&1
  local rc=$?
  log "RUN $1 rc=$rc"
  grep -E "TOKS_S|TOKS=|PHASE ms|INFO|FIRST20" $W/$1.log | tail -8 | tee -a $W/runner.log
}

probe(){ # $1 = tag, $2 = python, $3 = extra env
  sleep 25
  log "PROBE $1 starting"
  $BASE $3 $2 $PROBE > $W/probe_$1.log 2>&1
  log "PROBE $1 rc=$?"
  grep -E "M1=|probe env" $W/probe_$1.log | tee -a $W/runner.log
}

# ---- probe (rows off / rows on, both wheels) ----
probe off   $OFF_PY ""
probe on    $ON_PY  ""
probe offR  $OFF_PY "$ROWSON"
probe onR   $ON_PY  "$ROWSON"

# ---- anchor protocol (322 iterations = 161 verifies / ~510 tokens) ----
for i in 1 2 3; do
  run O$i $OFF_PY $B6 322 ""
  run N$i $ON_PY  $B6 322 "$ROWSON"
  run X$i $OFF_PY $B6 322 "$ROWSON"
done
# ---- phase decomposition (MTP6_MTPTRIM=1): 128 (64v) and 1024 (512v) ----
for i in 1 2 3; do
  run P64O$i $OFF_PY $B6P 128  "MTP6_MTPTRIM=1"
  run P64N$i $ON_PY  $B6P 128  "MTP6_MTPTRIM=1 $ROWSON"
  run P512O$i $OFF_PY $B6P 1024 "MTP6_MTPTRIM=1"
  run P512N$i $ON_PY  $B6P 1024 "MTP6_MTPTRIM=1 $ROWSON"
done

sleep 25
log "POST SBCAR: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors | cut -d' ' -f1)"
log "DONE"
touch $W/done
