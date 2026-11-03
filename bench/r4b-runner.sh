#!/bin/bash
# R4b: fused S=4 verify lever A/B. Lever = wheel (bias-free M=2..4 wide/multi-row
# QMV dispatch). OFF = the canonical baseline venv (expected ~46.49 t/s class);
# ON = the lever venv (canonical wheel + the gate fix; MLX_ROCM_QMV_ROWS=1 routes
# the bias-free M=4 GEMMs to the shared-weight-pass rows/mr paths).
# Identical env for both arms (config identity; MLX_ROCM_QMV_ROWS is inert in the
# canonical wheel - every linear in the model is bias-free).
# 25 s idle before the first run and between runs; all under gpu-bench.lock.
# Host-specific paths (venvs, model dir, lock) come from the environment:
#   OFF_PY / ON_PY / MODELS / LOCK
set -u
W=$(dirname "$0")
mkdir -p $W
cd $W

B6=$W/mtp-bench6.py
B6P=$W/mtp-bench6-phase.py
PROBE=$W/r4b_qmm_probe.py
ENV="env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 QMV_ROWS=1 QMV_ROWS_MIN_N=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1"

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
  $ENV $5 MTP_MAX_TOKENS=$4 $2 $3 > $W/$1.log 2>&1
  local rc=$?
  log "RUN $1 rc=$rc"
  grep -E "TOKS_S|TOKS=|PHASE ms|INFO|FIRST20" $W/$1.log | tail -8 | tee -a $W/runner.log
}

probe(){ # $1 = tag, $2 = python, $3 = env (default ENV; pass a stripped env for the rows-unset variant)
  local e=${3:-$ENV}
  sleep 25
  log "PROBE $1 starting"
  $e $2 $PROBE > $W/probe_$1.log 2>&1
  log "PROBE $1 rc=$?"
  grep -E "M1=|probe env" $W/probe_$1.log | tee -a $W/runner.log
}
ENVNOROWS="env MODELS=$MODELS LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 REFWD_M1=1"

# ---- probe (verify what the probe measures): rows set + rows unset ----
probe canon $OFF_PY $ENVNOROWS
probe r4b   $ON_PY  $ENVNOROWS
probe canonR $OFF_PY
probe r4bR   $ON_PY

# ---- anchor protocol (322 iterations = 161 verifies / ~510 tokens) ----
for i in 1 2 3; do
  run O$i $OFF_PY $B6 322 ""
  run N$i $ON_PY  $B6 322 ""
done
# ---- phase decomposition (MTP6_MTPTRIM=1): 128 (64v) and 1024 (512v) ----
for i in 1 2 3; do
  run P64O$i $OFF_PY $B6P 128 "MTP6_MTPTRIM=1"
  run P64N$i $ON_PY  $B6P 128 "MTP6_MTPTRIM=1"
  run P512O$i $OFF_PY $B6P 1024 "MTP6_MTPTRIM=1"
  run P512N$i $ON_PY  $B6P 1024 "MTP6_MTPTRIM=1"
done

sleep 25
log "POST SBCAR: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors | cut -d' ' -f1)"
log "DONE"
touch $W/done
