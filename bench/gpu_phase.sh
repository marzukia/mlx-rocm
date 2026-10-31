#!/bin/bash
# run-835 GPU phase (run ON guest-a): gates + bench for the dequant-hoist
# wheel (e144f39). Lock protocol: flock -n, on fail sleep 60s, retry up to
# 20x, log all fails.
#
# Usage: bash run-835_gpu_phase.sh <new-wheel-dir>
set -uo pipefail
MODELS="${MODELS:-/models}"
WHEELNW=$1
T=$MODELS/tmp/qmv-mr
H=$MODELS/tmp/qmv-mr-harness
LOCK=$MODELS/gpu-bench.lock
FL=$T/flock-fails.log
PYG4=$MODELS/venv-d/bin/python
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
WARMUP=10
REPS=50
mkdir -p $T/ref $T/new
: > $FL

try_locked() { # $1=tag, $2...=command; distinguishes lock-fail from cmd-fail
  local tag=$1; shift
  local i=0 rc
  rm -f $T/.ran-$tag
  while true; do
    flock -n $LOCK -c "touch $T/.ran-$tag; $*"
    rc=$?
    if [ -f $T/.ran-$tag ]; then
      rm -f $T/.ran-$tag
      return $rc
    fi
    i=$((i + 1))
    if [ $i -ge 20 ]; then
      echo "$(date -u +%FT%TZ) FLOCK-FAIL $tag attempt=$i GIVE-UP" | tee -a $FL
      return 1
    fi
    echo "$(date -u +%FT%TZ) FLOCK-FAIL $tag attempt=$i sleep-60" | tee -a $FL
    sleep 60
  done
}

echo "=== run-835 GPU phase start $(date -u +%FT%TZ) wheel=$WHEELNW ==="

# --- config probe (doc rule 6): GPU + wheel + gfx1201 --------------------
$PYG4 /tmp/gpuinfo.py > $T/new/gpuinfo.log 2>&1 || true
grep -c gfx1201 $WHEELNW/mlx/lib/libmlx.so > $T/new/gfx1201-count.txt
WHEELSHA=$(sha256sum $WHEELNW/mlx/lib/libmlx.so | cut -d' ' -f1)
echo "CONFIG: wheel_lib_sha256=$WHEELSHA gfx1201_strings=$(cat $T/new/gfx1201-count.txt)" | tee $T/new/config.log
cat $T/new/gpuinfo.log >> $T/new/config.log

# --- gate 1: bit-exact vs per-row wide kernel (16 refs) ------------------
# refs regenerated with the baseline wheel (venv-d, per-row wide kernel)
try_locked bitexact-ref "$PYG4 $H/qmv_mr_bitexact.py ref $T/ref/bitexact" \
  > $T/ref/bitexact.log 2>&1 || { echo "GATE1-REF-FAIL"; exit 1; }
grep "BITEXACT-GATE" $T/ref/bitexact.log
ls $T/ref/bitexact/*.npy | wc -l
PYTHONPATH=$WHEELNW try_locked bitexact-check \
  "$PYG4 $H/qmv_mr_bitexact.py check $T/ref/bitexact" \
  > $T/new/bitexact.log 2>&1 || { echo "GATE1-CHECK-FAIL"; exit 1; }
grep -E "BITEXACT-GATE|FAIL" $T/new/bitexact.log

# --- bench: 4 shapes x M=1..4, rocprofv3 kernel time (doc rule 2) --------
for spec in "4096 12288" "12288 4096" "4096 4096" "4096 248320"; do
  set -- $spec
  K=$1; N=$2
  for M in 1 2 3 4; do
    rm -rf $T/new/trace && mkdir -p $T/new/trace
    PYTHONPATH=$WHEELNW QMV_K=$K QMV_N=$N QMV_M=$M QMV_WARMUP=$WARMUP QMV_REPS=$REPS \
      try_locked "bench-${K}x${N}-M${M}" \
      "$ROCPROF -d $T/new/trace -f json --kernel-trace -- $PYG4 $H/qmv_mr_bench.py" \
      > $T/new/bench-${K}x${N}-M${M}.log 2>&1 || { echo "BENCH-FAIL $K x $N M=$M"; exit 1; }
    TRACE=$(ls $T/new/trace/*/*.json 2>/dev/null | head -1)
    [ -n "$TRACE" ] || { echo "TRACE-MISSING $K x $N M=$M"; exit 1; }
    python3 $H/qmv_mr_parse.py "$TRACE" $K $N $M $WARMUP $REPS \
      >> $T/new/bench-${K}x${N}-M${M}.log 2>&1 || { echo "PARSE-FAIL $K x $N M=$M"; exit 1; }
    grep -E "CONFIG:|QMV:" $T/new/bench-${K}x${N}-M${M}.log
  done
done

# --- gate 2/3: AR 1024-token greedy stream (MLX_MTP=0) -------------------
PYTHONPATH=$WHEELNW try_locked ar \
  "MLX_MTP=0 MTP_MODEL=$MODELS/qwen3-9b-mlx timeout 900 $PYG4 $H/mtp-bench3.py" \
  > $T/new/ar.log 2>&1 || { echo "GATE2-AR-FAIL"; exit 1; }
grep -E "^MTP=|^FIRST16:" $T/new/ar.log

# --- gate 2/3: stream comparison vs per-row wheel stream + baseline ------
grep "^ALLTOKS:" $T/new/ar.log > $T/new/alltoks
cmp $T/new/alltoks $MODELS/tmp/new-alltoks \
  && echo "AR-CMP vs per-row wheel stream (run-818 new-alltoks): IDENTICAL" \
  || echo "AR-CMP vs per-row wheel stream: DIFFER"
cmp $T/new/alltoks $MODELS/tmp/ref-alltoks \
  && echo "AR-CMP vs baseline stream (ref-alltoks): IDENTICAL" \
  || echo "AR-CMP vs baseline stream: DIFFER"

echo "=== run-835 GPU phase done $(date -u +%FT%TZ) ==="
