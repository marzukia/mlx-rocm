#!/bin/bash
# Step 1 (method-correction doc 2026-10-28, section 4 step 1): decode QMV
# gates + bench. GB/s from rocprofv3 kernel time + bytes moved (doc rule 2);
# bit-exact gate (doc rule 4); effective config printed + asserted (doc rule 6).
#
# Usage (ON the build VM):  bash qmv_mr_run.sh <tag> <python>
#   tag    : ref (current wheel: bitexact ref + baseline bench + AR stream)
#            new (new wheel: bitexact check + bench + AR stream)
#   python : interpreter with the wheel under test on its path
#
#   e.g. bash qmv_mr_run.sh ref $MODELS/venv-d/bin/python
#        bash qmv_mr_run.sh new $MODELS/venv-d/bin/python \
#            (with PYTHONPATH=$MODELS/wheel-qmv exported)
set -uo pipefail
MODELS="${MODELS:-/models}"
TAG=$1
PY=$2
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
T=$MODELS/tmp/qmv-mr/$TAG
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$MODELS/gpu-bench.lock
WARMUP=10
REPS=50
mkdir -p $T

echo "=== $TAG ENV: py=$PY wheel=$($PY -c 'import mlx.core as mx; print(mx.__version__)') ==="

# --- bit-exact gate (doc rule 4) -------------------------------------------
if [ "$TAG" = "ref" ]; then
  MODE=ref
else
  MODE=check
fi
flock -w 300 $LOCK -c "$PY $HERE/qmv_mr_bitexact.py $MODE $T/bitexact" \
  || { echo "FLOCK-FAIL bitexact"; exit 1; }

# --- per-shape bench (rocprofv3 kernel time, doc rule 2) -------------------
for spec in "4096 12288" "12288 4096" "4096 4096" "4096 248320"; do
  set -- $spec
  K=$1; N=$2
  for M in 1 2 3 4; do
    rm -rf $T/trace && mkdir -p $T/trace
    QMV_K=$K QMV_N=$N QMV_M=$M QMV_WARMUP=$WARMUP QMV_REPS=$REPS \
      flock -w 600 $LOCK -c "$ROCPROF -d $T/trace -f json --kernel-trace -- $PY $HERE/qmv_mr_bench.py" \
      || { echo "FLOCK-FAIL bench $K x $N M=$M"; exit 1; }
    TRACE=$(ls $T/trace/*/*.json 2>/dev/null | head -1)
    [ -n "$TRACE" ] || { echo "TRACE-MISSING $K x $N M=$M"; exit 1; }
    python3 $HERE/qmv_mr_parse.py "$TRACE" $K $N $M $WARMUP $REPS || exit 1
  done
done

# --- AR 1024-token greedy stream (doc rule 4) + tok/s record --------------
flock -w 900 $LOCK -c "MLX_MTP=0 MTP_MODEL=$MODELS/qwen3-9b-mlx timeout 900 $PY $HERE/mtp-bench3.py" \
  > $T/ar.log 2>&1 || { echo "FLOCK-FAIL ar"; exit 1; }
grep -E "^MTP=|^FIRST16:" $T/ar.log
echo "ALLTOKS saved: $T/ar.log"
echo "=== $TAG DONE ==="
