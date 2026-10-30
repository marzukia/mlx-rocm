#!/bin/bash
# Step 0 (method-correction doc 2026-10-28, section 4): in-guest streaming-read
# ceiling + runtime CU count. No model load.
#
# Usage (ON the build VM):  bash step0_run.sh
#   env: STEP0_NBYTES (default 4 GiB), STEP0_WARMUP (3), STEP0_ITERS (10),
#        STEP0_OUT (default $MODELS/tmp/step0)
#
# Timing: rocprofv3 kernel time only (doc rule 2). GB/s = bytes moved / kernel
# time. CU count: what the runtime reports (doc section 1 row 4).
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
MODELS="${MODELS:-/models}"
PY=$MODELS/venv-d/bin/python
OUT=${STEP0_OUT:-$MODELS/tmp/step0}
LOCK=$MODELS/gpu-bench.lock
HERE=$(cd "$(dirname "$0")" && pwd)
NBYTES=${STEP0_NBYTES:-$((4 * 1024 * 1024 * 1024))}
WARMUP=${STEP0_WARMUP:-3}
ITERS=${STEP0_ITERS:-10}
export STEP0_NBYTES=$NBYTES STEP0_WARMUP=$WARMUP STEP0_ITERS=$ITERS
mkdir -p $OUT

echo "ENV: venv=$PY rocm=10.1(LD_LIBRARY_PATH=$LD_LIBRARY_PATH) lock=$LOCK"

# --- runtime CU count (doc section 1 row 4: record what the runtime reports)
echo "--- rocminfo ---"
/opt/rocm/core-10.1/bin/rocminfo 2>/dev/null | grep -B1 -A4 "Marketing Name" | head -12
echo "--- amd-smi ---"
/opt/rocm/core-10.1/bin/amd-smi 2>/dev/null | head -14

# --- streaming-read ceiling (under flock, rocprofv3 kernel time)
rm -rf $OUT/trace && mkdir -p $OUT/trace
flock -w 60 $LOCK -c "/opt/rocm/core-10.1/bin/rocprofv3 -d $OUT/trace -f json --kernel-trace -- $PY $HERE/step0_streaming_read.py" \
  || { echo "FLOCK-FAIL"; exit 1; }

TRACE=$(ls $OUT/trace/*/*.json 2>/dev/null | head -1)
[ -n "$TRACE" ] || { echo "TRACE-MISSING"; exit 1; }
python3 $HERE/step0_parse.py "$TRACE" $ITERS $WARMUP $NBYTES
