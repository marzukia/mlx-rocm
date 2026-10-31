#!/bin/bash
# m5 ab: in-model chain A/B + direct re-run row-invariance (no model load).
MODELS="${MODELS:-/models}"
cd $MODELS/tmp/run-785
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib
PY=$MODELS/venv-d/bin/python
export MODE=ab
for attempt in 1 2 3 4 5 6; do
  flock -w 300 $MODELS/gpu-bench.lock -c "$PY diag-s62-m5.py > diag-m5-ab.log 2>&1; echo \$? > m5-ab-rc"
  rc=$?
  if [ $rc -eq 1 ]; then
    echo "FLOCK-FAIL attempt=$attempt $(date -u +%T), sleep 60" >> m5-run.log
    sleep 60
    continue
  fi
  echo "ab done pyrc=$(cat m5-ab-rc) attempt=$attempt $(date -u +%T)" >> m5-run.log
  exit 0
done
echo "ab FLOCK-FAIL after 6 attempts $(date -u +%T)" >> m5-run.log
exit 1
