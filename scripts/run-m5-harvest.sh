#!/bin/bash
# m5 harvest: one more deterministic m3-flow run, layer-3 block capture for A+B.
MODELS="${MODELS:-/models}"
cd $MODELS/tmp/run-785
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib
PY=$MODELS/venv-d/bin/python
export MODE=harvest
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  flock -w 600 $MODELS/gpu-bench.lock -c "$PY diag-s62-m5.py > diag-m5-harvest.log 2>&1; echo \$? > m5-harvest-rc"
  rc=$?
  if [ $rc -eq 1 ]; then
    echo "FLOCK-FAIL attempt=$attempt $(date -u +%T), sleep 60" >> m5-run.log
    sleep 60
    continue
  fi
  echo "harvest done pyrc=$(cat m5-harvest-rc) attempt=$attempt $(date -u +%T)" >> m5-run.log
  exit 0
done
echo "harvest FLOCK-FAIL after 12 attempts $(date -u +%T)" >> m5-run.log
exit 1
