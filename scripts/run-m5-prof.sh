#!/bin/bash
# m5 prof: per-spec rocprofv3 --kernel-trace dispatch table (no model load).
cd /mnt/models/tmp/an earlier run
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib
PY=/mnt/models/venv-d/bin/python
export MODE=prof
for attempt in 1 2 3 4 5 6; do
  flock -w 900 /mnt/models/gpu-bench.lock -c "$PY diag-s62-m5.py > diag-m5-prof.log 2>&1; echo \$? > m5-prof-rc"
  rc=$?
  if [ $rc -eq 1 ]; then
    echo "FLOCK-FAIL attempt=$attempt $(date -u +%T), sleep 60" >> m5-run.log
    sleep 60
    continue
  fi
  echo "prof done pyrc=$(cat m5-prof-rc) attempt=$attempt $(date -u +%T)" >> m5-run.log
  exit 0
done
echo "prof FLOCK-FAIL after 6 attempts $(date -u +%T)" >> m5-run.log
exit 1
