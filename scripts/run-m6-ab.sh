#!/bin/bash
# m6: focused rope_q A/B on the captured tensors (post RowInvariantRoPE
# patch). Uploads the script + the patched venv module is NOT needed -
# the script imports the venv's mlx_lm (the patch must be applied to
# the venv first: scripts/apply-m6-patch.py).
#
# Usage:
#   bash run-m6-ab.sh          # run (bounded flock -w 900, 8 retries)
#   bash run-m6-ab.sh check    # tail the log
#
# Exit: 0 = M6-AB-OK (all rope checks MATCH); 1 = a rope check DIFFed;
# 3 = unexpected.

export PATH=/opt/rocm/bin:/opt/rocm/core-10.1/bin:/opt/rocm/lib:/opt/rocm/core-10.1/lib:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib

OUT=${M6_OUT:-/mnt/models/tmp/an earlier run}
mkdir -p $OUT
LOCK=/mnt/models/gpu-bench.lock
PY=/mnt/models/venv-d/bin/python
SCRIPT=$OUT/diag-s62-m6.py

if [ "$1" = "check" ]; then
  tail -n 60 $OUT/ab.log 2>/dev/null
  exit 0
fi

[ -f $SCRIPT ] || { echo "missing $SCRIPT (upload scripts/diag-s62-m6.py first)"; exit 3; }
: > $OUT/ab.log
for attempt in 1 2 3 4 5 6 7 8; do
  flock -w 900 $LOCK env $PY $SCRIPT >> $OUT/ab.log 2>&1
  rc=$?
  if [ $rc -eq 1 ] && ! grep -q "M6-AB-OK\|Traceback" $OUT/ab.log 2>/dev/null; then
    echo "FLOCK-FAIL attempt=$attempt $(date -u +%T)" >> $OUT/ab.log
    sleep 60
    continue
  fi
  echo "done pyrc=$rc attempt=$attempt $(date -u +%T)" >> $OUT/ab.log
  break
done
tail -n 60 $OUT/ab.log
grep -q "M6-AB-OK" $OUT/ab.log
