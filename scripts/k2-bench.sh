#!/bin/bash
# K2 bench runner (guest side).
# usage: k2-bench.sh <venv> <model> <max_tokens> <n_runs> <tag>
V=$1; M=$2; T=$3; N=$4; TAG=$5
PY=$V/bin/python
for i in $(seq 1 10); do
  U=$(/opt/rocm/bin/amd-smi --rocm-smi 2>/dev/null | python3 -c "
import sys, re
for line in sys.stdin:
    if 'GPU%' in line or ('N/A' in line and '%' in line):
        m = re.findall(r'(\d+)\s*%', line)
        if m:
            print(m[-1]); break
else:
    print(0)
")
  if [ "${U:-0}" -lt 1 ]; then
    echo "gpu idle (util=${U}%)"
    break
  fi
  echo "gpu busy (util=${U}%), sleeping 90s (attempt $i/10)"
  sleep 90
done
cd /mnt/models
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib $PY gauntlet/bench.py "$M" "$T" "$N" "$TAG"
