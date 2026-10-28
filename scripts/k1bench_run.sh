#!/bin/sh
# k1bench_run.sh <baseline|k1-16|k1-32>
# GPU-serialized (flock, short-wait poll: re-poll if held >300s),
# python wrapped in `timeout 900`; results print before the known
# exit-time HIP teardown hang, so we kill the process group as soon
# as all 5 N lines land. NEVER waits beyond the timeout.
CFG=$1
cd /mnt/models/tmp
rm -f k1bench_$CFG.out
i=0
while :; do
  i=$((i+1))
  if flock -w 300 /mnt/models/gpu-bench.lock -c true; then
    break
  fi
  echo "lock held >300s (poll $i), re-polling"
done
# hold the lock for the whole bench (fd 9)
exec 9>>/mnt/models/gpu-bench.lock
flock -x 9
setsid sh -c "LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib timeout -s KILL 900 /mnt/models/venv-k1/bin/python k1_bench.py $CFG > k1bench_$CFG.out 2>&1" &
pid=$!
ok=0
for i in $(seq 1 190); do
  sleep 5
  n=$(grep -c "^CFG=$CFG N=" k1bench_$CFG.out 2>/dev/null || true)
  n=${n:-0}
  if [ "$n" -ge 5 ]; then ok=1; sleep 2; kill -9 -$pid 2>/dev/null; break; fi
  if ! kill -0 $pid 2>/dev/null; then break; fi
done
[ "$ok" = 1 ] || kill -9 -$pid 2>/dev/null
wait $pid 2>/dev/null
echo "DONE $CFG ok=$ok lines=$(grep -c "^CFG=$CFG N=" k1bench_$CFG.out 2>/dev/null || echo 0)"
