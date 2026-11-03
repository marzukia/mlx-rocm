#!/bin/bash
# Round 3: WCOLS tuning A/B (bit-identical geometry) on venv-ar,
# canonical bench /home/andryo/bench/mtp-bench6.py (md5 452d363d),
# standing config, 400 tokens, 3 runs per config.
LOGDIR=/home/andryo/arstep
mkdir -p $LOGDIR
exec 9>/home/andryo/gpu-bench.lock
DEADLINE=$(( $(date +%s) + 7200 ))
while true; do
  if flock -n 9; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "PENDING-LOCK: no lock within 2h"; exit 3; fi
  sleep 60
done
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ==="
md5sum /home/andryo/bench/mtp-bench6.py

run() {
  local label=$1 wcols=$2
  for i in 1 2 3; do
    echo "=== RUN $label.$i $(date -u +%H:%M:%SZ) ==="
    env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
        HCC_AMDGPU_TARGET=gfx1201 \
        MODELS=/mnt/models MLX_MTP=0 MTP_MAX_TOKENS=400 \
        MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1 \
        MLX_QMV_WCOLS=$wcols \
        timeout -s KILL 1500 /home/andryo/venv-ar/bin/python /home/andryo/bench/mtp-bench6.py \
        > $LOGDIR/$label.$i.log 2>&1
    echo "rc=$?"
    grep -E "^MTP=0" $LOGDIR/$label.$i.log
  done
}

run W4  4
run W16 16
echo "=== SESSION DONE $(date -u +%H:%M:%SZ) ==="
exec 9>&-
