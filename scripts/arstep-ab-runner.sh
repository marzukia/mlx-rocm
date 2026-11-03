#!/bin/bash
# AR step change A/B runner (guest-a). Canonical bench mtp-bench6.py,
# MLX_MTP=0, 400-token window (t1449 protocol), 3 runs per config, median.
#
#   A  venv-canon  + standing config (M=1 wide; REFWD no-op in this wheel)
#   B1 venv-ar     + standing config + MLX_ROCM_M1_LX=1 (M=1 int8-dot4)
#   B2 venv-ar     + standing config + MLX_ROCM_M1_ROWSMSG=1 (M=1 rows-msg)
#
# Standing config: MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1
# Lock: short-wait poll (60s), 3h deadline.
LOGDIR=<guest-home>/arstep
mkdir -p $LOGDIR
exec 9><guest-home>/gpu-bench.lock
DEADLINE=$(( $(date +%s) + 10800 ))
while true; do
  if flock -n 9; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "PENDING-LOCK: no lock within 3h"; exit 3; fi
  sleep 60
done
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ==="
md5sum <guest-home>/mtp-bench6.py

run() {  # $1=label  $2=venv  $3=extra env
  local label=$1 venv=$2 extra=$3
  for i in 1 2 3; do
    echo "=== RUN $label.$i $(date -u +%H:%M:%SZ) ==="
    env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
        HCC_AMDGPU_TARGET=gfx1201 \
        MODELS=<models-mount> MLX_MTP=0 MTP_MAX_TOKENS=400 \
        MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1 \
        $extra \
        timeout -s KILL 1500 $venv/bin/python <guest-home>/mtp-bench6.py \
        > $LOGDIR/$label.$i.log 2>&1
    echo "rc=$?"
    grep -E "^TOKS:|t/s|FIRST20:" $LOGDIR/$label.$i.log | head -5
  done
}

run A  <guest-home>/venv-canon ""
run B1 <guest-home>/venv-ar    "MLX_ROCM_M1_LX=1"
run B2 <guest-home>/venv-ar    "MLX_ROCM_M1_ROWSMSG=1"

echo "=== LOCK RELEASED $(date -u +%H:%M:%SZ) ==="
exec 9>&-
