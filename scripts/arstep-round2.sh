#!/bin/bash
# AR step change round-2 GPU session (guest-a), single lock hold:
#  1. CONTROL: venv-ar (patched wheel) standing config, 400 tokens —
#     isolates wheel/codegen effects from the gate effects.
#  2. per-shape M=1 QMM probes: baseline (venv-canon) vs m1lx / m1rowsmsg
#     / wcols-16 / wcols-4 (venv-ar).
#  3. MTP=4 guard: venv-canon, 1024 tokens (expect ~65.9-class).
LOGDIR=/home/andryo/arstep
mkdir -p $LOGDIR
exec 9>/home/andryo/gpu-bench.lock
DEADLINE=$(( $(date +%s) + 10800 ))
while true; do
  if flock -n 9; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "PENDING-LOCK: no lock within 3h"; exit 3; fi
  sleep 60
done
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ==="

# 1. CONTROL (3 runs, median)
for i in 1 2 3; do
  echo "=== CONTROL.$i $(date -u +%H:%M:%SZ) ==="
  env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
      HCC_AMDGPU_TARGET=gfx1201 MODELS=/mnt/models MLX_MTP=0 MTP_MAX_TOKENS=400 \
      MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1 \
      timeout -s KILL 1500 /home/andryo/venv-ar/bin/python /home/andryo/mtp-bench6.py \
      > $LOGDIR/C.$i.log 2>&1
  echo "rc=$?"
  grep -E "^MTP=0" $LOGDIR/C.$i.log
done

# 2. per-shape probes (baseline on venv-canon; variants on venv-ar)
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 \
    timeout -s KILL 900 /home/andryo/venv-canon/bin/python /home/andryo/arstep_qmm_probe.py baseline \
    > $LOGDIR/P-baseline.log 2>&1
echo "probe baseline rc=$?"; grep -E "^CFG|PROBE-DONE" $LOGDIR/P-baseline.log
for cfg in m1lx m1rowsmsg wcols-16 wcols-4; do
  env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 \
      timeout -s KILL 900 /home/andryo/venv-ar/bin/python /home/andryo/arstep_qmm_probe.py $cfg \
      > $LOGDIR/P-$cfg.log 2>&1
  echo "probe $cfg rc=$?"; grep -E "^CFG|PROBE-DONE" $LOGDIR/P-$cfg.log
done

# 3. MTP=4 guard (venv-canon, standing config, 1024 tokens)
echo "=== MTP4-GUARD $(date -u +%H:%M:%SZ) ==="
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
    HCC_AMDGPU_TARGET=gfx1201 MODELS=/mnt/models MLX_MTP=4 MTP_MAX_TOKENS=1024 \
    MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1 \
    timeout -s KILL 1800 /home/andryo/venv-canon/bin/python /home/andryo/mtp-bench6.py \
    > $LOGDIR/MTP4-guard.log 2>&1
echo "rc=$?"
grep -E "^MTP=4" $LOGDIR/MTP4-guard.log
echo "=== SESSION DONE $(date -u +%H:%M:%SZ) ==="
exec 9>&-
