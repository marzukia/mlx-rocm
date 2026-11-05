#!/bin/bash
# t2031 gen2 probe session B: bench A/B, BEST variant only. $1 = variant
# name (e.g. V1), $2 = extra env (e.g. "MLX_ROCM_QMM_M3_V1=1"). 13-var
# config-of-record (MTP6_TAPE=hybrid + MLX_ROCM_QMM_M3=1), 322 anchor,
# 3 rounds x 2 arms interleaved, 25 s idle before each run, per-run
# timeout 600, SCLK sampler every 10 s.
set -u
VNAME="$1"
VENV_EXTRA="$2"
exec 9>/home/andryo/gpu.lock
DEADLINE=$(( $(date +%s) + 5400 ))
while true; do
  if flock -n 9; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "PENDING-LOCK: no lock within 90m"; exit 3; fi
  sleep 15
done
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ==="
export MODELS=/mnt/models
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
export HCC_AMDGPU_TARGET=gfx1201
D=/home/andryo/gen2
L=$D/session-b.log
mkdir -p $D
cd $D
echo "=== GEN2 SESSION B START ($VNAME) $(date -u) ===" | tee $L
sclk() { m=$(grep "*" /sys/class/drm/card1/device/pp_dpm_sclk 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1); if [ -z "$m" ]; then m=$(timeout 15 rocm-smi --showclocks 2>/dev/null | grep -i "sclk\|GPU Clock" | head -1); fi; echo "$m"; }
( while true; do echo "$(date -u +%H:%M:%S) sclk=[$(sclk)]" >> $D/sclk-b.log; sleep 10; done ) &
SCLK_PID=$!
trap "kill $SCLK_PID 2>/dev/null" EXIT
{
echo "bench-md5: $(md5sum /home/andryo/gen2/mtp-bench6.py)"
echo "sbcar-md5-pre: $(md5sum /mnt/models/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-gen2-pre: $(md5sum /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-m3k-pre: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-pre: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
} | tee -a $L
CE="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1"
PY=/home/andryo/venv-gen2/bin/python
anchor() { # $1 = name, $2 = prompt env, $3 = extra env
  sleep 25
  echo "=== ANCHOR $1 $(date -u +%H:%M:%SZ) sclk=[$(sclk)] ===" | tee -a $L
  timeout 600 env $CE $2 $3 MTP_MAX_TOKENS=322 $PY /home/andryo/gen2/mtp-bench6.py > "$D/raw-$1.raw" 2>&1
  echo "ANCHOR-$1-rc=$?" | tee -a $L
  grep -E "INFO|MTP=3|FIRST20|ALLTOKS" "$D/raw-$1.raw" | head -40 >> $L 2>&1
  sleep 25
}
CANON=""
HELLO="MTP6_PROMPT=Hello"
for round in 1 2 3; do
  if [ $round = 1 ]; then ORD="C $VNAME"; fi
  if [ $round = 2 ]; then ORD="$VNAME C"; fi
  if [ $round = 3 ]; then ORD="C $VNAME"; fi
  for s in $ORD; do
    if [ $s = C ]; then
      anchor "canon-C_r$round" "$CANON" ""
    else
      anchor "canon-${VNAME}_r${round}" "$CANON" "$VENV_EXTRA"
    fi
    if [ $s = C ]; then
      anchor "hello-C_r$round" "$HELLO" ""
    else
      anchor "hello-${VNAME}_r${round}" "$HELLO" "$VENV_EXTRA"
    fi
  done
done
{
echo "sbcar-md5-post: $(md5sum /mnt/models/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-m3k-post: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-post: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
} | tee -a $L
echo "=== GEN2 SESSION B DONE $(date -u) ===" | tee -a $L
echo DONE > $D/.done-b
