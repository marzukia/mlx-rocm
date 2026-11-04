#!/bin/bash
# Stack-assembly battery: arm B (MTP6_TAPE=hybrid) vs R5 (MTP6_TAPE=r5),
# BOTH arms + draft gate (MTP6_DRAFTGATE=1 T0=16 + ngram n=4) + 4-bit MTP
# module (CH_MTP_4BIT=1, sidecar by path). Identical method to the R5 report:
# anchor MTP_MAX_TOKENS=322 + 64v phase MTP_MAX_TOKENS=128, 3 repeats
# interleaved B/R5, 25 s idle between runs, SCLK sampled before/after (and
# every 10 s) each run, all under flock on ~/gpu.lock.
#
# Runs per repeat: BAN/R5AN (anchor, dg harness, gate+4bit), BPH/R5PH
# (64v phase breakdown, phase harness, 4bit, no gate - the phase harness has
# no gate path, same as the R5 report's 64v method), BAN64/R5AN64 (64v with
# gate+4bit, dg harness region timing). One CTRL (phase harness, hybrid, bf16
# sidecar, no gate/4bit): the r4v fused-verify control sanity.
set -u
MODELS_DIR="${MODELS_DIR:?set by caller}"
W=$HOME/stack-bench
LOCK=$HOME/gpu.lock
M4="$MODELS_DIR/qwen3-9b-mlx/model-mtp-4bit.safetensors.disabled"
mkdir -p "$W"
cd "$W"
LOG="$W/stack-run.log"
exec 9>"$LOCK"
LOCKED=0
for i in $(seq 1 96); do
  if flock -x -w 1800 9; then
    LOCKED=1
    echo "LOCK-ACQ iter=$i $(date -u +%H:%M:%S)" >"$LOG"
    break
  fi
  echo "LOCK-BUSY iter=$i $(date -u +%H:%M:%S) holders=$(fuser $LOCK 2>/dev/null | tr '\n' ' ')" >>"$LOG"
  sleep 60
done
[ "$LOCKED" = "1" ] || { echo "LOCK-FAIL $(date -u +%H:%M:%S)" >>"$LOG"; exit 1; }
PY=$HOME/venv-r4v/bin/python
DG="$W/mtp-bench6-dg.py"
PH="$W/mtp-bench6-phase.py"
LM=$HOME/venv-r4v/lib/python3.14/site-packages/mlx/lib/libmlx.so
SB="$MODELS_DIR/qwen3-9b-mlx/model-mtp.safetensors"
echo "PRE $(date -u +%H:%M:%S) libmlx=$(md5sum "$LM" | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1) m4size=$(stat -c %s "$M4")" >>"$LOG"
sleep 25

sclk() { grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | head -1 | tr -d ' '; }

sampler() {
  while :; do
    echo "  sclk $(date -u +%H:%M:%S) $(sclk)" >>"$LOG"
    sleep 10
  done
}

# run <name> <maxtok> <benchfile> <tape> <gate:1|0> <fourbit:1|0>
run() {
  local name=$1 mt=$2 bf=$3 tape=$4 gate=$5 f4=$6
  sleep 25
  local env4=""
  [ "$f4" = "1" ] && env4="CH_MTP_4BIT=1 MTP4_SIDECAR=$M4"
  local envg=""
  [ "$gate" = "1" ] && envg="MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4"
  echo "RUN $(date -u +%H:%M:%S) $name tape=$tape gate=$gate f4=$f4 maxtok=$mt sclk=$(sclk)" >>"$LOG"
  sampler & local SP=$!
  env MODELS="$MODELS_DIR" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201 \
    MLX_MTP=4 GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 \
    MTP6_TAPE="$tape" $envg $env4 MTP_MAX_TOKENS="$mt" "$PY" "$bf" >"$W/$name.log" 2>&1
  local rc=$?
  kill "$SP" 2>/dev/null
  echo "DONE $(date -u +%H:%M:%S) $name rc=$rc sclk=$(sclk)" >>"$LOG"
}

# CTRL: the r4v fused-verify control sanity (bf16 sidecar, hybrid, no gate/4bit)
run CTRL 128 "$PH" hybrid 0 0

for i in 1 2 3; do
  run "BAN-$i" 322 "$DG" hybrid 1 1
  run "R5AN-$i" 322 "$DG" r5 1 1
  run "BPH-$i" 128 "$PH" hybrid 0 1
  run "R5PH-$i" 128 "$PH" r5 0 1
  run "BAN64-$i" 128 "$DG" hybrid 1 1
  run "R5AN64-$i" 128 "$DG" r5 1 1
done

echo "POST $(date -u +%H:%M:%S) libmlx=$(md5sum "$LM" | cut -d' ' -f1) sbcar=$(md5sum "$SB" | cut -d' ' -f1) m4size=$(stat -c %s "$M4")" >>"$LOG"
echo "ALL-DONE $(date -u +%H:%M:%S)" >>"$LOG"
