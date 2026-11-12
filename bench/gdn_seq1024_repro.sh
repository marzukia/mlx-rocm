#!/bin/bash
# W1 PHASE 2 @1024 REPRO driver (jimmy, 2026-11-12, op order "repro the 108 too").
# The EXACT G1 record-battery invocation (seq S=3 @1024, the B config, 3 rounds)
# RUN FROM THE IN-TREE FILES: bench/k2arm-pinned.py (this dir) + the in-tree
# model (models/qwen3_5.py), which this driver installs into the venv before the
# run. Provenance + expected output: REPORT-W1-PHASE2-REPRO-20261112.md (repo root).
#
# Env mapping (the box run state, neutral in-tree form; the doc states the mapping):
#   MODELS       - the model root containing qwen3-9b-mlx (box: the models mount;
#                  in-tree default /models, overridable via the env)
#   W1P2_VENV    - the venv holding the ceef7ef1 wheel (box: $HOME/venv-p2seq)
#   W1P2_GPU_LOCK- the shared card lock (box: $HOME/gpu.lock); one GPU job at a time
#   HCC_AMDGPU_TARGET=gfx1201, HIP_VISIBLE_DEVICES, LD_PRELOAD=libhsa-runtime64.so.1,
#                  TMPDIR - the box run env, all neutral values
# Clock: DPM AUTO (unpinned). This driver RECORDS the sclk state per round, never pins.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
ROUNDS=${ROUNDS:-3}
NTOK=${NTOK:-1024}
export MODELS=${MODELS:-/models}
export W1P2_VENV=${W1P2_VENV:-$HOME/venv-p2seq}
export W1P2_GPU_LOCK=${W1P2_GPU_LOCK:-$HOME/gpu.lock}
export LD_PRELOAD=libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} TMPDIR=${TMPDIR:-/var/tmp/scratch}
PY=$W1P2_VENV/bin/python3
[ -x "$PY" ] || { echo "VENV-MISSING: $PY (set W1P2_VENV)"; exit 1; }
# one GPU job at a time (the shared card lock, 300s wait)
exec 9>"$W1P2_GPU_LOCK"
flock -w 300 9 || { echo "LOCK_FAIL (a GPU job is running)"; exit 1; }
# in-tree model -> venv (the repro runs the in-tree model file, expect md5 303b245b)
VPLAT=$("$PY" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
MP=$W1P2_VENV/lib/$VPLAT/site-packages/mlx_lm/models/qwen3_5.py
cp "$ROOT/models/qwen3_5.py" "$MP"
echo "MODEL-INSTALLED $(md5sum "$MP" | cut -d' ' -f1) (expect 303b245b)"
# the record-config arm env (the G1 battery, verbatim)
export MLX_MTP=3 MTP6_TAPE=seq MTP6_DRAFT_HEAD=bf16 MTP6_HEAD_H=postnorm MTP6_MOD_BF16=1
mkdir -p "$TMPDIR"
for r in $(seq 1 "$ROUNDS"); do
  L=$TMPDIR/gdnseq1024-repro-r$r.log
  MTP_MAX_TOKENS=$NTOK timeout 1800 "$PY" "$HERE/k2arm-pinned.py" > "$L" 2>&1
  RC=$?
  S=$(grep -a "^ALLTOKS:" "$L" | sed "s/^ALLTOKS: *//" | tr -d '\n' | md5sum | cut -c1-8)
  M=$(grep -a "^MTP=" "$L" | head -1)
  I=$(grep -a "^INFO" "$L" | head -1 | grep -oaE "REFWDS=[0-9]+" | head -1)
  F=$(grep -a "^FIRST20:" "$L" | head -1)
  C=$(grep -m1 '\*' /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null | tr -d ' ')
  echo "REPRO-R$r rc=$RC stream=$S $M $I $F clock=${C:-unreadable}"
done
echo "REPRO-DONE $(date -u +%H:%M:%S)"
