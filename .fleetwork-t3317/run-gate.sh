#!/bin/bash
# t3317 gate runner: 27B MTP K=3, T3298_PREWARM + T3298_ROUNDS measured rounds,
# MLX_MMQ_CFG=1 (the [pcr] proof hook + the [cfgq4k] dispatch count), 5s sampler.
set -u
R=/home/andryo/gauntlet-results
B=/home/andryo/gauntlet-mlx-bench
D=/home/andryo/mlx-rocm-dev
PY=/home/andryo/gauntlet-vehicle-venv/bin/python
P4=/home/andryo/h27b-bench/prompts/p4096.txt
M=/home/andryo/models/Qwen3.8-27B-4bit
LOG=$R/t3317-$1-gate.log
SMI=$R/t3317-$1-smi.log
ENV="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo TMPDIR=/var/tmp/mlxbench CPATH=/home/andryo/h27b-bench/rocmroot/opt/rocm-7.2.0/include ROCM_HOME=/home/andryo/h27b-bench/rocmroot/opt/rocm ROCM_PATH=/home/andryo/h27b-bench/rocmroot/opt/rocm HCC_AMDGPU_TARGET=gfx1201 HIP_VISIBLE_DEVICES=1 ROCBLASLT_DIR=/home/andryo/h27b-bench/rocmbundle/rocblas/library LD_LIBRARY_PATH=/home/andryo/h27b-bench/rocmroot/usr/lib/x86_64-linux-gnu:/home/andryo/h27b-bench/rocmroot/opt/rocm-7.2.0/lib"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mem() { /usr/bin/free -g | awk "/^Mem:/{print \"hostmem avail=\" \$7 \"G\"}"; }
smi1() { rocm-smi --showuse --showmemuse 2>/dev/null | grep -E "GPU\[1\]" | tr "\t" " " >> "$LOG"; }
rm -f "$LOG" "$SMI"

exec 9>/home/andryo/gpu.lock
if ! flock -n 9; then log "LOCK-BUSY-ABORT"; exit 1; fi
log "=== T3317-$1-START $(date -u +%FT%TZ) pid=$$ (lock held)"
log "PRE-$(mem)"; smi1

sampler() {
  local cap="$1"
  for i in $(seq 1 "$cap"); do
    { date -u +%s; rocm-smi --showmemuse 2>/dev/null | grep "GPU\[1\]" | tr -s " \t" " "; } >> "$SMI"
    sleep 5
  done
}

sampler 500 & spid=$!
$ENV T3298_PREWARM=1 T3298_ROUNDS=$2 T3298_CLEAR_BEFORE_DECODE=1 T3298_CHUNK_SIZE=256 \
  MLX_MTP=4 MTP6_PREFILL_FILL=1 MTP_4BIT=1 MTP_MAX_TOKENS=256 \
  MTP6_DRAFT_HEAD=int2 MTP6_HEAD_H=postnorm MLX_MMQ_CFG=1 \
  MTP_MODEL="$M" MTP6_PROMPT="$(cat "$P4")" \
  TREE_Q5="$B/qwen3_5_tree.py" BENCH_PY="$D/t3298-k2arm-gate.py" \
  "$PY" "$B/wrap_mtp27b.py" > "$LOG.run" 2>&1
rc=$?
kill $spid 2>/dev/null; wait $spid 2>/dev/null
log "GATE-$1 rc=$rc"
grep -E "T3298-|INJECT_OK|SIDECAR|PREFILL_FILL|ERROR|Error|Traceback|HBM full|Unable to allocate" "$LOG.run" | tail -40 | while read -r l; do log "  $l"; done
log "PCRCOUNT=$(grep -c "\[pcr\]" "$LOG.run" 2>/dev/null)"
log "CFGQ4KCOUNT=$(grep -c "\[cfgq4k\]" "$LOG.run" 2>/dev/null)"
smi1
flock -u 9
log "=== T3317-$1-DONE $(date -u +%FT%TZ) (lock released)"
