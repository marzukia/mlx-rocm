#!/bin/bash
# t2230 bucket 1 - per-kernel profile of ONE AR decode step (non-GEMM breakout).
#
# Runs ON neon (the 0x7550 dGPU box), launched under flock on $LOCK:
#   flock /home/andryo/gpu.lock /home/andryo/t2230/nongemm_run.sh
#
# Arms (all under the lock, one holder):
#   SCLK0  - 25 s idle, busy-gated, idle SCLK baseline (telemetry caveat: idle)
#   C1..C3 - clean bench rounds, no profiler (step wall; median reported)
#   P1     - profiled run, MTP_MAX_TOKENS=1  (prefill + 1 AR step)
#   P2     - profiled run, MTP_MAX_TOKENS=322 (prefill + 322 AR steps)
#   SCLK1  - post SCLK (telemetry caveat: post-load state)
#
# Per-step kernel table = (P2 - P1) / 321 AR steps, parsed by nongemm_parse.py.
# (P1 carries prefill + 1 step; subtracting removes the prefill block, and
# the steady-state per-step average = (322 - 1) step delta.)
#
# Nothing here modifies the venv or the model; md5-verified pre (and the
# launcher verifies post).
set -u
T=${T2230_DIR:-/home/andryo/t2230}
LOCK=${T2230_LOCK:-/home/andryo/gpu.lock}
VENV=${T2230_VENV:-/home/andryo/venv-verify}
PY=$VENV/bin/python
BENCH=$T/mtp-bench6.py
LIBMLX=$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so
SOK=/mnt/models/qwen3-9b-mlx/model-mtp.safetensors
ROC=/opt/rocm/bin/rocprofv2
LOG=$T/nongemm.log

echo "t2230-run start $(date -u +%FT%TZ)"
echo "lock-holder-pre: $(fuser -v $LOCK 2>&1 | tail -n +2 | tr '\n' ' ')"
echo "holder-pid: $$"

# --- md5 gates (pre)
echo "--- md5 pre ---"
md5sum "$BENCH" "$LIBMLX" "$SOK" 2>&1 | tee -a "$LOG"
B=$(md5sum "$BENCH" | cut -d' ' -f1)
L=$(md5sum "$LIBMLX" | cut -d' ' -f1)
S=$(md5sum "$SOK" | cut -d' ' -f1)
[ "$B" = "54ce16ac29e6c6aad2c48a233ff70d85" ] || { echo "BENCH-MD5-FAIL $B"; exit 1; }
[ "$L" = "8acf13fec6fc1636026d8e1dc69cd497" ] || { echo "LIBMLX-MD5-FAIL $L"; exit 1; }
[ "$S" = "02a54359cda8b247f76b277dd9bc4d4c" ] || { echo "SIDE-MD5-FAIL $S"; exit 1; }

# --- PCI identity (work card = 0x7550, iGPU 0x13c0 never used)
for c in /sys/class/drm/card[0-9]; do
  n=$(basename "$c")
  d=$(cat "$c/device/device" 2>/dev/null || echo MISSING)
  echo "card-identity $n $(cat "$c/device/vendor" 2>/dev/null || echo MISSING) $d" | tee -a "$LOG"
done
DID=$(cat /sys/class/drm/card0/device/device 2>/dev/null || echo MISSING)
[ "$DID" = "0x7550" ] || { echo "CARD-IDENTITY-FAIL card0=$DID"; exit 1; }

# --- SCLK sampler (dGPU sysfs = card0 = 0x7550, rocm-smi device 0)
sclk() { grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null | head -1 | tr -d ' '; }
busy() {
  # GPU% on the dGPU; >5 = HARD FAILURE (sysfs busy at sample time)
  local p
  p=$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)
  [ -n "$p" ] && { echo "$p"; return; }
  /opt/rocm/bin/rocm-smi --showuse 2>/dev/null | awk 'NR>=8 && /GPU/ {print $NF; exit}' | tr -d '%'
}
sclk_gate() { # $1 = tag
  local p
  for i in 1 2 3; do
    p=$(busy)
    [ -n "$p" ] || p=0
    [ "$p" -le 5 ] 2>/dev/null || { echo "SYSFS-BUSY-HARD-FAIL $p"; exit 1; }
    sleep 5
  done
  echo "sclk $1 $(date -u +%H:%M:%S) $(sclk)" | tee -a "$LOG"
}

# --- bench runner (env -i, box GPU env per brief)
bench() { # $1 tag  $2 maxtok
  env -i \
    MODELS=/mnt/models \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
    HCC_AMDGPU_TARGET=gfx1201 \
    MLX_MTP=0 \
    MLX_ROCM_QMV_ROWS=1 \
    MLX_ROCM_QMM_M3=1 \
    MTP6_HEAD_H=postnorm \
    MLX_ROCM_SDPA_S4VA=1 \
    MLX_ROCM_REFWD_M1=1 \
    MLX_QMV_MR_LX=0 \
    MTP_MAX_TOKENS=$2 \
    "$PY" "$BENCH" > "$T/$1.log" 2>&1
  echo "bench $1 rc=$? $(grep -o 'TOKS=[0-9]* TOKS_S=[0-9.]*' "$T/$1.log" | head -1)" | tee -a "$LOG"
}

# --- profiled bench runner (rocprofv2 kernel trace, json)
profbench() { # $1 tag  $2 maxtok
  rm -f "$T/$1.json"
  "$ROC" -i "$T/$1.json" -- \
    env -i \
      MODELS=/mnt/models \
      LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
      HCC_AMDGPU_TARGET=gfx1201 \
      MLX_MTP=0 \
      MLX_ROCM_QMV_ROWS=1 \
      MLX_ROCM_QMM_M3=1 \
      MTP6_HEAD_H=postnorm \
      MLX_ROCM_SDPA_S4VA=1 \
      MLX_ROCM_REFWD_M1=1 \
      MLX_QMV_MR_LX=0 \
      MTP_MAX_TOKENS=$2 \
      "$PY" "$BENCH" > "$T/$1.log" 2>&1
  echo "profbench $1 rc=$? $(grep -o 'TOKS=[0-9]* TOKS_S=[0-9.]*' "$T/$1.log" | head -1)" | tee -a "$LOG"
}

# --- rocprofv2 format probe (no GPU work: a plain pass; calibrates the parser)
echo "--- rocprofv2 format probe ---" | tee -a "$LOG"
rm -f "$T/fmt.json"
"$ROC" -i "$T/fmt.json" -- /usr/bin/true > "$T/fmt.log" 2>&1
python3 - "$T/fmt.json" >> "$LOG" 2>&1 <<'PYEOF'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as e:
    print("FMT-PROBE-FAIL", e)
    sys.exit(0)
def shape(o, depth=0):
    pad = "  " * depth
    if isinstance(o, dict):
        for k in list(o)[:12]:
            v = o[k]
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}" + (f"[{len(v)}]" if isinstance(v, list) else ""))
                if depth < 3 and isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"{pad}  [0] keys: {sorted(v[0].keys())[:16]}")
            elif isinstance(v, (list, dict)) and depth < 3:
                shape(v, depth + 1)
shape(d)
PYEOF

# --- SCLK baseline (25 s idle, busy-gated)
echo "sclk0: waiting 25 s idle" | tee -a "$LOG"
sleep 25
sclk_gate SCLK0

# --- clean rounds (no profiler)
for r in 1 2 3; do
  bench "clean$r" 322
  sclk_gate "SCLK-C$r"
done

# --- profiled runs
profbench "p1" 1
profbench "p2" 322

sclk_gate SCLK1

# --- md5 gates (post)
echo "--- md5 post ---"
md5sum "$BENCH" "$LIBMLX" "$SOK" 2>&1 | tee -a "$LOG"
echo "t2230-run end $(date -u +%FT%TZ) rc=0"
