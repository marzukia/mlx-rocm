#!/bin/bash
# ceiling-report on-box session (guest-a, RX 9070 XT / gfx1201).
# Runs under ssh; the caller must have scp'd the three ceiling-*.py + this
# script + ceiling-kdredump.sh to $CEIL_DIR.
set -uo pipefail

D=${CEIL_DIR:-$HOME/ceil-report}
mkdir -p "$D"
L=$D/session.log
exec > >(tee "$L") 2>&1
# the in-lane reports' box env (mlp-gen2/qgd-gen2 protocol line)
export HCC_AMDGPU_TARGET=gfx1201
HSA=$(find / -maxdepth 4 -name "libhsa-runtime*" 2>/dev/null | head -1)
[ -n "$HSA" ] && export LD_PRELOAD=$HSA && echo "LD_PRELOAD=$HSA"
echo "SESSION_START $(date -u +%FT%TZ)"

# --- bounded lock retry: 20 x 30 s
exec 9>~/gpu.lock
LOCKED=0
for i in $(seq 1 20); do
  if flock -n 9; then echo "LOCK_GOT attempt=$i"; LOCKED=1; break; fi
  echo "LOCK_WAIT attempt=$i $(fuser ~/gpu.lock 2>/dev/null)"
  sleep 30
done
[ "$LOCKED" = 1 ] || { echo "LOCK_FAIL after 20x30s"; exit 1; }

# --- environment discovery
VENV=$(find ~ -maxdepth 3 -name "activate" -path "*venv-m3k*" 2>/dev/null | head -1)
VENV=${VENV%/activate}
MODEL=${MODELS:-$HOME}/model-mtp.safetensors
if [ ! -f "$MODEL" ]; then MODEL=$(find / -maxdepth 5 -name "model-mtp.safetensors" 2>/dev/null | head -1); fi
PROBE=$(find ~ -maxdepth 4 -name "verify-bw-perop-m3k.py" 2>/dev/null | head -1)
LIB=$(ls "$VENV"/lib/python*/site-packages/mlx/lib/libmlx.so 2>/dev/null | head -1)
echo "VENV=$VENV"
echo "MODEL=$MODEL"
echo "PROBE=$PROBE"
echo "LIB=$LIB"
[ -n "$VENV" ] && [ -n "$MODEL" ] || { echo "ENV_MISSING"; exit 1; }

# --- md5 gates (COR records)
md5sum "$MODEL" | sed 's/^/MD5_MODEL /'
md5sum "$LIB" | sed 's/^/MD5_LIB /'
[ -n "$PROBE" ] && md5sum "$PROBE" | sed 's/^/MD5_PROBE /'

# --- hardware record (for the occupancy arithmetic)
ROCMINFO=$(find / -maxdepth 6 -name rocminfo -type f 2>/dev/null | head -1)
echo "ROCMINFO=$ROCMINFO"
"$ROCMINFO" 2>/dev/null | grep -E "CU Count|Wavefront|L2 Cache Size|LDS|VGPR|SGPR|Maximum Threads|Compute Unit" | head -20 | sed 's/^/HW /'

# --- kernel resource re-dump (citable record from the COR wheel)
bash "$D/ceiling-kdredump.sh" "$LIB" 2>&1 | sed 's/^/KDR /'

# --- SCLK sampler (background)
( while :; do rocm-smi --showclocks 2>/dev/null | grep -E "SCLK|GPU" | head -3 | tr '\n' ' '; echo " $(date +%H:%M:%S)"; sleep 10; done ) > "$D/sclk.log" 2>&1 &
SAMP=$!

# --- floor microbench (wall-clock primary; rocprof kernel-time cross-check)
echo "=== FLOOR (wall) ==="
sleep 25
"$VENV/bin/python" "$D/ceiling-floor.py"

ROC=$(which rocprofv3 2>/dev/null || ls /opt/rocm*/bin/rocprofv3 2>/dev/null | head -1)
if [ -n "$ROC" ]; then
  echo "=== FLOOR (rocprof kernel-time cross-check) ==="
  sleep 25
  "$ROC" -d "$D/floor-prof" -f json --kernel-trace -- "$VENV/bin/python" "$D/ceiling-floor.py" >/dev/null 2>&1
  python3 - "$D" <<'EOF'
import json, glob, sys
d = sys.argv[1]
times = []
for f in glob.glob(d + "/*.json"):
    try:
        data = json.load(open(f))
    except Exception:
        continue
    entries = data.get("kernel_trace", []) if isinstance(data, dict) else data
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = (e.get("kernel_name") or e.get("name") or "").lower()
        if "reduce" in name or "sum" in name:
            t = e.get("end_time", 0) - e.get("start_time", 0)
            if t > 1000:  # the heavy whole-buffer reduction, ns
                times.append(t)
times.sort()
if times:
    # skip the first (warmup-adjacent) if we have >= 2
    med = times[len(times) // 2]
    gb = 6 * 1024 ** 3
    print(f"ROCPF_READ_GBPS={gb / (med * 1e-6) / 1e9:.1f} "
          f"(median sum-kernel {med / 1e6:.3f} ms, n={len(times)})")
else:
    print("ROCPF: no whole-buffer reduction found in trace")
EOF
fi

# --- 3-arm mlp + context classes
echo "=== MLP-3ARM ==="
sleep 25
CEIL_MODEL="$MODEL" "$VENV/bin/python" "$D/ceiling-mlp-3arm.py"

kill $SAMP 2>/dev/null
echo "SAMPLER_STOPPED"

flock -u 9 2>/dev/null
echo "LOCK_RELEASED"
echo "SESSION_DONE $(date -u +%FT%TZ)"
