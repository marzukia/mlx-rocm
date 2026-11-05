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
# fix (t2059): the t2048 find matched the doc dir /usr/share/doc/libhsa-runtime64-1
# (broken LD_PRELOAD); match real .so.1 files, system lib dir first.
HSA=$(find /usr/lib/x86_64-linux-gnu /opt/rocm*/lib -maxdepth 3 -name "libhsa-runtime64.so.1*" -type f 2>/dev/null | head -1)
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
# fix (t2059): ${VENV%/activate} left /bin in the path (LIB empty, python path
# bin/bin/python); strip the component properly.
VENV=${VENV%/activate}
VENV=${VENV%/bin}
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
# fix (t2059): this rocminfo prints "L2: 8192 KB" (not "L2 Cache Size"); the
# old pattern caught the CPU's "Compute Unit" lines instead.
"$ROCMINFO" 2>/dev/null | grep -E "CU Count|Wavefront|L2:|LDS" | head -20 | sed 's/^/HW /'

# fix (t2059): rocm-smi is not on PATH in the login shell; resolve it
RSMI=$(which rocm-smi 2>/dev/null || ls /opt/rocm*/bin/rocm-smi 2>/dev/null | head -1)
echo "RSMI=$RSMI"
# --- SCLK sampler (background)
( while :; do "$RSMI" --showclocks 2>/dev/null | grep -E "SCLK|GPU" | head -3 | tr '\n' ' '; echo " $(date +%H:%M:%S)"; sleep 10; done ) > "$D/sclk.log" 2>&1 &
SAMP=$!

# --- floor microbench (wall-clock primary; rocprof kernel-time cross-check)
echo "=== FLOOR (wall) ==="
sleep 25
# fix (t2059): if the 6 GiB pair cannot be allocated (co-resident VRAM),
# retry once at 4 GiB (the floor is size-invariant for a whole-buffer read)
FLNBYTES=$((6 * 1024 * 1024 * 1024))
"$VENV/bin/python" "$D/ceiling-floor.py" || {
  echo "FLOOR_RETRY nbytes=4GiB"
  sleep 25
  FLNBYTES=$((4 * 1024 * 1024 * 1024))
  FLOOR_NBYTES=$FLNBYTES "$VENV/bin/python" "$D/ceiling-floor.py"
}

ROC=$(which rocprofv3 2>/dev/null || which rocprofv2 2>/dev/null || which rocprof 2>/dev/null || ls /opt/rocm*/bin/rocprofv3 2>/dev/null | head -1)
if [ -n "$ROC" ]; then
  echo "=== FLOOR (rocprof kernel-time cross-check) ==="
  sleep 25
  FLOOR_NBYTES=$FLNBYTES "$ROC" -d "$D/floor-prof" -f json --kernel-trace -- "$VENV/bin/python" "$D/ceiling-floor.py" >/dev/null 2>&1
  FLNBYTES=$FLNBYTES python3 - "$D" <<'EOF'
import json, glob, os, sys
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
    gb = int(FLNBYTES)
    print(f"ROCPF_READ_GBPS={gb / (med * 1e-6) / 1e9:.1f} "
          f"(median sum-kernel {med / 1e6:.3f} ms, n={len(times)})")
else:
    print("ROCPF: no whole-buffer reduction found in trace")
EOF
fi

# --- 3-arm mlp + context classes
echo "=== MLP-3ARM ==="
sleep 25
# t2059: marker for the post-run .hsaco re-dump (the COR wheel is 100%
# hipRTC-JIT; the GEMM kernels' code objects land in /tmp/mlx/<ver>/hsaco/
# on first use, i.e. during this run)
date -u > "$D/mlpmark"
# fix (t2059): mlx_lm.load takes the model DIRECTORY (the in-lane probe loads
# $MODELS/qwen3-9b-mlx the same way); the .safetensors file is the md5 gate only
CEIL_MODEL="$MODELS/qwen3-9b-mlx" "$VENV/bin/python" "$D/ceiling-mlp-3arm.py"

# --- kernel resource re-dump from the JIT cache (after the mlp run created
# the GEMM kernels' .hsaco files)
echo "=== KDR (hsaco cache re-dump) ==="
bash "$D/ceiling-kdredump.sh" "$D/mlpmark" 2>&1 | sed 's/^/KDR /'

kill $SAMP 2>/dev/null
echo "SAMPLER_STOPPED"

# --- post md5 gate (sidecar untouched)
md5sum "$MODEL" | sed 's/^/MD5_MODEL_POST /'

flock -u 9 2>/dev/null
echo "LOCK_RELEASED"
echo "SESSION_DONE $(date -u +%FT%TZ)"
