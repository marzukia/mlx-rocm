#!/bin/bash
# S4(1) ONE lock-holding GPU session: sampler validation (known-nonzero
# baseline) -> sweep arm C -> sweep arm B -> AR arm C -> AR arm B.
# Arm C = wheel default (mr_gate=1 -> M=1 routes to mmq, the leg-4 wheel);
# arm B = MLX_KQUANT_FILE_MR=0 (kqfg single-row). No lib swap; the venv
# SoT is untouched (lib md5 ledgered pre/post).
set -uo pipefail
R=/home/andryo/s41
mkdir -p "$R" /var/tmp/s41
S=$R/session.log
PY=/home/andryo/venv-ar/bin/python
SMI=/opt/rocm/bin/rocm-smi
LIB=/home/andryo/venv-ar/lib/python3.14/site-packages/mlx/lib/libmlx.so
D=$R  # scripts live alongside the results
ENVBASE="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo TMPDIR=/var/tmp/s41 HCC_AMDGPU_TARGET=gfx1201 HIP_VISIBLE_DEVICES=0"
# memory-255 SDK-injected rocprof (1.1.0 build; the 1.3.2 tool of record is
# NO-TRACE on this box): per-kernel trace CSV at process exit, written to
# /home/andryo/marzuki-neon/<pid>_kernel_trace.csv
PROF_ENV="LD_LIBRARY_PATH=/home/andryo/rcp/new724/tree/opt/rocm-7.2.4/lib:/home/andryo/t2246/libs LD_PRELOAD=/home/andryo/rcp/new724/tree/opt/rocm-7.2.4/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so ROCPROFILER_LIBRARY_CTOR=1 ROCPROF_KERNEL_TRACE=1 ROCPROF_OUTPUT_FORMAT=csv"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$S"; }
vram_b() { "$SMI" --showmeminfo vram 2>/dev/null | grep "Used Memory (B)" | sed -n 's/.*Used Memory (B): \([0-9]*\).*/\1/p' | head -1; }
state() {
  log "STATE kfd=[$(fuser /dev/kfd 2>/dev/null | tr -d ' ')] vram_B=$(vram_b) lockpid=[$(fuser /home/andryo/gpu.lock 2>/dev/null | tr -d ' ')]"
}
clocks() {
  local TAG=$1
  "$SMI" --showclocks 2>/dev/null | grep -E "GPU\[0\]|low-power" > "$R/clocks-$TAG.txt" || true
  log "CLOCKS-$TAG $(tr '\n' '|' < "$R/clocks-$TAG.txt")"
}
runpy() {  # runpy <tag> <script> <mr-env:0|1> [extra-args...]
  local TAG=$1 SC=$2 MR=$3; shift 3
  log "RUN-START $TAG $* $(date -u +%FT%TZ)"
  ( sleep 30; clocks "$TAG-mid" ) &
  local MID=$!
  if [ "$MR" = "0" ]; then
    timeout -k 60 1800 $ENVBASE env $PROF_ENV env MLX_KQUANT_FILE_MR=0 $PY "$D/$SC" --tag "$TAG" "$@" > "$R/$TAG.out" 2> "$R/$TAG.err"
  else
    timeout -k 60 1800 $ENVBASE env $PROF_ENV $PY "$D/$SC" --tag "$TAG" "$@" > "$R/$TAG.out" 2> "$R/$TAG.err"
  fi
  local rc=$?
  kill "$MID" 2>/dev/null
  log "RUN-DONE $TAG rc=$rc $(date -u +%FT%TZ)"
  # collect the per-pid trace CSV (written at process exit, CWD-relative
  # marzuki-neon/ dir, or $HOME/marzuki-neon/)
  local PID TP
  PID=$(grep -a "pid=" "$R/$TAG.out" 2>/dev/null | head -1 | grep -oE "pid=[0-9]+" | cut -d= -f2)
  for TP in "$R/marzuki-neon/${PID}_kernel_trace.csv" "/home/andryo/marzuki-neon/${PID}_kernel_trace.csv"; do
    if [ -n "$PID" ] && [ -f "$TP" ]; then
      mv "$TP" "$R/trace-$TAG.csv"
      log "TRACE-$TAG rows=$(wc -l < "$R/trace-$TAG.csv")"
      break
    fi
  done
  if [ ! -f "$R/trace-$TAG.csv" ]; then
    log "TRACE-$TAG MISSING pid=$PID"
  fi
  grep -aE "cfgq2k|Error|error|fault" "$R/$TAG.out" "$R/$TAG.err" 2>/dev/null | tail -6 | tee -a "$S"
  log "SSPOOL-WARN-$TAG count=$(grep -ac SharedSignalPool "$R/$TAG.err" 2>/dev/null || true)"
  state
}

exec 9>/home/andryo/gpu.lock
flock -w 10800 9 || { log "S41-LOCK-TIMEOUT"; exit 1; }
log "S41-START $(date -u +%FT%TZ) pid=$$ (lock held)"
state
log "PRE-LIB-MD5 $(md5sum "$LIB" | cut -d' ' -f1)"
log "PRE-WHEEL $($ENVBASE $PY -c 'import mlx.core as mx; print(mx.__version__)' 2>&1 | grep -v AMDCUID | tail -1)"
clocks pre
# pre-state: the vehicle file
md5sum /home/andryo/models/Qwen3.8-27B-Q2_K-plain.gguf > "$R/gguf.md5" 2>&1 || true
sha256sum /home/andryo/models/Qwen3.8-27B-Q2_K-plain.gguf >> "$R/gguf.md5" 2>/dev/null || true
cat "$R/gguf.md5" | tee -a "$S"

# --- sampler validation: known-nonzero baseline (the 11-27 grep-bug fix:
# the VRAM number is extracted after "Memory (B):"; validate it reads
# nonzero against a known ~512MB GPU + ~384MB host allocation)
log "SVAL-START"
echo "python.*s41-samval.py" > "$R/pat-val.txt"
cat > "$R/s41-samval.py" <<'PYEOF'
import time
import mlx.core as mx
x = mx.zeros((16384, 16384))          # 512MB bf16 on GPU
y = bytearray(384 * 1024 * 1024)      # 384MB host
mx.eval(x)
print("SVAL-ALIVE pid-ok", flush=True)
time.sleep(45)
PYEOF
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo TMPDIR=/var/tmp/s41 HCC_AMDGPU_TARGET=gfx1201 $PY "$R/s41-samval.py" > "$R/samval.out" 2> "$R/samval.err" &
VPID=$!
sleep 6
bash "$D/s41-hostsample.sh" "$R/pat-val.txt" "$R/samp-val.log" 9 &
sleep 40
wait "$VPID" 2>/dev/null
log "SVAL-RESULT $(awk '{split($2,a,"="); if(a[1]=="vRSS_KB" && a[2]+0>m) m=a[2]} {split($5,b,"="); if(b[1]=="vram_B" && b[2]+0>M) M=b[2]} END{print "max_vRSS_KB=" m, "max_vram_B=" M}' "$R/samp-val.log")"
# gate: vRSS > 300MB (300000 KB) and vram > 400MB (400000000 B)
awk '{split($2,a,"="); if(a[1]=="vRSS_KB" && a[2]+0>m) m=a[2]} {split($5,b,"="); if(b[1]=="vram_B" && b[2]+0>M) M=b[2]} END{exit !(m>300000 && M>400000000)}' "$R/samp-val.log" \
  && log "SVAL-PASS" || log "SVAL-FAIL"
state

# --- step 1: pre-loaded sweep, arm C then arm B (same session, same
# weights file; per-arm pre-load = one file read + one H2D pass each)
clocks pre-sweep
runpy sweep-C s41-sweep.py 1
runpy sweep-B s41-sweep.py 0

# --- step 2: 64-token AR, arm C then arm B, host-sampled at 3s cadence
clocks pre-ar
log "RUN-START ar-C $(date -u +%FT%TZ)"
echo "python.*s41-ar.py --tag ar-C" > "$R/pat-C.txt"
bash "$D/s41-hostsample.sh" "$R/pat-C.txt" "$R/samp-C.log" 80 &
SPC=$!
( sleep 30; clocks "ar-C-mid" ) &
MID=$!
timeout -k 60 1800 $ENVBASE env $PROF_ENV $PY "$D/s41-ar.py" --tag ar-C > "$R/ar-C.out" 2> "$R/ar-C.err"
echo "ar-C rc=$?" >> "$R/session.log"
kill "$MID" "$SPC" 2>/dev/null
log "RUN-DONE ar-C $(date -u +%FT%TZ)"
PID=$(grep -a "pid=" "$R/ar-C.out" 2>/dev/null | head -1 | grep -oE "pid=[0-9]+" | cut -d= -f2)
for TP in "$R/marzuki-neon/${PID}_kernel_trace.csv" "/home/andryo/marzuki-neon/${PID}_kernel_trace.csv"; do
  if [ -n "$PID" ] && [ -f "$TP" ]; then
    mv "$TP" "$R/trace-ar-C.csv"
    log "TRACE-ar-C rows=$(wc -l < "$R/trace-ar-C.csv")"
    break
  fi
done
[ -f "$R/trace-ar-C.csv" ] || log "TRACE-ar-C MISSING pid=$PID"
grep -aE "cfgq2k|Error|error|fault" "$R/ar-C.out" "$R/ar-C.err" 2>/dev/null | tail -6 | tee -a "$S"
log "SSPOOL-WARN-ar-C count=$(grep -ac SharedSignalPool "$R/ar-C.err" 2>/dev/null || true)"
state
clocks post-ar-C

log "RUN-START ar-B $(date -u +%FT%TZ)"
echo "python.*s41-ar.py --tag ar-B" > "$R/pat-B.txt"
bash "$D/s41-hostsample.sh" "$R/pat-B.txt" "$R/samp-B.log" 80 &
SPB=$!
( sleep 30; clocks "ar-B-mid" ) &
MID=$!
timeout -k 60 1800 $ENVBASE env $PROF_ENV env MLX_KQUANT_FILE_MR=0 $PY "$D/s41-ar.py" --tag ar-B > "$R/ar-B.out" 2> "$R/ar-B.err"
echo "ar-B rc=$?" >> "$R/session.log"
kill "$MID" "$SPB" 2>/dev/null
log "RUN-DONE ar-B $(date -u +%FT%TZ)"
PID=$(grep -a "pid=" "$R/ar-B.out" 2>/dev/null | head -1 | grep -oE "pid=[0-9]+" | cut -d= -f2)
for TP in "$R/marzuki-neon/${PID}_kernel_trace.csv" "/home/andryo/marzuki-neon/${PID}_kernel_trace.csv"; do
  if [ -n "$PID" ] && [ -f "$TP" ]; then
    mv "$TP" "$R/trace-ar-B.csv"
    log "TRACE-ar-B rows=$(wc -l < "$R/trace-ar-B.csv")"
    break
  fi
done
[ -f "$R/trace-ar-B.csv" ] || log "TRACE-ar-B MISSING pid=$PID"
grep -aE "cfgq2k|Error|error|fault" "$R/ar-B.out" "$R/ar-B.err" 2>/dev/null | tail -6 | tee -a "$S"
log "SSPOOL-WARN-ar-B count=$(grep -ac SharedSignalPool "$R/ar-B.err" 2>/dev/null || true)"
state
clocks post

# --- post-state: venv SoT untouched
log "POST-LIB-MD5 $(md5sum "$LIB" | cut -d' ' -f1)"
log "POST-WHEEL $($ENVBASE $PY -c 'import mlx.core as mx; print(mx.__version__)' 2>&1 | grep -v AMDCUID | tail -1)"
state
log "S41-END $(date -u +%FT%TZ)"
flock -u 9
log "S41-LOCK-RELEASED"
