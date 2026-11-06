#!/bin/bash
# t2231 seam run on the box (re-dispatch of the cancelled instance).
# usage: seam_run_box.sh <ar|spec> <workdir> <venv-python> <rounds>
# t2246 deltas vs the t2231 version:
#  - LOCK is /home/andryo/gpu-bench.lock (the fleet lock; the t2231 version
#    used gpu.lock, which t2223/t2241 do not take -> co-tenancy risk).
#  - holder policy: record holder PIDs before each hold; all pids dead ->
#    BROKE-STALE-LOCK (flock releases on process death), proceed.
#  - sysfs busy (empty /sys/class/drm read) = HARD FAILURE, abort the arm.
#  - pre-calibration: one small rocprofv2 --kernel-trace run to verify the
#    marker kernel name before spending a full model-load round.
#  - md5 re-check of venv lib + bench + model sidecar around every arm.
set -uo pipefail
ARM="$1"; W="$2"; VPY="$3"; N="${4:-3}"
BENCH=<box-home>/m3k/bench/mtp-bench6.py
LOCK=<box-home>/gpu-bench.lock
SCLK=/opt/rocm-7.2.0/bin/rocm-smi
ROCP=/opt/rocm/bin/rocprofv2
VLIB="${VPY%/bin/python}/lib/python3.14/site-packages/mlx/lib/libmlx.so"

case "$ARM" in
  ar)   AENV="MLX_MTP=0 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMM_M3=1 MTP6_HEAD_H=postnorm MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0" ;;
  spec) AENV="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1" ;;
  *) echo "bad arm"; exit 1 ;;
esac

ENVX="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=<box-home> MODELS=<models-mount> HCC_AMDGPU_TARGET=gfx1201 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1"

sclk() { $SCLK --showclocks 2>/dev/null | grep -iE "sclk|pclk" | head -3; }

# sustained clock sampler: inside the lock hold, before the workload
sampler() { local f="$1"; for k in 1 2 3 4 5 6 7 8; do sleep 15; sclk >> "$f"; done; }

md5block() { # $1 label; prints md5 lines for bench/venv-lib/model sidecar
  echo "--- md5 $1 ---"
  md5sum "$BENCH" "$VLIB" <models-mount>/qwen3-9b-mlx/model-mtp.safetensors
  # card identity: card0 must be 0x7550 (work dGPU), card1 0x13c0 (iGPU, never used)
  for c in /sys/class/drm/card*/device; do
    echo "card-identity $(basename $(dirname $c)) $(cat $c/vendor 2>/dev/null | xargs) $(cat $c/device 2>/dev/null | xargs)"
  done
}

# holder policy: record holder PIDs before each hold; dead holder -> stale, log
# BROKE-STALE-LOCK and proceed (flock releases on process death).
holder_check() {
  local h dead p
  h=$(fuser "$LOCK" 2>/dev/null | tr -s ' ')
  echo "HOLDER: ${h:-none}"
  if [ -n "$h" ]; then
    dead=1
    for p in $h; do kill -0 "$p" 2>/dev/null && dead=0; done
    if [ "$dead" = "1" ]; then
      echo "BROKE-STALE-LOCK prev-holder=$h (all pids dead, lock free)"
    fi
  fi
}

# sysfs busy = HARD FAILURE: an exclusively-held card returns empty device reads.
sysfs_check() {
  local f v
  for f in /sys/class/drm/card*/device/device; do
    v=$(cat "$f" 2>/dev/null)
    if [ -z "$v" ]; then
      echo "SYSFS-BUSY-HARD-FAIL $?"
      exit 41
    fi
  done
}

cotentancy() {
  local s
  s=$(ps -eo pid,cmd | grep -E "mtp-bench|seam_step|tmax3r|nongemm" | grep -v grep | head -5)
  [ -n "$s" ] && echo "CO-TENANT: $s" || echo "CO-TENANT: none"
}

hold() { # $1 round-script path, $2 log path
  local t0 t1
  t0=$(date +%s)
  sysfs_check
  holder_check
  cotenancy
  flock "$LOCK" -c "bash $1" > "$2" 2>&1
  echo "exit=$? hold-wait-sec=$(( $(date +%s) - t0 ))"
}

md5block pre
# pre-calibration: verify the marker kernel name under this exact profiler/venv
# before spending a full model-load round on it.
echo "== CALIB $ARM $(date -u +%H:%M:%S)"
rm -rf "$W/calib_$ARM"
cat > "$W/calib_$ARM.sh" <<EOF
#!/bin/bash
$(ENVX) /opt/rocm/bin/rocprofv2 --kernel-trace -o $W/calib_$ARM "$VPY" <box-home>/seam-t2231-calib.py
EOF
chmod +x "$W/calib_$ARM.sh"
hold "$W/calib_$ARM.sh" "$W/calib_$ARM.log"
grep -acE "BOOT|M1|M2|M3" "$W/calib_$ARM.log" | sed 's/^/calib-marker-prints: /'
ls "$W/calib_$ARM" 2>/dev/null | head -5

echo "=== SCLK-IDLE pre $(date -u +%H:%M:%S)"
sleep 25; sclk

for i in $(seq 1 $N); do
  echo "== CLEAN $ARM round $i $(date -u +%H:%M:%S)"
  cat > "$W/round_$ARM_c$i.sh" <<EOF
#!/bin/bash
$(ENVX) $AENV MTP_MAX_TOKENS=322 "$VPY" $BENCH
EOF
  chmod +x "$W/round_$ARM_c$i.sh"
  ( sampler "$W/clean_$ARM_$i.sclk" ) &
  SPID=$!
  hold "$W/round_$ARM_c$i.sh" "$W/clean_$ARM_$i.log"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  grep -E "TOKS_S|INFO TAPEMODE" "$W/clean_$ARM_$i.log" | tail -3
done

for i in $(seq 1 $N); do
  echo "== PROF $ARM round $i $(date -u +%H:%M:%S)"
  rm -rf "$W/profout_$ARM_$i"
  cat > "$W/round_$ARM_p$i.sh" <<EOF
#!/bin/bash
$(ENVX) $AENV /opt/rocm/bin/rocprofv2 --kernel-trace -o $W/profout_$ARM_$i "$VPY" $W/seam_step.py
EOF
  chmod +x "$W/round_$ARM_p$i.sh"
  ( sampler "$W/prof_$ARM_$i.sclk" ) &
  SPID=$!
  hold "$W/round_$ARM_p$i.sh" "$W/prof_$ARM_$i.log"
  wait $SPID 2>/dev/null
  grep -E "STEP_KIND|STEP_WALL_MS|WARMUP_ITERS|GEN_MS" "$W/prof_$ARM_$i.log" | tail -5
done

md5block post
echo "=== SCLK-IDLE post $(date -u +%H:%M:%S)"
sleep 25; sclk
echo "=== DONE $ARM"
