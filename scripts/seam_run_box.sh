#!/bin/bash
# t2231 seam run on the box.
# usage: seam_run_box.sh <ar|spec> <workdir> <venv-python> <rounds>
set -uo pipefail
ARM="$1"; W="$2"; VPY="$3"; N="${4:-3}"
BENCH=<box-home>/m3k/bench/mtp-bench6.py
LOCK=<box-home>/gpu.lock
SCLK=/opt/rocm-7.2.0/bin/rocm-smi
ROCP=/opt/rocm/bin/rocprofv2

case "$ARM" in
  ar)   AENV="MLX_MTP=0 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMM_M3=1 MTP6_HEAD_H=postnorm MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0" ;;
  spec) AENV="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1" ;;
  *) echo "bad arm"; exit 1 ;;
esac

# base env prefix for GPU work (env -i per brief)
ENVX="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=<box-home> MODELS=<models-mount> HCC_AMDGPU_TARGET=gfx1201 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1"

sclk() { $SCLK --showclocks 2>/dev/null | grep -iE "sclk|pclk" | head -3; }

# sustained clock sampler: inside the lock hold, before the workload
sampler() { local f="$1"; for k in 1 2 3 4 5 6 7 8; do sleep 15; sclk >> "$f"; done; }

round_script() { # $1 = out script path, $2 = "clean"|"prof", $3 = round idx
  local out="$1" kind="$2" i="$3"
  if [ "$kind" = "clean" ]; then
    cat > "$out" <<EOF
#!/bin/bash
$(ENVX) $AENV MTP_MAX_TOKENS=322 "$VPY" $BENCH
EOF
  else
    cat > "$out" <<EOF
#!/bin/bash
$(ENVX) $AENV /opt/rocm/bin/rocprofv2 --kernel-trace -o $W/profout_$ARM_$i "$VPY" $W/seam_step.py
EOF
  fi
  chmod +x "$out"
}

echo "=== SCLK-IDLE pre $(date -u +%H:%M:%S)"
sleep 25; sclk

for i in $(seq 1 $N); do
  echo "== CLEAN $ARM round $i $(date -u +%H:%M:%S)"
  echo "HOLDER: $(fuser $LOCK 2>&1 | head -1 || true)"
  round_script "$W/round_$ARM_c$i.sh" clean "$i"
  ( sampler "$W/clean_$ARM_$i.sclk" ) &
  SPID=$!
  flock $LOCK -c "bash $W/round_$ARM_c$i.sh" > "$W/clean_$ARM_$i.log" 2>&1
  echo "exit=$?"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  grep -E "TOKS_S|INFO TAPEMODE" "$W/clean_$ARM_$i.log" | tail -3
done

# co-tenancy note: the pending prior-dispatch run (t2231 nongemm_run.sh) waits on
# gpu-bench.lock (held by the stuck tmax3r probe), so it cannot start inside our
# gpu.lock window; the stuck probe is host-bound (gpu_busy ~3% ambient).
for i in $(seq 1 $N); do
  echo "== PROF $ARM round $i $(date -u +%H:%M:%S)"
  echo "HOLDER: $(fuser $LOCK 2>&1 | head -1 || true)"
  rm -rf "$W/profout_$ARM_$i"
  round_script "$W/round_$ARM_p$i.sh" prof "$i"
  ( sampler "$W/prof_$ARM_$i.sclk" ) &
  SPID=$!
  flock $LOCK -c "bash $W/round_$ARM_p$i.sh" > "$W/prof_$ARM_$i.log" 2>&1
  echo "exit=$?"
  wait $SPID 2>/dev/null
  grep -E "STEP_KIND|STEP_WALL_MS|WARMUP_ITERS|GEN_MS" "$W/prof_$ARM_$i.log" | tail -5
done

echo "=== SCLK-IDLE post $(date -u +%H:%M:%S)"
sleep 25; sclk
echo "=== DONE $ARM"
