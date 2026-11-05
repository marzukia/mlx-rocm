#!/bin/bash
# t2031 gen2 session A: build the gen2 wheel ON guest-a into a FRESH venv,
# then run the 5-arm per-class bandwidth probe (C baseline + V1..V4).
# All 13 config-of-record envs ON in every arm; the variant under test is the
# extra env. flock, 25 s idle before each run, per-run timeout 600 s,
# session deadline 90 min.
#
# VENV: venv-gen2 = copy of venv-p3 (the 13-var config-of-record venv,
# GDN T3 / tape-1 / postnorm / draftgate / s4va / rows=1 / refwd_m1 / mr_lx=0,
# hybrid tape, MTP 3) with the gen2 wheel force-reinstalled on top. This is
# the exact m3k-protocol construction: venv-m3k is venv-p3 + the mr3 wheel;
# venv-gen2 is venv-p3 + the gen2 wheel (stock mr3 + the four variants).
# venv-m3k and venv-p3 are never modified (md5-of-record checked pre/post).
#
# The vendor tree at 7ea5bca is a PRUNED source export (no CMakeLists.txt /
# pyproject.toml / python/ scaffolding). The fork clone (the tree that built
# venv-m3k) carries the full scaffold; this script copies the scaffold
# files from it (only python/src/fast.cpp differs: the fork graft adds the
# GDN fused_step binding, which the venv needs).
set -u
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
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib
export CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0"
D=/home/andryo/gen2
L=$D/session-a.log
mkdir -p $D
cd $D
echo "=== GEN2 SESSION A START $(date -u) ===" | tee $L
# SCLK sampler (pp_dpm_sclk "*" line; fallback to rocm-smi). NOTE: the
# pp_dpm_sclk level index is NOT the MHz - the lm_head GB/s row of each
# probe is the real clock-drift control.
sclk() { m=$(grep "*" /sys/class/drm/card1/device/pp_dpm_sclk 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1); if [ -z "$m" ]; then m=$(timeout 15 rocm-smi --showclocks 2>/dev/null | grep -i "sclk\|GPU Clock" | head -1); fi; echo "$m"; }
( while true; do echo "$(date -u +%H:%M:%S) sclk=[$(sclk)]" >> $D/sclk-a.log; sleep 10; done ) &
SCLK_PID=$!
trap "kill $SCLK_PID 2>/dev/null" EXIT
{
echo "sbcar-md5-pre: $(md5sum /mnt/models/qwen3-9b-mlx/model-mtp.safetensors)"
echo "probe-md5: $(md5sum /home/andryo/m3k/verify-bw-perop-m3k.py)"
echo "bench-md5: $(md5sum /home/andryo/gen2/mtp-bench6.py)"
echo "libmlx-m3k-pre: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-pre: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "gen2-kernel-md5: $(md5sum /home/andryo/vendor-gen2/mlx/backend/rocm/quantized/qmv_wide_mr3_gen2_kernel.hip)"
echo "gen2-qmm-md5: $(md5sum /home/andryo/vendor-gen2/mlx/backend/rocm/quantized/qmm.hip)"
} | tee -a $L
cd /home/andryo/vendor-gen2
# Scaffold from the m3k fork clone (pruned vendor export has none).
for f in pyproject.toml setup.py CMakeLists.txt MANIFEST.in mlx.pc.in mlx.pc.in.in; do
  [ -e "$f" ] || cp /home/andryo/vendor-m3k/$f $f
done
for f in cmake python; do
  [ -e "$f" ] || cp -r /home/andryo/vendor-m3k/$f $f
done
# setup.py does `git rev-parse` -> the build tree must be a git repo.
git init -q 2>/dev/null || true
git add -A 2>/dev/null; git -c user.email=build@local -c user.name=build commit -qm build 2>/dev/null || true
# Fresh venv from venv-p3 (the 13-var config-of-record venv).
rm -rf /home/andryo/venv-gen2
cp -r /home/andryo/venv-p3 /home/andryo/venv-gen2
echo "VENV-GEN2-COPIED" | tee -a $L
rm -rf build
rm -f /home/andryo/wheelout-gen2/*.whl
/usr/bin/python3.14 -m pip wheel . --no-deps -w /home/andryo/wheelout-gen2/ > $D/build.log 2>&1
echo "BUILD=$?" | tee -a $L
whl=$(ls -t /home/andryo/wheelout-gen2/*.whl 2>/dev/null | head -1)
echo "WHL=$whl" | tee -a $L
if [ -z "$whl" ]; then echo "BUILD-FAIL"; tail -30 $D/build.log | tee -a $L; exit 2; fi
rm -rf /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx /home/andryo/venv-gen2/lib/python3.14/site-packages/bin
/usr/bin/python3.14 -m pip install --no-index --no-deps --upgrade --force-reinstall --target /home/andryo/venv-gen2/lib/python3.14/site-packages "$whl" > $D/install.log 2>&1
echo "INSTALL=$?" | tee -a $L
/home/andryo/venv-gen2/bin/python -c "import mlx.core as mx; print('IMPORT-OK', mx.__version__)" >> $L 2>&1
{
echo "libmlx-gen2: $(md5sum /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "gen2-env-strings: $(for v in V1 V2 V3 V4; do /usr/bin/strings /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx/lib/libmlx.so | grep -c "MLX_ROCM_QMM_M3_$v"; done | tr '\n' ' ')"
} | tee -a $L
PY=/home/andryo/venv-gen2/bin/python
CE="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0"
probe() { # $1 = name, $2 = extra env
  sleep 25
  echo "=== PROBE $1 $(date -u +%H:%M:%SZ) sclk=[$(sclk)] ===" | tee -a $L
  timeout 1200 env $CE $2 MTP_MODEL=/mnt/models/qwen3-9b-mlx $PY /home/andryo/m3k/verify-bw-perop-m3k.py > "$D/raw-probe-$1.raw" 2>&1
  echo "PROBE-$1-rc=$?" | tee -a $L
  grep -E "^(  (mlp|qmv_gdn|gdn_fused|attn|norms|lm_head) |LOAD_S|CTX=|FUSED_VERIFY_MS|TOTAL bytes)" "$D/raw-probe-$1.raw" >> $L 2>&1
  sleep 25
}
probe "C"   "MLX_ROCM_QMM_M3=1"
probe "V1"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QMM_M3_V1=1"
probe "V2"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QMM_M3_V2=1"
probe "V3"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QMM_M3_V3=1"
probe "V4"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QMM_M3_V4=1"
{
echo "sbcar-md5-post: $(md5sum /mnt/models/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-m3k-post: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-post: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
} | tee -a $L
echo "=== GEN2 SESSION A DONE $(date -u) ===" | tee -a $L
echo DONE > $D/.done-a
