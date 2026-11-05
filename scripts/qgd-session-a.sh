#!/bin/bash
# t2040 qgd-gen2 session A: build the qgd-gen2 wheel ON the neon box into a
# FRESH venv (venv-qgd = copy of venv-gen2 + force-reinstall), then run the
# 5-arm per-class bandwidth probe (C stock per-row baseline + QGD V1..V4).
# All 13 config-of-record envs ON in every arm; the variant under test is the
# extra env (MLX_ROCM_QGD_V1..V4, MLX_ROCM_QMM_M3=1 in every arm so the mlp
# class stays on the adopted mr3 kernel). flock, 25 s idle before each run,
# per-run timeout 900 s, session deadline 90 min.
#
# The variant kernels are the gen2 memory-structure changes ported onto the
# per-row (M = 1) qmv_wide kernel (the qmv_gdn class - GDN 4 in-proj +
# out_proj QMMs, 24 layers, M = 1; 4.59 ms at 36.3% of the 546 GB/s floor).
# All DEFAULT OFF: with all four envs unset the stock per-row path runs
# verbatim.
#
# vendor-qgd = copy of vendor-gen2 (the tree that built venv-gen2) with the
# three qgd files overwritten (qmm.hip carries the QGD dispatch; the mr3 gen2
# kernel is byte-synced from the branch; the qgd kernel file is new).
set -u
exec 9>/home/andryo/gpu.lock
DEADLINE=$(( $(date +%s) + 5400 ))
echo "=== FUSER-PRE $(date -u +%H:%M:%SZ) ===" | tee /home/andryo/qgd/session-a.log
fuser -v /home/andryo/gpu.lock >> /home/andryo/qgd/session-a.log 2>&1 || true
while true; do
  if flock -n 9; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "PENDING-LOCK: no lock within 90m" >> /home/andryo/qgd/session-a.log; exit 3; fi
  sleep 15
done
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ===" | tee -a /home/andryo/qgd/session-a.log
export MODELS=/mnt/models
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
export HCC_AMDGPU_TARGET=gfx1201
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib
export CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0"
D=/home/andryo/qgd
L=$D/session-a.log
mkdir -p $D
cd $D
echo "=== QGD SESSION A START $(date -u) ===" | tee -a $L
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
echo "qgd-kernel-md5: $(md5sum /home/andryo/qgd/qmv_wide_qgd_gen2_kernel.hip)"
echo "qgd-qmm-md5: $(md5sum /home/andryo/qgd/qmm.hip)"
echo "libmlx-gen2 (venv-gen2, source of the copy): $(md5sum /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-m3k-pre: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-pre: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
} | tee -a $L
# vendor-qgd = vendor-gen2 + the three qgd files from this branch.
rm -rf /home/andryo/vendor-qgd
cp -r /home/andryo/vendor-gen2 /home/andryo/vendor-qgd
cp /home/andryo/qgd/qmm.hip /home/andryo/vendor-qgd/mlx/backend/rocm/quantized/qmm.hip
cp /home/andryo/qgd/qmv_wide_qgd_gen2_kernel.hip /home/andryo/vendor-qgd/mlx/backend/rocm/quantized/qmv_wide_qgd_gen2_kernel.hip
cp /home/andryo/qgd/qmv_wide_mr3_gen2_kernel.hip /home/andryo/vendor-qgd/mlx/backend/rocm/quantized/qmv_wide_mr3_gen2_kernel.hip
echo "vendor-qgd-ready" | tee -a $L
cd /home/andryo/vendor-qgd
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
# Fresh venv from venv-gen2 (the 13-var config-of-record venv + stock mr3 +
# the four mlp gen2 variants). venv-gen2 / venv-m3k / venv-p3 are never
# modified (md5-of-record checked pre/post).
rm -rf /home/andryo/venv-qgd
cp -r /home/andryo/venv-gen2 /home/andryo/venv-qgd
echo "VENV-QGD-COPIED" | tee -a $L
echo "libmlx-qgd-copy (must equal venv-gen2 md5): $(md5sum /home/andryo/venv-qgd/lib/python3.14/site-packages/mlx/lib/libmlx.so)" | tee -a $L
rm -rf build
rm -f /home/andryo/wheelout-qgd/*.whl
mkdir -p /home/andryo/wheelout-qgd
/usr/bin/python3.14 -m pip wheel . --no-deps -w /home/andryo/wheelout-qgd/ > $D/build.log 2>&1
echo "BUILD=$?" | tee -a $L
whl=$(ls -t /home/andryo/wheelout-qgd/*.whl 2>/dev/null | head -1)
echo "WHL=$whl" | tee -a $L
if [ -z "$whl" ]; then echo "BUILD-FAIL" | tee -a $L; tail -30 $D/build.log | tee -a $L; exit 2; fi
# Pip gotcha (proven t2031): a plain pip install of a cached wheel does NOT
# replace the package - force-reinstall, then VERIFY the libmlx md5 changed.
rm -rf /home/andryo/venv-qgd/lib/python3.14/site-packages/mlx /home/andryo/venv-qgd/lib/python3.14/site-packages/bin
/usr/bin/python3.14 -m pip install --no-index --no-deps --upgrade --force-reinstall --target /home/andryo/venv-qgd/lib/python3.14/site-packages "$whl" > $D/install.log 2>&1
echo "INSTALL=$?" | tee -a $L
/home/andryo/venv-qgd/bin/python -c "import mlx.core as mx; print('IMPORT-OK', mx.__version__)" >> $L 2>&1
{
echo "libmlx-qgd-post (must differ from the copy md5 above): $(md5sum /home/andryo/venv-qgd/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "qgd-env-strings: $(for v in V1 V2 V3 V4; do /usr/bin/strings /home/andryo/venv-qgd/lib/python3.14/site-packages/mlx/lib/libmlx.so | grep -c "MLX_ROCM_QGD_$v"; done | tr '\n' ' ')"
} | tee -a $L
PY=/home/andryo/venv-qgd/bin/python
CE="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0"
probe() { # $1 = name, $2 = extra env
  sleep 25
  echo "=== PROBE $1 $(date -u +%H:%M:%SZ) sclk=[$(sclk)] ===" | tee -a $L
  timeout 900 env $CE $2 MTP_MODEL=/mnt/models/qwen3-9b-mlx $PY /home/andryo/m3k/verify-bw-perop-m3k.py > "$D/raw-probe-$1.raw" 2>&1
  echo "PROBE-$1-rc=$?" | tee -a $L
  grep -E "^(  (mlp|qmv_gdn|gdn_fused|attn|norms|lm_head) |LOAD_S|CTX=|FUSED_VERIFY_MS|TOTAL bytes)" "$D/raw-probe-$1.raw" >> $L 2>&1
  sleep 25
}
probe "C"   "MLX_ROCM_QMM_M3=1"
probe "Q1"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_V1=1"
probe "Q2"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_V2=1"
probe "Q3"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_V3=1"
probe "Q4"  "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_V4=1"
{
echo "sbcar-md5-post: $(md5sum /mnt/models/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-gen2-post: $(md5sum /home/andryo/venv-gen2/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-m3k-post: $(md5sum /home/andryo/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-post: $(md5sum /home/andryo/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
} | tee -a $L
echo "=== QGD SESSION A DONE $(date -u) ===" | tee -a $L
echo DONE > $D/.done-a
