#!/bin/bash
# t2050 qgd-ksplit session (runs ON the GPU box; ssh in as the box user).
#
# Phase A (no lock, CPU-only): vendor-ksplit prep (copy of the gen2 vendor
# tree + qmm.hip overwrite + the new ksplit kernel file), venv-ksplit (copy
# of venv-m3k + force-reinstall of the wheel), md5 + env-string + symbol
# verification, base-tree spot check (4 files vs the branch copies).
# Phase B (flock ~/gpu.lock, bounded retry 20x30s): 25 s idle + SCLK sample,
# bit-identity + correctness check (stock x2 + ksplit + 2x compare), the
# per-class probe (verify-bw-perop-m3k.py; C x3 / K x3, 13 config-of-record
# envs + MLX_ROCM_QMM_M3=1 on in every arm, 25 s idle before each run),
# then the full canonical bench (mtp-bench6.py, MTP_MAX_TOKENS=322 anchor,
# both prompts, 3 runs per arm per prompt, C/K interleaved per prompt,
# 25 s idle before each run). C arm = config-of-record + M3 (stock path);
# K arm = config-of-record + M3 + MLX_ROCM_QGD_KSPLIT=1.
#
# Never touched: venv-m3k, venv-p3, venv-r4v, the m3k probe script, the
# model (md5-of-record verified pre + post in every phase).
set -u
H=$HOME
MNT=/mnt
export MODELS="$MNT/models"
HSALIB=$(ldconfig -p 2>/dev/null | grep 'hsa-runtime64' | head -1 | awk '{print $NF}')
export LD_PRELOAD=$HSALIB
export HCC_AMDGPU_TARGET=gfx1201
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib
export CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0"
D=$H/qkd
L=$D/session.log
mkdir -p $D
echo "=== QKD SESSION START $(date -u) ===" | tee -a $L
sclk() { m=$(grep "*" /sys/class/drm/card1/device/pp_dpm_sclk 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1); if [ -z "$m" ]; then m=$(timeout 15 rocm-smi --showclocks 2>/dev/null | grep -i "sclk\|GPU Clock" | head -1); fi; echo "$m"; }
( while true; do echo "$(date -u +%H:%M:%S) sclk=[$(sclk)]" >> $D/sclk.log; sleep 10; done ) &
SCLK_PID=$!
trap "kill $SCLK_PID 2>/dev/null" EXIT

{
echo "sbcar-md5-pre: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-m3k-pre: $(md5sum $H/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-pre: $(md5sum $H/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-r4v-pre: $(md5sum $H/venv-r4v/lib/python3.14/site-packages/mlx/lib/libmlx.so 2>/dev/null)"
echo "probe-md5: $(md5sum $H/m3k/verify-bw-perop-m3k.py)"
echo "bench6-box-md5: $(md5sum $H/m3k/bench6.py 2>/dev/null)"
echo "bench6-uploaded-md5: $(md5sum $D/bench6.py)"
echo "bc-uploaded-md5: $(md5sum $D/qkd-ksplit-bitcheck.py)"
} | tee -a $L

# ---- Phase A (no lock) -------------------------------------------------
# vendor-ksplit = the gen2 vendor tree + qmm.hip (branch: gen2 dispatch +
# ksplit dispatch) + the new ksplit kernel file.
rm -rf $H/vendor-ksplit
cp -r $H/vendor-gen2 $H/vendor-ksplit
cp $D/qmm.hip $H/vendor-ksplit/mlx/backend/rocm/quantized/qmm.hip
cp $D/qmv_tiled_ksplit_kernel.hip $H/vendor-ksplit/mlx/backend/rocm/quantized/qmv_tiled_ksplit_kernel.hip
# Base-tree spot check: the 4 shared files must byte-match the branch.
for f in qmv_tiled_kernel.hip qdequant.hpp qmv_tiled_qgd_gen2_kernel.hip qmv_wide_mr3_gen2_kernel.hip; do
  a=$(md5sum $H/vendor-ksplit/mlx/backend/rocm/quantized/$f | awk '{print $1}')
  b=$(md5sum $D/verify/$f | awk '{print $1}')
  echo "treecheck $f: ${a} ${b} $([ "$a" = "$b" ] && echo PASS || echo FAIL)" | tee -a $L
done
cd $H/vendor-ksplit
for f in pyproject.toml setup.py CMakeLists.txt MANIFEST.in mlx.pc.in mlx.pc.in.in; do
  [ -e "$f" ] || cp $H/vendor-m3k/$f $f
done
for f in cmake python; do
  [ -e "$f" ] || cp -r $H/vendor-m3k/$f $f
done
git init -q 2>/dev/null || true
git add -A 2>/dev/null; git -c user.email=build@local -c user.name=build commit -qm build 2>/dev/null || true
# Fresh venv from venv-m3k (the canonical fold venv; never modified).
rm -rf $H/venv-ksplit
cp -r $H/venv-m3k $H/venv-ksplit
echo "venv-ksplit-copy: $(md5sum $H/venv-ksplit/lib/python3.14/site-packages/mlx/lib/libmlx.so)" | tee -a $L
rm -rf build
rm -f $H/wheelout-ksplit/*.whl
mkdir -p $H/wheelout-ksplit
timeout 3600 /usr/bin/python3.14 -m pip wheel . --no-deps -w $H/wheelout-ksplit/ > $D/build.log 2>&1
echo "BUILD=$?" | tee -a $L
whl=$(ls -t $H/wheelout-ksplit/*.whl 2>/dev/null | head -1)
echo "WHL=$whl" | tee -a $L
if [ -z "$whl" ]; then echo "BUILD-FAIL" | tee -a $L; tail -30 $D/build.log | tee -a $L; exit 2; fi
rm -rf $H/venv-ksplit/lib/python3.14/site-packages/mlx $H/venv-ksplit/lib/python3.14/site-packages/bin
/usr/bin/python3.14 -m pip install --no-index --no-deps --upgrade --force-reinstall --target $H/venv-ksplit/lib/python3.14/site-packages "$whl" > $D/install.log 2>&1
echo "INSTALL=$?" | tee -a $L
$H/venv-ksplit/bin/python -c "import mlx.core as mx; print('IMPORT-OK', mx.__version__)" >> $L 2>&1
{
echo "libmlx-ksplit-post: $(md5sum $H/venv-ksplit/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "ksplit-env-string: $(strings $H/venv-ksplit/lib/python3.14/site-packages/mlx/lib/libmlx.so | grep -c 'MLX_ROCM_QGD_KSPLIT')"
echo "ksplit-kernel-symbols: $(for v in qmv_tiled_ksplit_kernel qmv_tiled_ksplit_reduce_kernel; do strings $H/venv-ksplit/lib/python3.14/site-packages/mlx/lib/libmlx.so | grep -c "$v"; done | tr '\n' ' ')"
} | tee -a $L
PY=$H/venv-ksplit/bin/python

# ---- Phase B (flock) ---------------------------------------------------
exec 9>$H/gpu.lock
echo "=== FUSER-PRE $(date -u +%H:%M:%SZ) ===" | tee -a $L
fuser -v $H/gpu.lock >> $L 2>&1 || true
LOCKED=0
for i in $(seq 1 20); do
  if flock -n 9; then LOCKED=1; break; fi
  echo "lock-wait $i/20 $(date -u +%H:%M:%SZ)" | tee -a $L
  sleep 30
done
if [ "$LOCKED" != "1" ]; then
  echo "PENDING-LOCK: no lock within 20x30s" | tee -a $L
  exit 3
fi
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ===" | tee -a $L
sleep 25
echo "pre-bitcheck sclk=[$(sclk)]" | tee -a $L

# Bit-identity + correctness: stock x2 (bit-identical across processes =
# the unset-env verbatim guarantee), ksplit vs stock (close, not bit-
# identical by construction on the split shapes; the K = 1024 fallback
# shape must be bit-identical even with the env set).
CEBC="MTP_MODEL=$MODELS/qwen3-9b-mlx"
timeout 1800 env MODE=stock $CEBC $PY $D/qkd-ksplit-bitcheck.py > $D/bc-stock1.jsonl 2> $D/bc-stock1.err
echo "BC-STOCK1-rc=$? lines=$(grep -c '^{' $D/bc-stock1.jsonl)" | tee -a $L
timeout 1800 env MODE=stock $CEBC $PY $D/qkd-ksplit-bitcheck.py > $D/bc-stock2.jsonl 2> $D/bc-stock2.err
echo "BC-STOCK2-rc=$? lines=$(grep -c '^{' $D/bc-stock2.jsonl)" | tee -a $L
timeout 1800 env MODE=ksplit MLX_ROCM_QGD_KSPLIT=1 REF_FILE=$D/bc-stock1.jsonl $CEBC $PY $D/qkd-ksplit-bitcheck.py > $D/bc-ksplit.jsonl 2> $D/bc-ksplit.err
echo "BC-KSPLIT-rc=$? lines=$(grep -c '^{' $D/bc-ksplit.jsonl)" | tee -a $L
timeout 600 env MODE=compare REF_FILE=$D/bc-stock1.jsonl NEW_FILE=$D/bc-stock2.jsonl $PY $D/qkd-ksplit-bitcheck.py > $D/bc-cmp-same.log 2>&1
echo "BC-CMP-SAME-rc=$?" | tee -a $L
timeout 600 env MODE=compare REF_FILE=$D/bc-stock1.jsonl NEW_FILE=$D/bc-ksplit.jsonl $PY $D/qkd-ksplit-bitcheck.py > $D/bc-cmp-ksplit.log 2>&1
echo "BC-CMP-KSPLIT-rc=$?" | tee -a $L
grep -E '^(COMPARE|FALLBACK|OPS_DONE)' $D/bc-cmp-same.log >> $L 2>&1
grep -E '^(COMPARE|FALLBACK|OPS_DONE)' $D/bc-cmp-ksplit.log >> $L 2>&1
grep -E '^(OPS_DONE|WARN)' $D/bc-*.err >> $L 2>&1
grep '^OPS_DONE' $D/bc-*.jsonl >> $L 2>&1

# Per-class probe: C x3 / K x3 interleaved, 25 s idle before each.
CE="MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0"
probe() { # $1 = name, $2 = extra env
  sleep 25
  echo "=== PROBE $1 $(date -u +%H:%M:%SZ) sclk=[$(sclk)] ===" | tee -a $L
  timeout 900 env $CE $2 MTP_MODEL=$MODELS/qwen3-9b-mlx $PY $H/m3k/verify-bw-perop-m3k.py > "$D/raw-probe-$1.raw" 2>&1
  echo "PROBE-$1-rc=$?" | tee -a $L
  grep -E "^(  (mlp|qmv_gdn|gdn_fused|attn|norms|lm_head) |LOAD_S|CTX=|FUSED_VERIFY_MS|TOTAL bytes)" "$D/raw-probe-$1.raw" >> $L 2>&1
  sleep 25
}
for i in 1 2 3; do
  probe "C$i" "MLX_ROCM_QMM_M3=1"
  probe "K$i" "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_KSPLIT=1"
done

# Canonical bench: 322 anchor, both prompts, 3 runs per arm per prompt,
# C/K interleaved within each prompt, 25 s idle before each run.
bench() { # $1 = name, $2 = extra env, $3 = extra prompt env
  sleep 25
  echo "=== BENCH $1 $(date -u +%H:%M:%SZ) sclk=[$(sclk)] ===" | tee -a $L
  timeout 1200 env $CE $2 $3 MTP_MAX_TOKENS=322 MTP_MODEL=$MODELS/qwen3-9b-mlx $PY $D/bench6.py > "$D/raw-bench-$1.raw" 2>&1
  echo "BENCH-$1-rc=$?" | tee -a $L
  grep -E "^(TOKS|TOKS_S|VERIFIES|TOK_PER_STEP|FIRST20|LOAD_S)" "$D/raw-bench-$1.raw" >> $L 2>&1
  sleep 25
}
for i in 1 2 3; do
  bench "CC$i" "MLX_ROCM_QMM_M3=1" ""
  bench "CK$i" "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_KSPLIT=1" ""
done
for i in 1 2 3; do
  bench "HC$i" "MLX_ROCM_QMM_M3=1" "MTP6_PROMPT=Hello"
  bench "HK$i" "MLX_ROCM_QMM_M3=1 MLX_ROCM_QGD_KSPLIT=1" "MTP6_PROMPT=Hello"
done

{
echo "sbcar-md5-post: $(md5sum $MODELS/qwen3-9b-mlx/model-mtp.safetensors)"
echo "libmlx-m3k-post: $(md5sum $H/venv-m3k/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-p3-post: $(md5sum $H/venv-p3/lib/python3.14/site-packages/mlx/lib/libmlx.so)"
echo "libmlx-r4v-post: $(md5sum $H/venv-r4v/lib/python3.14/site-packages/mlx/lib/libmlx.so 2>/dev/null)"
} | tee -a $L
echo "=== QKD SESSION DONE $(date -u) ===" | tee -a $L
echo DONE > $D/.done
exit 0
