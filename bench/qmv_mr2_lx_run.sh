#!/bin/bash
# t889 (method-correction doc 2026-10-28, section 4 step 1, lever 1
# follow-up; REPORT.md section 6.4/7.6 pointer): t852 LX-hoist A/B gates.
#
# (a) LOSSLESS: MR+hoist kernel bit-exact vs the per-row qmv_wide_kernel
#     at identical inputs (maxdiff 0.0, raw-uint16 equality) — the
#     reference is generated in the SAME wheel with MLX_QMV_NO_MR=1
#     (per-row wide kernel), so both sides are the same code.
# (b) GAIN >= 3%: per-combo kernel-time gain, LX=1 (hoist) vs LX=0
#     (t825 path), one fresh process per env (t835 convention), identical
#     combo sequence, per-position pairing; the down shape (K=12288,
#     wide fallback for both envs) is the drift control.
# (c) CONFIG: printed + asserted per run (doc rule 6) — the CONFIG lines
#     carry the wheel version + MLX_* env; the parser prints the kernel
#     names actually dispatched; this runner asserts the expected kernel
#     family per shape + the template instantiation from the full symbol
#     name (mangled Lb0/Lb1 or demangled — non-vacuous on this
#     toolchain's mangled names; t903 MINOR-2 fix).
#
# Usage (ON marzuki-neon):
#   bash qmv_mr2_lx_run.sh <python>
#     python: interpreter with the wheel under test on its path
#            (venv-g4 python + PYTHONPATH=/mnt/models/wheel-qmv2)
set -uo pipefail
PY=$1
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
T=/mnt/models/tmp/qmv-mr2-lx
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=/mnt/models/gpu-bench.lock
WARMUP=10
REPS=50
mkdir -p $T

# Orphan-lock class (t852 incident): kill our children on any exit.
CHILDREN=""
trap 'for c in $CHILDREN; do kill $c 2>/dev/null; done; wait 2>/dev/null' EXIT INT TERM

echo "=== t889 ENV: py=$PY wheel=$($PY -c 'import mlx.core as mx; print(mx.__version__)') ==="
$PY -c "import mlx.core as mx; print('DEVICE:', mx.rocm.device_name() if hasattr(mx.rocm,'device_name') else 'n/a')" 2>/dev/null || true

# --- gate (a): lossless — bit-exact vs per-row wide kernel ---------------
flock -w 1200 $LOCK -c "MLX_QMV_NO_MR=1 $PY $HERE/qmv_mr_bitexact.py ref $T/bitexact" \
  > $T/bitexact-ref.log 2>&1 || { echo "FLOCK-FAIL bitexact-ref"; exit 1; }
flock -w 1200 $LOCK -c "MLX_QMV_MR_LX=1 $PY $HERE/qmv_mr_bitexact.py check $T/bitexact" \
  > $T/bitexact-check.log 2>&1 || { echo "FLOCK-FAIL bitexact-check"; exit 1; }
grep -E "^CONFIG:|^BITEXACT-GATE:" $T/bitexact-ref.log $T/bitexact-check.log
# gate (a) verdict: 16/16 combos (4 shapes x M=1..4) bit-exact, and the
# ref run actually ran with NO_MR=1 (per-row wide kernel reference).
NPASS=$(grep -c "maxdiff=0.0 bitexact=True PASS" $T/bitexact-check.log)
[ "$NPASS" = "16" ] || { echo "BITEXACT-GATE-FAIL: $NPASS/16"; exit 1; }
grep -q "'MLX_QMV_NO_MR': '1'" $T/bitexact-ref.log || { echo "CONFIG-ASSERT-FAIL: ref env"; exit 1; }
grep -q "'MLX_QMV_MR_LX': '1'" $T/bitexact-check.log || { echo "CONFIG-ASSERT-FAIL: check env"; exit 1; }
echo "BITEXACT-GATE: PASS ($NPASS/16)"

# --- gate (b): A/B (one fresh process per env) ---------------------------
for LX in 0 1; do
  rm -rf $T/trace$LX && mkdir -p $T/trace$LX
  MLX_QMV_MR_LX=$LX QMV_WARMUP=$WARMUP QMV_REPS=$REPS \
    flock -w 1200 $LOCK -c "$ROCPROF -d $T/trace$LX -f json --kernel-trace -- $PY $HERE/qmv_mr2_lx_bench.py" \
    > $T/bench$LX.log 2>&1 || { echo "FLOCK-FAIL bench LX=$LX"; exit 1; }
  TRACE=$(ls $T/trace$LX/*/*.json 2>/dev/null | head -1)
  [ -n "$TRACE" ] || { echo "TRACE-MISSING LX=$LX"; exit 1; }
  mkdir -p $T/lx$LX
  python3 $HERE/qmv_mr2_lx_parse.py "$TRACE" $WARMUP $REPS $T/lx$LX > $T/parse$LX.log 2>&1 || { echo "PARSE-FAIL LX=$LX"; exit 1; }
  mv $T/lx$LX/combos.json $T/combos$LX.json
done
grep -E "^CONFIG:" $T/bench0.log $T/bench1.log
grep -E "^COMBO:" $T/parse0.log $T/parse1.log

# --- gate (c): config assertions (kernel names actually dispatched) ------
# The template-instantiation check is non-vacuous on the mangled names
# rocprofv3 actually reports (mangled Lb0 = t825 path / Lb1 = hoist, or
# demangled ", false>" / ", true>"; t903 MINOR-2 fix — the t889 check
# was a no-op for mangled names). Proven non-vacuous offline by
# tests/test_qmv_mr2_lx_instantiation.py.
python3 $HERE/qmv_mr2_lx_instantiation.py $T/combos0.json $T/combos1.json || { echo "CONFIG-ASSERT-FAIL"; exit 1; }

# --- gate (b) verdict ----------------------------------------------------
python3 $HERE/qmv_mr2_lx_analyze.py $T/combos0.json $T/combos1.json > $T/gain.log 2>&1
RC=$?
cat $T/gain.log
echo "=== t889 DONE (gain-gate rc=$RC) ==="
exit $RC
