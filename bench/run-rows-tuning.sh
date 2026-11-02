#!/bin/bash
# t1354 rows-tuning A/B battery (run ON guest-a, RX 9070 XT / gfx1201):
# the M=4 verify GEMM knob battery for the qmv_rows_msg kernel variants
# (patches/qmv-rows-tuning.diff; REPORT-rows-tuning.md).
#
# Arms (one fresh process per env, canonical tape bench):
#   B0  rows OFF (gate default) — the wheel regression gate: must
#       reproduce the m4 B0 reference (70.02 tok/s, verify 25.66 ms,
#       pair 44.85 ms) within +/-3% (all tuning code is default-off,
#       so B0 is behaviorally the m4 wheel).
#   R0  rows ON, base kernel (all knobs unset — the shipped rows path)
#   R1  rows ON + MLX_QMV_ROWS_WS=32   (pre-decoded weight tiles)
#   R2  rows ON + MLX_QMV_ROWS_KBLOCK=2 (two k-tiles per lane step)
#   R3  rows ON + MLX_QMV_ROWS_WARPS=4  (4 waves per block)
#
# Gates (per arm, per config doc rule 6):
#   (1) tape bench md5 == 2c2d6c90c2349ae087e53e5d6c60018e (canonical,
#       guest-pinned); the model sidecar model-mtp.safetensors md5 is
#       pinned at start and re-checked after the battery (no model drift).
#   (2) B0 gate: tok/s within +/-3% of 70.02 and verify ms within +/-3%
#       of 25.66 (the m4 B0 reference; REPORT-gemms-m4.md section 2).
#   (3) bit-exactness: R1/R2/R3 FIRST20 + ALLTOKS md5 == R0 (the
#       variants are bit-identical by construction — the unit gate
#       tests/test_qmv_rows_tuning_math.py; any divergence = FAIL and
#       the arm is not a speed data point).
#   (4) n_refwd sanity: 95 <= n_refwd <= 107 (m4/t1307 band 100-102).
#
# Target: a rows arm (R0-R3) with verify ms < 22.25 (below the current
# best 22.24/22.25). The winner (if any) is the tuning config for the
# next round (the WMMA M<=4 kernel, REPORT-gemms-m4.md section 4, is
# the follow-up either way).
#
# Usage (ON guest-a):
#   bash bench/run-rows-tuning.sh <venv-py> [model-dir] [tape-bench]
#     venv-py:    the interpreter with the built wheel (venv-rs)
#     model-dir:  default $MODELS/qwen3-9b-mlx
#     tape-bench: default $HOME/t1325-canon/mtp-bench-k3-tape.py
set -uo pipefail
PY=$1
MODEL=${2:-${MODELS:-models}/qwen3-9b-mlx}
TAPE=${3:-$HOME/t1325-canon/mtp-bench-k3-tape.py}
MODELS=$(dirname "$MODEL")
T=$MODELS/tmp/t1354-rows-tuning
LOCK=$MODELS/gpu-bench.lock
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
B0_TOKS_REF=70.02
B0_VERIFY_REF=25.66
TAPE_MD5_REF=2c2d6c90c2349ae087e53e5d6c60018e
mkdir -p $T

# Orphan-lock class (run-852 incident): kill our children on any exit.
CHILDREN=""
trap 'for c in $CHILDREN; do kill $c 2>/dev/null; done; wait 2>/dev/null' EXIT INT TERM

echo "=== t1354 ENV: py=$PY model=$MODEL"
echo "=== wheel: $($PY -c 'import mlx.core as mx; print(mx.__version__)')"
$PY -c "import mlx.core as mx; print('DEVICE:', mx.rocm.device_name() if hasattr(mx.rocm,'device_name') else 'n/a')" 2>/dev/null || true

# --- gate (1): the canonical bench + the model sidecar pin -----------
[ -f "$TAPE" ] || { echo "TAPE-MISSING: $TAPE"; exit 1; }
TAPE_MD5=$(md5sum "$TAPE" | cut -d' ' -f1)
[ "$TAPE_MD5" = "$TAPE_MD5_REF" ] || { echo "TAPE-MD5-FAIL: $TAPE_MD5 != $TAPE_MD5_REF"; exit 1; }
echo "TAPE: ok ($TAPE_MD5)"
SIDECAR=$MODEL/model-mtp.safetensors
[ -f "$SIDECAR" ] || { echo "SIDECAR-MISSING: $SIDECAR"; exit 1; }
SIDECAR_MD5=$(md5sum "$SIDECAR" | cut -d' ' -f1)
echo "SIDECAR: pinned $SIDECAR_MD5"

# the lock is the run gate (gpu held by another task -> FLOCK-FAIL)
flock -w 300 $LOCK -c true || { echo "FLOCK-FAIL: GPU busy (lock held)"; exit 1; }

run_arm() { # name rows knobs...
  local name=$1; shift
  local envs=()
  local rows=$1; shift
  [ -n "$rows" ] && envs+=("$rows")
  for k in "$@"; do envs+=("$k"); done
  if [ ${#envs[@]} -gt 0 ]; then
    env "${envs[@]}" MTP_MODEL=$MODEL MLX_ROCM_GDN_TAPE=1 \
      $PY "$TAPE" > $T/$name.log 2>&1 &
  else
    MTP_MODEL=$MODEL MLX_ROCM_GDN_TAPE=1 $PY "$TAPE" > $T/$name.log 2>&1 &
  fi
  CHILDREN="$CHILDREN $!"
  wait $! 2>/dev/null
  grep -E "^MTP=4|^WALL_MS|^PER_PAIR_MS|^TAPE |^FIRST20" $T/$name.log | sed "s/^/$name /"
}

parse() { # name field -> value (verify ms / toks/s / n_refwd)
  local name=$1 f=$2
  case $f in
    verify) grep "^WALL_MS" $T/$name.log 2>/dev/null | head -1 | sed -E 's/.*verify=([0-9.]+).*/\1/' ;;
    toks)   grep "^MTP=4" $T/$name.log 2>/dev/null | head -1 | sed -E 's/.*TOKS_S=([0-9.]+).*/\1/' ;;
    nref)   grep "^WALL_MS" $T/$name.log 2>/dev/null | head -1 | sed -E 's/.*n_refwd=([0-9]+).*/\1/' ;;
  esac
}

# --- gate (2): B0 (rows OFF) — the wheel regression gate --------------
# B0 = MLX_ROCM_QMV_ROWS UNSET (the dispatch gate is getenv != nullptr;
# an empty value would still engage the rows path)
run_arm B0 ""
V_B0=$(parse B0 verify); T_B0=$(parse B0 toks); N_B0=$(parse B0 nref)
echo "B0: toks=$T_B0 verify=$V_B0 n_refwd=$N_B0"
python3 - "$T_B0" "$V_B0" <<'EOF' || { echo "B0-GATE-FAIL"; exit 1; }
import sys
t, v = float(sys.argv[1]), float(sys.argv[2])
assert abs(t - 70.02) / 70.02 <= 0.03, f"B0 toks/s {t} off by >3% vs 70.02"
assert abs(v - 25.66) / 25.66 <= 0.03, f"B0 verify {v} off by >3% vs 25.66"
print("B0-GATE: PASS (toks/s within +/-%.1f%%, verify within +/-%.1f%%)" %
      (abs(t - 70.02) / 70.02 * 100, abs(v - 25.66) / 25.66 * 100))
EOF

# --- the rows arms ----------------------------------------------------
run_arm R0 "MLX_ROCM_QMV_ROWS=1" "MLX_ROCM_QMV_ROWS_MIN_N=1"
run_arm R1 "MLX_ROCM_QMV_ROWS=1" "MLX_ROCM_QMV_ROWS_MIN_N=1" "MLX_QMV_ROWS_WS=32"
run_arm R2 "MLX_ROCM_QMV_ROWS=1" "MLX_ROCM_QMV_ROWS_MIN_N=1" "MLX_QMV_ROWS_KBLOCK=2"
run_arm R3 "MLX_ROCM_QMV_ROWS=1" "MLX_ROCM_QMV_ROWS_MIN_N=1" "MLX_QMV_ROWS_WARPS=4"

# --- gate (3): bit-exactness vs R0 (md5 of the token streams) ---------
MD5_R0=$(grep "^ALLTOKS:" $T/R0.log | md5sum | cut -d' ' -f1)
F20_R0=$(grep "^FIRST20:" $T/R0.log | md5sum | cut -d' ' -f1)
GATE3=PASS
for A in R1 R2 R3; do
  MD5_A=$(grep "^ALLTOKS:" $T/$A.log | md5sum | cut -d' ' -f1)
  F20_A=$(grep "^FIRST20:" $T/$A.log | md5sum | cut -d' ' -f1)
  if [ "$MD5_A" != "$MD5_R0" ] || [ "$F20_A" != "$F20_R0" ]; then
    GATE3=FAIL
    echo "BITEXACT-FAIL: $A diverges from R0 (ALLTOKS ${MD5_A:0:8} vs ${MD5_R0:0:8})"
  fi
done
echo "BITEXACT-GATE: $GATE3 (vs R0 ALLTOKS ${MD5_R0:0:12} / FIRST20 ${F20_R0:0:12})"

# --- gate (4): n_refwd band -------------------------------------------
for A in R0 R1 R2 R3; do
  N=$(parse $A nref)
  if [ -n "$N" ] && { [ "$N" -lt 95 ] || [ "$N" -gt 107 ]; }; then
    echo "NREFWD-FAIL: $A n_refwd=$N outside 95..107"
  fi
done

# --- summary table -----------------------------------------------------
echo
echo "=== t1354 rows-tuning summary (verify ms / pair ms / tok/s) ==="
printf "%-4s %-10s %-10s %-10s %-8s\n" arm toks/s verify pair n_refwd
for A in B0 R0 R1 R2 R3; do
  printf "%-4s %-10s %-10s %-10s %-8s\n" "$A" "$(parse $A toks)" "$(parse $A verify)" \
    "$(grep "^PER_PAIR_MS" $T/$A.log 2>/dev/null | head -1 | sed -E 's/.*wall=([0-9.]+).*/\1/')" \
    "$(parse $A nref)"
done
[ "$GATE3" = PASS ] || echo "VERDICT: GATE3 FAIL — the diverging arm(s) are NOT speed data points"
# the target check: any rows arm < 22.25 verify ms
for A in R0 R1 R2 R3; do
  V=$(parse $A verify)
  [ -z "$V" ] && continue
  if python3 -c "import sys; sys.exit(0 if float('$V') < 22.25 else 1)"; then
    echo "TARGET MET: $A verify $V ms < 22.25"
  fi
done
# sidecar re-check (no model drift during the battery)
SIDECAR_MD5_AFTER=$(md5sum "$SIDECAR" | cut -d' ' -f1)
[ "$SIDECAR_MD5" = "$SIDECAR_MD5_AFTER" ] || echo "SIDECAR-DRIFT: $SIDECAR_MD5 -> $SIDECAR_MD5_AFTER"
echo "SIDECAR: ok ($SIDECAR_MD5_AFTER)"
