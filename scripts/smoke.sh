#!/usr/bin/env bash
# mlx-rocm MVP smoke harness.
#
#   1. neon check: ssh reachability + rocm-smi (record GPU + toolchain)
#   2. local (hydrogen): build the GEMM (host path) + run a small GEMM
#   3. neon (marzuki-neon, gfx1201): transfer tree, build with ROCm,
#      run the GPU GEMM + fp8 probe
#
# The neon phase is budgeted ~120 s total; if neon is unreachable or the
# budget runs out, the failure mode is recorded and the script continues
# (local results are still reported). Exit: 0 = local PASS & neon PASS,
# 1 = local FAIL, 2 = local PASS but neon failed/skipped.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BUILD_DIR="$ROOT/build"
RES_DIR="$ROOT/scripts/smoke-results"
mkdir -p "$RES_DIR"
LOG="$RES_DIR/smoke.log"

NEON_USER="${NEON_USER:-andryo}"
NEON_HOST="${NEON_HOST:-100.91.94.56}"
NEON_BUDGET="${NEON_BUDGET:-120}"
NEON_REMOTE_DIR="/tmp/mlx-rocm-smoke"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

: > "$LOG"
log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }
step() { log "STEP: $*"; }

NEON_START=$(date +%s)
now() { date +%s; }
within_budget() { [ $(( $(now) - NEON_START )) -lt "$NEON_BUDGET" ]; }

LOCAL_STATUS=FAIL
NEON_STATUS=UNKNOWN
NEON_REASON=""

# ---------------------------------------------------------------- neon check
step "neon check: ssh ${NEON_USER}@${NEON_HOST} + rocm-smi"
NEON_INFO=$(
  "${SSH[@]}" "$NEON_USER@$NEON_HOST" \
    'rocm-smi 2>/dev/null | sed -n "/Concise Info/,/====/p" | tail -6; echo ---; command -v hipcc cmake | tr "\n" " "; echo; hipcc --version 2>/dev/null | head -1' \
    2>&1
)
NEON_RC=$?
if [ $NEON_RC -eq 0 ] && ! grep -qE "Connection refused|No route to host|timed out|Could not resolve" <<<"$NEON_INFO"; then
  NEON_STATUS=OK
  log "neon reachable; GPU/toolchain info:"
  sed 's/^/    /' <<<"$NEON_INFO" | tee -a "$LOG"
else
  NEON_STATUS=DOWN
  NEON_REASON="ssh failed (rc=$NEON_RC): $(grep -m1 -E "Connection|route|timed out|resolve|refused" <<<"$NEON_INFO" | head -1)"
  log "neon check failed: $NEON_REASON"
fi

# ---------------------------------------------------------------- local build
step "local build (host GEMM; ROCm optional)"
if cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1 \
   && cmake --build "$BUILD_DIR" -j >>"$LOG" 2>&1 \
   && [ -x "$BUILD_DIR/gemm_smoke" ]; then
  log "local build: ok"
else
  log "local build: FAILED (see $LOG)"
  exit 1
fi

# ---------------------------------------------------------------- local run
step "local run: small host GEMM (bf16 256^3)"
if "$BUILD_DIR/gemm_smoke" host >>"$LOG" 2>&1; then
  LOCAL_STATUS=PASS
  log "local GEMM: PASS"
else
  log "local GEMM: FAIL (see $LOG)"
fi

# ---------------------------------------------------------------- neon GPU GEMM
if [ "$NEON_STATUS" = OK ]; then
  if within_budget; then
    step "neon: transfer tree + build + GPU GEMM (budget ${NEON_BUDGET}s, elapsed $(( $(now) - NEON_START ))s)"
    if tar -C "$ROOT" --exclude .git --exclude build --exclude scripts/smoke-results --exclude dist \
        -cf - . \
      | "${SSH[@]}" "$NEON_USER@$NEON_HOST" \
          "rm -rf $NEON_REMOTE_DIR && mkdir -p $NEON_REMOTE_DIR && tar xf - -C $NEON_REMOTE_DIR"; then
      log "transfer: ok ($(( $(now) - NEON_START ))s elapsed)"
      if within_budget; then
        NEON_RUN=$(
          "${SSH[@]}" "$NEON_USER@$NEON_HOST" \
            "cd $NEON_REMOTE_DIR && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLX_ROCM_WITH_HIP=ON 2>&1 | tail -2 && cmake --build build -j 2>&1 | tail -6 && ./build/gemm_smoke hip 2>&1" \
            2>&1
        )
        NEON_RUN_RC=$?
        sed 's/^/    /' <<<"$NEON_RUN" | tee -a "$LOG"
        if [ $NEON_RUN_RC -eq 0 ] && grep -q "gemm_smoke: PASS" <<<"$NEON_RUN"; then
          NEON_STATUS=PASS
          NEON_REASON=""
        else
          NEON_STATUS=FAIL
          NEON_REASON="remote build/run rc=$NEON_RUN_RC (last: $(grep -m1 -E "FAIL|error|Error" <<<"$NEON_RUN" | head -1))"
        fi
      else
        NEON_STATUS=SKIPPED
        NEON_REASON="budget exhausted before remote build"
      fi
    else
      NEON_STATUS=FAIL
      NEON_REASON="tree transfer failed"
    fi
  else
    NEON_STATUS=SKIPPED
    NEON_REASON="budget exhausted before transfer"
  fi
else
  NEON_STATUS=SKIPPED
  NEON_REASON="$NEON_REASON (neon unreachable — GPU phase skipped)"
fi

# ---------------------------------------------------------------- summary
log "=================================================="
log "SUMMARY: local=$LOCAL_STATUS neon=$NEON_STATUS ${NEON_REASON:+($NEON_REASON)}"
log "log: $LOG"

[ "$LOCAL_STATUS" = PASS ] || exit 1
[ "$NEON_STATUS" = PASS ] || exit 2
exit 0
