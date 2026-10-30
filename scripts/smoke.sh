#!/usr/bin/env bash
# mlx-rocm MVP smoke harness.
#
#   1. the build VM check: ssh reachability + rocm-smi (record GPU + toolchain)
#   2. local (host-a): build the GEMM (host path) + run a small GEMM
#   3. the build VM (the build VM, gfx1201): transfer tree, build with ROCm,
#      run the GPU GEMM + fp8 probe
#
# The the build VM phase is budgeted ~120 s total; if the build VM is unreachable or the
# budget runs out, the failure mode is recorded and the script continues
# (local results are still reported). Exit: 0 = local PASS & the build VM PASS,
# 1 = local FAIL, 2 = local PASS but the build VM failed/skipped.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BUILD_DIR="$ROOT/build"
RES_DIR="$ROOT/scripts/smoke-results"
mkdir -p "$RES_DIR"
LOG="$RES_DIR/smoke.log"

GUESTA_USER="${GUESTA_USER:-operator}"
GUESTA_HOST="${GUESTA_HOST:-<tailnet-ip>}"
GUESTA_BUDGET="${GUESTA_BUDGET:-120}"
GUESTA_REMOTE_DIR="/tmp/mlx-rocm-smoke"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

: > "$LOG"
log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }
step() { log "STEP: $*"; }

GUESTA_START=$(date +%s)
now() { date +%s; }
within_budget() { [ $(( $(now) - GUESTA_START )) -lt "$GUESTA_BUDGET" ]; }

LOCAL_STATUS=FAIL
GUESTA_STATUS=UNKNOWN
GUESTA_REASON=""

# ---------------------------------------------------------------- the build VM check
step "the build VM check: ssh ${GUESTA_USER}@${GUESTA_HOST} + rocm-smi"
GUESTA_INFO=$(
  "${SSH[@]}" "$GUESTA_USER@$GUESTA_HOST" \
    'rocm-smi 2>/dev/null | sed -n "/Concise Info/,/====/p" | tail -6; echo ---; command -v hipcc cmake | tr "\n" " "; echo; hipcc --version 2>/dev/null | head -1' \
    2>&1
)
GUESTA_RC=$?
if [ $GUESTA_RC -eq 0 ] && ! grep -qE "Connection refused|No route to host|timed out|Could not resolve" <<<"$GUESTA_INFO"; then
  GUESTA_STATUS=OK
  log "the build VM reachable; GPU/toolchain info:"
  sed 's/^/    /' <<<"$GUESTA_INFO" | tee -a "$LOG"
else
  GUESTA_STATUS=DOWN
  GUESTA_REASON="ssh failed (rc=$GUESTA_RC): $(grep -m1 -E "Connection|route|timed out|resolve|refused" <<<"$GUESTA_INFO" | head -1)"
  log "the build VM check failed: $GUESTA_REASON"
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

# ---------------------------------------------------------------- the build VM GPU GEMM
if [ "$GUESTA_STATUS" = OK ]; then
  if within_budget; then
    step "the build VM: transfer tree + build + GPU GEMM (budget ${GUESTA_BUDGET}s, elapsed $(( $(now) - GUESTA_START ))s)"
    if tar -C "$ROOT" --exclude .git --exclude build --exclude scripts/smoke-results --exclude dist \
        -cf - . \
      | "${SSH[@]}" "$GUESTA_USER@$GUESTA_HOST" \
          "rm -rf $GUESTA_REMOTE_DIR && mkdir -p $GUESTA_REMOTE_DIR && tar xf - -C $GUESTA_REMOTE_DIR"; then
      log "transfer: ok ($(( $(now) - GUESTA_START ))s elapsed)"
      if within_budget; then
        GUESTA_RUN=$(
          "${SSH[@]}" "$GUESTA_USER@$GUESTA_HOST" \
            "cd $GUESTA_REMOTE_DIR && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLX_ROCM_WITH_HIP=ON 2>&1 | tail -2 && cmake --build build -j 2>&1 | tail -6 && ./build/gemm_smoke hip 2>&1" \
            2>&1
        )
        GUESTA_RUN_RC=$?
        sed 's/^/    /' <<<"$GUESTA_RUN" | tee -a "$LOG"
        if [ $GUESTA_RUN_RC -eq 0 ] && grep -q "gemm_smoke: PASS" <<<"$GUESTA_RUN"; then
          GUESTA_STATUS=PASS
          GUESTA_REASON=""
        else
          GUESTA_STATUS=FAIL
          GUESTA_REASON="remote build/run rc=$GUESTA_RUN_RC (last: $(grep -m1 -E "FAIL|error|Error" <<<"$GUESTA_RUN" | head -1))"
        fi
      else
        GUESTA_STATUS=SKIPPED
        GUESTA_REASON="budget exhausted before remote build"
      fi
    else
      GUESTA_STATUS=FAIL
      GUESTA_REASON="tree transfer failed"
    fi
  else
    GUESTA_STATUS=SKIPPED
    GUESTA_REASON="budget exhausted before transfer"
  fi
else
  GUESTA_STATUS=SKIPPED
  GUESTA_REASON="$GUESTA_REASON (the build VM unreachable — GPU phase skipped)"
fi

# ---------------------------------------------------------------- summary
log "=================================================="
log "SUMMARY: local=$LOCAL_STATUS the build VM=$GUESTA_STATUS ${GUESTA_REASON:+($GUESTA_REASON)}"
log "log: $LOG"

[ "$LOCAL_STATUS" = PASS ] || exit 1
[ "$GUESTA_STATUS" = PASS ] || exit 2
exit 0
