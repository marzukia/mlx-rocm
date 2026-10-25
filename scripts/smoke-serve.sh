#!/usr/bin/env bash
# Phase 1 serve smoke (run ON marzuki-neon): qmlx-serve with a 9B-class dense Qwen3.
# Exit criteria (PLAN.md Phase 1): served, no NaNs, no OOM->SEGV single-process.
set -euo pipefail

: "${QMLX_SERVE_BIN:?path to the qmlx-serve entrypoint on neon}"
: "${QMLX_SERVE_MODEL:?model ref/name for a 9B-class dense Qwen3 (4bit qmm expected on gfx1201)}"
: "${QMLX_SERVE_PORT:-8080}"
# Safe default until the gfx1201 fp8 prefill is proven clean on ROCm 7.7 (docs/plan.md)
export MLX_ROCM_GEMM_BF16="${MLX_ROCM_GEMM_BF16:-1}"

"$QMLX_SERVE_BIN" --model "$QMLX_SERVE_MODEL" --port "$QMLX_SERVE_PORT" &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

# wait for readiness, then a generation probe
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${QMLX_SERVE_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 5
  kill -0 "$PID" || { echo "FAIL: qmlx-serve exited during startup"; exit 1; }
done
curl -sf "http://127.0.0.1:${QMLX_SERVE_PORT}/healthz" >/dev/null || { echo "FAIL: not ready in 10 min"; exit 1; }
echo "ok: qmlx-serve ready on :${QMLX_SERVE_PORT}"
echo "SERVE SMOKE: ready (generation probe: issue a 16-token completion against the API)"
