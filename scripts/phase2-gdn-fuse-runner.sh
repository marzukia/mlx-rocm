#!/usr/bin/env bash
# Phase 2 GDN-fuse: lossless gate + A/B bench under the GPU lock.
#
# Protocol (plan.md conventions):
#   * fresh process per measured path; WARM pass (discarded) inside each
#     process, then the MEASURE pass
#   * decode tok/s for ALL steps + steady-state window (first 56 excluded)
#   * graph decode OFF (MLX_ROCM_GRAPH_DECODE not set) on both arms
#   * lossless: byte-identical greedy stream (14tok + 29K) across processes,
#     per-layer GDN state match across processes (n asserted), fused-path
#     call counter > 0 (non-vacuity)
set -uo pipefail
MODELS="${MODELS:-/models}"

LOCK=$MODELS/gpu-bench.lock
BASE=$MODELS/tmp/phase2-fuse
VENV_F=${BASE}/venv-fused
# Both arms use the SAME build (venv-fused, which has both the chain and the
# fused op) so the only difference is MLX_ROCM_GDN_FUSE. The earlier
# venv-m8 (baseline run-1012 wheel) vs venv-fused comparison mismatched at the
# final step because the two .so BUILDS differ in codegen for the chain path
# (accumulating by step 256); within one build chain == fused is bit-identical.
VENV_UF=${VENV_F}
MODEL=$MODELS/qwen3-9b-mlx
OUT=${BASE}/out
PY=${BASE}/phase2-gdn-fuse-bench.py

echo "=== phase2-gdn-fuse runner $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
mkdir -p "$OUT"

sha() { sha256sum "$1" | cut -c1-16; }
SO_F=$(sha ${VENV_F}/lib/python3.12/site-packages/mlx/lib/libmlx.so)
echo "so_sha256 both-arms=${SO_F}"
# The .so must contain the fused op (both arms share it; the F branch selects
# it via the env var, verified by the non-vacuous fused_calls>0 below).
${VENV_F}/bin/python -c "import mlx.core as mx,os; os.environ['LD_LIBRARY_PATH']='/opt/rocm/core-10.1/lib'" 2>/dev/null
env LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ${VENV_F}/bin/python -c "import mlx.core as mx; assert hasattr(mx.fast,'gdn_fused_step'), 'fused op missing'" \
  || { echo "FATAL: venv-fused .so lacks gdn_fused_step"; exit 1; }
grep -q "gdn_fused_step" ${VENV_F}/lib/python3.12/site-packages/mlx_lm/models/qwen3_5.py \
  || { echo "FATAL: venv-fused model not patched"; exit 1; }

flock -w 5400 "$LOCK" bash -s <<EOF
set -e
rc=0
run_one() {
  local tag=\$1 py=\$2 ctx=\$3; shift 3
  local log=${OUT}/\${tag}_ctx\${ctx}.txt
  echo "--- \${tag} ctx=\${ctx} \$(date -u +%H:%M:%S) ---"
  env "\$@" \
      LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib \
      P2_PROMPT_TOKENS=\$ctx P2_MAX_TOKENS=256 P2_SKIP=56 P2_WARM=1 \
      P2_STATE_OUT=${OUT}/\${tag}_ctx\${ctx}_state.jsonl \
      P2_STREAM_OUT=${OUT}/\${tag}_ctx\${ctx}_stream.txt \
      GD_MODEL=$MODEL \
      "\$py" "$PY" >"\$log" 2>&1
  local r=\$?
  [ \$r -eq 0 ] || { echo "FAIL \${tag} ctx=\${ctx} rc=\${r}"; tail -5 "\$log"; }
  grep -E "MEASURE|STATE-CAPTURE" "\$log" | head -3
  return \$r
}
for ctx in 14 29000; do
  run_one UF ${VENV_UF}/bin/python \${ctx} VENV_ID=unfused || rc=1
  run_one F  ${VENV_F}/bin/python  \${ctx} VENV_ID=fused MLX_ROCM_GDN_FUSE=1 || rc=1
  echo '--- lossless gate ctx=\${ctx} ---'
  if cmp -s ${OUT}/UF_ctx\${ctx}_stream.txt ${OUT}/F_ctx\${ctx}_stream.txt; then
    echo 'STREAM-IDENTICAL PASS ctx='\${ctx}' tokens='\$(wc -w < ${OUT}/UF_ctx\${ctx}_stream.txt)
  else
    echo 'STREAM-IDENTICAL FAIL ctx='\${ctx}; rc=1
  fi
  python3 ${BASE}/compare_states.py \
    ${OUT}/UF_ctx\${ctx}_state.jsonl ${OUT}/F_ctx\${ctx}_state.jsonl || rc=1
  FC=\$(grep -o 'fused_calls=[0-9]*' ${OUT}/F_ctx\${ctx}.txt | grep -o '[0-9]*' | tail -1)
  if [ "\${FC:-0}" -gt 0 ]; then
    echo 'NON-VACUOUS PASS ctx='\${ctx}' fused_calls='\${FC}
  else
    echo 'NON-VACUOUS FAIL ctx='\${ctx}' fused_calls='\${FC}; rc=1
  fi
done
exit \$rc
EOF
RC=$?

echo "=== A/B decode tok/s (graph decode OFF both arms) ==="
for ctx in 14 29000; do
  for arm in UF F; do
    line=$(grep -h "MEASURE" ${OUT}/${arm}_ctx${ctx}.txt | head -1)
    echo "  ${arm} ctx=${ctx}: $(echo "$line" | grep -o 'decode_tok_s_all=[0-9.]*\|decode_tok_s_window=[0-9.]*' | tr '\n' ' ')"
  done
done
echo "=== exit ${RC} ==="
exit $RC
