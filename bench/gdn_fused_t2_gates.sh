# !/bin/bash GDN fused T=2 verification gates (run on guest-a, new wheel installed). $1 = python (venv), $2 = model dir NOTE: the old MLX_ROCM_GDN_FUSE_T2=0 kill-switch (routed T=2 to the op's 4-op fallback chain, which is 1-D-only in gdn_prep and faults on the GPU for T>1) has been removed from use_fallback; lossless A/B gating is FUSE=0 (model default path) vs FUSE=1 (fused T<=2).
set -u
PY=$1; MODEL=$2
cd "$(dirname "$0")"
B="mtp-bench3-side.py"
mkdir -p gdn_gates

run() { # name mode fuse
  local name=$1 mode=$2 fuse=$3
  echo "=== GATE RUN: $name (MLX_MTP=$mode FUSE=$fuse)"
  MLX_MTP=$mode MLX_MTP_DRAFT=2 MLX_ROCM_GDN_FUSE=$fuse \
  MTP_MODEL=$MODEL MTP_SIDECAR=model-mtp.safetensors \
  $PY $B > "gdn_gates/gate_$name.log" 2>&1
  grep -E "TOKS_S|GDN_FUSE_CALLS" "gdn_gates/gate_$name.log"
}

# Gate 1: lossless A/B (both must be byte-identical token streams)
run g1_default 2 0     # fully default chain (model default path, MTP=2)
run g1_t2on    2 1     # S=1+S=2 fused (new kernel, MTP=2)

echo "=== Gate 1 token-stream comparison:"
A=$(grep "^ALLTOKS:" gdn_gates/gate_g1_default.log | md5sum | cut -d' ' -f1)
C=$(grep "^ALLTOKS:" gdn_gates/gate_g1_t2on.log | md5sum | cut -d' ' -f1)
echo "md5: default=$A t2on=$C"
[ "$A" = "$C" ] && echo "GATE1 PASS (byte-identical)" || echo "GATE1 FAIL"

# Gate 2: fused-AR unchanged (MTP=0, S=1 fused)
run g2_fused_ar 0 1

# Gate 3: MTP-on speed (fused, back-to-back repeat for stability)
run g3_mtp_t2on 2 1
echo "=== Gate 3 tok/s (compare g1/g3 vs g2):"
grep -H "TOKS_S" gdn_gates/gate_g1_default.log gdn_gates/gate_g1_t2on.log gdn_gates/gate_g2_fused_ar.log gdn_gates/gate_g3_mtp_t2on.log
