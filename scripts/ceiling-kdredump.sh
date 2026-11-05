#!/bin/bash
# ceiling-kdredump (t2059 rewrite): the COR wheel is 100% hipRTC-JIT —
# libmlx.so carries NO device code (verified: 0 AMDGPU ELFs embedded).
# The code objects that actually run are the .hsaco files the wheel's
# jit_module writes on first use to /tmp/mlx/<wheel-ver>/hsaco/gfx1201/.
# Dump their resource notes from that cache (same wheel ver, same machine
# toolchain, same gfx1201 — the kernels this wheel executes).
set -uo pipefail
MARK=${1:?usage: ceiling-kdredump.sh <marker-file>   (marker = file created before the mlp run)}
CACHE=/tmp/mlx/0.32.3.dev20261104+75ca881/hsaco/gfx1201
[ -d "$CACHE" ] || CACHE=$(ls -t -d /tmp/mlx/*/hsaco/gfx1201 2>/dev/null | head -1)
[ -n "$CACHE" ] || { echo "NO_CACHE_DIR"; exit 1; }
LLVM=$(ls /usr/lib/rocm-*/llvm/bin/llvm-readobj 2>/dev/null | head -1)
[ -n "$LLVM" ] || LLVM=$(which llvm-readobj)
echo "LLVM=$LLVM"
echo "CACHE=$CACHE"
ls -la "$CACHE" 2>/dev/null | grep -E "hsaco$|\.txt$" | head -40
NEW=$(find "$CACHE" -name "*.hsaco" -newer "$MARK" 2>/dev/null)
if [ -z "$NEW" ]; then
  # fall back to name-matched entries (qmm / qmv / gdn fused)
  NEW=$(ls "$CACHE"/*.hsaco 2>/dev/null | grep -iE "qmv|qmm|gdn|quant" | head -20)
fi
echo "--- dumping: $(echo "$NEW" | wc -l) files"
for f in $NEW; do
  [ -e "$f" ] || continue
  SYMS=$("$LLVM" --symbols "$f" 2>/dev/null | grep -E "qmv|qmm|gdn" | head -3)
  if [ -n "$SYMS" ]; then
    echo "== $f =="
    echo "$SYMS"
    "$LLVM" --notes "$f" 2>/dev/null | grep -E "Function Name|vgpr_count|sgpr_count|agpr_count|private_segment_size|group_segment_size|maximum_workgroup_size" | head -80
  fi
done
echo "KDR_DONE"
