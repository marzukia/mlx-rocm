#!/bin/bash
# qmv_rows_bitexact_disasm: compile-verify the rows-kernel restructure against
# the one-row qmv_wide_kernel (the contract reference's kernel).
#
# Compiles BOTH kernels standalone (ROCm 10.x clang 23, -O3, gfx1201),
# disassembles with llvm-objdump, and compares the per-row FP op SEQUENCE
# (normalized), not just FMA counts. Run on the GPU guest (compile only;
# no GPU needed, but keep the lane pattern: no GPU work here).
#
# Args: <kernel-dir> (dir with qmv_rows_msg_kernel.hip)
#       <qmm.hip path> (source of qmv_wide_kernel)
#
# Exit 0 = per-row FP sequences match for all 18 instantiations.
set -u
KDIR=$1
QMM=$2
W=/tmp/qmvbe-disasm
mkdir -p $W
cd $W

CXX=/usr/bin/hipcc
OBJDUMP=/usr/lib/rocm/llvm/bin/llvm-objdump

# 1. Extract qmv_wide_kernel (the function text only) from qmm.hip.
python3 - <<EOF
import re
src = open("$QMM").read()
m = re.search(r"template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>\n__global__ void __launch_bounds__\(1024\) qmv_wide_kernel\(", src)
assert m, "qmv_wide_kernel not found"
start = m.start()
# find the closing brace of the function: scan brace depth from the first '{'
i = src.index("{", start)
depth = 0
for j in range(i, len(src)):
    if src[j] == "{":
        depth += 1
    elif src[j] == "}":
        depth -= 1
        if depth == 0:
            end = j + 1
            break
fn = src[start:end]
open("qmv_wide_extract.hip", "w").write(
    "namespace mlx::core::rocm {\n" + fn + "\n} // namespace\n")
print("extracted qmv_wide_kernel:", len(fn), "bytes")
EOF

# 2. Wrapper: load_scale_value (affine) + both kernels + 18 instantiations.
cat > wrapper.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

template <typename ScaleT, int GROUP_SIZE, bool AFFINE>
__device__ inline float load_scale_value(ScaleT raw) {
  if constexpr (AFFINE)
    return static_cast<float>(raw);
  else
    return 0.0f; // non-affine path not used by this harness
}
#include "qmv_wide_extract.hip"
#include "qmv_rows_msg_kernel.hip"

#define INST_WIDE(T, GS, BITS)                                              \
  template __global__ void mlx::core::rocm::qmv_wide_kernel<T, float, GS,   \
                                                            BITS>(          \
      const T*, const uint8_t*, const float*, const float*, T*, int, int,   \
      int, bool);

#define INST_ROWS(T, GS, BITS)                                              \
  template __global__ void mlx::core::rocm::qmv_rows_msg_kernel<T, float,   \
                                                                 GS, BITS>( \
      const T*, const uint8_t*, const float*, const float*, T*, int, int,   \
      int, bool);

EOF
# 18 instantiations each (f32/bf16/f16 x GS 32/64/128 x BITS 4/8).
for T in float hip_bfloat16 __half; do
  for GS in 32 64 128; do
    for BITS in 4 8; do
      echo "INST_WIDE($T, $GS, $BITS);" >> wrapper.hip
      echo "INST_ROWS($T, $GS, $BITS);" >> wrapper.hip
    done
  done
done

# 3. Compile both (one translation unit, -O3, gfx1201) — the same
#    pipeline as the wheel build (hipcc -O3 -std=c++20 --offload-arch=gfx1201).
cp $KDIR/qmv_rows_msg_kernel.hip .
$CXX -O3 -std=c++20 --offload-arch=gfx1201 -c wrapper.hip -o both.o 2> compile.log
if [ $? -ne 0 ]; then
  echo "COMPILE FAIL:"
  grep -iE 'error' compile.log | head -20
  exit 2
fi
echo "compile OK"

# 4. Carve the GPU ELF out of the .hip_fatbin section (the .o carries
# the x86 device stubs only; the AMDGPU code is embedded as a fatbin).
python3 - <<'EOF'
import struct
d = open("both.o", "rb").read()
offs = []
i = 0
while True:
    i = d.find(b"\x7fELF", i)
    if i < 0:
        break
    offs.append(i)
    i += 1
for o in offs:
    if o == 0:
        continue
    ei = d[o : o + 64]
    if len(ei) < 64 or ei[4] != 2:  # 64-bit
        continue
    em = struct.unpack("<H", ei[18:20])[0]
    if em != 224:  # AMDGPU e_machine
        continue
    shoff = struct.unpack("<Q", ei[40:48])[0]
    shentsize = struct.unpack("<H", ei[58:60])[0]
    shnum = struct.unpack("<H", ei[60:62])[0]
    end = shoff + shnum * shentsize
    phoff = struct.unpack("<Q", ei[32:40])[0]
    phentsize = struct.unpack("<H", ei[54:56])[0]
    phnum = struct.unpack("<H", ei[56:58])[0]
    end = max(end, phoff + phnum * phentsize)
    for s in range(shnum):
        h = shoff + s * shentsize
        if h + 64 > len(d) - o:
            break
        sh = d[o + h : o + h + 64]
        off = struct.unpack("<Q", sh[24:32])[0]
        size = struct.unpack("<Q", sh[32:40])[0]
        end = max(end, off + size)
    open("gpu.elf", "wb").write(d[o : o + end])
    print(f"carved GPU ELF at {o}, {end - o} bytes")
    break
else:
    raise SystemExit("no embedded AMDGPU ELF found")
EOF

# 5. Disassemble both kernels.
$OBJDUMP -d gpu.elf > both.sass 2>/dev/null
python3 "$3" both.sass
rc=$?
echo "disasm-compare rc=$rc"
exit $rc
