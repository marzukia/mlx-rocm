#!/bin/bash
# ceiling-kdredump: extract embedded amdgcn code objects from the COR wheel's
# libmlx.so and llvm-readobj --notes the gdn_fused_step_tape + qmv_tiled
# kernels, so the occupancy/in-flight arithmetic is cited from the exact
# wheel under test.
set -uo pipefail
LIB=${1:?usage: ceiling-kdredump.sh <libmlx.so>}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python3 - "$LIB" "$WORK" <<'EOF'
import sys, glob, struct
lib, work = sys.argv[1], sys.argv[2]
data = open(lib, "rb").read()
MAGIC = b"\x7fELF"
i, n = 0, 0
while True:
    i = data.find(MAGIC, i)
    if i < 0:
        break
    # find the end of this embedded ELF: scan for the next magic at a
    # plausible offset; we take 64 MiB max and let llvm-readobj reject bad
    # tails (it reads only valid sections)
    j = data.find(MAGIC, i + 4)
    end = j if j > 0 else len(data)
    chunk = data[i:min(end, i + 64 * 1024 * 1024)]
    # keep only elf64-amdgpu: e_ident[16] + e_type(2) + e_machine(2);
    # e_machine == 277 (0x115) is AMDGPU
    if len(chunk) < 20 or struct.unpack_from("<H", chunk, 18)[0] != 277:
        i = i + 4
        continue
    f = f"{work}/co{n}.elf"
    open(f, "wb").write(chunk)
    n += 1
    i = i + 4
print(f"extracted={n}", flush=True)
EOF

LLVM=$(ls /usr/lib/rocm/llvm/bin/llvm-readobj 2>/dev/null || which llvm-readobj)
echo "LLVM=$LLVM"
for f in "$WORK"/co*.elf; do
  [ -e "$f" ] || continue
  if "$LLVM" --symbols "$f" 2>/dev/null | grep -q "gdn_fused_step_tape\|qmv_tiled_kernel"; then
    echo "== $f =="
    "$LLVM" --notes "$f" 2>/dev/null | grep -E "Function Name|vgpr_count|sgpr_count|agpr_count|private_segment_size|group_segment_size|maximum_workgroup_size" | head -60
  fi
done
echo "KDR_DONE"
