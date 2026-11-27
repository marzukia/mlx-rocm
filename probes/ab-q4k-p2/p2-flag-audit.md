# A/B lane - P2 CPU line 1: compile/link flag audit (our lib-ab3 mmq TU vs their libggml-hip Q4_K TU)

Provenance: both lines verbatim from the build trees on the box (not reconstructed).
- Ours: /home/guest/attport-jimmy/build/temp.linux-x86_64-cpython-314/mlx.core/mlx/backend/rocm/CMakeFiles/mlx_hip_objects.dir/build.make (rule for hip_objs/quantized/mmq/mmq.o) + mlx.core/CMakeFiles/mlx.dir/link.txt; built by ab3-build.sh -> attport-jimmy-build.sh (`pip wheel`, CMAKE_ARGS `-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0`), lib-ab3.so md5 4738da138f1e0d8613fa90da06a73355.
- Theirs: /home/guest/perclass-probe/llama.cpp/build/compile_commands.json (entry for ggml-cuda/template-instances/mmq-instance-q4_k.cu, Nov 26 14:59) + the CMake link step for ggml-hip; libggml-hip.so.0 -> .0.28.0, md5 a2f1e90e7a9e41ed34025643280155bb.

## Compile line (mmq Q4_K translation unit)

Ours (hipcc driver wrapper, single arch, c++20, no explicit -O, 1 define):
    /usr/bin/hipcc -c mmq.hip -o mmq.o -fPIC -DMLX_USE_ROCM --offload-arch=gfx1201 -I<tree> -I... -I/opt/rocm-7.2.0/include -std=c++20

Theirs (direct clang++, dual arch, gnu++17, explicit -O3 -DNDEBUG, ~60 defines):
    /usr/lib/rocm/llvm/bin/clang++ -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_FA_*=... (49 FA-related defines: 48 pair switches, q4_0-q4_0/q8_0-q8_0/f16-f16/bf16-bf16 enabled + the FA_QUANTS list) -DGGML_HIP_NO_VMM -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_HIP -DUSE_PROF_API=1 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Dggml_hip_EXPORTS -I<ggml> -I<include> -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1201 --offload-arch=gfx1036 -fPIC -o mmq-instance-q4_k.cu.o -x hip -c mmq-instance-q4_k.cu

One-line delta: **hipcc wrapper + gfx1201-only + c++20 + default opt + 1 define**  vs  **direct clang++ + gfx1201/gfx1036 dual + gnu++17 + explicit -O3 -DNDEBUG + ~61 defines (incl. GGML_HIP_NO_VMM, GGML_SCHED_MAX_COPIES=4, USE_PROF_API=1)**.
No -mllvm flags on either side; no -ggml-hip-... tuning flags on either side. The 49 FA-related defines select flash-attention quant pairs only (q4_0/q8_0/f16/bf16) and do not touch the mmq Q4_K path. Total -D count: 61.

## Link line

Ours (plain c++ shared link of cpp objects + static hip archive + ROCm runtimes):
    /usr/bin/c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libmlx.so -o libmlx.so <~150 .cpp objects> -ldl -lpthread libopenblas.so libgguflib.a mlx/backend/rocm/libmlx_rocm_kernels.a libamdhip64.so librocblas.so.5.6 libhiprand.so.1.1 libhiprtc.so libhipblaslt.so

Theirs (clang++ device link of the hip objects, dual arch, explicit rtlib/unwindlib):
    /usr/lib/rocm/llvm/bin/clang++ -fPIC -O3 -DNDEBUG --offload-arch=gfx1201 --offload-arch=gfx1036 -shared --hip-link --rtlib=compiler-rt -unwindlib=libgcc -Wl,-soname,libggml-hip.so.0 <ggml-cuda objects> -o libggml-hip.so.0.28.0

One-line delta: ours = host `c++` linking a **static archive** of hip objects (device code embedded at compile time; no --hip-link, no rtlib/unwindlib selection); theirs = `clang++ --hip-link` device link with `--rtlib=compiler-rt -unwindlib=libgcc` and the dual arch flags repeated.

## Toolchain note

- Our /usr/bin/hipcc at build time (Nov 18, via CMAKE_PREFIX_PATH=/opt/rocm-7.2.0) is the ROCm 7.2.0 toolchain; the box's /usr/bin/hipcc now resolves to ROCm 7.15 / clang 23.0.0rocm7.14. The ISA extraction below therefore uses /usr/lib/rocm/llvm/bin/llvm-{objdump,readobj,nm} (clang-23 / rocm-7.14) = the SAME toolchain as their Nov 26 build, so their object is disassembled with the matching toolchain; our object is disassembled with a newer toolchain than the one that built it (7.2.0 -> 7.14) — noted as a caveat.
- Both .so files are fat: the AMDGPU code objects are embedded; extraction = carve the embedded ELF (e_machine=0xE0 in this toolchain, not 0x91) per section, then llvm-nm / llvm-objdump / llvm-readobj --notes per object.
