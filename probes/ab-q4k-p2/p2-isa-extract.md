# A/B lane - P2 CPU line 2: ISA extraction from both .so files (Q4_K mul_mat_q kernels, gfx1201)

Method: carve the embedded AMDGPU code objects out of each .so (embedded ELFs with e_machine=0xE0 in this toolchain — not 0x91; verified via llvm-readobj "Machine: EM_AMDGPU (0xE0)"), then per object: llvm-nm (symbols), llvm-readobj --notes (kernel resource metadata: vgpr_count/sgpr_count/vgpr_spill_count/private_segment_fixed_size/group_segment_fixed_size), llvm-objdump -d (per-kernel instruction counts). Tools: /usr/lib/rocm/llvm/bin/llvm-{objdump,readobj,nm} = clang 23.0.0 / ROCm 7.14 = the same toolchain as their Nov 26 build. Carve scripts on box: /home/andryo/gemm1-leg2/{isacarve3.py,kstats.py}.

Objects: 46 embedded objects in lib-ab3.so (md5 4738da13); the Q4_K gfx1201 object = co-ab3-40.elf (our ggml enum: GGML_TYPE_Q4_K=10; 8 q4_K strings). 282 objects in libggml-hip.so.0 (md5 a2f1e90e, dual-arch, 1036/1201 pairs); the Q4_K gfx1201 object = co-thr-231.elf (their enum: GGML_TYPE_Q4_K=12; 17 q4_K strings), paired with co-thr-230.elf (gfx1036).

## Finding 0 (both): the Q4_K dot has NO dp4a and NO mfma instructions

The `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a` "dp4a" is a TILE-LAYOUT name (`mmq_get_dp4a_tile_x_sizes`, mmq-vec-dot.cuh:905); the inner product is `vec_dot_q4_K_q8_1_impl_mmq` = f32/half arithmetic. Confirmed in ISA: `v_dp4a*` count = 0 and `v_mfma*` count = 0 in BOTH the ours and theirs Q4_K gfx1201 objects (whole-object grep over the full disassembly). The Q4_K path is pure VALU (f32/mix + integer dequant) in both builds. (dp4a instructions exist only in the pure-int8 q8_0-family dot variants, not the K-quant path.)

## Finding 1: per-(J,fallback) device-code map — the two builds disagree on which configs have real code

7-instruction kernels (VGPR 39, SCR 16B, 32-512B LDS) are no-device-code stubs: body is the s_getpc_b64 / s_add / s_swappc_b64 immediate-exit pattern (verbatim, theirs J=40 fb=1):

    s_getpc_b64 s[2:3]
    s_sext_i32_i16 s3, s3
    s_add_co_u32 s2, s2, 0xffff4d7c
    s_add_co_ci_u32 s3, s3, lit(0xffffffff)
    s_add_nc_u64 s[8:9], s[0:1], 0xc0
    s_mov_b32 s32, 0
    s_swappc_b64 s[30:31], s[2:3]

A launched stub computes nothing and writes no output.

| J   | fb=0 ours        | fb=0 theirs       | fb=1 ours        | fb=1 theirs       |
|-----|------------------|-------------------|------------------|-------------------|
| 8   | stub             | stub             | stub             | stub              |
| 16  | REAL 256/1652    | stub             | REAL 256/1732    | REAL 89/865       |
| 24  | stub             | stub             | stub             | stub              |
| 32  | REAL 256/3051    | stub             | REAL 212/2756    | REAL 108/1134     |
| 40  | stub             | stub             | stub             | stub              |
| 48  | REAL 256/4464    | stub             | stub             | stub              |
| 56  | stub             | stub             | stub             | stub              |
| 64  | REAL 256/5765    | stub             | REAL 133/4858    | REAL 157/1651     |
| 72  | stub             | stub             | stub             | stub              |
| 80  | REAL 256/7098    | stub             | stub             | stub              |
| 88  | stub             | stub             | stub             | stub              |
| 96  | stub             | stub             | stub             | stub              |
| 104 | stub             | stub             | stub             | stub              |
| 112 | stub             | stub             | stub             | stub              |
| 120 | stub             | stub             | stub             | stub              |
| 128 | **stub**         | **stub**         | **stub**         | **REAL 256/2744** |

(cell = VGPR / instruction count; "stub" = 7-instruction no-device-code kernel)

- J=128 — the config the runtime selects at runtime (the in-run hook prints `J=128 I=128` on every P2 class, and the host picker loops J=8..128 by best-ntiles, mmq.cu:141) — is a no-device-code stub in OURS for BOTH fb values, and real in THEIRS only for fb=1 (256 VGPR / 2744 inst).
- fb=0: ours real at J=16,32,48,64,80; theirs ALL stubs.
- fb=1: ours real at J=16,32,64; theirs real at J=16,32,64,128.

## Finding 2: resource table for the real kernels (VGPR / SGPR / scratch B / LDS B / inst / VGPR-spill count / dp4a-instr)

Ours (co-ab3-40.elf, type 10):
| J  | fb | VGPR | SGPR | SCR  | LDS | INST | SPILLm |
|----|----|------|------|------|-----|------|--------|
| 16 | 1  | 256  | 33   | 100  | 64  | 1732 | 24     |
| 32 | 1  | 212  | 33   | 0    | 128 | 2756 | 0      |
| 64 | 1  | 133  | 33   | 0    | 256 | 4858 | 0      |
| 16 | 0  | 256  | 31   | 52   | 64  | 1652 | 12     |
| 32 | 0  | 256  | 31   | 768  | 128 | 3051 | 192    |
| 48 | 0  | 256  | 31   | 1336 | 192 | 4464 | 358    |
| 64 | 0  | 256  | 31   | 1388 | 256 | 5765 | 495    |
| 80 | 0  | 256  | 31   | 2040 | 320 | 7098 | 680    |

Theirs (co-thr-231.elf, type 12) — all fb=0 kernels are stubs; real kernels are fb=1 only:
| J  | fb | VGPR | SGPR | SCR | LDS | INST | SPILLm |
|----|----|------|------|-----|-----|------|--------|
| 16 | 1  | 89   | 34   | 0   | 64  | 865  | 0      |
| 32 | 1  | 108  | 34   | 0   | 128 | 1134 | 0      |
| 64 | 1  | 157  | 34   | 0   | 256 | 1651 | 0      |
| 128| 1  | 256  | 34   | 8   | 512 | 2744 | 1      |

Notes:
- Theirs is leaner in the same real configs (e.g. J=16 fb=1: 89 VGPR vs our 256, 0 spills vs our 24; J=64 fb=1: 157/1651 vs our 133/4858 — ours uses 3x the instructions). Consistent with the compile-flag delta (their explicit -O3 -DNDEBUG, gnu++17 vs our hipcc-default c++20) plus the different surrounding template environment.
- Ours fb=0 J=48/80 have no real counterpart in theirs (stub on their side — all their fb=0 are stubs); theirs fb=1 J=128 has no counterpart in ours (stub on our side).
- Ours fb=0 kernels spill heavily (up to 680 VGPR spills / 2040B scratch at J=80); theirs real kernels spill 0-1.
- Both use s_clause branch hints + v_dual_mov_b32 (RDNA3 idioms); no mfma, no dp4a.

## Caveats
- Our object was built by ROCm 7.2.0 hipcc and disassembled with ROCm 7.14 llvm tools; their object was built AND disassembled with 7.14 tools. Instruction/resource counts from metadata (llvm-readobj --notes) are authoritative per-object; no cross-toolchain reinterpretation is involved.
- The stub determination (7-instruction immediate-exit body) is from the disassembly, not an assumption.
- Whether a stub is a compiler "couldn't generate" artifact or a deliberate config gate is NOT distinguishable from the object alone; the host-side picker (mmq.cu:141 best-ntiles over J=8..128) is the arbiter of which (J,fb) actually launches.
