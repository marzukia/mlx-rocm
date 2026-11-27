# A/B lane - P2 deliverable card (their Q4_K dp4a dot standalone at the six trio classes + CPU lines)

Record: the P2 continuation of the Q4_K A/B lane per the 14:29Z ruling (P1 banked at e589d3a). P2 GPU session 2026-11-27T18:00:10Z-18:00:38Z on the GPU box dGPU (HIP_VISIBLE_DEVICES=0, gfx1201 16GB), flock on /home/guest/gpu.lock. CPU lines (flag audit + ISA extract) from the box build trees, no GPU.

## Provenance (binding)

- lib-ab3.so md5 4738da138f1e0d8613fa90da06a73355 (identity printed in-session on every class, P2-IDENTITY lines); venv SoT lib 070122d8b617093da104a1b5626cadbd saved/swapped/restored, md5-verified (SAVED/RESTORED lines).
- Their build of record: /home/guest/perclass-probe/llama.cpp, libggml-hip.so.0 -> .0.28.0 md5 a2f1e90e7a9e41ed34025643280155bb (built Nov 26 14:59, clang-23/rocm-7.14, dual-arch gfx1201+gfx1036). This is the exact .so the P1-named "harness512 build" loads: llama-harness512 (md5 fbc898cfb6367c3944eff62dfee3e867) is a dynamically-linked executable with no embedded AMDGPU code; `ldd` resolves libggml-hip.so.0 -> /home/guest/perclass-probe/llama.cpp/build/bin/libggml-hip.so.0.
- ISA tools: /usr/lib/rocm/llvm/bin/llvm-{objdump,readobj,nm} = clang 23.0.0 / ROCm 7.14 = the SAME toolchain as their build. Our object was built by ROCm 7.2.0 hipcc (CMAKE_PREFIX_PATH=/opt/rocm-7.2.0); disassembled with the 7.14 tools (caveat noted in the extract).
- Box state after P2-END: venv imports mlx.core OK, /dev/kfd present (no leftover contexts), lock released.
- Artifacts: probes/ab-q4k-p2/{p2-probe.py, p2-session.sh, p2-results.txt, p2-flag-audit.md, p2-isa-extract.md}.

## P2 spec

Their Q4_K dp4a dot (lib-ab3: their 3 .cuh + our mma.cuh + the re-applied vec-dot adaptations, zero-init/I_MAJOR/93-index) IN ISOLATION via the mmq_ab_gemm hook, at the SIX trio classes (M=512, SoT shapes: N=1024/4096/12288/8192/12288/248320 x K=4096, plus N=4096 x K=12288), corr vs the refgen2-math dequantized CPU reference, before any timing.

## P2 result — the six-class table

| class          | N x K      | sync wall | maxrel | rmsrel | frac_zero | verdict |
|----------------|------------|-----------|--------|--------|-----------|---------|
| n1024k4096     | 1024x4096  | 0.001     | 1.198  | 1.226  | 0.5000    | CHECK   |
| n4096k4096     | 4096x4096  | 0.000     | 1.279  | 1.061  | 0.8750    | CHECK   |
| n4096k12288    | 4096x12288 | 0.000     | 1.000  | 1.061  | 0.8750    | CHECK   |
| n8192k4096     | 8192x4096  | 0.001     | 1.000  | 1.031  | 0.9375    | CHECK   |
| n12288k4096    | 12288x4096 | 0.000     | 1.290  | 1.020  | 0.9583    | CHECK   |
| n248320k4096   | 248320x4096| 0.001    | 1.161  | 1.001  | 0.9979    | CHECK   |

- 6/6 CHECK, 6/6 no hang, 6/6 hook ok, 6/6 sync rc=0. Every run: hook config line `J=128 I=128 nb0=2` (the runtime's natural J selection).
- The partial-write signature is universal and sharper than P1's 1/16: the NONZERO count is a fixed ~512 rows on every class (1024*0.5, 4096*0.125, 8192*0.0625, 12288*0.0417, 248320*0.0021 all = ~512) — one fixed block of the output is written, the rest stays zero, regardless of N.
- n248320k4096 (the large-N class; 23s run wall, dominated by the 3.97M-block CPU dequant) does NOT hang: launch->sync 0.001s.
- The n8192k4096 output is byte-identical to P1's exact-hang-shape run (same out8/y8 bytes) — the standalone defect is stable across sessions.

## Decision-tree outcome: (ii)

Per the pre-registered map (P1 card): (ii) the partial-write pattern on ALL six classes -> **the standalone context is not faithful (harness or kernel tiling in isolation) -> the hang is PLUMBING (full-GEMM context), carrying the degraded-oracle caveat.**

The degraded-oracle caveat, made concrete by the CPU lines: the runtime selects J=128 (the hook prints it on every run; the host picker loops J=8..128 by best-ntiles, mmq.cu:141), and the ISA extract shows **J=128 is a no-device-code stub in OURS for BOTH fallback values** (7-instruction immediate-exit kernel; their J=128 is real only for fb=1). A stub main kernel writes nothing; the fixed ~512 nonzero rows therefore come from a companion path (e.g. the stream_k_fixup kernel, which has real code in both builds), not the dp4a dot. Hypothesis, labeled as such — the object-level facts (stub map, resource table) are in the extract. Consequence: the standalone "no hang" is REAL but WEAK evidence — the standalone dot does not exercise the J=128 arithmetic in our build, so the in-model hang is attributed to the full-GEMM context (plumbing) on the strength of the outcome map, not on a faithful standalone oracle.

## CPU line 1 — compile/link flag audit (one-line deltas)

- Compile: **hipcc wrapper + gfx1201-only + c++20 + default opt + 1 define** vs **direct clang++ + gfx1201/gfx1036 dual + gnu++17 + explicit -O3 -DNDEBUG + 61 defines (incl. GGML_HIP_NO_VMM, GGML_SCHED_MAX_COPIES=4, USE_PROF_API=1)**; no -mllvm on either side; the 49 FA-related defines select FA quant pairs only and do not touch the mmq Q4_K path.
- Link: plain `c++ -shared` over a static hip archive + ROCm runtimes (no --hip-link, no rtlib/unwindlib selection) vs `clang++ --hip-link --rtlib=compiler-rt -unwindlib=libgcc` with the dual-arch flags repeated.
- Verbatim lines + build-tree provenance: probes/ab-q4k-p2/p2-flag-audit.md.

## CPU line 2 — ISA extract (key findings)

- The "dp4a" in `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a` is a tile-LAYOUT name (`mmq_get_dp4a_tile_x_sizes`); the inner product is `vec_dot_q4_K_q8_1_impl_mmq` (f32/half). ISA confirms: **0 v_dp4a and 0 v_mfma instructions in BOTH Q4_K gfx1201 objects** — the Q4_K path is pure VALU (f32/mix + integer dequant) in both builds.
- Per-(J,fb) device-code map (full table in the extract): J=128 stub/stub in ours vs real(256 VGPR, 2744 inst)/stub in theirs; theirs fb=0 entirely stubs; ours fb=0 real at J=16/32/48/64/80 with heavy spills (up to 680 VGPR spills / 2040B scratch at J=80); in the same real configs theirs is leaner (J=16 fb=1: 89 VGPR / 865 inst / 0 spill vs our 256 / 1732 / 24).
- Both objects use RDNA3 idioms (s_clause branch hints, v_dual_mov_b32).
- Full extract + method + caveats: probes/ab-q4k-p2/p2-isa-extract.md.

## VERDICT

**Outcome (ii), with the degraded-oracle caveat.** Their Q4_K dp4a dot in our framework, standalone at all six trio classes: no hang anywhere (incl. the large-N class, sync 0.001s), but 6/6 CHECK with a universal fixed-~512-row partial write (frac_zero 0.5 -> 0.9979), byte-stable across sessions and J. The standalone oracle is NOT faithful — the runtime-selected J=128 main kernel is a no-device-code stub in our lib-ab3 build — so the in-model hang is attributed to the full-GEMM context (plumbing: launcher/grid/shape interaction in the in-model GEMM path), not to the dot intrinsic. The A/B comparison of the dot arithmetic itself is inconclusive from standalone runs in our build (the J=128 arithmetic never executes); a faithful A/B would require either a non-stub config (J<=64 forced, fb per the map) or the in-model context.

## Next-lane suggestion (not ruled)

To close the A/B on the arithmetic: force MLX_MMQ_J to a real config on both sides (ours: J=64 fb=1 or J=80 fb=0; theirs: J=128 fb=1 or J=64 fb=1 — same (J,fb) pair) at one or two trio classes, and check whether ours still partial-writes with a real main kernel; plus a host-side audit of why the compiler (rocm-7.2.0 hipcc, default opt, c++20, our .cuh mix) emitted no-device-code stubs for J=128 while theirs (clang++ -O3, gnu++17) did not — the launch-bounds/occupancy macro values in the mixed-tree mmq.cuh are the suspect.
