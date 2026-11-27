# A/B lane - P1 standalone result (Q4_K dp4a dot at the exact hang shape)

Record: t3598 (targeted probe, the (a)-round follow-up per piboy ruling 2026-11-27 14:29Z) was cancelled clean by the warden cancel-80 fallback at 17:46:01Z (null wall -> 3h13m from the 14:33Z start). The P1 GPU session had been relaunched at 17:41:56Z and completed its runs before the kill; the orchestrator recovered the session output, fixed the last script bug (a `.sqrt()` on a numpy scalar), and re-ran the session at 17:48:30Z and 17:53:23Z to get the full P1 verdict lines. This card is the banked P1 state of record for the P2 continuation.

## P1 spec (from the 14:29Z ruling)

ARM P1 = their Q4_K dp4a dot (lib-ab3, their 3 .cuh + our mma.cuh + zero-init/I_MAJOR/93-index re-applied vec-dot 0435c478) IN ISOLATION at the EXACT hang shape: M=512 N=8192 K=4096 (the first [cfgq4k] config from the ab3 in-model hang). Direct ctypes call to mmq_ab_gemm on a 512x4096 bf16 x (ref2-x.npy) vs the dequantized W (bfiledump-N8192k4096.bin, refgen2 math) CPU reference. Verdict lines: P1-DONE (wall, corr) or P1-HANG (timeout rc).

## Provenance (binding)

- Box: the GPU box dGPU (HIP_VISIBLE_DEVICES=0, gfx1201 16GB), ROCm 7.2.4 (new724 tree), venv /home/guest/venv-hbmfix.
- lib-ab3.so + core-ab3.so md5 4738da138f1e0d8613fa90da06a73355 (identity printed in-session, P1-IDENTITY line); venv SoT lib 070122d8b617093da104a1b5626cadbd saved/swapped/restored by the session script, md5-verified.
- GPU lock /home/guest/gpu.lock held by the session (flock); lock free + /dev/kfd clean after P1-END.
- Harness: /home/guest/gemm1-leg2/{p1-probe.py, p1-session.sh, p1-session.out, p1-j128.log, p1-j64.log}.
- Runs: J=128 (natural) and J=64 (MLX_MMQ_J=64), env -i, HCC_AMDGPU_TARGET=gfx1201, MLX_MMQ_CFG=1.

## P1 result (verbatim, 17:53:23Z run)

```
P1-START 2026-11-27T17:53:23Z
SAVED lib=070122d8b617093da104a1b5626cadbd
SWAPPED lib=4738da138f1e0d8613fa90da06a73355
P1-RUN-j128 2026-11-27T17:53:24Z
P1-SYNC rc=0 wall=0.001
P1-CORR maxrel=1 rmsrel=1.031
P1-OUTSTAT out_absmax=5.204 frac_zero=0.9375 out8=[-0.3069 -1.4104 -1.3136 -0.5027 -1.2693  1.1578  0.6176 -1.3619] y8=[ 1.7792 -0.196  -0.5723 -0.3802  0.8816  0.3409  1.4536  0.3712]
P1-DONE wall=0.001 verdict=CHECK
P1-RUN-j64 2026-11-27T17:53:25Z
P1-SYNC rc=0 wall=0.000
P1-CORR maxrel=1 rmsrel=1.031
P1-OUTSTAT out_absmax=5.204 frac_zero=0.9375 out8=[-0.3069 -1.4104 -1.3136 -0.5027 -1.2693  1.1578  0.6176 -1.3619] y8=[ 1.7792 -0.196  -0.5723 -0.3802  0.8816  0.3409  1.4536  0.3712]
P1-DONE wall=0.000 verdict=CHECK
P1-END 2026-11-27T17:53:26Z
```

The [abq4k] config lines from the runs: `M=512 N=8192 K=4096 ne0=4096 ne1=512 nb0=2 J=128 I=128` (natural) and `... J=64 I=64` (forced). The mmq_ab_gemm hook returned ok (P1-HOOK-RC 1 = c_bool true) in both runs.

## Reading

- NO HANG: at the exact in-model hang shape, the dot standalone launches, the hook returns ok, and hipStreamSynchronize completes in 0.000-0.001s. The P1-HANGS branch of the pre-registered decision tree (dot intrinsic-dead in our framework -> P3 cross-feed + CPU lines) is NOT taken. P3 is dropped from the continuation.
- CORR is BAD: maxrel=1.0, rmsrel=1.031 against the <2e-2 PASS bar -> verdict CHECK, not PASS.
- The failure shape is a PARTIAL WRITE: frac_zero=0.9375 = exactly 15/16 of the output buffer is zero; 1/16 of the elements carry O(1)-class values (out_absmax 5.204, y_ref absmax 16.65). The nonzero pattern is deterministic and byte-identical between the J=128 and J=64 runs (J does not change which 1/16 lands).
- maxrel landing exactly 1.0 is consistent with the nonzero 1/16 being a small subset of wrong values, not a scaled version of the reference (a pure scale c would give maxrel = rmsrel = |c-1|; here 1.0 vs 1.031).
- The standalone dot at the hang shape therefore EXHIBITS A DEFECT (1/16 partial write) in ~1ms where the in-model context EXHIBITS A HANG at the same shape: same shape, different symptom. The discriminating question the P2 six-class run answers: is the 1/16 pattern shape-specific (only M=512 N=8192 K=4096) or universal across the standalone context (all six trio classes).

## P2 pre-registered outcome map (per the 14:29Z ruling)

- (i) all six classes full-corr PASS -> the standalone oracle is faithful and the hang shape is special: the 1/16 partial write is shape-specific -> name the locus (which launcher arg / LDS / grid / shape triggers the in-model hang and the standalone partial write) via a code read of our launcher arg computation for the ab kernel vs the hook call, CPU file:line.
- (ii) the 1/16 (or any partial-write) pattern on all six classes -> the standalone context is not faithful (harness or kernel tiling in isolation) -> the hang is PLUMBING (full-GEMM context), the plumbing statement carries the degraded-oracle caveat.
- (iii) mixed -> per-class locus table.

Either way the terminal card carries: per-class corr + OUTSTAT lines, the plumbing statement (or the locus name), the CPU lines (compile/link flag audit of our mmq TU vs theirs; ISA extract of the dp4a dot from both .so - ab3 vs their harness512 build /home/guest/perclass-probe/llama-harness512 md5 fbc898cf - VGPR/scratch/spills/instruction count), and the decision-tree outcome (i)/(ii)/(iii) named.

## Box hygiene

After the 17:53:23Z run: LOCK-FREE, /dev/kfd empty, venv restored to SoT 070122d8. The probe's P3 pieces are located for reference: /home/guest/perclass-probe/llama-harness512 (their M=512 harness binary) + /home/guest/perclass-probe/Qwen3.5-9B-xfeed-wq4.gguf (transposed GGUF, our-orientation bytes) - P3 dropped per the tree, pieces banked in case a re-fire is ruled.
