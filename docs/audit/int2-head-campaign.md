# Stream 2 — int2 draft-head campaign (digest)

Consolidated narrative digest for the int2 (2-bit) MTP draft-head lane: port (R4c) -> vectorization (int2w) -> VGPR v2 -> VGPR v3 -> step-level A/B -> cheap-head follow-up. The root REPORT-*/NOTES* files of this stream (REPORT-mtp-r4c- int2-head.md, REPORT-int2-wmma.md, REPORT-int2-wmma-vgpr.md, REPORT-int2- wmma-vgpr-v3.md, REPORT-int2w-step.md, REPORT-cheap-head.md, NOTES-int2vgpr.md, NOTES-int2vgpr-v3.md) are consolidated into this digest (deleted by the PR). Box: guest-a (RX 9070 XT / gfx1201, 32 CU, ROCm 10.1); cheap-head ran on neon (same GPU, bare metal). Model: qwen3-9b-mlx (4-bit lm_head, group 64, N=248320, K=4096). Protocol: flock, 25 s idle, sclk locked high, medians of 3, explicit per-arm head selection.

## 1. R4c — scalar int2 port: lossless, wheel-clean, LOSES the speed gate

The int2 g128-asymmetric draft head (quarter-split 2-bit pack, per-group `s = max((hi-lo)/3, 1e-8)`, `z = clamp(round(-lo/s), 0, 3)`, dequant algebra `out += dot_g*s - sx_g*(z*s)`) was ported as a new scalar kernel (`lm_head_int2`, grid (N/256)x256, one column per thread, scalar dequant-MAC, 8-round shared-memory top-8 selection), with host guards hardened to `N % 256 == 0` and `K % 512 == 0` (the old N%64 guard left two hazards: dead shared columns winning top-8 slots, and dead sub-blocks writing past the [N/64, 8] candidate arrays).

- **Head-traffic probe (the diagnosis):** 4-bit shared head 1 row = 963 us
  (probe) / 949 us (warm xcheck); int2 scalar = 19200 us / 18743 us — **19.75x slower** in a tight warm loop (reproduced as 18.59 ms in later lanes). The kernel is compute-bound (54 GMAC/s effective; the 242 MB packed read would take ~0.5 ms at HBM speed): the 2-bit traffic saving never engages at M<=4. argmax agreement 201/256 (78.5%); max abs logit diff 1.935.
- **A/B (322-iter anchor):** OFF 48.93 / ON 27.32 (0.56x) / CTRL 48.93 / CTRL2
  26.13; 512v protocol: OFF 40.31 / ON 24.23 (0.60x; ON rows PENDING
  re-measure — the phase bench's reference quantizer had been running a broken `scale = 4*s, zs = (4+z)*s` revision that rescales every group 4x plus a per-element offset, distorting the logit ranking; the reference was corrected to `s`, `z*s` to match the kernel).
- **Lossless PASS** (FIRST20 identical across all 12 MTP=4 runs + 2 AR runs; AR
  ON 64.56 vs OFF 64.42; the task's 74.4-class AR expectation did not reproduce for the canonical wheel either — session drift, recorded).
- Verdict: **NO ADOPT** (scalar int2); the lever is OFF by default. Follow-up
  identified: a vectorized int2-dequant kernel.

## 2. int2w — vectorized int2: 9.73 ms/row (10.2x), gate fail, gap identified

`lm_head_int2_wide`: block (32,32) = 1024 threads (the 2048-thread variant is not launchable — AMD workgroups cap at 1024), 2 blocks/CU, x staged f32 in LDS, per-element decode fused into the MAC loop, per-32 top-8 selection (strictly refines the scalar's per-64 top-8; `sel_match = 0/256` is a structural artifact, not a bug). Per-row (warm loop): **9.73 ms = 10.2x the 4-bit anchor (0.953 ms) — GATE FAIL**; per-column logits are **bit-identical** to the scalar port (max abs/rel diff 0.0 over 16 random rows). Counter evidence: full occupancy (2048/2048 threads), not weight-bandwidth-bound (26 GB/s = ~24.5x under the 640 GB/s HBM peak); the structural gap = **128 per-element LDS x-reads per lane per column** (the 4-bit kernel preloads x into VGPRs, 0 LDS reads in the FMA loop) + 8 extra 4-lane `__shfl_xor` group reductions. Identified next lane: preload the lane's 128 x-values into VGPRs.

## 3. VGPR v2 — 1.35 ms (1.42x), 7.2x faster, occupancy lost the row

The 128-value VGPR preload (`float xv[4][32]`, staging hoisted out of the grid-stride loop) cut per-row to **1.354 / 1.351 ms (1.42x the anchor — GATE FAIL)**. The inner loop won: per-column int2 174.5 ns vs 4-bit 184.4 ns (5.4% faster per wave; 254 MB at 1.354 ms = 188 GB/s vs 533 GB/s). The occupancy lost the row: 192 VGPR/lane (spilling) -> 1 block/CU (32 waves vs 48); row cost = (48/32) x (174.5/184.4) = 1.42x, exactly the measured gap. Next lane: get to <= ~96 VGPR to restore 2 blocks/CU (~0.68 ms/row projected).

## 4. VGPR v3c — 570.8 us (0.60x the anchor): GATE PASS

v3c: 16 KB bank-spread x layout (`k2 = p*1024 + 32j + 4gl + l3` for `k = p*1024 + 128gl + 32l3 + j`; a 32-lane load at fixed (c,b) hits 32 distinct banks in one LDS cycle), 32-byte weight window preloaded once per column (2 x 16B, L1-hot; the per-chunk u32 load had made the compiler re-issue the window 8x redundantly — 9664 us/row), 74 VGPR / 22 SGPR / 0 scratch, 2 blocks/CU (64 waves). **570.8 / 570.4 / 571.8 us = 0.60x the anchor (953 us), 2.39x faster than v2, 2.01x faster than v3b (1149.7 us); GATE PASS.** 16-row logit bit-identity re-verified (max_abs_diff = 0.0). Key engineering finding: **occupancy on this driver is structure-sensitive, not a function of the .s metadata counts** — an empirical sweep (512-thread blocks, LDS = 0) showed v3t (86 VGPR / 24 SGPR, fewer of every observable resource than v3i's 89/26) capping at 48 waves while v3i fit 64; no single-resource cap explains both rows. Every candidate therefore had to be verified with the occ probe before a wheel build.

## 5. Step-level A/B (t1796) — 0% draft acceptance: NO ADOPT

Arms (R4b config of record, 512-verify protocol): A (4-bit head) **40.30 t/s**, 2.438 tok/step, D1 72.9%, 1628 tokens; B (int2w head) **27.07 t/s**, 1.000 tok/step, **0% acceptance**, 1024 tokens. Ladder 0.67x (-32.7%). Lossless gate FAIL (reported with reason): first divergence at token 32 — B's stream is the 1-token/step greedy-equivalent stream. Diagnosis (three independent ways, 64 greedy AR hidden states, same wheel): **P1** the int2 draft head (both ops, with the bench's 4-bit re-rank) matches the 4-bit verifier's argmax **0/64** steps (both zero-point conventions tested); **P2** the MTP state is healthy (4-bit d1 = 56.2% under the B-state) — the failure is isolated to the int2 head; **P3** a manual float int2 GEMV from the exact int2 values also matches 0/64, and the wide top-32 candidate set covers the 4-bit argmax 0/64. Root cause: **the 2-bit g128 quantization itself is a poor approximation of this 4-bit lm_head** (consistent with the 1.935 max abs logit diff) — a quantizer change (group size / level count / verifier-matched quantizer), not a kernel change, would be needed. The 570.8 us kernel-speed win says nothing about draft acceptance; the acceptance gap dominates: 0.67x end-to-end. **NO ADOPT** for int2w as the MTP draft head; the lever stays an explicit opt-in, off by default.

## 6. Cheap-head follow-up (neon) — 4-bit MTP module ADOPTED; int2 confirmed dead

K=3 chain per-op + draft-chain wall + 256-token acceptance runs (config B regime), across venv-r4v / venv-kfix / venv-int2w:

- **4-bit MTP module: ADOPT.** The sidecar MTP module rebuilds quantized
  (`CH_MTP_4BIT` — NOT the wheel's `MTP_4BIT`, which would shape-mismatch the bf16 sidecar at load): mtp traffic 2.92 GiB -> 411 MB per call; mtp_head 1.81-1.84 -> 0.49-0.69 ms; chain -3.61 ms on r4v (-33%) and -4.0 ms on the kfix-class (-42%); acceptance r4v +8.1% (47.52, D2 63.5, D3 54.1), kfix-class +1.7/+2.6%. Caveat: on the kfix-class wheel D1 drops 5.5pp (76.6 -> 71.1) and one stop-boundary token divergence per 400 — D drift not strictly neutral on that wheel; the r4v shipping wheel is the adoption basis.
- **int2w: NOT ADOPT** — 0% draft acceptance (D1=D2=D3=0), tok/s -37%, chain
  only -1.58 ms. The t1796 zero-acceptance finding reproduces on the new wheel: int2 draft quality for this 9B model is zero.
- **Batched argmax: INFEASIBLE as specified.** The K=3 chain is a strict
  serial dependency chain (`mtp_k` requires argmax of step k-1; the MTP input embedding depends on the previously SELECTED token — no linear-algebra pass materializes mtp_2/mtp_3 without the selections). The stated savings do not exist in the measured regime: launches are asynchronous (no per-call wall to remove), and the only input shared by the 3 GEMVs is the head weight matrix, which is read once PER ROW (an M=3 call reads 3 x (K x N/4) bytes — no traffic saved). Measured: M=3 ~= 2.2x a single M=1 head on r4v.
- **144 t/s contribution:** at tok/step 2.4 the spec step wall must be <= 16.7
  ms; measured spec step wall 45-49 ms. 4-bit-MTP removes ~4.0 ms of the ~28.5 ms gap (~14%); the binding constraint is the **verify/commit wall (~27.5 ms)** — a different lever. The draft chain (5.57 ms) is inside the 3-6 ms budget once 4-bit-MTP is adopted.

## Verdicts in one line

int2 scalar: lossless + wheel-clean but 19.75x per row, 0.56x A/B — NO ADOPT; int2w vectorized: 10.2x, gap = per-element LDS x-reads; VGPR v2: 1.42x, occupancy lost the row; VGPR v3c: **570.8 us = 0.60x, GATE PASS** (bit- identical); step-level A/B: **0% draft acceptance, 0.67x — NO ADOPT, the 2-bit quantization is too coarse for top-1 on this head**; 4-bit MTP module: **ADOPT** (chain -33% to -42%, r4v basis, drift caveat on kfix-class); batched argmax: INFEASIBLE (serial chain).
