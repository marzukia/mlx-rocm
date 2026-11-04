# REPORT-m3-verify-eff: M=3 (S=3) verify pass efficiency — env sweep, two code-first candidates, A/B (t1964, guest-a RX 9070 XT / gfx1201, ROCm 10.1, 2026-11-04)

## Result (one line)

[TO FILL after the Phase 3 battery: best arm, tok/s delta vs the 84.11 BASE, verify-wall delta vs the 22.30 ms BASE, FIRST20 verdict, adopt/reject against the ≥30% wall cut / ≥+10% t/s bar.]

## 1. Mission

Close the M=3 (S=3) verify pass efficiency gap: the S=3 verify pass runs at **41.9% of the HBM floor** (target 66%+). The verify pass is the per-step bottleneck of the 84+ t/s MTP stack (config-of-record: `MLX_MTP=3 MTP6_TAPE=1 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1`, anchor `MTP_MAX_TOKENS=322`, median of 3). Two phases: (1) env-knob sweep at the 322 anchor on the existing wheel; (2) one env-gated code-first candidate per hot kernel class (QMM quantized matmul; GDN fused step), default-off, then a full A/B measurement.

## 2. Setup and identity

Machine: guest-a (<tailnet-ip>, RX 9070 XT / gfx1201, ROCm 10.1 guest userland); flock on `~/gpu.lock`, 25 s idle before/after every arm, SCLK sampled every 10 s. Stack: `stack-assembly-bench` @ b273877, branch `m3-verify-eff-probe`. Wheels: venv-p3 libmlx.so md5 460c803d66c43ebfce8997ef08c912db (the config-of-record wheel, untouched); venv-m3c = hardlink copy of venv-p3 with the candidate wheel installed (libmlx.so md5 74be126a8ec611588922e439edd6fe64; the p3 copy remains 460c803d after install). Model sidecar md5 verified. Harness: `mtp-bench6.py` (anchor) and `roofline-s3-perop.py` (per-class probes), both from the verified stack checkout.

## 3. Phase 1 — env-knob sweep on the existing wheel (median of 3, 322 anchor)

| setting | tok/s | verify ms | S1-corr ms | mlp | qmv_gdn | gdn_fused | attn |
|---|---|---|---|---|---|---|---|
| BASE (MR16 + lx-hoist, config-of-record) | 84.11 | 22.30 | 12.28 | 12.41 | 6.02 | 5.49 | 2.09 |
| MR8 | 82.04 (−2.5%) | 23.10 | 11.94 | 12.81 | 6.24 | 5.62 | 2.16 |
| MR16 | 83.61 (≈BASE) | 22.36 | 12.23 | 12.37 | 6.14 | 5.50 | 2.09 |
| **LX0 (lx-hoist OFF)** | **88.98 (+5.8%)** | **20.02 (−10.2%)** | **11.70 (−4.7%)** | 11.34 | 5.55 | 5.02 | 1.94 |
| MR12 | **BROKEN** (see 3.1) | — | — | — | — | — | — |

FIRST20 for every working setting = `271 1919 ×19` (identical to the acceptance reference; the S=3 loop lock is config-level, see REPORT-acceptance-t3). All rounds rc=0, SCLK stable.

**Finding: the MR in-kernel lx hoist is NET-NEGATIVE on gfx1201 for M≤4 shapes.** Turning it off (`MLX_QMV_MR_LX=0`) gives +5.8% tok/s and −10.2% verify wall with FIRST20 unchanged (the hoist is a math-identical re-read optimization; its cost is the in-k `__syncthreads()` per k-tile, which outweighs the saved lx re-reads at these shapes). MR8 is worse than MR16 (−2.5%): fewer rows per block does not pay on this part.

### 3.1 MR12 defect (pre-existing knob safety bug, documented)

`MLX_QMV_MR_COLS=12` crashes the GPU (`HSA_STATUS_ERROR_MEMORY_FAULT` in `qmv_wide_mr_kernel<hip_bfloat16, hip_bfloat16, 64, 4, true>`, grid=[10944,12,1], workgroup=[32,12,1], faulting addr 0x79a21c000000). Root cause: mcols=12 does not divide N=4096 (nor 8192/12288); the kernel's **weight read is unguarded** (`wr = w + col * row_bytes` for col ≥ N/… → OOB), only the store is guarded by `valid_col`. The MR_COLS knob should either validate `N % cols == 0` or clamp; as shipped it can OOB-fault (and hung the GPU, requiring a reset) for cols ∉ {divisors of N}. Not fixed in this run (out of scope; recorded for the kernel-maintainers lane).

## 4. Phase 2 — code-first candidates (env-gated, default-off)

Two candidates, one per hot kernel class, both default-off (unset = the shipped path runs verbatim, zero behavior change), both bit-identical by construction to the reference path (same values, same FMA order per output element):

### 4.1 QMM: `qmv_wide_mr_nostage_kernel` (new file `qmv_wide_mr_nostage_kernel.hip`, dispatch gate `MLX_QMV_MR_NOSTAGE` in `qmm.hip`)

The staged MR kernel's x-staging is capped at 32 KB of LDS (dispatch gate `mr_lds <= 32768`), so K > 4096 shapes — the 4-bit Qwen3-9B `down_proj` (K=12288, N=4096, 25.2 MB packed per layer) — fall back to the **per-row** `qmv_wide_kernel` at M>1: the 25 MB weight tile is re-read **M times (3× at M=3)**. The nostage variant removes the x staging entirely (x rows, M×K×2 B, are L2-resident and read directly from global), so the 32 KB cap no longer applies: one weight pass for up to 4 rows at ANY K. Per-row math is unchanged (weight tile dequantized ONCE per k-tile, MAC across all M rows; exact per-row FMA order; optional lx hoist with the identical FADD chain order) → every row stays bit-identical to the per-row kernel. Dispatch: only replaces the per-row fallback when `mr_lds > 32768 && M <= 4` (staged-MR shapes untouched); `MLX_QMV_MR_NOSTAGE` unset = per-row path verbatim.

### 4.2 GDN: 2-heads-per-512-thread block (`gdn_fused_step_tape_2h_kernel` + `launch_gdn_fused_step_tape_2h`, gate `MLX_ROCM_GDN_TAPE_2H` in `gdn_fused_step_tape.hip`)

The T=3 tape kernel launches 32 blocks of 256 threads (one per v-head). The 2H variant packs 2 heads per 512-thread block (16 blocks, grid = num_v_heads/2, requires `num_v_heads % 2 == 0`): per-head shared buffers are `[2][...]` indexed by `head_off = threadIdx.x/256`, the 256-thread-internal index is `tith = threadIdx.x % 256` (exactly the original kernel's `tid` for that head's 256 threads), and per-head math is bit-identical (same `tith` mapping, per-head reduction buffers, per-head rms via the index-aware `gdnt_rms_norm_row_i` helper). `MLX_ROCM_GDN_TAPE_2H` unset (or odd head count) = the shipped 256-thread kernel verbatim.

## 5. Phase 3 — A/B battery (3 rounds × 3 arms interleaved + per-class probes, 322 anchor, config-of-record)

Arms: A = venv-p3 + `MLX_QMV_MR_LX=0` (the Phase 1 winner, existing wheel); B = venv-m3c + `MLX_QMV_MR_LX=0` + `MLX_QMV_MR_NOSTAGE=1`; B2 = venv-m3c + all three + `MLX_ROCM_GDN_TAPE_2H=1`. Pre-battery sanity: one anchor per arm, all rc=0, FIRST20 = the 271/1919 reference for all three (candidate wheel + both candidates are token-identical).

[TO FILL: per-arm medians (tok/s, verify ms, S1-corr ms, per-class probe ms), B vs A (isolate QMM nostage), B2 vs B (isolate GDN 2H), SCLK/round stability, FIRST20 across rounds.]

## 6. Adoption decision

[TO FILL: adopt/reject per candidate against the bar (verify wall ≥30% cut OR t/s ≥+10% same-wheel); the LX0 env knob is adopted as the immediate config-of-record change (Phase 1, +5.8% t/s / −10.2% verify wall, existing wheel, FIRST20-identical).]

## 7. Follow-ups

- MR_COLS knob safety: validate `N % cols == 0` (or clamp to a valid divisor) before dispatch; MR12 currently OOB-faults and hung the GPU (3.1).
- lx-hoist: default the hoist OFF for M≤4 on gfx1201 (or make the K-divisibility gate the only path); the in-k barrier cost dominates at these shapes (Phase 1).
- [TO FILL: per-candidate follow-ups from Phase 3.]
