# REPORT-m3-verify-eff: M=3 (S=3) verify pass efficiency — env sweep, two code-first candidates, A/B (t1964, guest-a RX 9070 XT / gfx1201, ROCm 10.1, 2026-11-04)

## Result (one line)

Best arm B (venv-m3c + `MLX_QMV_MR_LX=0` + `MLX_QMV_MR_NOSTAGE=1`): **91.29 t/s** median (+8.5% vs the 84.11 BASE), verify wall **18.95 ms** (−15.0% vs the 22.30 ms BASE), FIRST20 identical in all 12 runs; both code candidates **REJECTED** against the bar (QMM nostage: +1.8% t/s / −3.8% wall vs A; GDN 2H: −0.2% / +0.6% vs B), the **LX0 env knob ADOPTED** (arm A 89.70 t/s = +6.6% vs BASE, wall −11.7%, existing wheel).

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

Anchors (median of 3, round-interleaved R1: A B B2 / R2: B2 A B / R3: B B2 A, 322 anchor, config-of-record envs + per-arm extras):

| arm | r1 | r2 | r3 | median t/s | verify ms | S1-corr ms |
|---|---|---|---|---|---|---|
| A (venv-p3, LX=0) | 89.86 | 89.70 | 89.53 | **89.70** | 19.70 | 11.69 |
| B (venv-m3c, LX=0 + NOSTAGE) | 91.34 | 91.29 | 91.01 | **91.29** | 18.95 | 11.69 |
| B2 (venv-m3c, + TAPE_2H) | 91.09 | 91.12 | 90.86 | **91.09** | 19.06 | 11.69 |

Per-class probes (`roofline-s3-perop.py`, per-class wall ms):

| probe | verify | S1-corr | mlp | qmv_gdn | gdn_fused | attn |
|---|---|---|---|---|---|---|
| A | 19.70 | 11.69 | 10.96 | 5.54 | 5.02 | 1.94 |
| B | 18.95 | 11.69 | 10.23 | 5.59 | 5.06 | 1.94 |
| B2 | 19.06 | 11.69 | 10.29 | 5.52 | 5.07 | 1.94 |

**B vs A (isolates the QMM nostage kernel): +1.8% t/s, verify wall −0.75 ms (−3.8%)** — the win sits entirely in the mlp class (down_proj, 10.96 → 10.23 ms, −0.73 ms); qmv_gdn is +0.05 ms (noise), S1-corr unchanged. The K>4096 per-row fallback weight re-read was far cheaper than the design assumed: at M=3 the 25.2 MB packed down_proj weight is mostly L2-resident across the M passes, so removing the extra 2 re-reads saved ~0.7 ms of a 19.7 ms wall.

**B2 vs B (isolates the GDN 2H kernel): −0.2% t/s, verify wall +0.11 ms (+0.6%)** — gdn_fused 5.06 → 5.07 ms = net-neutral. The shipped T=3 launch is already 32 blocks × 256 threads (one per v-head, one per CU on the 32-CU part); packing 2 heads per 512-thread block buys no occupancy and the per-head `[2][...]` shared-buffer indexing adds a small overhead.

Stability: round spreads 0.37% (A) / 0.36% (B) / 0.29% (B2); SCLK 71/71 samples at 2200 MHz (10 s cadence, 16:48:36–17:00:27Z, all under load); 25 s idle before/after every arm; all 12 runs rc=0; FIRST20 = `271 1919 ×19` in every anchor. Provenance: the battery completed on guest-a before the t1964 cancel (16:48–17:00Z); the runner nohup log was lost in the cancel, but all 12 raw files were verified complete (full output through the TEXT/ALLTOKS tail, FIRST20 reference, probe RESULT_JSON present) and the wheel md5s (460c803d / 74be126a, r4v 8acf13fe untouched) + model sidecar md5 were re-verified before this run; one sanity anchor was re-run per arm (B 91.28 / B2 91.05 / A 89.80 t/s, all rc=0, FIRST20 = the 271/1919 reference) and matches the battery within round spread.

## 6. Adoption decision

Bar: verify wall ≥30% cut OR t/s ≥+10% same-wheel, per candidate; FIRST20 is the silent gate (all arms token-identical, so the gate never fired).

- **QMM nostage (`MLX_QMV_MR_NOSTAGE`): REJECT.** B vs A = +1.8% t/s / −3.8% wall — both legs of the bar missed by a wide margin. The kernel is bit-identical by construction and default-off, so it causes no harm; it simply does not pay at M=3 on gfx1201 (the re-reads were L2-cheap). Best available stack number stays arm A = 89.70 t/s (existing wheel + LX0); B's 91.29 (+8.5% vs BASE) is below the +10% bar, so no code change is adopted into the config-of-record.
- **GDN 2H (`MLX_ROCM_GDN_TAPE_2H`): REJECT.** B2 vs B = −0.2% t/s / +0.6% wall — net-neutral-to-negative; default-off stays.
- **LX0 env knob (`MLX_QMV_MR_LX=0`): ADOPT** as the config-of-record env change — Phase 1 (+5.8% t/s / −10.2% verify wall) and the battery (arm A 89.70 t/s = +6.6% vs BASE, wall −11.7%) agree, existing wheel, FIRST20 identical.

## 7. Follow-ups

- MR_COLS knob safety: validate `N % cols == 0` (or clamp to a valid divisor) before dispatch; MR12 currently OOB-faults and hung the GPU (3.1).
- lx-hoist: default the hoist OFF for M≤4 on gfx1201 (or make the K-divisibility gate the only path); the in-k barrier cost dominates at these shapes (Phase 1).
- QMM nostage: the re-read saving scales with M and shrinks with L2 residency, so the kernel becomes interesting only where the K>4096 fallback persists AND the weight tile is not L2-resident (larger M; wider-down projections; higher-precision weights). At M=3 / 4-bit / gfx1201 the measured win is 0.75 ms of a 19.7 ms wall. Keep the gate default-off; do not re-measure at M=3.
- GDN 2H: the T=3 launch is already at full CU occupancy (32 v-heads on 32 CUs), so 2H's premise (under-occupied 256-thread blocks) does not hold; the measured cost is the per-head shared-buffer indexing. If revisited, profile launch/occupancy on a part with >32 CUs or at a T where block count < CU count before further work. Keep default-off.
- Config-of-record update from this run: add `MLX_QMV_MR_LX=0` (LX0 adopted); best measured stack 89.70 t/s (120.4% of AR 74.44), verify wall 19.70 ms.
