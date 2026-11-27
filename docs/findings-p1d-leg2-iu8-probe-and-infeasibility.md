# P1-d leg 2 findings — iu8 WMMA layout probe + defect-1 fix infeasibility

Date: 2026-11-27. Companion to `findings-p1d-q2k-mmq-root-cause.md` (leg 1).

Leg-2 task: apply the root-cause card fixes (edits #1-#7), build the wheel, land the
G1 verdict (rel <= 1e-3). **Verdict: G1 CANNOT be landed with the card's fix as
specified.** The card's defect-2 premise (C fragment hole) is refuted by a definitive
probe, and the card's defect-1 fix is structurally infeasible in the mmq scale
structure. Details below. No wheel was built with the infeasible fix (a build would
guarantee a G1 fail at rel ~ 1.5 and burn the only wall-optimal build slot).

## 1. iu8 WMMA layout probe — DEFINITIVE (defect 2 refuted)

Probe: `probes/iu8wmma_probe.cu` (built on neon with hipcc --offload-arch=gfx1201,
16.5 s). Design: 68 one-hot-lane patterns (all 32 lanes' 8-byte groups) through the
`v_wmma_16x16x16_ldmatrix` intrinsic, decoded with the kernel's exact load pattern
(A: `get_m h=0` from words k0, k0+2; `h=1` from words k0+1, k0+3; B symmetric).

Result: `probes/iu8wmma_probe.out`. Decoded: **C layout = 93 layout**
(m = 8*(lane>>4) + l, n = lane%16). All 256 cells of the C tile form a complete
16x16 grid with 0 mismatches. Both h=0 and h=1 sides are live (one-hot patterns with
both sides nonzero produce C on both halves — no 8/16 hole).

**Consequences:**
- Card defect 2 (8/16 C-fragment hole for iu8) is REFUTED. Edits #5/#6 (the conditional
  C-layout fixes) are NOT needed — exactly as the card itself conditioned them on the
  probe showing a difference.
- F2 (the "zero rows 8-15" signature) is NOT a C-fragment artifact. It is a product of
  the defect-1 transposed read: the loader reads block (kb0*N + i) per tile row i, and
  the kernel's 256 "K" values are actually one block's 256 N-values. The zero rows are
  a staging/scale artifact of that misread, not a builtin hole.
- The existing C writeback (i = i0 + 16n + 8h + l, row r -> lane r%8 (h=0) and
  r%8+16 (h=1), word r/8) is CORRECT and matches the probe. Keep it.

## 2. Defect 1 — the file is [K, N/256]; the fix as written is INFEASIBLE

Ground truth (kqfg, bit-exact): `blk = w + (uf*NBS + cf)*bb`, uf = k in 0..K-1,
cf = 0..NBS-1: block (k, cf) = 256 N-values of ONE K-row. Weight(n, k) =
block(k, n/256) element (n%256).

The mmq x-tile is 64 N-rows x 256 K. For N-row n, the 256 K-values live in **256
different blocks** (block (kb0*256 + k, n/256) for k = 0..255), one code + one
scale-group + one (d, min) pair per K-element.

The card's fix says: read the per-k block, stage 1 code per sram element (256 K ->
64 words — fits), and read scale/dm "from the per-k block (scale group elem/16 and
that block's dm)".

**The scale does not fit.** The vec-dot applies scale post-mma at one slot per 16 K
(`x_dm[i*sram_stride + k0/4]`, k0 = 0..63 -> 16 slots, slot s covers K 16s..16s+15).
In the file layout, the 16 K of slot s come from 16 different blocks with 16 different
(d, sc, min, mn) values. One half2 slot holds one (d*sc, min*mn) pair. Per-k scale
requires 256 slots per row (512 halfs = 256 words); the x_dm region has 32 words
(64 halfs). 8x too small.

**The iu8 builtin cannot carry it either.** Post-mma scaling is per-16-K (the mma sums
over k; a per-k f16 factor like d_k*dy_k cannot be applied after the sum). The iu8
intrinsic is integer-only (A and B are iu8), so a per-k f16 scale (d_k, min_k) cannot
enter the mma at all. The only iu8 escape is 1-K-per-mma-call (15 zero padding,
~3 mma per K, ~3.1M mma for a 9B GEMV — feasible in instruction count and 16 KB sram
at I=16) but that is a full rewrite of the Q2_K mmq path, not edits #1/#2.

Requantization in a pre-pass (gather 256 K-values per N-row, repack to a standard
[N, K/256] Q2_K block) loses precision: the 256 gathered values sit on 256 different
2-bit grids (one per source block); repacking onto 16 new group grids gives
~10-30% per-element error -> GEMV relative error ~ 0.1, 100x over the 1e-3 gate. Dead.

## 3. What WOULD work (for a follow-up leg, not this one)

Option A — f16/bf16 WMMA rewrite of the Q2_K GEMV path:
- A[m][k] = q_k * sc_k[m/16] (f16, exact for 0..45); B = y-code (f16, exact ints).
- Per-k f16 pre-scales: dy' = d*dy, my' = d*my, dy'' = min*dy, my'' = min*my
  (4 f16 per K = 512 words/row at I=16; total ~ 37 KB at I=16 -> I=8 with half the
  C-tile wasted, ~18.6 KB, or a tighter budget).
- 1-K-per-mma-call (15 zero pad), 2-3 mma per K, ~3.1M mma for 9B GEMV (~2-3 ms).
- Cost: full rewrite of Q2_K loader + vec-dot + a new I=16/8 config case + writeback
  mapping. ~400-700 lines of new device code. F16 double-rounding per B-term puts the
  expected GEMV rel error at ~1e-3 (marginal vs the gate).
Option B — iu8 1-K-per-mma variant: same shape, integer A (q*sc, mn), 3 mma per K,
  per-k f16 scales stored in sram (16 KB at I=16), integer-exact mma (error only from
  the stored f16 pre-scales, ~5e-4 per term — a better G1 margin than option A).
Both are a new leg with its own build budget, not the card's 7 edits.

## 4. G1 verdict

**FAIL (not landable as specified).** The card's defect-1 fix cannot represent per-k
scale in the mmq scale structure; any build of it would ship ~1.5-2.0 relative error
(3/4 wrong scale factors per word + the K-coverage fix only). The C-layout half of the
card (defect 2, edits #5/#6) is refuted and was never needed.

Practical note: for M=1 GEMV (the G1 regime), the kqfg path is the ground-truth
kernel and serves it exactly (0 error vs itself). The mmq port's value is speed at
M>1 / GEMM; that requires the option A/B rearchitecture leg.

## Reproducibility

- Probe source: `probes/iu8wmma_probe.cu` (one-hot-lane design, 68 patterns).
- Probe output: `probes/iu8wmma_probe.out` (full 68-pattern decode, 0 mismatches).
- Decoder: `probes/decode_iu8_probe.py` (decodes (m, n) per cell, checks 16x16 grid
  completeness + 93/77 layout hypotheses).
- Build on neon: `hipcc -O3 -std=c++17 --offload-arch=gfx1201 -x hip` (16.5 s).
