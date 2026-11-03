# MTP fix round 1 — report

Date: 2026-11-03. Guest: guest-a (RX 9070 XT, **16 GB SKU R907XTGL**, gfx1201, 32 CU, 64 KB LDS, DPM performance=high, sclk 3.15–3.17 GHz under load). Model: `qwen3-9b-mlx` (4-bit, 15-tensor sidecar). Wheels: canonical `mlx-0.32.3.dev20261102+75ca881` in `venv-canon`; patched `mlx-0.32.3.dev20261103+ed40b1a` in `venv-patch1` (canonical + `lm_head_int2` kernel, tape op already in the canonical source, env-default-off).

Protocol (identical for every run): `bench/mtp-bench6.py` (md5 452d363 canonical, patched variant on the same loop), MTP=4, greedy, 512 verifies (ctx 1024), rows-ON + REFWD_M1 + GDN_FUSE standing config, `flock ~/gpu-bench.lock`, GPU at full clock, no build in parallel, 25 s idle between runs (HBM reclaim). 3-run medians where marked.

## Bottom line

| config | t/s | Δ vs canon | D1PCT/D2PCT/D3PCT | verdict |
|---|---|---|---|---|
| **PRE canonical** (savestate + 4-bit head, 3-run median 37.53/37.42/37.40) | **37.42** | — | 72.7/61.0/60.8 | baseline |
| AR canonical | 80.61 | — | — | floor 73.45: **no regression** |
| AR patched wheel | 80.63 | +0.02% | — | floor: **no regression** |
| tape + 4-bit head (S==4-gated) | 39.16 | +4.7% | 76.6/61.5/48.5 | speed win, **losslessness gate fails** |
| tape + int2 head (3-run median 23.87/23.76/23.77) | 23.77 | −36.5% | 76.6/61.5/48.5 | **net negative** |
| savestate + 4-bit head + mtp-trim | 36.57 | −2.3% | 75.6/58.1/51.1 | **net negative** |
| savestate + int2 head + mtp-trim | 22.72 | −39.3% | 75.6/58.1/51.1 | **net negative** |

Target 144.3 t/s: **not met** (best 39.16, −73%). Per the negative-result rule the default path reverts to the canonical config; all fixes remain on the branch as env-switched experimental features (`MTP6_TAPE`, `MTP6_DRAFT_HEAD`, `MLX_ROCM_GDN_TAPE` default-off).

## Losslessness

The brief's gate (MTP=4 ALLTOKS == MTP=0 ALLTOKS, same wheel) is **not satisfiable on this wheel by any S=4-verify config, including the canonical one**:

- canonical MTP=4 vs canonical AR (same wheel): first divergence at **token 299**
- savestate + trim vs AR: divergence at token 148
- tape + 4-bit head vs AR: divergence at **token 32**

All configs share the FIRST20 anchor (`271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`). The divergence is the fused GDN kernel's T=1 vs T=4 inexactness (the report-note flagged by the brief): the batched S=4 verify leaves the GDN state numerically different from the sequential S=1 chain. Canonical MTP=4 only diverges on full-accept steps (restore on rejection returns the pre-verify state) → late divergence (299); the tape additionally uses the batched row state on rejected steps → early divergence (32). This is a property of the wheel's GDN kernel, not of any change in this round. Greedy-AR (MTP=0) is the lossless reference; MTP output matches the FIRST20 anchor and the divergence profile is the same mechanism.

## Fix #1 — GDN state-tape commit (kill rejection re-forwards)

Change: bench-only (no C++ change). Mode 4 sets `MLX_ROCM_GDN_TAPE=1` (the wheel's TAPE branch, pre-existing, env-default-off). Commit = per-row tape select (`state_tape[krow]` / `conv_state_tape[krow]`, B-dim restored via reshape) + KV `trim(3 − krow)`, no save/restore, no re-forward of the accepted prefix; the correction token takes one S=1 forward. Wheel model file: TAPE branch gated `S == 4` (S=1 forwards take the fused op) — measured no difference (23.73 vs 23.77, noise), kept for the record.

Measured: 39.16 t/s with the 4-bit head (+4.7% vs canonical) — the only speed win of the round, but (a) the losslessness gate above fails (token-32 divergence) and (b) D3PCT falls 60.8 → 48.5 (batched-row h reuse changes the draft h rows). **Verdict: revert (losslessness gate); keep as experimental.**

## Fix #2 — int2 g128-asymmetric draft head (port of radiance)

Change: `lm_head_int2` HIP kernel (one thread per output column; per-64-column candidates via 8-round shared-memory max; row-quarter packing geometry: packed byte `p ∈ [0, K/4)` carries elements `p, K/4+p, K/2+p, 3K/4+p`; group G = (q, gl) spans plane `G/8` of bytes `[G%8·128, +128)`; dequant algebra `out += d·s − sx·(z·s)`), C++/pybind plumbing, bench quantizer + top-32 exact re-rank (32-row 4-bit re-dequant, 16 GB card — no 2 GB bf16 copy retained).

Correctness: standalone argmax == 4-bit head argmax (the re-rank recovers the exact 4-bit argmax — the bench's "bf16" head is the 4-bit QuantizedLinear; tape+int2 and tape+4-bit runs are bit-identical end-to-end), candidate set complete (3880/3880 per-block maxima, val↔col pairing 0 mismatches).

Measured: **−36.5% (tape) / −39.3% (savestate) vs the 4-bit head**. The int2 path costs ~2.8 ms per draft more than the 4-bit dequant head (per draft: int2 kernel + argsort(31040) + 3× take + 32-row re-dequant + dot ≈ 3 ms, vs the optimized 4-bit dequant+matvec). Radiance's 473 µs/call number does not transfer to this stack (python-side op chain + this kernel). HBM traffic *does* fall (3× ~1 GB int2 vs 3× ~4.5 GB 4-bit-head per step) — the regression is launch-count/latency-bound, not bandwidth-bound. **Verdict: net negative; revert from the default path, keep as experimental (MTP6_DRAFT_HEAD=int2).**

## Fix #3 — draft-cache trim on rejection

Change: per-branch `mtp_cache[0].trim(3 − krow)` (draft only; target untouched). Measured with the savestate path: 36.57 t/s (−2.3%), D1PCT 72.7 → 75.6 — the acceptance gain (−27 re-forwards) does not offset the trim cost. D2PCT/D3PCT fall (61.0/60.8 → 58.1/51.1). **Verdict: net negative; revert from the default path.**

## Per-phase (ms/step, 512 steps)

| phase | canonical | tape + int2 |
|---|---|---|
| draft | 31.8 (37.4%) | 82.3 (60.3%) |
| verify | 0.55 (0.6%) | 1.2 (0.9%) |
| commit (rejection cascade) | 52.7 (61.9%) | 52.8 (38.8%) |

The tape's commit phase is *not* cheaper in wall time than the canonical cascade: one fused S=1 correction forward + tape select (52.8 ms over 395 rejections ≈ 68 ms per rejection) ≈ the canonical 2–4 re-forwards (same 52.7 ms) — the re-forward saving is real (1 vs 2–4 forwards) but the per-forward cost on this wheel is high and the draft-phase cost (fix #2) dominates the net.

## HBM estimate (per step)

Canonical: verify (4.5 GB 4-bit weights + GDN save/restore 2×50 MB on 77% of steps) + 3× 4.5 GB draft head + KV ≈ **~18–19 GB** (the gap report's ~34 GB figure includes the save/restore re-copy at the old re-forward count). Patched tape + int2: verify (+206 MB tape outputs, −save/restore) + 3× 1 GB int2 head ≈ **~7 GB** — a ~2.5× traffic reduction, *but* wall time regresses (latency-bound, see fix #2).

## Commits (branch `mtp-fix-round1`, base 565c9b4)

- `565c9b4` vendor sync to vendor-canon (canonical wheel source)
- `7455906` fix round 1: tape commit, int2 head (kernel + plumbing), mtp trim, phase bench
- `059245e` fixes: tape B-dim restore, rejection S=1 forward, int2 row-quarter packing
  geometry, dequant algebra (s, z·s), re-rank argmax→token-column map, MTP6_TAPE switch + savestate A/B fallback

## Standing numbers after the round

Canonical config (PRE) stands: **MTP=4 37.42 t/s** (3-run median), **AR 80.61 t/s** (above the 73.45 floor; the patched wheel's AR 80.63 confirms the int2 additions cause no AR regression). FIRST20 anchor intact on all lossless-reference runs.
