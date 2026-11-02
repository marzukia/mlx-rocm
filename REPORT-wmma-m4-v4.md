# REPORT-wmma-m4-v4: M=4-native zero-LDS WMMA kernel — NEGATIVE (loses at the dominant class)

## Verdict

**NEGATIVE. Not shipped (default OFF; opt-in gate only).** Three structurally different M=4-native zero-LDS WMMA kernels (v4 scalar ILP-2, v4b 128-k super-chunk 16B-vector ILP-8, v4c 3-stage register-pipelined 16B-vector) are **correct** — unit gate 24/24 (worst maxdiff 0.007812 ≤ 1e-2), 200-token A/B **bit-identical (200/200)**, SASS verified (`v_wmma_f32_16x16x16_f16` present, no `ds_`/LDS, no spills) — but **all three converge on the same timing and lose to rows scalar FMA at the dominant class**:

| metric | rows | wmma (v4c) | delta |
|---|---|---|---|
| isolated K=4096 N=12288 (median, same run) | 146µs | 376µs | **+158%** |
| isolated K=4096 N=4096 | 78µs | 96µs | +23% |
| isolated K=12288 N=4096 | 156µs | 200µs | +28% |
| fwd4 full-model M=4 forward (median of 10) | 24.07ms | 48.18ms | **+100%** |
| battery (canonical K=3 MTP bench, pair ms, 2 runs/arm) | **48.98 / 49.05 (R1)** | **75.27 / 75.05 (R2)** | **+53% loss** (B0 default: 52.40 / 52.48) |

Battery arms all produced identical FIRST20 tokens and TAPE n_ok=128 n_fb=0 (correctness held in the real model stream).

## K-scan (N=12288, rows vs wmma-1wave, same run)

| K | rows | wmma | delta |
|---|---|---|---|
| 512 | 69µs | 46µs | **−33% win** |
| 1024 | 65µs | 56µs | −14% win |
| 2048 | 93µs | 97µs | parity |
| 4096 | 146µs | 376µs | **+158% loss** |

The win band is K ≤ 1024 only. In the 9B model M=4 verify runs are dominated by the K=4096 / K=12288 in_proj-class shapes — the loss band.

## Why it loses (mechanism, probe-verified)

1. **The mma unit is fast.** 1-wave ILP probe: 21.0 / 12.0 / 10.9 / 10.4 / 10.1 ns per mma at ILP 1/2/4/8/16 — the WMMA pipe saturates at ILP ≥ 4. Total mma work for the class (768 waves × 128 chunks × 2 = 196,608 mma) is ≈ 32µs on the CU matrix pipe spread over the 64 CUs (12 waves/CU × 256 mma × 10.4 ns = 31.9 µs; serial it is 2.04 ms) — not the cost either way.
2. **The kernel structure is cheap.** 1-wave decomposition probe (L2-resident data): full 32-chunk (loads + 16-nibble dequant + 2 mma ILP-2) = 78 cycles/chunk; loads+dequant alone 61; 2 mma alone 23.
3. **The cost is the B fetch pattern.** Measured 557 cycles/chunk in the real kernel vs 78 in the probe = HBM line-fetch latency, ~2.6-4x the rows pattern. rows (msg morphology) loads B lane-strided 128-bit: 32 lanes fetch 512B contiguous per 1024-k wave instruction = 16 32B lines per 32-k chunk. The WMMA B fragment layout — lane L owns n = L&15, k = 8*(L>>4)+e — forces each lane to read its OWN column's words: 32 lanes × 16B = 32 non-adjacent 32B lines per 32-k chunk (+ 8 A lines + ~4 scale lines ≈ 44 line fetches vs ~17). Each line fetch is a full HBM latency.
4. **No software structure fixes it.** v4 (scalar, ILP-2), v4b (128-k super-chunks, 16B vectors, ILP-8) and v4c (explicit 3-stage register pipeline, 2 chunks ahead) all measure 376-380µs. The pipeline window (2 chunks — the probe measures 78 cycles/chunk for the full 32-chunk decomposition, i.e. ~156 for 2 chunks, with the loads+dequant + 2-mma split alone at 84) is << HBM latency (~500 cycles), so steady-state per chunk ≈ latency/2. A 128-k super-chunk (8 mma, ILP-8) issues 8× the loads per step — same line traffic, same latency.
5. **LDS staging (v2) is the coalesced alternative and it also loses** (+95µs at the dominant class): the barrier + LDS round-trip does not beat the scalar FMA either. The loss is structural: the WMMA B fragment (16 n per lane group, column-major) is incompatible with coalesced packed-column weight access on this toolchain, and the mma unit speed (10ns/mma) does not pay back the ~2.6x line-fetch multiplier at K=4096.

## Wave counts

waves 1/2/4 at K=4096 N=12288: 371 / 386 / 389µs — monotonically worse (consistent with v2; no inter-wave coupling in v4, waves are pure scheduling granularity). The dispatch default (WAVES=2) is unchanged; the v4 round measured 1/2/4 explicitly.

## What was done (v4 workstream)

- Fragment layouts probe-verified on gfx1201 (A row-major: (L&15, 8*(L>>4)+e); B col-major: (8*(L>>4)+e, L&15); ACC: row 8*(L>>4)+e, col L&15 — M≤4 rows on lanes 0..15 elements 0..M-1: independent per-thread stores, no barrier needed).
- v4/v4b/v4c kernels grafted into the wheel tree, incremental wheel builds, SASS verified each time (LLVM-23 gfx1201 mnemonics: `global_load_b32/b128`, `global_store_b16`, `v_wmma_f32_16x16x16_f16`; no `ds_*`, no spills).
- v4b fixed a v4 bug found in review: K%32==16 tail read 16 k out of range; v4c uses a single 16-wide tail tile and clamps the last chunk's second scale index.
- Unit gate (24 combos, incl. K%32==16 shapes), fwd4, 200-token A/B, battery — all run in a clean scratch venv (venv-v4, wheel extracted; canonical venvs venv-canon / venv-wrc / venv-ra untouched). Sidecar pin: model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c active, 4-bit sidecar .disabled, other shards parked — verified before the battery and re-verified after (unchanged).

## Recommendation

- **Keep `MLX_QMV_WMMA` default OFF. M=4 verify stays on the rows (msg) path.**
- The opt-in gate remains available (zero behavior change when unset; correctness is bit-identical — 200/200 tokens, 24/24 unit combos).
- A K ≤ 1024 gate (where WMMA wins up to −33%) is a possible follow-up, but M=4 at K≤1024 is rare in the 9B model; not worth shipping now.
- v2 (LDS-staged, +95µs at the dominant class) remains the best WMMA variant; it is not shipped either.

## Files

- `patches/qmv-wmma-m4/qmv_wmma_m4_v4c_kernel.hip` — final kernel (3-stage pipeline; v4/v4b superseded). The b16-tail (K%32==16) scale/bias read is clamped to ng-1 (latent OOB for shapes where kt/GROUP_SIZE == ng); the benched wheel predates the clamp and the battery shapes (K=512/1536/4096/12288) have K%32==0 and never enter the tail.
- `patches/qmv-wmma-m4/qmv-wmma-m4-v2-dispatch.diff` — dispatch gate, unchanged across the v4 family (same file name, same gate: opt-in, bits==4, M==4, N%16==0, K%16==0).
