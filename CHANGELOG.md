# Changelog

All notable changes to `mlx-rocm` are documented here.

## [0.3.0-alpha] - 2026-11-19

The 0.3.0 alpha cut — **released**: published to PyPI 2026-11-20 (`pip install mlx-rocm` resolves `0.3.0`); merged to main `3e84320e` (PR #40). API and performance are not stable; expect breaking changes. Everything below the 0.2.0-alpha line stands; 0.3.0 lands the 3.7K PP lane stack on main (the #36 ONE-PR record, `f5e7cd88`) + the version bump (the alpha designation is carried by these release docs, not the version string; the v0.3.0 tag is pending the owner's push).

### Added

- **Q4_K mmq port (#36)**: the `Q4_K` mmq path (the 4-byte-into-2-byte store-width defect fixed in the PATCH81 in-place bf16 store chain) — the in-tree `Q4_K_M` path for the 9B work model.
- **Attention port (#36)**: the RDNA WMMA `mma-f16` flash-attention port (RDNA WMMA mma-f16 flash attention, gated behind `MLX_SDPA_FATTN`, ON-by-default in the merged tree per op letter (a) = ADOPT).
- **HBM edge fix (`hbmfix`, #36)**: the sync `hipMalloc` fallback when the HSA async pool fails to extend (the HBM-full-after-reclaim edge) + the `sync_alloc` free tag + the reclaim/fallback diagnostics (physical HBM + arena + hip error at reclaim/fallback).
- **MTP decode path**: the MTP K=3 record protocol (`MLX_MTP=3` + `MTP6_PREFILL_FILL=1`; 111.01 t/s decode banked on main; in-tree W1 record 111.84 t/s, `docs/w1/REPORT-W1-RECORD-11184-20261114.md`). Enablement is a flag-class recipe — see `docs/mtp-enablement.md`.
- **Prefill stack (3.7K parity work, on main at `f5e7cd88`)**: the #36 ONE-PR consolidated lane stack — vendor-sync + S1 SDPA fix (the Q staging per-D-chunk fix) + hbmfix + mmq Q4_K port + P1B4 scratch fix (pool-routed scratch in `mmq_q4k_g64`, killing the per-call `hipMalloc`/`hipFree` churn; `MLX_ROCM_MMQ_Q4K_ALLOC=0` raw fallback) + attention port. See the `[3.7K PP lane merge]` section below for the record (3084.9 t/s @4K adopted / 2861.4 second axis vs the 3683.1 bar).
- **The 0.3.0-alpha release prep**: the `CHANGELOG` section, the version bump to `0.3.0` (the alpha designation is in the release docs; the v0.3.0 tag is pending the owner's push), the README pass (hardware line, known-limits, `BUILD-RECIPE.md` pointer, benchmark section), `docs/env-flags.md` (every env flag in the tree), `docs/mtp-enablement.md`, the benchmark charts under `docs/diagrams/` (the llama.cpp anchor, 2026-11-19), and the root-cluster move to `docs/notes/`.

### Changed

- `pyproject.toml` 0.2.0 → 0.3.0 (the wheel metadata is derived from it at build time; the alpha designation is in the release docs, not the version string; the v0.3.0 tag is pending the owner's push).
- The root cluster (`PLAN.md`, the `REPORT*.md` / `NOTES-*.md` / `WALL-NOTE.md` files, `size-ratchet.json`) moved to `docs/notes/` (moved, not deleted; `vendor.json` stays at the root — `scripts/build-wheel.sh` reads it).
- `scripts/size_ratchet.py` ratchet path follows the move (`docs/notes/size-ratchet.json`).

## [3.7K PP lane merge] - 2026-11-19

Post-0.2.0-alpha lane work: the 3.7K PP prefill lane, merged by op call (#36 consolidated ONE-PR + #37 qmm-pad e2e, 2026-11-19).

### Added

- **Consolidated lane stack (#36)**: vendor-sync + the S1 SDPA Q-staging fix + hbmfix diagnostics + the Q4_K mmq port (the PATCH81 in-place bf16 store chain: the 4-byte-into-2-byte store-width defect) + the P1B4 scratch fix + the attention port (RDNA WMMA mma-f16 flash attention, gated behind MLX_SDPA_FATTN) + plan-D (staging-retention cap, logits-bf16 gate, per-shape repack caps).
- **qmm-pad e2e (#37)**: MLX_ROCM_INPROJ_PAD (default OFF): GDN in_proj_b/a via one fused GEMM padded to N=128 (existing kernels only). e2e unmeasured at merge (the pad-branch python fix + the e2e re-run land post-merge) - no win claimed at the default.
- **Repack byte-cap suffix parse**: MLX_ROCM_MMQ_REPACK_MAX_BYTES now accepts case-insensitive SI (1000-base) k/m/g suffixes (the t3171 F1-medium fix).

### Record (measured, x3 median, full gate set, 2026-11-19)

- Adopted record **3084.9 t/s @4K** (the f2c zero-mode axis; the FLAG-OFF record state; record vehicle lib 070122d8), second axis **2861.4 t/s** (the pcm bench class). Bar **3683.1 t/s** = the external 500x sustained stream. Gates: argmax=89 all 6 runs, faults 0, sclk boost class, SoT restore verified. The fixed record protocol (22:2xZ) never changes.
- Flag states: MLX_SDPA_FATTN ON-by-default in the merged tree (op letter (a) = ADOPT); the flag-ON vs flag-OFF boost A/B rides the parity re-stamp. MLX_ROCM_INPROJ_PAD default OFF (e2e unmeasured, no win claimed).
- Merge-gate status: **cleared** — the fixed-protocol re-stamp on the merged tree 3719.25 t/s = 100.98% of the 3683.1 bar (like-for-like = the working judge per the 2026-11-19 parity ruling); the lane merged to main `f5e7cd88` (2026-11-19 22:37Z).

Provenance for the 0.2.0-alpha numbers: the in-repo docs/w1/`REPORT-W1-PHASE1-20261112.md` and docs/w1/`REPORT-W1-PHASE2-REPRO-20261112.md`.

## [0.2.0-alpha] - 2026-11-13

The alpha cut. API and performance are not stable; expect breaking changes. Record config: 9B-hybrid (GDN + FA) 4-bit, K=2 MTP spec decode, the sequential GDN kernel engaged, 1024 tokens per round, on the validated gfx1201 guest (ROCm 10.1).

### Added

- **MTP speculative decoding with exact GDN state rollback** (#26): K=1/K=2 spec-decode at 92.23 t/s @1024, token-identical to plain decode; the 9B in-tree model (`models/qwen3_5.py`) and the K=1..K=3 bench (`bench/k2arm-pinned.py`).
- **W1 sequential GDN kernel** (#30): the llama.cpp GDN design port, bit-exact; engages under `MTP6_TAPE=seq` + `MLX_ROCM_GDN_SEQ=1` (default off). 97.52 -> 107.56 t/s @1024 certified, lossless 3/3, the in-tree repro is bit-identical, 84.8% of the 126.9 t/s same-GPU reference.
- **slot2 fused sequential slot** (#30): the 2<=S<=4 per-row state design, `MLX_ROCM_GDN_SLOT2=1` (default off).
- **Half-precision (bf16) draft module** (#30): the post-load-eval lazy cast (`MTP6_MOD_BF16`, default off) + on-device token passing in the bench.
- **S=4 verify-attention dispatch** (the `MLX_ROCM_SDPA_S4VA` flag): the S=4 (K=3 spec-decode verify batch) 1-pass vector-kernel extension; default off, bit-identical with the env unset.
- **GDN fused T=4** (#7): S=4 MTP verify on the fused path (bit-exact, lossless; K=3 +1.1%).
- **Graph decode** (#2, #3): re-delivery of the 16 in-graph H2D staging values pre-launch; the deterministic 2nd-capture launch fault fixed.
- **Bench harness + record battery**: `bench/k2arm-pinned.py` (the K=1..K=3 MTP probe with the record config), `bench/gdn_seq1024_repro.sh` (the 108 number repro driver), docs/w1/`REPORT-W1-PHASE1-20261112.md` + docs/w1/`REPORT-W1-PHASE2-REPRO-20261112.md`.
- **The 0.1.1/0.1.2 cycle**: non-GEMM fusion (k2, #2 — 9B neutral, 0.6B +17-18%, adopted), the GDN port (k3, #3) + the chunked-copy segfault fix, K1 QMV int8-activation fix (N=12288), the PR3-HYBRID fused wheel reproducibility (#4), `build-wheel.sh` verifying the `vendor.json` pin, the migration to GitHub (0.1.2).

### Fixed

- **The gather-slice commit fix (P3.2b, #30)**: the mode-4 `commit()` int-index take -> the slice rebind (K3 path only): S4 -10.11 ms/step in-session, lossless 5/5.
- The `gdn_prep` fault: a host `Data*` passed as a device ptr (14 kernel-arg ptrs); gates G1/G2.
- The `copy_general_general` segfault in the GDN host-shadow chunked copy (no per-chunk `hipStreamDestroy`).
- MTP full-state restore (keys + values + offsets) in the verify path; the `rope_q` row-invariance lossless gate fix.

### Measured, not re-stamped (the absolute re-stamp is pending the next card session; 107.56 stands as the certified record)

- **MTP prefill-fill + streaming catch-up**: same-session 12-arm battery (2026-11-12): K2 prefill-fill +2.87% (TPK 2.297), K2 streaming +1.64% (TPK 2.332), K3 streaming TPK 2.670 (best tokens/step on the card, +6.01% within-session). The in-tree bench is still the pre-fill state.
- **The ~120 / 126.9 target**: the campaign ruling — the code-side path is closed on the named levers; the residual gap to the same-GPU 126.9 t/s reference is the weight-class boundary.

### Not in 0.2.0 (in flight)

- **W1 M=3 slot interior A/B/C**: issues #27/#28/#29, open.
- 27B+ / MoE validation (Phase 2); the native mxfp4 kernel (the opt-in dequant->bf16 GEMM path only); non-gfx1201 archs.

## [0.1.2] / [0.1.1] / [0.1.0]

See the git tags (`v0.1.0`, `v0.1.1`, `v0.1.2`). 0.1.0 = the first public release (the scaffold: the vendored ROCm backend tree + the standalone hipBLASLt GEMM + the smoke plumbing, pre-alpha). 0.1.1 = the GDN fused decode kernel + ent4 entropy compression release line. 0.1.2 = the migration from the private GitLab to GitHub.
