# Changelog

All notable changes to `mlx-rocm` are documented here.

Provenance for the 0.2.0-alpha numbers: the in-repo
`REPORT-W1-PHASE1-20261112.md` and `REPORT-W1-PHASE2-REPRO-20261112.md`.

## [0.2.0-alpha] - 2026-11-13

The alpha cut. API and performance are not stable; expect breaking changes.
Record config: 9B-hybrid (GDN + FA) 4-bit, K=2 MTP spec decode, the
sequential GDN kernel engaged, 1024 tokens per round, on the validated
gfx1201 guest (ROCm 10.1).

### Added

- **MTP speculative decoding with exact GDN state rollback** (#26):
  K=1/K=2 spec-decode at 92.23 t/s @1024, token-identical to plain decode;
  the 9B in-tree model (`models/qwen3_5.py`) and the K=1..K=3 bench
  (`bench/k2arm-pinned.py`).
- **W1 sequential GDN kernel** (#30): the llama.cpp GDN design port,
  bit-exact; engages under `MTP6_TAPE=seq` + `MLX_ROCM_GDN_SEQ=1`
  (default off). 97.52 -> 107.56 t/s @1024 certified, lossless 3/3, the
  in-tree repro is bit-identical, 84.8% of the 126.9 t/s same-GPU
  reference.
- **slot2 fused sequential slot** (#30): the 2<=S<=4 per-row state design,
  `MLX_ROCM_GDN_SLOT2=1` (default off).
- **Half-precision (bf16) draft module** (#30): the post-load-eval lazy
  cast (`MTP6_MOD_BF16`, default off) + on-device token passing in the
  bench.
- **S=4 verify-attention dispatch** (the `MLX_ROCM_SDPA_S4VA` flag): the
  S=4 (K=3 spec-decode verify batch) 1-pass vector-kernel extension;
  default off, bit-identical with the env unset.
- **GDN fused T=4** (#7): S=4 MTP verify on the fused path (bit-exact,
  lossless; K=3 +1.1%).
- **Graph decode** (#2, #3): re-delivery of the 16 in-graph H2D staging
  values pre-launch; the deterministic 2nd-capture launch fault fixed.
- **Bench harness + record battery**: `bench/k2arm-pinned.py` (the
  K=1..K=3 MTP probe with the record config), `bench/gdn_seq1024_repro.sh`
  (the 108 number repro driver), `REPORT-W1-PHASE1-20261112.md` +
  `REPORT-W1-PHASE2-REPRO-20261112.md`.
- **The 0.1.1/0.1.2 cycle**: non-GEMM fusion (k2, #2 — 9B neutral, 0.6B
  +17-18%, adopted), the GDN port (k3, #3) + the chunked-copy segfault
  fix, K1 QMV int8-activation fix (N=12288), the PR3-HYBRID fused wheel
  reproducibility (#4), `build-wheel.sh` verifying the `vendor.json` pin,
  the migration to GitHub (0.1.2).

### Fixed

- **The gather-slice commit fix (P3.2b, #30)**: the mode-4 `commit()`
  int-index take -> the slice rebind (K3 path only): S4 -10.11 ms/step
  in-session, lossless 5/5.
- The `gdn_prep` fault: a host `Data*` passed as a device ptr (14
  kernel-arg ptrs); gates G1/G2.
- The `copy_general_general` segfault in the GDN host-shadow chunked copy
  (no per-chunk `hipStreamDestroy`).
- MTP full-state restore (keys + values + offsets) in the verify path; the
  `rope_q` row-invariance lossless gate fix.

### Measured, not re-stamped (the absolute re-stamp is pending the next card session; 107.56 stands as the certified record)

- **MTP prefill-fill + streaming catch-up**: same-session 12-arm battery
  (2026-11-12): K2 prefill-fill +2.87% (TPK 2.297), K2 streaming +1.64%
  (TPK 2.332), K3 streaming TPK 2.670 (best tokens/step on the card,
  +6.01% within-session). The in-tree bench is still the pre-fill state.
- **The ~120 / 126.9 target**: the campaign ruling — the code-side path
  is closed on the named levers; the residual gap to the same-GPU 126.9
  t/s reference is the weight-class boundary.

### Not in 0.2.0 (in flight)

- **W1 M=3 slot interior A/B/C**: issues #27/#28/#29, open.
- 27B+ / MoE validation (Phase 2); the native mxfp4 kernel (the opt-in
  dequant->bf16 GEMM path only); non-gfx1201 archs.

## [0.1.2] / [0.1.1] / [0.1.0]

See the git tags (`v0.1.0`, `v0.1.1`, `v0.1.2`). 0.1.0 = the first public
release (the scaffold: the vendored ROCm backend tree + the standalone
hipBLASLt GEMM + the smoke plumbing, pre-alpha). 0.1.1 = the GDN fused
decode kernel + ent4 entropy compression release line. 0.1.2 = the
migration from the private GitLab to GitHub.
