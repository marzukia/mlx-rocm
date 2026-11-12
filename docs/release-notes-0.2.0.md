# mlx-rocm 0.2.0-alpha

**ALPHA designation.** 0.2.0-alpha: API and performance not stable; the wheel
build is guest-a-only (ROCm gfx1201); the vendor pin is advisory - see
`docs/alpha-checklist.md`.

## What is in 0.2.0

The 0.2.0-alpha cut carries the W1 line (PR #30) on top of the 0.1.x cycle.
Record config: 9B-hybrid (GDN + FA) 4-bit, K=2 MTP spec decode, the
sequential GDN kernel engaged, 1024 tokens per round, validated on gfx1201
(ROCm 10.1 userland).

- **MTP speculative decoding with exact GDN state rollback** (#26):
  K=1/K=2 spec-decode, 92.23 t/s @1024, token-identical to plain decode;
  the 9B model is in-tree (`models/qwen3_5.py`), the K=1..K=3 bench is
  in-tree (`bench/k2arm-pinned.py`).
- **The W1 sequential GDN kernel** (#30): the llama.cpp GDN design port,
  bit-exact; engages under `MTP6_TAPE=seq` + `MLX_ROCM_GDN_SEQ=1`
  (default off). 97.52 -> 107.56 t/s @1024 certified, lossless 3/3, the
  in-tree repro is bit-identical, 84.8% of the 126.9 t/s same-GPU
  reference.
- **slot2 fused sequential slot** (#30): the 2<=S<=4 per-row state design,
  `MLX_ROCM_GDN_SLOT2=1` (default off).
- **The half-precision (bf16) draft module** (#30): the post-load-eval
  lazy cast (`MTP6_MOD_BF16`, default off) + on-device token passing in
  the bench.
- **The S=4 verify-attention dispatch** (the `MLX_ROCM_SDPA_S4VA` flag):
  the S=4 verify batch goes to the 1-pass vector kernel; default off,
  bit-identical with the env unset.
- **GDN fused T=4** (#7): S=4 MTP verify on the fused path (bit-exact,
  lossless; K=3 +1.1%).
- **Graph decode** (#2, #3): the 16 in-graph H2D staging values are
  re-delivered pre-launch; the deterministic 2nd-capture launch fault is
  fixed.
- **The gather-slice commit fix (P3.2b)**: the mode-4 `commit()`
  int-index take -> the slice rebind (K3 path only): S4 -10.11 ms/step
  in-session, lossless 5/5.
- **The bench harness + record battery**: `bench/gdn_seq1024_repro.sh`
  (the 108 number repro driver) + `REPORT-W1-PHASE1-20261112.md` +
  `REPORT-W1-PHASE2-REPRO-20261112.md`.

## The number, reproducible in-tree

The record is 107.56 t/s @1024 (record config, 3/3 lossless, stream
`64b1b6da`). The in-tree repro is `bash bench/gdn_seq1024_repro.sh` (the
exact record-battery invocation run from the in-tree files; identity
clauses + same-session relative rate clause, the in-tree run recorded in
`REPORT-W1-PHASE2-REPRO-20261112.md`).

## Not in 0.2.0 (in flight)

- **MTP prefill-fill + streaming catch-up**: measured, not re-stamped. Same-session 12-arm battery (2026-11-12): K2 prefill-fill +2.87% (TPK 2.297), K2 streaming +1.64% (TPK 2.332), K3 streaming TPK 2.670 (best tokens/step on the card, +6.01% within-session). The absolute re-stamp is pending the next card session — 107.56 t/s stands as the certified record. The in-tree bench is still the pre-fill state.
- **W1 M=3 slot interior A/B/C**: issues #27/#28/#29, open.
- 27B+ / MoE validation (Phase 2), the native mxfp4 kernel, non-gfx1201
  archs.

## The ~120 / 126.9 target state

The campaign ruling: the code-side path is closed on the named levers;
the residual gap to the same-GPU 126.9 t/s reference is the weight-class
boundary (9B-hybrid class vs the reference). 0.2.0 stands at 107.56 t/s
@1024 (84.8%).

## Known limitations

The open `docs/alpha-checklist.md` gaps as of this cut (the checklist is
re-assessed 2026-11-12 against `main` @ `aed6a25f` + PR #31; since the
2026-10-26 snapshot the ROCm-version story was settled to 10.1, the
license fixed to MIT, the supported-models table + conversion recipe and
`docs/limitations.md` landed, MTP K=2 + the W1 line are in-tree, and the
GPU validation is recorded on the validated guest):

- **Install path** — `pip install mlx-rocm` (PyPI `0.0.0`, `0.1.2`,
  uploaded 2026-10-30); the full `mlx` wheel is built on the validated
  guest (`scripts/build-wheel.sh`). The empirical 0.2.0 wheel build on the
  box is the follow-up (the pin is resolvable + the gate is fail-loud).
- **fp8 e4m3 long prefill on gfx1201** — the probe passes (max_rel_err
  0.0057) but real long-context prefill is unvalidated;
  `MLX_ROCM_GEMM_BF16=1` is the shipped safe default
  (`scripts/smoke-serve.sh`).
- **GDN long prefill** — the per-timestep fallback is fine for decode
  (T=1) but slow for long prefill (T=4096 is ~32k small kernel launches);
  the fused GDN port (K3, merged) covers the decode path and the W1
  sequential GDN kernel is in the certified record.
- **NO_GPU op set** — FFT, the LAPACK family (LUF/QRF/SVD/Inverse/Eig/Eigh)
  and BlockMaskedMM fall back to CPU.
- **27B+ / 9B-hybrid-large / MoE** — Phase 2; not validated.
- **Non-gfx1201 archs** — only gfx1201 is validated in this repo.
- **mxfp4** — the opt-in dequant->bf16 GEMM prefill path only; no native
  mxfp4 kernel.
- **GPU smoke matrix** — the checks exist (`tests/test_gpu.py`,
  `scripts/smoke.py`, `scripts/smoke-serve.sh`) but no matrix documents
  check x ROCm version x box.
- **Vendor re-pin cadence** — `build-wheel.sh` verifies the
  `vendor.json` sha at build time (fail-loud); the pin was re-pinned
  2026-11-13 to a sha resolvable in the public fork; the cadence is
  "re-pin in a PR when the base moves" (no calendar rule).

## Building

`scripts/build-wheel.sh` (run on a gfx1201 box, ROCm 10.1 userland;
`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1` — see `docs/quickstart.md`)
clones the pinned vendor fork (= this repo), verifies the checkout
against the `vendor.json` sha (the pin is resolvable in the public fork;
fail-loud otherwise), and builds the wheel from the fetched pinned tree.
The committed `vendor/mlx-rocm-upstream/` is a partial, byte-verified
reference snapshot (no build manifest) — not the build input. The wheel
metadata is derived from `pyproject.toml` at build time (setuptools); the
version in this release is `0.2.0`. `BUILD-RECIPE.md` is the 2026-11-01
historical provenance snapshot (80.93 tok/s); the single canonical wheel
recipe is this path.

## Provenance

The in-repo `REPORT-W1-PHASE1-20261112.md` (the format-change battery) and
`REPORT-W1-PHASE2-REPRO-20261112.md` (the 108 number, the in-tree repro,
the record-battery driver). `docs/limitations.md` for the full what-works /
in-flight / not-supported list.
