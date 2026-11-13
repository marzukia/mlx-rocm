# REPORT-RELEASE020-20261112 (op order 2026-11-12 items 3+4: tee up the 0.2.0-alpha release)

Date: 2026-11-13 (Z). Worker: cloudyfromff7. Base: main @ aed6a25f (PR #30 merged).
Branch: `release/0.2.0`. PR: **#31** (opened; not tagged, not published — the
release cut needs the tag `v0.2.0-alpha` + the operator tap). Head SHA at
PR open: **9dc8e93** (this report lands as the follow-up commit on the PR).

**Fix round (2026-11-13):** the accumulated reviewer findings (D1-D4, all
terminal 23:0xZ) were fixed on this branch after this report — the
build-story re-pin (vendor.json → a resolvable sha; the committed partial
tree documented as the reference, not the build input), the stage/version/
numbers refresh, the checklist re-assessment + 0.2.0=alpha designation,
the persona reword, the 15-line wrap reflow, and the battery-number
release-notes update. See `REPORT-ALPHA-FIXROUND-20261112.md`.

## 1. Version-declaration inventory (every site, before/after)

| Site | Before | After | Note |
|---|---|---|---|
| `pyproject.toml:7` | `version = "0.1.0"` | `version = "0.2.0"` | the single source of the wheel version |
| `pyproject.toml:8` | description "(pre-alpha)" | "(alpha)" | the alpha designation in the metadata |
| `pyproject.toml:15` | classifier "2 - Pre-Alpha" | "3 - Alpha" | the alpha designation in the metadata |
| `src/python/mlx_rocm/__init__.py:8` | `__version__ = "0.0.1"` | `__version__ = "0.2.0"` | the 0.0.1 had diverged from pyproject pre-PR; now aligned |
| `src/python/mlx_rocm.egg-info/PKG-INFO` | Version 0.1.0 + pre-alpha + old README copy | regenerated from the new pyproject + README | tracked build byproduct (repo precedent 9caab6a); SOURCES.txt untouched |
| `uv.lock` | root package `version = "0.1.0"` | `version = "0.2.0"` | via `uv lock` (repo precedent bf49583: "uv lock after 0.1.0 version bump") |
| `README.md:3,11-15` | "Stage: **pre-alpha** … the alpha cut is planned once decode reaches close to the card's ceiling" | "Stage: **alpha** (0.2.0-alpha)" + the 0.2.0 standing (107.56 t/s @1024 record, 84.8% of the 126.9 t/s same-GPU reference) | the historical 0.1.0 references stay (they record the first public release) |
| `vendor/mlx-rocm-upstream/mlx/version.h` | MLX 0.32.3 | NOT bumped | the upstream MLX version of the vendored tree, not our release version |
| `scripts/build-wheel.sh` | no version embedded | — | the wheel metadata is derived from pyproject.toml at build time (setuptools `pip wheel`); no separate version file |
| `CMakeLists.txt` (root), `size-ratchet.json`, `MANIFEST.in`, `.github/workflows/ci.yml` | no version embedded | — | checked, nothing to bump |
| ROCm 10.1.0 references in `docs/plan.md`, `docs/quickstart.md` | ROCm toolchain version | NOT touched | not our version |
| `docs/alpha-checklist.md` | quotes "v0.0.1"/"0.1.0" | NOT touched | historical assessment snapshot (2026-10-26); the release notes note which gaps closed since |

## 2. CHANGELOG.md (new, repo root) — content

`[0.2.0-alpha] - 2026-11-13`, derived from git log v0.1.0..HEAD (77
commits: the 77-commit main base + the 4 release commits; the `v0.1.0`
tag sits on the detached scaffold lineage, so the range count = the
full branch history).
Added: MTP spec decode with exact GDN state rollback (#26, 92.23 t/s @1024,
token-identical to plain decode); the W1 sequential GDN kernel (#30, the
llama.cpp GDN design port, bit-exact, `MTP6_TAPE=seq` + `MLX_ROCM_GDN_SEQ=1`
default-off; 97.52 -> 107.56 t/s @1024 certified, lossless 3/3, in-tree repro
bit-identical, 84.8% of the 126.9 t/s same-GPU reference); slot2 (2<=S<=4,
`MLX_ROCM_GDN_SLOT2=1` default-off); the bf16 draft module (`MTP6_MOD_BF16`
default-off) + on-device token passing; the `MLX_ROCM_SDPA_S4VA` S=4
verify-attention dispatch (default-off, bit-identical with the env unset);
GDN fused T=4 (#7); graph decode (#2/#3); the bench harness + record battery
(`bench/k2arm-pinned.py`, `bench/gdn_seq1024_repro.sh`, the REPORT-W1-* docs);
the 0.1.1/0.1.2 cycle items. Fixed: the gather-slice commit fix (P3.2b,
K3-path-only, S4 -10.11 ms/step in-session, lossless 5/5); the `gdn_prep`
fault; the `copy_general_general` segfault; the MTP full-state restore +
`rope_q` gate fix. Not in 0.2.0 (in flight): MTP prefill-fill + streaming
catch-up (verified absent from main — the in-tree bench is the pre-fill
state 11766c82 class; no fill/streaming code in `models/qwen3_5.py`); the W1
M=3 slot interior A/B/C (issues #27/#28/#29, open); 27B+/MoE (Phase 2), the
native mxfp4 kernel, non-gfx1201 archs. Plus a one-paragraph 0.1.2/0.1.1/0.1.0
summary pointing at the git tags. Public-safe numbers only (rate + config
class; provenance = the in-repo REPORT-W1-* files, no box paths).

## 3. Release notes (new, `docs/release-notes-0.2.0.md`)

The GitHub release body draft: the explicit ALPHA designation (0.2.0-alpha:
API and performance not stable; the wheel build is guest-a-only (ROCm
gfx1201); the vendor pin is advisory — see docs/alpha-checklist.md), the
feature list, the reproducible-in-tree number block
(`bash bench/gdn_seq1024_repro.sh`, stream `64b1b6da`), the not-in-0.2.0
list, the known-limitations section = the still-open alpha-checklist gaps
(no install path; fp8 e4m3 long prefill unvalidated with the
`MLX_ROCM_GEMM_BF16=1` safe default; the GDN long-prefill fallback; the
NO_GPU op set; 27B+/MoE; non-gfx1201; mxfp4 opt-in path only; the GPU smoke
matrix; the vendor re-pin cadence) with the since-closed items noted
(ROCm-version story, license MIT, supported-models table + conversion recipe,
`docs/limitations.md`, MTP K=2 in-tree, the GPU validation recorded, the
`build-wheel.sh` pin verification).

## 4. PR + head SHA

PR: **marzukia/mlx-rocm #31** (base main aed6a25f, head `release/0.2.0` @
9dc8e93 at PR open; this report = the follow-up commit). Body carries the op
order citation, "worker: cloudyfromff7, reviewer pass to follow", the alpha
designation, the build/tag flow (no tag, no publish), and the check-line
results below.

## 5. Check-line results

- **F (format, CI verbatim on the audited state): PASS.** `uv sync --locked`
  PASS; `uv run ruff check .` PASS (all checks passed); `uv run ruff format
  --check .` PASS (103 files already formatted); `uv run pytest -m 'not gpu'`
  25 passed + 1 skipped; `uv run python scripts/size_ratchet.py --base
  origin/main` OK (4 ratcheted).
- **P (persona-leak): PASS.** 8-pattern battery over the diff's added lines:
  marzuki-* hostnames 0; 100.64.0.0/10 0; andryo 0; neonvm@ 0;
  192.168.122.x 0; /mnt/models 0; *.junkyard.sh 0; @ag.mrzk.io in tree
  content 0 (the git author lines are the exempt mandated fleet identity);
  full-tree: 1 pre-existing self-referential pattern-list line in
  `REPORT.md` (the battery's own description text), not in this diff.
- **W (line-wrap): PASS.** `scripts/line_wrap_check.py` is ABSENT at main
  (verified) — the manual equivalent detector (an indented continuation after
  a complete sentence, uppercase start) over the new + touched .md docs +
  PKG-INFO + the PR body: 0 hits after fixing 2 (one in the new release
  notes, one pre-existing in README mirrored to the PKG-INFO copy).

## 6. Build/tag flow (stated, not executed)

`scripts/build-wheel.sh` (gfx1201 box, ROCm 10.1 userland) clones the pinned
vendor fork (= this repo), verifies the checkout against the `vendor.json`
sha, and runs `python -m pip wheel .` — the wheel metadata is derived from
`pyproject.toml` at build time (setuptools), so the 0.2.0 in this PR
reaches the wheel with no separate file. The release cut needs: the tag
`v0.2.0-alpha` + the operator tap (the human-gated merge of #31 first).
Nothing tagged, nothing published by this worker.
