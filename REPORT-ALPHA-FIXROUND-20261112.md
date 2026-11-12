# REPORT-ALPHA-FIXROUND-20261112.md (op order 22:5xZ item 4: the alpha fix round on the accumulated reviewer findings)

Date: 2026-11-13 (Z). Worker: cloudyfromff7. Base: main @ `aed6a25f` (PR #30
merged 2026-11-12 22:37Z). Branch: `release/0.2.0`. PR: **#31** (the
fix-round commits land on the open PR; no new PR).

Scope: D1 findings A1-A23 + the stale stage/version/numbers story + the
15-line wrap list (D1 terminal 23:0xZ); D2 F-1..F-18 (2026-11-12 ruling);
D3 main-tree REPORT.md reword (the 3 historical commit messages are the
op decision item — delivered as the prepared-not-executed deliverable);
D4 F1-F25.

## 1. D1 — the build story (A1-A5)

**Choice (stated, within the bounds): the PREFERRED option** — re-pin
`vendor.json` to a sha that resolves in the public fork; the script builds
from the fetched pinned tree; the committed partial tree is documented as
the byte-verified *reference* (not the build input); `BUILD-RECIPE.md` is
bannered as the 2026-11-01 historical provenance snapshot (80.93 tok/s)
and the single canonical wheel recipe is `scripts/build-wheel.sh` +
`vendor.json`. The alternative (restoring a build manifest into the
committed tree) was not chosen: it would fabricate a build surface the
proven tree never had (its README already documents "no build manifest"),
and the repo root IS the buildable `pip wheel .` source with the ROCm
backend grafted in — the pin can simply point at the public repo.

Evidence:

- **Pin resolvability (verified before committing):** `git ls-remote
  https://github.com/marzukia/mlx-rocm main` = `aed6a25fb1d38bd3593845108f0243fc5aa015b2`
  (2026-11-13 03:2xZ); `git -C <clone --depth 1 --branch main> rev-parse
  aed6a25f...^{commit}` = `aed6a25f...` (the `scripts/build-wheel.sh`
  gate line, dry-run). `vendor.json` re-pinned: primary.sha =
  `aed6a25f...`, ref `main` == the pin (A3 closed; the "authoritative
  source" claim is now the buildable pinned tree, and the committed
  partial tree is labeled reference).
- **Fail-loud gate:** `scripts/build-wheel.sh` (lines 26-27/42-47)
  unchanged in logic — `git rev-parse --verify $SHA^{commit}` (exit 1 on
  an unresolvable pin) + `head_sha == $SHA` (exit 1 on a checkout
  mismatch). With the resolvable pin, the gate passes on the public repo;
  with a bad pin it fails loud with the correct error. The documented
  wheel path now builds or fails loud with a CORRECT error. (The
  empirical wheel build on the validated gfx1201 box is the follow-up
  box session — not possible on this host; the gate here is the
  code/doc consistency + the resolvable pin, per the brief.)
- **README (A1):** the "python setup.py bdist_wheel from
  vendor/mlx-rocm-upstream" claim is replaced by `bash
  scripts/build-wheel.sh` (guest-a-only, clones the pinned tree) + the
  statement that the committed tree is a partial byte-verified *reference
  snapshot* with no build manifest and NOT the build input. The
  `vendor.json` inventory line now states the pin is advisory + the
  build-time sha verification + fail-loud.
- **vendor/mlx-rocm-upstream/README.md (A16/A17):** the "Repository" row
  now names the public vendor ref (this repo, `main` @ the re-pinned sha,
  resolvable in the public fork) + a "This tree" row (partial, NOT the
  build input); the build block replaces `setup.py bdist_wheel` with
  `scripts/build-wheel.sh` + the canonical CPython 3.12 venv /
  `/opt/rocm/core-10.1` pointer; "ROCm 7.x (TheRock/Fedora)" → "the
  verified toolchain is ROCm 10.1 userland (core at
  `/opt/rocm/core-10.1`; 6.4 and 7.7 historical)".
- **BUILD-RECIPE.md (A4/A5):** banner added (historical 2026-11-01
  provenance, the 80.93 tok/s record; the `75ca881` private-fork sha is
  absent from the public history so step 1 is not reproducible from the
  public repo; the canonical recipe is `scripts/build-wheel.sh` +
  `vendor.json`); step 1 flagged "historical; not reproducible" with the
  canonical-path pointer.

## 2. D1 — stale stage/version/numbers (fixed per the report's fix list)

- **Stage:** `pyproject.toml` (0.2.0, "3 - Alpha", "(alpha)" summary) + `__init__.py` 0.2.0 + PKG-INFO regenerated (README copy re-synced with this fix round) + `uv.lock` 0.2.0 (the PR #31 open state, unchanged). The README stage line: 0.2.0-alpha, the 0.1.0 "not a stability promise" sentence removed, the alpha-cut condition (decode close to the card ceiling) explicitly stated as **unmet** (107.56 t/s @1024, 84.8% of the 126.9 reference).
- **PyPI 0.1.0-vs-0.1.2 mismatch:** the README now states the public line
  0.0.0 → 0.1.0 → 0.1.1 → 0.1.2 → 0.2.0-alpha and that the PyPI uploads
  are `0.0.0` + `0.1.2` (uploaded 2026-10-30; 0.1.0/0.1.1 are git-only
  tags, never published); the "published 0.1.0 wheel" claim fixed to the
  published-wheels statement. The release notes / limitations publish
  state aligned to the same story.
- **k2-era numbers:** README supported-models table now carries AR 97.52 /
  MTP K=2 107.56 t/s @1024 (W1, #30; 84.8% of the 126.9 reference);
  `docs/limitations.md` re-assessed (9B-hybrid supported, MTP K=2 + W1 in
  "What works", K1 closed as shipped, G1 retired, publish state real,
  the battery in flight); `docs/audit/README.md` stack table now marked
  superseded by the W1 107.56 record (69.43 kept as the 2026-11-07
  checkpoint).
- **Current numbers (public-safe, no box state):** the 107.56 t/s @1024
  certified record (W1, #30) + the 2026-11-12 12-arm battery same-session
  pairs (K2 prefill-fill +2.87% TPK 2.297; K2 streaming +1.64% TPK
  2.332; K3 streaming TPK 2.670, +6.01% within-session; absolute
  re-stamp pending, 107.56 stands) — stated in CHANGELOG (new "Measured,
  not re-stamped" section), release notes (the "Not in 0.2.0" MTP item
  + a "The ~120 / 126.9 target state" section with the campaign ruling:
  code-side path closed on the named levers, residual = the
  weight-class boundary), and limitations.

## 3. D2 — alpha-checklist (F-1..F-18)

`docs/alpha-checklist.md` fully re-assessed 2026-11-12/13 against
`aed6a25f` + PR #31: the 2026-10-26 assessment bannered as the
historical snapshot with its 4 superseded headline claims named; the
**ALPHA designation section added at the top** (0.2.0 = the alpha cut;
the alpha-cut condition unmet — 107.56 vs 126.9); every table row
re-ruled (identity/docs, packaging, tests, CI gates, models,
quantisation, benchmarks, known limitations, housekeeping) with the
superseded evidence pointers (the c51a835 CI run → the aed6a25 run; the
PyPI "held, unreleased" → `0.0.0` + `0.1.2` uploaded 2026-10-30; the WIP
branches → merged/absent; G1 → retired) and honest residuals (the
rulesets 404 evidence pointer, the smoke matrix, the quantization-modes
page, the empirical wheel build as the follow-up box session). Top-5
rewritten to the #30-era state with the residual list. The F-15
superseded marker (audit README), F-11 (PLAN.md duplicated "Minimum"
block + the stale "Current state" line), F-12 (the MTP "blocked on the
M3 stream-TLS fix" open questions → SUPERSEDED notes in docs/plan.md
§6 table + §Open questions 4 and the PLAN.md Phase 2 line) all applied.

## 4. D3 / D4

- **D3-1 (REPORT.md:251)** — the persona pattern list reworded to the
  placeholder form exactly as the report specified (the 3 historical
  commit messages `c9e79c5`/`f689285`/`679d59d` are the op decision item
  — the reword patch is prepared, not executed: a `git rebase`/filter
  rewrite of those 3 messages to placeholder form + force-push is the
  op's tap; it is not in this PR's diff).
- **D4 wraps (16) / D1 wraps (15)** — all joins applied (PLAN.md, docs/plan.md:166, docs/audit/{gdn-t2-fast, k3-gdn-rereview, mtp-bench, mtp-numpy-rca (2), opt-gauntlet-areaB-backend (2), pr-2-rereview, pr-2-review, pr-6-rereview-r2}, docs/entropy4.md; docs/alpha-checklist:83 was removed by the re-assessment rewrite; README:32 was fixed in the PR #31 open commit). Detector re-run: **0** in-scope hits.
- **docs/vendor-overlay-k1.md** — status line → superseded / re-pinned
  (the K1 fix + kernel are in the committed vendor tree; the pin
  resolves; the 2026-11-01 fork-push failure kept as historical
  provenance).

## Check lines

1. **CI verbatim green (this head):** `uv sync --locked` (resolved,
   7 packages audited); `uv run ruff check .` — all checks passed;
   `uv run ruff format --check .` — 104 files already formatted;
   `uv run pytest -m 'not gpu'` — 25 passed, 1 skipped; `uv run python
   scripts/size_ratchet.py --base origin/main` — SIZE GATE: OK (4
   ratcheted).
2. **line-wrap, no-exemption:** **0** in-scope hits (the
   manual-equivalent detector over all .md, incl. fenced-exclusion +
   bullet/prefix exclusion; the BUILD-RECIPE.md fenced command
   continuations are code-argument continuations, cited).
3. **persona-leak:** **0** content hits (all 8 patterns over the tree;
   the only greps that match are the reports' own zero-count check-line
   lines, as in the prior round; git author lines exempt; no
   guest-a/box state in any new number statement — the battery is
   same-session + TPK + % form).
4. **pin resolvability:** `git ls-remote ... main` = the pinned sha;
   `rev-parse --verify` dry-run = the pinned sha (the gate's exact
   command). The committed tree remains byte-verified per its README
   (untouched by this round).

## Deliverable (prepared, not executed — the op decision item)

The 3 historical commit messages on main (`c9e79c5`/`f689285`/`679d59d`,
persona pattern hits in the pattern-list form) would require a history
rewrite + force-push of main: **not executed** (main is protected;
force-push = the op's tap). The prepared form: rewrite the 3 messages'
pattern-list lines to the placeholder form (the REPORT.md:251 reword,
already in-tree, is the canonical form), via a filter-branch patch;
execute only on the op's explicit go.
