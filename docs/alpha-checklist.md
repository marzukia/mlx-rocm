# Alpha-state assessment — mlx-rocm

**Alpha definition:** a first-time external user can install it, load a model, run
generation, and trust the numbers.

**ALPHA designation: 0.2.0 = the alpha cut** (0.2.0-alpha). The alpha-cut
condition — decode reaching close to the card's ceiling (~120 tok/s,
byte-identical vs autoregressive) — is **unmet**: the standing record is
107.56 t/s @1024 (84.8% of the same-GPU 126.9 t/s reference). The
designation is carried as the alpha cut regardless; API and performance are
not stable.

Re-assessed 2026-11-12/13 against `main` @ `aed6a25f` (PR #30 merged
2026-11-12 22:37Z) + PR #31 head. Status per item: **DONE / PARTIAL /
TODO**, with an evidence pointer and an effort estimate (person-days, pd).
The 2026-10-26 assessment (against `c51a835`) is the historical snapshot —
its four headline claims ("no install path, no published wheel, no published
numbers on main, GPU lane mid-flight (run-480 RUN 9)") were all superseded
before this re-assessment.

**Headline: the repo is an installable product in alpha flight.** The
install path exists (`pip install mlx-rocm`; PyPI `0.0.0` + `0.1.2`
uploaded 2026-10-30; `docs/quickstart.md`), the wheel build is
sha-verified and fail-loud (`scripts/build-wheel.sh` + `vendor.json`, the
pin re-pinned 2026-11-13 to a sha resolvable in the public fork), the
numbers are on `main` (`docs/audit/README.md` stack-of-record table,
`pins/`, the root `REPORT-W1-*` files), and the GPU validation is recorded
(107.56 t/s @1024 certified, #30 merged, CI green 2026-11-12).

---

## 1. Identity / docs

| Item | Status | Evidence | Effort |
|---|---|---|---|
| README external quality | **DONE** | `README.md` — user-facing: stage line (0.2.0-alpha), supported-models table (the #30-era record), the conversion recipe, the build story (the fetched-pinned-tree wheel path; the committed partial reference tree). | — |
| Quick start | **DONE** | `docs/quickstart.md` — the verified build + smoke path (6 sections, incl. the model-files-needed list with placeholders, the `QMLX_SERVE_PORT` doc, the canonical 3.12 venv Python). | — |
| Install guide | **DONE** | PyPI `mlx-rocm` (`0.0.0`, `0.1.2`, uploaded 2026-10-30) + `pip install mlx-rocm` (README) + the quickstart + the conversion section. | — |

## 2. Packaging

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Wheel build reproducible | **DONE** (the gate) | `scripts/build-wheel.sh` clones the `vendor.json`-pinned tree, sha-verifies the checkout, and fails loud on a mismatch or an unresolvable pin; `vendor.json` re-pinned 2026-11-13 to `aed6a25f` (resolves in the public fork; `ref main` == the pin). The committed `vendor/mlx-rocm-upstream/` is a byte-verified partial reference snapshot (no build manifest), documented as NOT the build input. Build is guest-a-only (ROCm + `gfx1201`), no CI job (by design — GPU is guest-a). | 1 pd (an empirical wheel build on the validated guest — the follow-up box session; the code/doc consistency + the resolvable pin are in place) |
| Versioning | **DONE** | `pyproject.toml`: `version = "0.2.0"`, classifier "3 - Alpha"; the single version story (0.0.0 → 0.1.0 → 0.1.1 → 0.1.2 → 0.2.0-alpha, PyPI: `0.0.0` + `0.1.2`) is stated in the README; `CHANGELOG.md` + `docs/release-notes-0.2.0.md` land in PR #31. | — |
| pyproject | **DONE** (for the scaffold) | `pyproject.toml`: setuptools backend, `requires-python >=3.11`, dev group (ruff, pytest), package-data for `*.so`, documented workaround for the license-table form (uv 0.10.9 vs setuptools 84). Correct for the *skeleton* package (`mlx_rocm` ctypes bindings); the full `mlx` wheel is built from the pinned tree, not this pyproject. | — |

## 3. Tests

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Hermetic CI status | **DONE** | `.github/workflows/ci.yml`: `uv sync --locked`, ruff check + format, `pytest -m "not gpu"`, size gate. `tests/test_gates.py` pins the load-bearing facts. CI green on `main` (`aed6a25` success, 2026-11-12 22:37Z). | — |
| GPU smoke matrix documented | **PARTIAL** | `docs/quickstart.md` documents the smoke ladder + the single verified ROCm 10.1; a check × ROCm-version × box matrix still does not exist. | 0.5 pd (one table) |

## 4. CI gates

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Main green | **DONE** | GitHub Actions: `main` @ `aed6a25` CI **success** (2026-11-12T22:37Z, PR #30 merge). | — |
| Required checks | **PARTIAL** (evidence pointer stale) | CI wiring + main-green hold; the 2026-10-26 branch-protection evidence no longer verifies — `branches/main/protection` 404s (protection may have moved to rulesets). No required check for the GPU lane — by design (`@pytest.mark.gpu` skips off-GPU; AGENTS.md). | 0.25 pd (re-verify via the rulesets API, refresh the evidence pointer) |

## 5. Models

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Documented supported set | **DONE** | README supported-models table with per-model status, now carrying the #30-era record (AR 97.52 / MTP 107.56 t/s @1024 certified; 84.8% of the 126.9 reference) + the hardware/ROCm compatibility statement (gfx1201 only). | — |
| Conversion recipes | **DONE** | README "Converting a model" (`mlx_lm.convert` bf16/4bit/8bit + `mlx-community/*` refs); linked from `docs/quickstart.md` (the model-files-needed section, placeholders). | — |

## 6. Quantisation

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Supported modes | **PARTIAL** | From the vendored tree: dense bf16/fp16 GEMM, 4/6/8-bit **affine** QMM, fp8 e4m3 non-affine, mxfp8/mxfp4/nvfp4 prefill via dequant→bf16 GEMM; NO_GPU gaps: FFT, LAPACK, BlockMaskedMM. Still no user-facing quantization-modes summary page. | 0.5 pd (one page) |
| 4-bit affine status | **DONE** | 4-bit qmm validated + benched (`docs/audit/k2-non-gemm-fusion.md`; the W1 4-bit records on `main`); the K1 N=12288 fix + kernel are in the committed vendor tree (`docs/vendor-overlay-k1.md`). | — |
| FP8 caveat | **DONE** (documented) | The known gfx1201 fp8 e4m3 activation-GEMM driver bug is documented (README "gfx1201 notes", `docs/plan.md` §4.2, `docs/limitations.md`) and enforced in code (`scripts/smoke-serve.sh` ships `MLX_ROCM_GEMM_BF16=1`); the real prefill validation is still open. | 0.5 pd (close out the prefill test) |
| MXFP4 in flight | **PARTIAL** | mxfp4 prefill exists only as the opt-in dequant→bf16 GEMM path (`qmm.hip:3475-3499`); no native mxfp4 kernel; the K1 kernel is shipped in-tree. | 1 pd (land k4 + document) |

## 7. Benchmarks

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Published numbers | **DONE** | Numbers are on `main`: `docs/audit/README.md` (the stack-of-record table — superseded-by marker to the W1 record), `pins/` (the 69.43 and 92.23 wheels), the root `REPORT-W1-PHASE1/PHASE2-20261112.md` (the 107.56 t/s @1024 certified record, 84.8% of the 126.9 reference). No `reports/README.md`-style index (the audit README is de facto the index). | 0.25 pd (a one-line index pointer) |
| Method | **PARTIAL** | Protocols are in the audit docs; no canonical "how we benchmark" page on `main`. | 0.5 pd (promote the protocol to a `main` doc) |

## 8. Known limitations

**DONE (2026-11-12 re-assessment).** `docs/limitations.md` exists and is
refreshed to the #30-era state: MTP K=2 + W1 in "What works" (the
107.56 record + the same-session battery), 9B-hybrid supported, G1
retired (NO-GO), K1 closed as shipped, the publish state real (PyPI
`0.0.0` + `0.1.2`), the GDN long-prefill caveat post-#30.

## 9. Housekeeping

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Branch cleanup (k1/k2/k3/g1) | **DONE** | `k2-non-gemm-fusion` (#2) and `k3-gdn-port` (#3) merged; k1/g1/alpha branches absent from origin (the K1 kernel + fix are in the committed vendor tree; G1 retired — NO-GO). | — |
| .gitignore | **DONE** | `.gitignore`: build artifacts, `*.whl`, `*.so`, `scripts/smoke-results/`, `.venv`; explicit note that `vendor/` is NOT ignored (committed, byte-verified). No leaks observed on `main`. | — |
| LICENSE | **DONE** | `LICENSE` is MIT (Apple MLX, © 2023); README "MIT (MLX upstream license)", `pyproject.toml` "License :: OSI Approved :: MIT License" — the three historical claims fixed to MIT. | — |
| Vendor sync process | **DONE** (the pin) | `vendor.json` pins the public fork `main` @ the sha (re-pinned 2026-11-13 to a resolvable sha; `ref` == the pin), the committed partial tree documented as the byte-verified reference (not the build input), `build-wheel.sh` sha-verifies + fails loud, `tests/test_gates.py` pins the facts. Residual: the re-pin cadence is stated ("re-pin in a PR when the base moves"), not a calendar rule. | — |
| Doc consistency | **DONE** | Single ROCm 10.1 story with explicit supersession notes (`vendor.json`, README, `docs/plan.md`); the license fixed to MIT; `PLAN.md` current-state line refreshed to the #30-era state. | — |

---

## Top-5 critical path to alpha (re-assessed 2026-11-12)

1. **GPU validation on the validated guest — DONE** (#30 merged 2026-11-12;
   107.56 t/s @1024 certified, lossless 3/3, in-tree repro bit-identical;
   the `scripts/smoke.py` + `scripts/smoke-serve.sh` ladder documented in
   `docs/quickstart.md`).
2. **Installable: DONE** (PyPI `0.0.0` + `0.1.2`, the quickstart, the
   conversion recipe, the model-files-needed section).
3. **Numbers on `main`: DONE** (the audit index, `pins/`, the root
   `REPORT-W1-*` files, the 107.56 record).
4. **Vendor re-pin + reproducible build: DONE at the gate** (the pin
   resolvable in the public fork, the sha-verified fail-loud gate, the
   committed tree documented as the partial reference); the empirical
   wheel build on the validated guest is the follow-up box session.
5. **Docs + housekeeping: DONE** (the MIT license, the single ROCm 10.1
   story, the supported-models table + conversion recipe,
   `docs/limitations.md` refreshed, the WIP branches resolved).

**Residual to a green alpha gate:** the empirical wheel build on the
validated guest (item 4 follow-up); the check × ROCm-version × box smoke
matrix (item 3/§3); the rulesets evidence pointer (§4); the
quantization-modes page + the canonical bench-method page (§6/§7).
