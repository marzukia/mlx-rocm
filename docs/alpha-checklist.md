# Alpha-state assessment — mlx-rocm

**Alpha definition:** a first-time external user can install it, load a model, run
generation, and trust the numbers.

Assessed 2026-10-26 against `main` @ `c51a835` (merge of PR #1, 2026-10-25) plus the
WIP branches on the mirror/origin. Status per item: **DONE / PARTIAL / TODO**, with an
evidence pointer and an effort estimate (person-days, pd).

**Headline: the repo is a validated *scaffold*, not an installable product.** The
vendored ROCm backend is adopted and byte-verified, hermetic CI is green, and the
build/smoke plumbing exists — but there is no install path for an external user, no
published wheel, no published numbers on `main`, and the GPU validation lane (guest-a)
is still mid-flight (run-480 RUN 9: wheel build #5 in progress, 2026-10-26).

---

## 1. Identity / docs

| Item | Status | Evidence | Effort |
|---|---|---|---|
| README external quality | **PARTIAL** | `README.md` — accurate and well-cited, but explicitly "MVP skeleton (v0.0.1)": describes the scaffold (vendor tree + standalone hipBLASLt GEMM + `smoke.sh`), not a user-facing product. "The wheel does not yet bundle the full `mlx` C++ core — that is the next milestone." No "what can I run with this" section. | 0.5 pd (rewrite as user-facing once Phase 0/1 land) |
| Quick start | **TODO** | No zero→generation path anywhere. Closest: `README.md` "Build" (host-only cmake, ROCm build, wheel `CMAKE_ARGS` one-liner) — all aimed at *building on guest-a*, not at a user installing. | 0.5 pd (after the wheel exists) |
| Install guide | **TODO** | No documented install: no PyPI release (PLAN.md open question 2 defers it to Phase 2+), no wheel artifact location, no "pip install <wheel>" + "run a model" section. `scripts/build-wheel.sh` produces `dist/*.whl` on guest-a but nothing tells a user where to get it. | 1 pd (artifact + doc + quick start) |

## 2. Packaging

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Wheel build reproducible | **PARTIAL** | `scripts/build-wheel.sh` clones the vendor fork and **checks out the branch ref** (`git fetch --depth 1 origin "$REF" && git checkout FETCH_HEAD`) — the `vendor.json` sha (the vendor ref) is advisory, not enforced; a branch move silently changes the build. The committed `vendor/mlx-rocm-upstream/` is byte-verified vs the sha (`vendor/mlx-rocm-upstream/README.md`), but the build script doesn't use it (it clones to `$MLX_VENDOR_DIR`). Build is guest-a-only (ROCm + `gfx1201`), no CI job, no pinned toolchain/ROCm version in the script. | 1 pd (checkout the pinned sha, or build from the committed vendor tree; pin ROCm version) |
| Versioning | **PARTIAL** | `pyproject.toml`: `version = "0.0.1"`, classifier "Development Status :: 1 - Planning". No PyPI release, no changelog, no versioning policy. PyPI slot `mlx-rocm` is held (docs/plan.md §5). | 0.5 pd (policy + first release on Phase 0/1 pass) |
| pyproject | **DONE** (for the scaffold) | `pyproject.toml`: setuptools backend, `requires-python >=3.11`, dev group (ruff, pytest), package-data for `*.so`, documented workaround for the license-table form (uv 0.10.9 vs setuptools 84). Correct for the *skeleton* package (`mlx_rocm` ctypes bindings); the full `mlx` wheel is built from the vendor tree, not this pyproject. | — |

## 3. Tests

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Hermetic CI status | **DONE** | `.github/workflows/ci.yml`: ruff check + format, `pytest -m "not gpu"`, size gate. `tests/test_gates.py` pins the load-bearing facts (vendor pin well-formed, build flags, gfx1201 naming, script inventory, CI wiring). CI green on `main` (`c51a835` success, 2026-10-25; see §4). | — |
| GPU smoke matrix documented | **TODO** | The checks exist but no matrix documents them: `tests/test_gpu.py` (is_available, matmul-vs-CPU, softmax — Phase 0 exit criteria only; file header says Phase 1 "extends this file" — not yet), `scripts/smoke.py` (Phase 0 + optional 0.6B generate), `scripts/smoke-serve.sh` (Phase 1 9B serve, `MLX_ROCM_GEMM_BF16=1` default). Nothing documents *which* checks run *when* on *which* ROCm version — and the ROCm version is itself inconsistent across docs (6.4 in README/vendor.json, 7.7 in AGENTS/PLAN, 10.1 in the run-480 guest VM). | 0.5 pd (one table: check × ROCm version × box; fix the version story) |

## 4. CI gates

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Main green | **DONE** | GitHub Actions: `main` @ `c51a835` CI **success** (2026-10-25T08:39Z); `3ab5099` success. Latest runs on WIP branches (k2 `228b29f`, k3 `dcf6a9b`, `d03cd21`) all success. | — |
| Required checks | **DONE** | Branch protection on `main`: enabled; required status checks `lint-and-check` + `size`, enforcement level `everyone` (GitHub API, 2026-10-26). No direct pushes (AGENTS.md). Note: no required check for the GPU lane — by design (GPU runs on guest-a, `@pytest.mark.gpu` skips off-GPU; AGENTS.md). | — |

## 5. Models

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Documented supported set | **PARTIAL** | The *plan* defines the set: Qwen3-0.6B (bf16 + 8bit, Phase 0 smoke — `scripts/smoke.py`), 9B-class dense Qwen3 (Phase 1 minimum — PLAN.md "Definition of minimum"), 27B+/9B-hybrid + MoE (Phase 2, deferred). But there is no user-facing "supported models" list, no per-model status (which are validated on gfx1201, which are expected-to-work, which are known-broken), and no hardware/ROCm compatibility statement. | 0.5 pd (one table, filled as validation lands) |
| Conversion recipes | **TODO** | No documented recipe for getting a model into the wheel (e.g. `mlx_lm.convert` to 4bit qmm, HuggingFace `*-mlx` refs). The k2 bench report references `qwen3-9b-mlx` / `qwen3-06b-mlx` model dirs on the guest (`/models/`) without saying how they were produced. | 0.5 pd |

## 6. Quantisation

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Supported modes | **PARTIAL** | From the vendored tree (`vendor/mlx-rocm-upstream/mlx/backend/rocm/`): dense bf16/fp16 GEMM (hipBLASLt + rocBLAS fallback, `gemms/`), 4/6/8-bit **affine** QMM (WMMA dense `qmm_wmma_dense_kernel` + wide QMV decode + int8-dot QMV, `quantized/qmm.hip`), fp8 e4m3 non-affine, mxfp8/mxfp4/nvfp4 prefill via dequant→bf16 GEMM fast path (`qmm.hip:3475` — "the only functional mxfp8 prefill GEMM on RDNA4 today"). NO_GPU gaps: FFT, LAPACK (LUF/QRF/SVD/Inverse/Eig/Eigh), BlockMaskedMM, GatedDeltaUpdate (per-timestep fallback — slow for long prefill). None of this is summarized in a user-facing doc. | 0.5 pd (one page) |
| 4-bit affine status | **PARTIAL** | The default 4-bit affine path is in and tuned for Rdna4 (WMMA allowlist includes gfx1200/1201, `qmm.hip:620-631`; wide QMV decode "DEFAULT ON", `qmm.hip:3898`). **Not yet validated on gfx1201** — Phase 1 step 1 (QMM 4bit numerics vs CPU) is open; the k1 branch (`k1-qmv-gemv-bandwidth`, `99cddb5`) fixed the int8-activation QMV bug at N=12288 but "no bench yet". | 1 pd (guest-a validation lane) |
| FP8 caveat | **DONE** (documented) | The known gfx1201 fp8 e4m3 activation-GEMM driver bug is documented in three places: `README.md` "gfx1201 notes" (probe passes, max_rel_err 0.0057, but long-context prefill "may be kernel-buggy"), `docs/plan.md` §4.2, and enforced in code: `scripts/smoke-serve.sh` exports `MLX_ROCM_GEMM_BF16=1` as the safe default; `vendor.json` records the probe PASS. Caveat: the fp8 *prefill* validation on the real box is still open (Phase 1 step 3). | 0.5 pd (close out the prefill test, then update the doc) |
| MXFP4 in flight | **PARTIAL** | mxfp4 prefill exists only as the opt-in dequant→bf16 GEMM path (`qmm.hip:3475-3499`); no native mxfp4 kernel. The k1 branch (int8-activation QMV) is the in-flight work adjacent to it. No user-facing statement of what mxfp4/mxfp8 means for a user today. | 1 pd (land k1 + document) |

## 7. Benchmarks

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Published numbers | **PARTIAL** (not on `main`) | Real numbers exist but live on WIP branches, not `main`: `reports/k2-non-gemm-fusion.md` (on `k2-non-gemm-fusion`): 9B-4bit A/B 55.9–57.8 vs 51.1–55.5 tok/s; 0.6B-8bit 207.6/214.3 vs 177.2/180.6 tok/s (+17–18%); dispatch-count delta from rocprofv3. Also `reports/` on `opt-gauntlet-areaA/B`, `low-hanging-fruit-battery`, `<opt>-steal-recon`, `k3-gdn-port`. Prior-art numbers (Strix Halo 151.2 tok/s @ 0.6B; RX 9060 ~104 tok/s @ 0.8B fp8) are in `docs/plan.md` §3. An external user on `main` sees **zero** published numbers. | 0.5 pd (land `reports/` + a `reports/README.md` index on main) |
| Method | **PARTIAL** | The k2 report documents a solid protocol (200-tok warm decode, 16-tok warmup, fresh process, flock, timeout-300, same runner both arms, re-anchored baselines with confounding noted). But it's branch-local, and there is no canonical "how we benchmark" doc on `main` (hardware, ROCm version, warmup, locking, which tok/s definition). | 0.5 pd (promote the k2 protocol to a `main` doc) |

## 8. Known limitations

**TODO — no honest limitations list exists on `main`.** The ingredients are scattered across
`docs/plan.md` (hazards §2.1, prior-art issue M1, Phase 2 open items) and `README.md`
(fp8 note). Consolidated, the honest list is:

1. **No install path yet** — the wheel is built on guest-a; nothing is published (PyPI slot
   held but unreleased; PLAN.md open question 2 defers it).
2. **fp8 e4m3 prefill on gfx1201** — possible driver kernel bug; `MLX_ROCM_GEMM_BF16=1`
   is the shipped default until the real-box prefill test proves it clean (docs/plan.md §4.2).
3. **GatedDeltaUpdate** — runs via a backend-agnostic per-timestep CPU-style fallback;
   fine for decode (T=1), slow for long prefill (T=4096 ≈ 32k small kernel launches).
   Main *performance* gap for the 9B hybrid (docs/plan.md §1.3).
4. **NO_GPU op set** — FFT, LAPACK family (LUF/QRF/SVD/Inverse/Eig/Eigh), BlockMaskedMM
   fall back to CPU (docs/plan.md §2 table).
5. **Dual large-model load → SIGSEGV** (exit 139) on unified-memory APUs — resource
   overcommit, not a single-stream kernel bug; not the blocking case for a single
   discrete dGPU, but the one to watch if two models run (docs/plan.md §3.1, fork RESULTS.md).
6. **Allocator hazards on discrete GPUs** — BAR read ~54 MB/s vs hipMemcpy 50.4 GB/s
   (pinned `host_stage` bounce engineered around it); async `hipMallocAsync` pool off by
   default (caused gfx page faults / IMA 700 on MI300X) (docs/plan.md §2.1).
7. **Wavefront 32/64** — a fatbin pinning `WARP_SIZE` to 64 mis-launches on RDNA; the
   fork queries `warpSize` at runtime and defaults RDNA to 32 (docs/plan.md §2.1).
8. **MTP spec decode** — blocked on the M3 stream-TLS fix (`There is no Stream(cpu, N)`
   in `mlx/backend/cpu/encoder.cpp`); Phase 2 at the earliest (docs/plan.md §Phase 2).
9. **ROCm version story is unsettled** — README/vendor.json say 6.4 (the probe
   validation), AGENTS/PLAN say 7.7 (the target), the run-480 guest VM runs 10.1. A user
   cannot tell which ROCm they need.
10. **GPU validation is in flight** — Phase 0/1 exit criteria (is_available, numerics,
    0.6B sane, 9B served, no NaNs, no OOM→SEGV) are not yet recorded as a pass on
    guest-a (run-480 RUN 9: wheel build #5 running, 2026-10-26).

Effort to publish this list: 0.5 pd (promote to `docs/limitations.md` + README link).

## 9. Housekeeping

| Item | Status | Evidence | Effort |
|---|---|---|---|
| Branch cleanup (k1/k2/k3/g1) | **TODO** | All four WIP branches exist on the bare mirror **and** origin (GitHub, 2026-10-26). PR state: #2 (`k2-non-gemm-fusion`) and #3 (`k3-gdn-port`) **open** (CI green, k2's bench verdict "ADOPT"); `k1-qmv-gemv-bandwidth` and `g1-graph-decode` have **no PR** (k1: N=12288 fix landed, "no bench yet"; g1: ROCm 10.1 `hipStreamIsCapturing` fix, pilot wiring). Additionally ~30 `fleetwork-salvage/*` + `feature/*` + `notes/*` + `run-480-report` branches clutter the mirror. | 1 pd (PR k1/g1 or archive; salvage-branch sweep) |
| .gitignore | **DONE** | `.gitignore`: build artifacts, `*.whl`, `*.so`, `scripts/smoke-results/`, `.venv`; explicit note that `vendor/` is NOT ignored (committed, byte-verified). No leaks observed on `main`. | — |
| LICENSE | **PARTIAL** | `LICENSE` is **MIT** (Apple MLX, © 2023), and `vendor/mlx-rocm-upstream/LICENSE.upstream` is the same MIT text — but `README.md` says "Apache-2.0 (MLX upstream license; see `LICENSE`)", the `pyproject.toml` classifier says "OSI Approved :: Apache Software License", and `vendor/mlx-rocm-upstream/README.md` says "Apache-2.0 (see `LICENSE.upstream`)". The claim and the file disagree. | 0.25 pd (pick one; MLX upstream is MIT — fix the three claims) |
| Vendor sync process | **PARTIAL** | Process is documented (AGENTS.md: "re-pin in a PR when the base moves"; `vendor.json` primary/fallback + build flags; committed byte-verified tree; `tests/test_gates.py` pins the facts). Gaps: (a) `build-wheel.sh` checks out the **branch** (`rocm-current`), not the pinned sha — the pin is advisory; (b) the pin (the vendor ref) is 2026-09-28 (~4 weeks old), the branch head is still the same sha (no drift yet), but the ROCm-base head it forward-merges is 2026-07-19 (~3 months) and the MLX base it merges onto (`59d600b5`, 2026-09-17) is ~5 weeks behind MLX main (`56c2bf9`, 2026-10-23). No re-pin cadence or "how fresh is too stale" rule. | 1 pd (sha-pinned checkout + re-pin PR when the base moves) |
| Doc consistency | **PARTIAL** | Three plan docs: `PLAN.md` (living plan, "Current state 2026-10-25: GPU validation still to be done on guest-a"), `docs/plan.md` (research report, 2026-10-25), `README.md` (scaffold, "ROCm 6.4 userland"). Inconsistencies: (1) ROCm version — 6.4 (README + vendor.json validation note) vs 7.7 (AGENTS/PLAN/plan) vs 10.1 (run-480 guest VM, `REPORT.md` on `run-480-report`); (2) license — Apache-2.0 claimed, MIT file (§9 above); (3) PLAN.md's "current state" predates the run-480 VM runs (wheel build #5 in flight 2026-10-26) — stale by a day, and it says "submitter is building the MVP on guest-a" while the guest VM lane is now the active path. | 0.5 pd (single source of truth for version + state; refresh PLAN.md) |

---

## Top-5 critical path to alpha

Ordered; blockers named.

1. **Land Phase 0/1 GPU validation on guest-a (the wheel that works).**
   Build the wheel, pass `scripts/smoke.py` (is_available, matmul/softmax, 0.6B sane) and
   `scripts/smoke-serve.sh` (9B dense Qwen3 served, no NaNs, no OOM→SEGV), and record
   the pass. *Blockers:* the guest is busy with other lanes (no guest GPU work this
   lane — run-480 RUN 9 left wheel build #5 running detached); the ROCm version must be
   settled first (6.4 vs 7.7 vs 10.1) or the pass proves the wrong thing; the 4-bit
   QMM + fp8-prefill + allocator checks (Phase 1 steps 1–4) are all still open.
2. **Make it installable: artifact + quick start + install guide.**
   A first-time user needs one documented path: get the wheel (artifact location or the
   held PyPI slot), `pip install` it, load a model, generate. *Blockers:* no published
   wheel or artifact location today; the build is guest-a-only with no pinned ROCm
   version; `build-wheel.sh` checks out the moving branch, so "the wheel" is not
   reproducible from `vendor.json` alone; the README/PLAN defer PyPI to Phase 2+ —
   alpha needs at minimum a documented wheel artifact + quick start.
3. **Publish the numbers: land `reports/` + benchmark method + GPU smoke matrix on `main`.**
   "Trust the numbers" fails today because every published number (k2 A/B, gauntlet,
   <opt>) lives on WIP branches. *Blockers:* `reports/` is not on `main`; the GPU
   smoke matrix (check × ROCm version × box) is undocumented; the numbers' ROCm-version
   provenance is ambiguous (10.1 guest vs 7.7 target) until item 1 settles it.
4. **Vendor re-pin + reproducible build.**
   Re-pin the vendor ref (2026-09-28) against the ROCm-base head (2026-07-19) / MLX
   main (2026-10-23, `56c2bf9`), make `build-wheel.sh` check out the pinned sha (or
   build from the committed byte-verified tree), and pin the ROCm toolchain version.
   *Blockers:* protected `main` + owner merge gate (branch + PR, CI green, adversarial
   review by a different agent, peer merges `--squash`); the committed vendor tree and
   `tests/test_gates.py` facts must move in the same PR; the guest build lane is the
   only place a re-pinned tree can be validated.
5. **Docs + housekeeping: license, ROCm version, supported models, branch sweep.**
   Fix the Apache-2.0-vs-MIT claim (README + pyproject + vendor README), one ROCm
   version everywhere, a user-facing supported-models table + conversion recipe, a
   `docs/limitations.md` (the list in §8), and close out the WIP branches (k1/g1 have
   no PR; k2/k3 PRs #2/#3 are open and green). *Blockers:* protected `main` + owner
   merge gate for every fix; the limitations/supported-models content depends on item 1
   (you can't honestly document what works until the validation pass is recorded).
