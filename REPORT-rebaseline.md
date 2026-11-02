# REPORT-rebaseline — canonical-wheel re-baseline round (build + 2×2 GPU anchor runs)

**Task:** restore the canonical reference wheel so the mission anchors
(69.48 B0 / 75.00 rows-ON tok/s; verify 25.71 → 22.13 ms; pair
44.90 → 41.35 ms) are reproducible again.

**Verdict: NO-GO.** The wheel rebuilt byte-faithfully from the canonical
tree (91d7df5 vend tree + the canonical graft set) runs B0 at 59.47
(−14.4%) and R1 rows-ON at 64.54 (−13.9%) — both far outside the ±3%
bands. The in-session noise is 0.05–0.09%, the tape engages 128/128 on
all four runs, and the run signature (93/59/27 D-ACC, 101/99 n_refwd,
34.4/30.8 verify) is the exact rows-tuning control-matrix signature
(56.3–60.2 band). The anchors are NOT restored; per the round rule the
baseline is NOT re-calibrated by decree. The pinned wheel/venv record is
frozen for the next attempt (e.g., after a guest-box state reset).

---

## 1. Control-matrix reference (REPORT-rows-tuning-gpu.md, t1355)

| wheel (built) | tok/s | verify ms | n_refwd | D-ACC |
|---|---|---|---|---|
| 03:05Z wheel (lost to venv overwrites) — historical | **70.20** | **25.60** | 100 | 90/56/28 |
| `+75ca881` + tuning (t1355 battery) | 59.54 / 59.40 | 34.25 / 34.37 | 101 | 93/59/27 |
| `+73346f3` (WMMA-v3 tree) | 59.46 | 34.32 | 101 | 93/59/27 |
| `+75ca881` wheelout-rs (02:10Z, no tuning) | 56.26 | 36.35 | 101 | 93/59/27 |
| `+75ca881` wheelout-v3 | 59.42 | 34.37 | 101 | 93/59/27 |
| `+75ca881` wheelout-tape (Nov-1) | 60.23 | 34.46 | 98 | 100/66/30 |

Established facts carried in: in-session noise ~0.2%; the 70.2 baseline
was last reproduced 03:05Z; every wheel on the guest since lands
56.3–60.2 with D-ACC 93/59/27 (or 100/66/30 for the Nov-1 tape
wheel) — a wheel-independent, guest-state-level difference.

## 2. Build (guest box, CPU-only phase, no GPU lock)

- Fresh pristine clone of the fork at `75ca881b2e82f7b4069c8564a203792aa16d8a29`
  (verified: `git rev-parse HEAD` = the pin; clean status).
- Graft 1 — the `worker/rows-statistical @91d7df5` committed vend tree:
  **738 files, every file md5-verified** against the branch manifest.
- Graft 2 — the canonical guest graft delta (the working-tree delta of
  the guest's canonical build tree vs pristine 75ca881, minus the
  `.bak`/`.pre-wsfix` junk): **39 files, every file md5-verified**.
  Includes `gdn_fused_step.hip`, `gdn_fused_step_tape.hip` (+ CMake
  entries), the `fast.{h,cpp}` / `fast_primitives.h` /
  `python/src/fast.cpp` op wiring, the verify-attention-s4 SDPA patch,
  `gated_delta_update.cpp`, the `qmm.hip` rows/hoist dispatch
  (both gates default-off), `qmv_lx_kernel.hip`,
  `qmv_wide_mr_kernel.hip`, `qmv_rows_msg_kernel.hip` (carries the
  inert t1334 `qmv_rows_msg_hoist_kernel` append; the base rows kernel
  section is byte-identical to the 91d7df5 committed version — diff is
  append-only), and the `cuda`/`cpu`-simd files.
- **Vend-tree defect found and fixed:** the committed 91d7df5
  `mlx/backend/rocm/device/fp16_math.hpp` carries a failed
  `s/t1/run-1/` scrub artifact (`bfloat162float` →
  `bfloarun-162float`, 1 site; the same artifact is present in 5
  `cuda` files, which are byte-identical to the guest's canonical
  build-tree copies and are not compiled in a ROCm build — left as-is).
  The ROCm-compiled `fp16_math.hpp` was restored from pristine 75ca881
  (md5 `84f17f85…` = the canonical build tree's copy) before building.
  The first build attempt (with the artifact) died on exactly that line
  — `error: expected ';' after top level declarator` at line 151.
- Build command:
  `CMAKE_PREFIX_PATH=/opt/rocm CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS" /usr/bin/python3 -m pip wheel . --no-cache-dir`.
  Two deviations from the task-stated recipe, both forced:
  (a) `--no-build-isolation` is not runnable on this box (no
  scikit-build-core in the system python; the task command falls back
  to the legacy setup.py path) — the proven recipe
  (`build-rs3.log`, the 02:10Z canonical wheel) is the isolated pip
  wheel build, used here; (b) the first isolated attempt was a silent
  pip **wheel-cache reuse** (identical version string
  `dev20261102+75ca881`; 0 compiled objects) — re-run with
  `--no-cache-dir`, verified a genuine compile (186 fresh object
  files, full build dir).
- Final tree manifest: 1863 files md5-listed (`canon-tree.md5`).

## 3. Wheel / venv record (pinned, frozen)

| item | value |
|---|---|
| venv | `venv-canon` — NEW dedicated venv on the guest box (CPython 3.14.8, same layout/versions as the anchor venvs; created as a read-only copy of the anchor venv — the source venv was never touched — then the mlx package replaced by zipfile extraction of this wheel; `qwen3_5.py` = the proven patched copy, md5 `7763fbbb498129ac2750811be9154b5b`, FUSE_BRANCH_PATCH + TAPE_BRANCH_PATCH present). **No other run may modify it.** |
| wheel name | `mlx-0.32.3.dev20261102+75ca881-cp314-cp314-linux_x86_64.whl` (22,479,213 B) |
| wheel sha256 | `5bc7ca20509a5beebb894828cc654135568eeb87fdb31bd7a080822cda12d18d` |
| wheel md5 | `5e5ee87841d076dac238f4d8c9779476` |
| libmlx.so md5 | `b1de59fda0e4fc607d620e287deeaaf8` |
| bench script | `mtp-bench-k3-tape.py` md5 `2c2d6c90c2349ae087e53e5d6c60018e` — VERIFIED before use |
| commit guard | `gdn_tape_commit.py` md5 `4a262a0d9239eea7326606cdfb177c60` |
| sidecar | `model-mtp.safetensors` active (973,164,115 B); 4-bit sidecar `.disabled` — gate PASS |

## 4. Capability probe (PASS)

qwen3_5.py dispatch (grep of `mx.fast.*`): exactly `gdn_fused_step` and
`gdn_fused_step_tape` (the tape op is the superset selected inside the
FUSE branch → both `MLX_ROCM_GDN_TAPE=1` and `MLX_ROCM_GDN_FUSE=1`
required; TAPE alone silently runs all-fallback).

```
version: 0.32.3.dev20261102+75ca881
hasattr(mx.fast, 'gdn_fused_step_tape'): True   (strings in venv libmlx.so: 45)
hasattr(mx.fast, 'gdn_fused_step'): True        (strings in venv libmlx.so: 92)
CAPABILITY PROBE: PASS
```

## 5. Anchor re-verification (GPU, exclusive window)

Lock `gpu-bench.lock`: acquired on the first 15 s poll, held across all
four runs, released after the last arm (verified free afterward; no
orphan GPU processes left). Env per run: `MODELS=<model mount>`,
`LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`,
`MLX_MTP=4`. B0 = rows OFF (tape env only). R1 = statistical rows env
(`MLX_ROCM_QMV_ROWS=1`, `MLX_ROCM_QMV_ROWS_MIN_N=1`, per the t1307
canonical run). Two runs per arm:

| arm | tok/s | verify ms | pair ms | n_refwd | D-ACC | TAPE n_ok |
|---|---|---|---|---|---|---|
| B0-1 | 59.45 | 34.34 | 53.62 | 101 | 93/59/27 | 128/128 |
| B0-2 | 59.48 | 34.41 | 53.59 | 101 | 93/59/27 | 128/128 |
| **B0 median** | **59.47** | **34.38** | **53.61** | 101 | — | — |
| R1-1 | 64.57 | 30.76 | 49.85 | 99 | 95/61/29 | 128/128 |
| R1-2 | 64.51 | 30.80 | 49.90 | 99 | 95/61/29 | 128/128 |
| **R1 median** | **64.54** | **30.78** | **49.88** | 99 | — | — |

### Delta table vs the mission anchors

| metric | anchor | measured | delta | band | gate |
|---|---|---|---|---|---|
| B0 tok/s | 69.48 | 59.47 | **−14.4%** | 67.40–71.56 | **FAIL** |
| B0 verify ms | 25.71 | 34.38 | +8.67 ms (+33.7%) | ~25.7 ±3% | FAIL |
| B0 pair ms | 44.90 | 53.61 | +8.70 ms (+19.4%) | ~44.9 ±3% | FAIL |
| R1 tok/s | 75.00 | 64.54 | **−13.9%** | 72.75–77.25 | **FAIL** |
| R1 verify ms | 22.13 | 30.78 | +8.65 ms (+39.1%) | ~22.1 ±3% | FAIL |
| R1 pair ms | 41.35 | 49.88 | +8.53 ms (+20.6%) | ~41.3 ±3% | FAIL |
| n_ok | 128/128 | 128/128 ×4 | — | required | PASS |
| in-session noise | — | B0 0.05%, R1 0.09% | — | — | stable |
| R1/B0 win ratio | +7.2% (t1307) | +8.55% | preserved in relative terms | — | — |

Supporting observations:

- The B0 single-step ms (9.75) matches the canonical run (9.75)
  exactly; the entire delta sits in verify/GDN (34.38 vs 25.71,
  +8.67 ms) — the same verify-path-specific pattern the t1355 control
  matrix documented.
- This wheel's B0 (59.47, 93/59/27, 101 n_refwd, 34.38 verify) is
  within 0.2% of the t1355 battery wheel (59.54/59.40, 93/59/27,
  101, 34.25/34.37) — the same signature as every wheel on the guest
  box since ~09:41Z, i.e. the wheel-independent guest-state class.
- Post-run forensics: no orphan GPU processes; the HSA runtime
  (`libhsa-runtime64.so.1.21.0`) and ROCm userland library mtimes
  predate the drift window (Oct 6 / Jan 10) — no library replacement
  found; the 09:41–09:52Z concurrent activity (vllm anchor installs)
  left only user-space scripts, no ROCm library changes; DPM was
  verified default in the t1355 round. The drift window is therefore
  between 03:05Z (last 70.20) and ~09:41Z, unexplained by the
  inspectable file state.

## 6. Verdict and standing record

- **NO-GO** (B0 and R1 both far off; n_ok gates passed, so the runs are
  valid measurements, not protocol failures). **Not re-baselined by
  decree** — the 69.48/75.00 calibration stands.
- The pinned record (section 3) is the canonical-tree wheel of this
  round: `venv-canon` is FROZEN (no other run may modify it) and is
  the ready vehicle for the next attempt — it reproduces the canonical
  graft set byte-for-byte (modulo the inert default-off t1334 hoist
  append, never launched in any arm). If a guest-box state reset
  (reboot / driver reload) restores the 03:05Z behavior, this exact
  wheel+venv should be re-benched before any new build is attempted.
- Next (operator call, per t1355 section 3): (a) reset the guest-box
  GPU/driver state and re-run this pinned wheel, or (b) re-calibrate
  the B0 reference + target to the current stack — both move the 100+
  tok/s target framing; this round deliberately does neither.

## 7. Persona scrub

Scrubbed before commit: 1 hostname literal (→ "the guest box"), 3
guest-user path literals (→ `venv-canon` / "on the guest box"), 1
model-mount literal (→ `<model mount>` / env `MODELS`), 1 tailnet IP
literal (→ "the guest box"). 0 pattern hits on the committed file.
