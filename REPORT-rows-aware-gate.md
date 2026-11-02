# REPORT-rows-aware-gate — the rows-aware REFWD_M1 gate + stacked A/B (t1391 fix round)

**Task:** one conjunct in the graft engage gate — engage the M=1 32-col lx re-route ONLY when rows is OFF — then the standard 4-arm battery on one wheel, 2 runs/arm, medians. Fix for the `REPORT-stack-ab.md` finding (R3 −1.51% below R0: the M=1 re-route re-routes the M=1 QMMs onto the slow lx path while rows is ON, refwd +1.9 ms, n_refwd +12, D-ACC collapse 95/61/29 → 85/45/17).

**Verdict: the collision is FIXED. R3 = 65.58 tok/s, +9.90% vs R0 (was −1.51% on the candidate wheel) and above R2 (65.555, +0.04%, within noise — the stacked arm now takes rows' win with zero toggle penalty).** R3's signature is bit-identical to R2's (D-ACC 95/61/29, n_refwd 99, refwd 9.08 ms, first20 = R0's stream) and R1's standalone signature is unchanged (n_refwd 94, D-ACC 99/62/34): the conjunct landed exactly where it should. New best standing this session: 65.58 tok/s (R3, STACKED — the first stacked standing of the program).

---

## 1. Setup (verified before the battery)

- **Build tree:** the graft build tree of the wheel-reconcile round (the 1910-file manifest tree, `wheel-recon/src` on the guest box). After the one-conjunct change, the md5 manifest vs the pre-change 1910-file manifest differs in **exactly one file** (`qmm.hip`); all 1910 files present. - **Graft diff (one-liner):** the K1 engage gate in `quantized/qmm.hip` gains a `!rows_active` conjunct on the M=1 REFWD_M1 re-route term, with `rows_active` the rows dispatch's own predicate (`MLX_ROCM_QMV_ROWS` set) — one new static, one changed expression, nothing else; default-inert (rows OFF → the gate is exactly the candidate wheel's). - **Wheel:** `mlx-0.32.3.dev20261102+75ca881-cp314-cp314-linux_x86_64.whl` (22,479,133 B; sha256 `3f29fa58f5a4ab94a80361edbfb62e7d50dc957240381ca868e0e60dc16ce0f0`; distinct from the candidate `f6873c65…`). Built with the canonical recipe (`CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -Drocwmma_DIR=/opt/rocm/lib/cmake/rocwmma" pip wheel . --no-deps`, incremental in the graft build tree; `qmm.o` recompiled + relinked). - **venv:** FRESH scratch `venv-ra` — read-only copy of the wheel-reconcile round's scratch venv (`venv-wrc`, `libmlx.so` md5 `5bf011c3…`), the `mlx` package replaced by zipfile extraction of the new wheel (`libmlx.so` md5 `93810e871f7bc977e723d5288a3b73e2`). One operational note: the first extraction pass swept the adjacent `mlx_lm` package (name prefix match); it was restored verbatim from the untouched read-only source before any GPU run, and the first (failed, import-error) battery pass at 16:30:58Z was discarded — the data below is the clean 16:34:54Z pass. `venv-wrc` re-verified unchanged after (`libmlx.so` still `5bf011c3…`); `venv-canon` untouched (`b1de59fd…`). - **Wheel verification (pre-battery):** `strings` on the new `libmlx.so`: `MLX_ROCM_REFWD_M1` (1), `MLX_ROCM_QMV_ROWS` (3), `MLX_QMV_LX` (1), `MLX_ROCM_GDN_TAPE` (3), `MLX_ROCM_GDN_FUSE` (3), tape frame present (`gdn_fused_step_tape` 45). Capability probe (the t1375 script, scratch venv): **4/4 ops ok** — `gdn_fused_step_tape`, `gdn_fused_step`, `gdn_prep`, `gdn_norm_gated`. - **Bench:** canonical `mtp-bench-k3-tape.py` md5 `2c2d6c90c2349ae087e53e5d6c60018e` (the guest bench dir used by the prior rounds); guard `gdn_tape_commit.py` md5 `4a262a0d…`. Driver + bench UNMODIFIED — env toggles only, no method variation. - **Sidecar pin:** `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` active, the 4-bit sidecar `.disabled`, all other shards parked — verified before AND after, unchanged. - **Run env per run:** `MODELS=<model mount>`, `LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`, `MLX_MTP=4`, `MLX_ROCM_GDN_TAPE=1`, `MLX_ROCM_GDN_FUSE=1`, plus the arm toggles. Arms exactly: R0 = nothing extra; R1 = `MLX_ROCM_REFWD_M1=1`; R2 = `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`; R3 = R1+R2. - **GPU lane:** a sibling long-lived vllm-serve job held the lock at script start; respected per protocol, lock acquired at 16:34:54Z on the 15 s poll (the sibling released), held across all eight runs, released at exit (verified free; no orphan GPU processes).

## 2. The graft (the changed site, `quantized/qmm.hip` K1 section)

Captured as `patches/graft-rows-aware-m1.diff` — the exact delta between the pre-change and post-change K1 site, both file states committed as reference artifacts: `patches/rows-aware/qmm.pre-rowsaware.hip` (md5 `43c2a009f743bb313e2a9295a7a91097` — the qmm.hip entry of the 1910-file manifest of the wheel-reconcile round, i.e. the graft build tree) and `patches/rows-aware/qmm.post-rowsaware.hip` (md5 `0425ba4561d76cfe21ee6339076101dc`, the tree that built the battery wheel). Verified at capture: applying the patch to the committed pre file yields the committed post file byte-identical. The change is a 7-line comment + the `rows_active` static + the reworked `lx_engage` expression gaining the `!rows_active` conjunct; default-inert: rows OFF ⇒ `rows_active` false ⇒ `lx_engage` reduces exactly to `(lx_env != nullptr) || (refwd_m1 && M == 1)`. Scope note: the build tree is not reproducible from the repo's captured diffs alone (the 39-file canonical graft of the pinned wheel is not committed to the repo; the 1910-file manifest is the tree's identity anchor, as in the wheel-reconcile round) — the patch is anchored to the manifest-identified tree, not to repo base + `graft-refwd-m1-wheel.diff`.

```cpp
  static const bool rows_active =
      (std::getenv("MLX_ROCM_QMV_ROWS") != nullptr);
  const bool lx_engage = (lx_env != nullptr) ||
      (refwd_m1 && M == 1 && !rows_active);
```

`rows_active` is the rows dispatch's own predicate (its `use_rows_qmv` env gate, read at the K1 site — the dispatch defines the same getenv check below). With rows OFF, `!rows_active` is true and `lx_engage` reduces exactly to the candidate wheel's `(lx_env != nullptr) || (refwd_m1 && M == 1)`. With rows ON, the M=1 32-col lx re-route does not engage: the M=1 QMMs stay on the wide default (where the rows win lives) and the M=2..4 verify QMMs take the rows kernel — no toggle collision.

## 3. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | single ms (med) | verify ms (med) | refwd ms (med) | n_refwd | D-ACC | TAPE n_ok | first20 (both runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 (gate) | 60.59 | 58.76 | **59.675** | 10.075 (9.75/10.40) | 33.435 (33.36/33.51) | 9.685 (9.27/10.10) | 101 | 93/59/27 | 128/128 | `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` (identical) |
| R1 (REFWD_M1) | 60.43 | 61.81 | **61.12** | 9.735 | 33.99 (34.53/33.45) | 9.36 | 94 | 99/62/34 | 128/128 | `271 1919 369 264 15352 10597 29006 314 264 6278 20853 318 68 1257 2487 11819 220 16 20 11` (identical) |
| R2 (rows ON) | 65.53 | 65.58 | **65.555** | 9.74 | 30.05 | 9.08 | 99 | 95/61/29 | 128/128 | `271 1919 369 264 15352 11 70042 …` (identical, = R0's stream) |
| R3 (STACKED) | 65.58 | 65.58 | **65.58** | 9.73 | 30.04 (30.02/30.06) | 9.085 | 99 | 95/61/29 | 128/128 | `271 1919 369 264 15352 11 70042 …` (identical, = R0's stream) |

In-run spread: R0 3.07% (R0-2 soft — box wander), R1 2.26%, R2 0.08%, R3 0.00% (medians are the data points). TAPE_CALLS per arm: R0 5496, R1 5328, R2 5448, R3 5448 (matching the prior rounds' per-arm signatures).

**The fix, by signature (not just by number):** on the candidate wheel, stacked R3 collapsed (refwd 11.02 ms, n_refwd 111, D-ACC 85/45/17, first20 diverged). On this wheel R3 is **bit-signature-identical to R2** — n_refwd 99, D-ACC 95/61/29, refwd 9.08 ms, first20 = R0's undiverged stream — i.e. with rows ON the M=1 QMMs took exactly R2's path (the re-route suppressed) while the M=4 verify QMMs took the rows kernel (verify 30.04 ms, best of the rows arms). And with rows OFF (R1) the re-route still engages exactly as before (n_refwd 94, D-ACC 99/62/34 — the candidate wheel's R1 signature). The conjunct landed where it should: both arms keep their wins, the collision is gone.

## 4. Gates

| gate | result |
|---|---|
| (1) bench md5 + sidecar pin (before/after) | PASS — `2c2d6c90…` / `4a262a0d…`; `02a54359…` unchanged, 4-bit `.disabled` |
| (2) **R0 absolute window 57.33–60.98 (drift headroom 61.5)** | **PASS** — both runs inside (60.59 / 58.76); median 59.675 |
| (3) within-arm md5 determinism (FIRST20, 2 runs) | PASS — all four arm pairs identical |
| (4) TAPE n_ok=128 n_fb=0 on all eight runs | PASS (silent gate) |

**Cross-session drift note:** this session's R0 (59.675) vs the stack-ab round's R0 (60.515): −1.39%; R2 65.555 vs 64.705: +1.31% — expected box wander (the t1379 DPM-sclk floor). The absolute window absorbed it; the in-session R0 is the reference for the relative arms. Vs the pre-change anchors: R0 59.675 is −14.1% vs 69.48; R3 65.58 is −12.5% vs 75.00 — the absolute gap remains the guest clock state, not the wheel (the relative ratios are the data).

## 5. Relative verdicts

| comparison | value | vs the candidate wheel's stacked R3 (−1.51%) |
|---|---|---|
| R1/R0 | 61.12/59.675 = **+2.42%** | the re-route's standalone win, signature unchanged (n_refwd 101→94, D-ACC 93/59/27→99/62/34) |
| R2/R0 | 65.555/59.675 = **+9.85%** | the rows win (verify −3.39 ms, 33.435→30.05) |
| **R3/R0 (STACKED)** | **65.58/59.675 = +9.90%** | **regression → best arm** (was 59.60 / −1.51% on the candidate wheel) |
| R3/R2 | 65.58/65.555 = **+0.04%** | R3 ≥ R2 (within in-run noise); the stacked arm pays no toggle penalty and keeps the rows win |

**Is the collision fixed? YES.** The stacked arm went from −1.51% below R0 (refwd +1.9 ms, n_refwd +12, D-ACC collapse) to +9.90% above R0 with R2's exact M=1 signature and the rows kernel's verify (30.04 ms). The levers now compose by construction: rows ON takes rows' win; the M=1 re-route's standalone win (R1, +2.42%) applies on the rows-OFF arms. The two env toggles no longer fight over the M=1 QMMs.

## 6. Bottom line

- **Stacked (R3) = 65.58 tok/s — +9.90% vs R0, above R2, the first stacked standing of the program** (verify 30.04 ms, refwd 9.08 ms, n_refwd 99, D-ACC 95/61/29, TAPE 128/128 n_fb=0). The t1391 collision is fixed by one conjunct: the M=1 32-col lx re-route engages only when rows is OFF (`!rows_active` on the `refwd_m1 && M == 1` term), default-inert with rows OFF. - Standing config is now the **stacked env arm** (rows ON + REFWD_M1 set; the re-route is inert under rows by the wheel's own gate — the env toggle is harmless and keeps the arm ready for a rows-OFF A/B in the same wheel). The 100+ gap is unchanged in kind: the guest DPM-sclk floor remains the absolute limiter (−12.5% vs the 75.00 anchor). - All four gates PASS; the lossless/TAPE gate is silent (128/128 n_fb=0 on all eight runs).

## 7. Exit state

Lock released (verified free), no orphan GPU processes, sidecar unchanged (md5 re-verified after), parked shards untouched, `venv-wrc` and `venv-canon` unmodified (md5 re-verified after). Guest left clean.

## 8. Persona scrub

Scrubbed the report + commit message: foreign hostnames, tailnet/LAN IP ranges, foreign user + home paths, the model-mount path, codenames — 0 hits.
