# REPORT-stack-ab — STACKED A/B (rows + REFWD_M1) on the candidate unified wheel

**Task:** the mission's best-stack number — one wheel (the candidate unified
wheel, `REPORT-wheel-reconcile.md`), env toggles only, no method variation,
2 runs/arm, medians. Arms: R0 gate (both off), R1 REFWD_M1, R2 rows
canonical, R3 the stacked arm.

**Verdict: GATE PASS (R0 median 60.515, inside the 57.33–60.98 window). The
levers DO NOT compose on the candidate wheel: R1 +1.57%, R2 +6.93%, but
R3 = −1.51% — a regression below R0. The M=1 re-route collides with the
rows M=1 path (refwd ms +1.9 vs R2, n_refwd 99→111, D-ACC 95/61/29→
85/45/17). New best standing this session: 64.71 tok/s (R2). Next lever:
guest clock lock first; if the collision persists post-fix, a rows-aware
REFWD_M1 gate (wheel change) — the stacked env arm is dead on this wheel.**

---

## 1. Setup (verified before the battery)

- **Wheel:** the candidate unified wheel — the scratch venv from the
  wheel-reconcile round, found by `libmlx.so` md5
  `5bf011c3cc3eb6502475fae61e4ef3af` (the wheel-recon round's venv; no
  recreation needed), `strings` shows `MLX_ROCM_REFWD_M1` (1, 0 in the
  canonical lib). `venv-canon` untouched.
- **Bench:** canonical `mtp-bench-k3-tape.py` md5
  `2c2d6c90c2349ae087e53e5d6c60018e` (the wheel-reconcile round's bench
  dir, the same canonical copy as the rows-stack round); guard
  `gdn_tape_commit.py` md5 `4a262a0d…` (the canonical pin). Driver and
  bench unmodified.
- **Sidecar pin:** `model-mtp.safetensors` md5
  `02a54359cda8b247f76b277dd9bc4d4c` active; the 4-bit sidecar
  `.disabled`; all other shards parked — verified before AND after the
  battery, unchanged.
- **Tape frame on ALL arms:** `MLX_ROCM_GDN_TAPE=1 AND
  MLX_ROCM_GDN_FUSE=1`; result `TAPE n_ok=128 n_fb=0` on all eight runs.
- **Run env per run:** `MODELS=<model mount>`,
  `LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`, `MLX_MTP=4`
  + the tape frame. Arms exactly: R0 = nothing extra; R1 =
  `MLX_ROCM_REFWD_M1=1`; R2 = `MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1`; R3 = R1+R2.
- **GPU lane:** lock free at the first 15 s poll (no sibling holder to
  respect); held across all eight runs; released at exit (verified free
  afterward; no orphan GPU processes). Battery duration ~1.5 min.

## 2. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | single ms (med) | verify ms (med) | refwd ms (med) | n_refwd | D-ACC | TAPE n_ok | first20 (both runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 (gate) | 60.94 | 60.09 | **60.515** | 9.745 | 33.44 (33.07/33.81) | 9.265 | 101 | 93/59/27 | 128/128 | `bda3b13c…` (identical) |
| R1 (REFWD_M1) | 61.85 | 61.08 | **61.465** | 9.735 | 33.68 (33.34/34.01) | 9.365 | 94 | 99/62/34 | 128/128 | `dcc79e37…` (identical) |
| R2 (rows ON) | 64.90 | 64.51 | **64.705** | 9.97 (10.21/9.73) | 30.46 (30.08/30.83) | 9.095 | 99 | 95/61/29 | 128/128 | `bda3b13c…` (identical) |
| R3 (STACKED) | 59.50 | 59.70 | **59.600** | 9.725 | 29.64 (29.68/29.59) | 11.02 | 111 | 85/45/17 | 128/128 | `16bd20da…` (identical) |

In-run noise: R0 1.40%, R1 1.25%, R2 0.60%, R3 0.34% (R0-2 a touch soft;
medians are the data points). R2's first20 is bit-identical to R0's
(`bda3b13c…`); R1's and R3's streams differ (as expected — the M=1 re-route
changes acceptance, R3's D-ACC drop is visible in the stream).

## 3. Gates

| gate | result |
|---|---|
| (1) bench md5 + sidecar pin (before/after) | PASS |
| (2) **R0 absolute window 57.33–60.98 (drift headroom 61.5)** | **PASS** — both runs (60.94/60.09) inside; median 60.515 |
| (3) within-arm md5 determinism (FIRST20, 2 runs) | PASS — all four arm pairs identical |
| (4) TAPE n_ok=128 n_fb=0 on all eight runs | PASS |

**Cross-session drift note:** in-session R0 (60.515) vs the wheel-reconcile
round's R0 (60.945): −0.70% — the expected box wander (the t1379 DPM-sclk
observation). The absolute window absorbed the drift; R1's R0-1 (60.94)
matches the prior round's R0-1 (60.97) to 0.5%. The in-session R0 is the
reference for the relative arms, as briefed.

## 4. Relative verdicts

| comparison | value | expectation | verdict |
|---|---|---|---|
| R1/R0 | 61.465/60.515 = **+1.57%** | ~+4.4% (prior same-wheel-class A/B 55.93→58.42) | lever WORKS (signature: n_refwd 101→94 exactly −7, D-ACC 93/59/27→99/62/34, all matching the prior round) but net gain is ⅓ of the prior round's — the candidate wheel's canonical verify layer (mr2) already absorbs part of the M=1 re-route benefit |
| R2/R0 | 64.705/60.515 = **+6.93%** | ~+8.6% (rows-stack: 65.85 vs 60.65) | lever WORKS; verify −2.98 ms (33.44→30.46), n_refwd 101→99, D-ACC 95/61/29 — the rows win, at 80% of the prior round's ratio (this session's R0 baseline is also 0.2% lower) |
| R3/R0 | 59.600/60.515 = **−1.51%** | ~68–69 if the levers compose | **REGRESSION** — below R0 |

**Do the levers compose? NO.** Compose expectations: multiplicative
60.515 × 1.0157 × 1.0693 = **65.72**; additive 60.515 + 0.95 + 4.19 =
**65.66**. Observed R3: **59.60** — **−9.3% below** the compose
expectation.

**Mechanism (from the per-component rows):** R3's verify is the best of all
arms (29.64 ms, even below R2's 30.46 — rows still fast on the M=4
verify QMMs), but the stacked arm's refwd pass degrades: refwd ms
9.095 (R2) → **11.02 (R3), +1.9 ms**, n_refwd 99 → **111 (+12)**, and
draft accuracy collapses 95/61/29 → **85/45/17**. With rows ON, the
REFWD_M1 engage re-routes M=1 QMMs away from the rows path onto the 32-col
lx path: the wheel loses the rows M=1 benefit in single/refwd, pays a
slower refwd pass with more refwd iterations, and generates weaker drafts —
a double penalty that swamps the M=1 re-route's standalone +1.6%. The two
toggles fight over the same M=1 QMMs; rows ON should win that dispatch,
but the graft's engage gate does not know about rows.

## 5. New best standing (this session, candidate wheel)

- **64.71 tok/s (R2, rows ON)** — verify 30.46 ms, pair 49.75 ms (wall),
  single 9.97 ms, n_refwd 99, D-ACC 95/61/29, TAPE 128/128 n_fb=0.
- Vs the prior best standing 65.88 (rows-stack round, pinned wheel): −1.78%
  — consistent with this session's lower R0 baseline (60.515 vs 60.65) and
  the lower R2/R0 ratio (+6.9% vs +8.6%); the guest state wanders
  (t1379 DPM-sclk floor). The stacked arm (59.60) is not a standing.
- Vs the pre-change anchors: R0 60.515 is −13.0% vs 69.48; R2 64.71 is
  −13.4% vs 75.00 — the absolute gap is the guest-state (clock) gap, not
  a wheel gap (the same pinned-wheel round ran 65.88 with the same
  +8.6% ratio a session earlier).

## 6. Next-lever recommendation (100+ target)

1. **Unblock the box, not the wheel.** The absolute gap (−13% vs the
   pre-change anchors, R0 and R2 alike) is the guest DPM-sclk floor
   (t1379: sclk stuck at 600 MHz under load, force knobs root-only). The
   **operator reboot / guest-root clock lock** remains the single
   prerequisite for any 100+ math: at pre-change state the same levers
   were 69.48 base / 75.00 rows — the 100+ gap was ~10 ms of
   verify+refwd, not 45 ms. Re-run this same battery (candidate wheel,
   `venv-wrc`, this exact script) in the post-clock-lock state before
   drawing further wheel conclusions.
2. **If the R3 collision persists post-fix:** the stacked env arm is dead
   on this wheel. The wheel-level fix is a **rows-aware REFWD_M1 gate** —
   engage the 32-col lx re-route only when rows is OFF (the graft's K1
   gate is one conjunct away: `lx_engage = … && !rows_active`); that would
   let the stack take rows' M=1 win AND the re-route's refwd win without
   the toggle collision. This is the next wheel graft to test, pending
   (1).
3. **Until then, the standing config is R2 (rows ON, REFWD_M1 unset):
   64.71 tok/s.** R1's standalone +1.6% on this wheel is real but small;
   do not stack the two env toggles on this wheel.

## 7. Exit state

Lock released (verified free), no orphan GPU processes, sidecar unchanged
(md5 re-verified after), parked shards untouched, `venv-canon` unmodified.
Guest left clean.

## 8. Persona scrub

Scrubbed before commit: guest user / home-dir / model-mount literals →
"the guest box" / "the scratch venv from the wheel-reconcile round" /
`<model mount>` / `MODELS`. 0 pattern hits on the committed file.
