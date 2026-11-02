# REPORT-rows-stack — rows battery on the pinned wheel, VALID current guest state

**Task:** re-confirm the rows battery (one wheel, 2 runs/arm, medians) on the
pinned canonical wheel (sha256 `5bc7ca20…`, md5 `5e5ee878…`, libmlx
`b1de59fd…` — all re-verified on-box) in the current guest state, with env
toggles only, no method variation. The 69.48/75.00 pre-change band is NOT
reachable in the current guest state (RCA verdict: NOT FIXABLE IN PLACE,
operator reboot pending); the VALID baseline is the pinned-wheel current-state
band (B0 59.17/59.25, R1 64.53/64.58, σ < 0.6% across 6 runs).

**Verdict: GATE PASS (R0 median 60.65, in the 57.4–60.9 window). Rows-ON
canonical reproduces the +8.6% win (+8.55% rebaseline). The WARPS4 +1.6%
signal from the t1355 battery is NOT re-confirmed (median delta +0.04%,
within noise). New best standing: 65.88 tok/s (R2, = R1 within noise).**

---

## 1. Setup (verified before the battery)

- Pinned venv intact: `venv-canon`, `mlx 0.32.3.dev20261102+75ca881`,
  `libmlx.so` md5 `b1de59fda0e4fc607d620e287deeaaf8` (the frozen record).
- Canonical tape bench md5 `2c2d6c90c2349ae087e53e5d6c60018e` ok; model
  sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c`
  pinned at start and unchanged after the battery (no drift); 4-bit sidecar
  `.disabled`, other shards parked — untouched.
- Tape frame on ALL arms: `MLX_ROCM_GDN_TAPE=1 AND MLX_ROCM_GDN_FUSE=1`
  (the driver fix carried from the rows-tuning round); result: `TAPE
  n_ok=128 n_fb=0` on all six runs.
- Run env per run: `MODELS=<model mount>`, `LD_PRELOAD=libhsa-runtime64`,
  `HCC_AMDGPU_TARGET=gfx1201`, `MLX_MTP=4`. Arms are exactly the rows-battery
  env definitions: R0 = no rows env (rows OFF, gate); R1 =
  `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`; R2 = R1 +
  `MLX_QMV_ROWS_WARPS=4`. The driver and bench were not modified.
- GPU lane: lock acquired on the first 15 s poll (12:44Z, free at check; no
  holder to respect), held across all six runs, released 12:45Z; no orphan
  GPU processes at exit (verified). Battery duration ~1 min per 3-arm round.

## 2. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | verify ms (med) | pair ms (med) | n_refwd | D-ACC | TAPE n_ok |
|---|---|---|---|---|---|---|---|---|
| R0 (rows OFF, GATE) | 60.84 | 60.45 | **60.65** | 33.35 (33.17/33.52) | 52.56 | 101 | 93/59/27 | 128/128 |
| R1 (rows ON canonical) | 65.99 | 65.71 | **65.85** | 29.91 (29.84/29.97) | 48.89 | 99 | 95/61/29 | 128/128 |
| R2 (rows ON + WARPS4) | 65.77 | 65.98 | **65.88** | 29.87 (29.89/29.85) | 48.87 | 99 | 95/61/29 | 128/128 |

Single-step ms: 9.72–9.74 across all arms (constant). In-run noise: R0
0.64%, R1 0.43%, R2 0.32% (a touch above the 0.2% t1355 in-session figure,
but the medians are the data points).

## 3. Gates

| gate | result |
|---|---|
| (1) bench md5 + sidecar pin (before/after) | PASS |
| (2) **R0 median 59.1–59.2 ±3% (window 57.4–60.9)** | **PASS** — median 60.65, both runs (60.84/60.45) inside the window (upper edge 60.9) |
| (3) within-arm md5 determinism (FIRST20 + ALLTOKS stream) | PASS — R0 pair `c11b54d3…`, R1 pair `f0e30930…`, R2 pair `f0e30930…`; all pairs identical within arm (R1 = R2 bit-identical by construction, as expected) |
| (4) TAPE n_ok=128 n_fb=0 on all six runs | PASS |

**Cross-session drift note:** this round sits ~+2.0% above the rebaseline
round (R0 60.65 vs 59.47; R1 65.85 vs 64.54) — the guest state wanders
(consistent with the DPM-sclk observation of the t1379 round); the R1/R0 win
ratio is preserved (+8.6% vs +8.55%). The gate window (57.4–60.9, from
59.17/59.25 ±3%) absorbed the drift; R0 max run 60.84 remains inside.

## 4. WARPS4 re-confirm verdict

**NOT re-confirmed.** R2 vs R1: median tok/s +0.04% (65.88 vs 65.85), verify
−0.03 ms (29.87 vs 29.91), pair −0.02 ms — all within the in-run noise
(0.3–0.6%), i.e. neutral. The t1355 +1.6% (65.58 vs 64.53) was a single-run
packing probe on a different wheel/session and does not carry to the pinned
wheel. WARPS4 is a wash on this state; it is not a lever.

## 5. New best standing (current guest state, pinned wheel)

- **65.88 tok/s (R2 / R1, rows ON, WARPS=4 optional-neutral)**, verify
  29.87–29.91 ms, pair 48.87 ms, n_refwd 99, D-ACC 95/61/29.
- The rows win vs the gate arm: +8.6% (R1/R0 65.85/60.65) — consistent with
  the +8.55% rebaseline and the +7.2% original statistical win.
- The 75.00 mission anchor remains unreachable in this guest state (−12.1%
  vs 65.88); the 100+ target is untouched, pending the operator reboot
  decision (reboot + this same pinned wheel+venv is the ready re-bench
  vehicle per the rebaseline round).

## 6. Exit state

Lock released, no orphan GPU processes, sidecar unchanged, parked shards
untouched, venv-canon unmodified. Guest left clean.

## 7. Persona scrub

Scrubbed before commit: guest user / home-dir / model-mount literals →
"the guest box" / `venv-canon` / `<model mount>` / `MODELS`. 0 pattern hits
on the committed file.
