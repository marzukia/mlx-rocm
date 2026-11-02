# REPORT-rows-tuning-gpu — M=4 verify GEMM knob battery (GPU round)

**Task:** run `bench/run-rows-tuning.sh` on guest-a (RX 9070 XT / gfx1201)
with the guest-built wheel (sha256 `86b7598f…`, `mlx-0.32.3.dev20261102+75ca881`),
exclusive GPU window, and deliver the wall-identification verdict of
REPORT-rows-tuning.md section 4.

**Verdict: the battery ran (exclusive window acquired 11:00Z, arms
11:30–11:45Z, `gpu-bench.lock` held for the whole window, released
11:47Z), but the B0 gate FAILED — 59.4–59.5 tok/s vs the 70.02 ±3%
reference. The run is INVALID per the gate. The failure is NOT in the
tuning knobs: a five-wheel control matrix shows the 70.2 baseline is not
attainable on ANY current wheel on guest-a (all 56.3–60.2 tok/s, verify
34.4–36.4 ms vs 25.66); the 03:05Z baseline wheel was lost to venv
overwrites by the concurrent WMMA work. The A/B table (R0 vs R1/R2/R3,
same wheel) is the valid output of this round: **R2 (KBLOCK=2) is
neutral → the wall is scalar-FMA-throughput bound, not load-latency
bound → the WMMA M≤4 kernel (REPORT-gemms-m4.md section 4) is the
decisive lever**, per section 4 of the CPU report.

---

## 1. Setup (verified)

- Guest wheel located by sha256 (`86b7598f1499cbf0…bd5cb`, 22,516,552
  bytes) in the t1354 scratch `wheelout/`; installed into the uv-managed
  venv by `zipfile` extraction over site-packages (the established
  no-pip method; the prior wheel `…+73346f3` was displaced).
- Import + symbol verification PASS: `mlx.core` resolves
  `0.32.3.dev20261102+75ca881`; `nm` shows `qmv_rows_msg_ws32_kernel`
  and `qmv_rows_msg_kb2_kernel` instantiated at 36 template combos each,
  base kernel signature ends in the trailing `int` (the `warps` arg).
- Canonical tape bench md5 `2c2d6c90c2349ae087e53e5d6c60018e` ok; model
  sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c`
  pinned at start and unchanged after the battery (no drift).
- **Script defect found and fixed (this commit):** `run_arm` exports
  `MLX_ROCM_GDN_TAPE=1` but NOT `MLX_ROCM_GDN_FUSE=1`, and the tape
  dispatch branch in the model requires BOTH (qwen3_5.py: the tape op
  is a superset selected inside the FUSE branch). Without FUSE the
  tape never engages: the first battery attempt produced
  `TAPE n_ok=0 n_fb=128`, `n_refwd=408`, 32.5 tok/s — an all-fallback
  run (the t1354 CPU round hit the same defect; its 10:29–10:30Z
  `canon` logs show the identical `last_tape missing (tape dispatch
  not engaged)` signature). Fixed in `bench/run-rows-tuning.sh`
  (`run_arm` now exports `MLX_ROCM_GDN_FUSE=1` alongside `TAPE=1`).
- Exclusive window: the lock was held by a 4h placeholder (orphaned
  `sleep`, PPID=1) from the WMMA task's 09:14Z reservation; acquired
  11:00Z (the reservation lapsed early — the WMMA task's own runs do
  not hold this lock), held via a nohup'd flock holder for the whole
  battery, released 11:47Z. The vLLM-anchor agent's waiting flocks were
  never killed; the lock returned to the queue on release.

## 2. Gates

| gate | result |
|---|---|
| (1) bench md5 + sidecar pin | PASS (re-check after battery: no drift) |
| (2) **B0 = 70.02 ±3%** | **FAIL** — 59.54 / 59.40 tok/s (−15%), verify 34.25/34.37 vs 25.66 ms |
| (3) bit-exactness R1/R2/R3 vs R0 (FIRST20 + ALLTOKS md5) | **PASS** — all four arms md5-identical (`aee9db45…` / `bda3b13c…`) |
| (4) n_refwd 95..107 | PASS — all arms 99 |
| target: any rows arm verify < 22.25 ms | NOT MET (but the run is invalid for gate-2 reasons) |

**The run is INVALID per gate (2).** The A/B table below is same-wheel
data and is valid for the wall-identification question (which is a
relative one); it is NOT valid for the 22.25 absolute target or the
70.02 baseline comparison.

## 3. B0 gate failure — root cause (wheel-baseline loss, not the knobs)

The 70.02 m4 reference was last reproduced 03:05Z (t1325 re-bench,
70.20 tok/s, verify 25.60, D-ACC 90/56/28, n_refwd 100). That run's
wheel is no longer recoverable: the guest venv was re-built 09:43Z and
10:18Z by the concurrent WMMA task (its `73346f3` ref is now a
09:55Z "wmma v3" commit; the original ref's wheel was overwritten), and
no wheel on disk reproduces the 03:05 numbers. In-session noise is
~0.2% (B0 repeat 59.54 → 59.40), so the gap is not thermal/session
drift; the 03:05 run's D-ACC signature (90/56/28) matches no available
wheel (all current wheels: 93/59/27, or 100/66/30 for the Nov-1 tape
wheel), confirming a wheel-level numeric difference.

**Control matrix** (all: B0 protocol, rows OFF, tape ON, same session,
lock held; 03:05Z row = historical log):

| wheel (built) | tok/s | verify ms | n_refwd | D-ACC |
|---|---|---|---|---|
| 03:05Z wheel (lost to overwrite) — historical | **70.20** | **25.60** | 100 | 90/56/28 |
| `+75ca881` + tuning (this battery, `86b7598f…`) | 59.54 / 59.40 | 34.25 / 34.37 | 101 | 93/59/27 |
| `+73346f3` (WMMA-v3 tree, 10:01Z rebuild) | 59.46 | 34.32 | 101 | 93/59/27 |
| `+75ca881` wheelout-rs (02:10Z, no tuning) | 56.26 | 36.35 | 101 | 93/59/27 |
| `+75ca881` wheelout-v3 (01:12Z) | 59.42 | 34.37 | 101 | 93/59/27 |
| `+75ca881` wheelout-tape (Nov-1 22:10Z) | 60.23 | 34.46 | 98 | 100/66/30 |
| `+75ca881` Nov-1 15:06Z (wheels/) | n/a — predates the `gdn_fused_step_tape` op; cannot run the tape protocol | | | |

Observations: (a) the tuning patch is NOT the B0 regressor — the
unpatched 02:10Z wheel is 5.5% SLOWER than the patched wheel (build
tree drift between 02:10Z and 10:19Z on the same base ref); (b) every
available wheel lands 56.3–60.2, none within the gate band
(67.9–72.1); (c) the single-step ms is near-constant (9.7–10.5) across
all runs while verify is 34.4–36.4 — the drift is in the verify/GDN
path specifically, consistent with a build-tree (kernel-code) change
rather than clock/throttle drift (driver state checked: DPM default, no
forced performance level, no thermal-throttle events, no clock caps).

**Consequence:** the m4 70.02 / 22.25 calibration is stale for the
current guest wheel stack. The next round must re-baseline: either
rebuild the 03:05-era wheel (the bit-exact t1334 wheel lineage — its
original build is gone from disk) or re-calibrate the B0 reference +
target to the current stack (70.02 → ~59.5; 22.25 → the current R0
verify, which is 30.8 on this wheel) — an operator call, since it
moves the 100+ tok/s target framing.

## 4. A/B table (same wheel, `86b7598f…`; valid for the wall question)

| arm | knob | tok/s | verify ms | pair ms | n_refwd | single ms |
|---|---|---|---|---|---|---|
| B0 | rows OFF | 59.54 (59.40 repeat) | 34.25 | 53.53 | 101 | 9.75 |
| R0 | rows ON, base | 64.53 | 30.77 | 49.88 | 99 | 9.76 |
| R1 | WS=32 | 64.45 | 30.84 | 49.94 | 99 | 9.74 |
| R2 | KBLOCK=2 | 64.46 | 30.83 | 49.93 | 99 | 9.75 |
| R3 | WARPS=4 | 65.58 | 30.09 | 49.08 | 99 | 9.74 |

Deltas vs R0: R1 +0.07 ms (−0.1%), R2 +0.06 ms (−0.07%), R3 −0.68 ms
(+1.6% tok/s). Bit-exactness PASS (gate 3); n_refwd band PASS (gate 4,
all 99).

## 5. Wall-identification verdict (section 4 of the CPU report)

- **R2 (KBLOCK=2, the only knob with a physical path to a win if
  load-latency bound): NEUTRAL** (+0.06 ms, within the 0.2% in-session
  noise). 2× the weight/x loads in flight changed nothing → the kernel
  is not hiding load latency behind the FMA pipe; **the wall is
  scalar-FMA-throughput bound**.
- R1 (WS=32): neutral (+0.07 ms) — expected; the −75% decode-ALU cut
  does not help when the FMA pipe is the wall (consistent with the
  t1334 hoist negative experiment).
- R3 (WARPS=4): +1.6% (verify −0.68 ms, pair −0.80 ms) — a small
  block-granularity signal (128-thread blocks vs 192), above the
  0.2% noise but below any meaningful lever; note for re-confirmation
  in the next round, NOT a recommendation (it was a packing probe).

**Per section 4: all FMA-relevant arms neutral → FMA-throughput bound →
the WMMA M≤4 kernel (4–8× the FMA throughput of the matrix units) is
the decisive lever.** The knobs infrastructure (default-off dispatch,
bit-exact contract, this battery) is in place for the WMMA round.

## 6. Spill / occupancy observation

No profiler run in this window (the battery is the tape bench; the
occupancy-probe tooling belongs to the WMMA task). Indirect: R1 (the
128-float `wdec` pre-decode, 768 VGPRs/block worst case) shows NO
regression — but at the FMA wall a decode-side spill would only cost
the (unused) ALU overlap, so this is not a clean spill test. R2 adds
no registers (two tiles in flight). No occupancy numbers are claimed.

## 7. What's delivered / preserved

- The A/B table (section 4 above) + the bit-exactness / n_refwd gates
  (PASS) + the wall verdict (section 5): **FMA-throughput bound, WMMA
  is the lever.**
- The B0-gate finding (section 3): the 70.02 baseline is not
  reproducible on the current guest wheel stack (five-wheel control
  matrix); the 03:05Z baseline wheel is lost to venv overwrites — the
  next round must re-baseline (rebuild the bit-exact wheel lineage, or
  re-calibrate the gate + target — operator call).
- The `bench/run-rows-tuning.sh` fix: `run_arm` now exports
  `MLX_ROCM_GDN_FUSE=1` (the tape dispatch requires both envs; without
  it the battery silently runs all-fallback, 32 tok/s — the first
  battery attempt of this round and the t1354 10:29Z `canon` logs).
- Guest state: battery wheel `86b7598f…` is installed in the guest
  venv (symbol-verified); `gpu-bench.lock` released, no orphan
  processes left, control venvs cleaned up, model sidecar unchanged.

**Next:** re-baseline decision (section 3) → the WMMA M≤4 kernel
(REPORT-gemms-m4.md section 4) is the decisive lever either way.
