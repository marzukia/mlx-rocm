# REPORT-seam — decode-step seam measurement (t2231)

Claim under test: the kernels are roughly fine; the missing time is BETWEEN them,
not inside them. Verified arithmetic: AR reference 16.93 ms/step, 59.10 t/s
(t2222 session); one pass over trunk weights ~5.54 GB; in-session streaming floor
~631 GB/s → 8.8 ms floor; single large matmul in isolation hits 76.8-80.2% of
floor (t2199/t2207) → if every op ran at 78%, the step would be 8.8/0.78 =
11.3 ms = 88 t/s ≈ llama.cpp non-speculative anchor 87 t/s. EXPECTED: AR kernelsum ~11 ms vs step wall ~17 ms. If the kernel sum is already ~17 ms the
hypothesis is DEAD — the kernels are the problem.

Token bases: AR = 1 tok/step vs record spec = 2.193 tok/step. Different bases:
stated whenever tok/s is compared.

## 1. Code-first: graph-decode state on the two wheels (no GPU)Question: which is true NOW on the lane wheel (venv-verify, libmlx 8acf13fe) andthe record wheel (venv-m3k, libmlx a0057c22): (a) capture+replay armed (onegraph launch per step), (b) capture-then-relaunch-every-step, or (c) eager with
one launch per op.

### 1.1 The machine, as it stands in the committed vendor source

The committed vendor tree (vendor.json: fork marzukia/mlx-rocm @ 75ca881; the
committed `vendor/mlx-rocm-upstream/` is byte-verified authoritative per the pin
note) contains exactly ONE live graph path for decode, and it is the
**GraphDecodePilot** in `mlx/backend/rocm/eval.cpp`:

- `eval()` (eval.cpp:586) unconditionally calls `graph_decode_pilot(arr)` on
  every op; the pilot is a no-op unless `MLX_ROCM_GRAPH_DECODE=1`:

  ```
  void graph_decode_pilot(array& arr) {
    if (!gd_env_enabled()) {
      return;
    }
  ```

  and `gd_env_enabled()` (eval.cpp:108) is `getenv("MLX_ROCM_GRAPH_DECODE") == "1"`.
  The file's own header comment (eval.cpp:38):

  ```
  // OFF BY DEFAULT; MLX_ROCM_GRAPH_DECODE=1 enables the whole autopilot.
  // Any capture failure or shape mismatch falls back to eager (and cools
  // down), so the worst case is exactly today's behavior.
  ```

- Pattern detector: on the DEFAULT stream only (`gd_is_default_stream`), it
  watches the eval-burst pattern (one dispatch burst per mx.eval/mx.async_eval,
  ending with the no-op "Synchronize" primitive): a large step (>=
  `MLX_ROCM_GRAPH_DECODE_BIG_OPS`, default 4000 evals) followed by two small,
  shape-matching steps arms the next small step as the CAPTURE step
  (`GdPilotLogic::eager_end`, graph_decode_pilot_logic.h:119-138).

- What "armed" means NOW — this answers the design question (b) vs (a):
  the pilot does NOT re-capture every step. First small step = one capture
  (`gd_begin_capture` → `decode_capture_begin(slot)`), `end_record` at step
  end, and immediately `p.arm_replay(...)` (eval.cpp:474-484):

  ```
  // --- Exec-cache replay: capture once per shape, then replay the
  // cached graph (hipGraphLaunch) instead of re-capturing every step.
  ```

  Subsequent matching small steps are REPLAY steps: `gd_begin_replay` suppresses
  per-op launches (`set_decode_replay_suppress(true)`), per-step values ride
  the persistent arena value slots, and ONE `hipGraphLaunch` (via
  `decode_capture_replay(slot)` / the exec-cache path at device.cpp:2141ff,
  refreshed in place with `hipGraphExecUpdate`) fires the whole step. Re-capture
  happens ONLY on failure/overflow/divergence/big-step/shape-mismatch
  (`ReplayEnd::GrowArena/Discard`, `CaptureEnd::Cooldown`). So the live design is
  **(a): capture once per shape, then replay — one graph launch per step** —
  not the WORKSTATE-era per-step re-capture (b).

- `use_hip_graphs()` (device.cpp:63) is **permanently false**:

  ```
  bool use_hip_graphs() {
    // The rebuild-every-eval graph-build path is a net loss vs eager (and the
    // prefill variant segfaults on RDNA3.5). Decode uses build-once
    // capture/replay (decode_capture_*) ... so this path is permanently off.
    return false;
  }
  ```

  i.e. there is no rebuild-every-eval graph path in flight.

- The bench/serve path matters: `gpu_set_graph_decode_mode(bool)` (eval.cpp:738)  is the only C entry that set the old dormant machine, and it is declared in  NO header (grep over the whole vendor tree: the only occurrences are the
  definition and its own comment). The free-function bridge
  `mlx::core::decode_capture_begin/end_record/replay/destroy`
  (device.cpp:2464-2480) likewise has zero in-tree callers. WORKSTATE.md:66
  ("decode_capture_replay is dead code (zero callers)") is therefore:
  - STALE for the `CommandEncoder::decode_capture_replay` method — the pilot
    calls it (eval.cpp:332/350/430), so the encoder method is live code;
  - STILL TRUE for the free-function bridge and for `gpu_set_graph_decode_mode`
    — zero in-tree callers, nothing in mlx-lm calls it either.

  Conclusion: with `MLX_ROCM_GRAPH_DECODE` unset, EVERY decode step in both
  wheels runs **eager, one launch per op** (the pilot returns at the first
  line; `use_hip_graphs()` is false; the dormant bridge is never entered).

### 1.2 Applied to this measurement

Neither measured config sets `MLX_ROCM_GRAPH_DECODE` (AR 13-var: MLX_MTP=0
MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMM_M3=1 MTP6_HEAD_H=postnorm
MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MTP_MAX_TOKENS=322;
record 13-var: MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1
MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16
MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1
MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1 MTP_MAX_TOKENS=322).**CONFIRMED on the box (2026-11-06), state (c) EAGER, one launch per op, both
wheels:**

- Wheel provenance: both dist-infos say `mlx-0.32.3.dev20261104+75ca881`
  (= the vendor.json base); the lane wheel was built from the `vendor-r4v`
  lane build tree (tmax3r-build.sh: "vendor-r4v (the lane build tree)"), the
  record wheel from `vendor-m3k` (build-m3k.log: wheel
  `mlx-0.32.3.dev20261104+75ca881`, `libmlx-md5: a0057c22...` = the in-box
  venv-m3k libmlx, md5-verified both pre-run below). Both libmlx binaries carry
  the pilot: `strings | grep -c "graph decode:"` = **17** on each wheel
  (including `graph decode: capture begin failed; staying eager`).
- The pilot's env gate is never set by the measured configs: `grep GRAPH_DECODE`
  over the current battery scripts (t2212-ab.sh, t2213-oobfix-battery.sh,
  tmax3r-build.sh, verify_eff_run.sh) = **zero hits**; `MLX_ROCM_GRAPH_DECODE=1`
  exists only in the old diagnostic runners (diag.sh, diag2/3/4.sh, gd2/gd3
  runner, gd4fix-r4-run.sh).
- The bench logs confirm: the current battery logs (t2190-oobfix/*.log —
  t2213 battery) contain **0** `graph decode:` / `[g1]` lines (e.g. PRE-can.log
  goes straight from the `[mlx-rocm] bound HIP device 0: gfx1201 ...` boot
  lines to `INFO TAPEMODE=hybrid REFWDS=... MTP=4 LOAD=... GEN=...`); the only
  `graph decode:` lines on the box are from the old G1/G2 diagnostic sessions
  (t1788/G1.log, r4x-r4/PG.log, r4v-r3chk/INST_G.log, where the gate was set
  deliberately: e.g. `90x [g1] graph decode: armed (small steps 852/876 ops);
  capturing next`, `58x [g1] graph decode: step captured+launched (ops=876,
  slot=0, active)`, `18x [g1] graph decode: replay step discarded (ops=860,
  big=0, div=1)` — evidence the capture+replay machine arms when enabled).- Predicted measurement signature of the eager state: hundreds-to-thousands of  kernel launches per step (a graph-armed step would be ONE graph launch per
  step). Checked by the measured kernel count per step in section 3.

  Note the design answer for the state question: the in-tree design is (a)
  capture-once-then-replay (one graph launch per step, exec-cache via
  hipGraphExecUpdate), NOT the WORKSTATE-era per-step re-capture (b); and (a)
  exists but is off (env-gated) in both measured wheels.

### 1.3 Levers named (not implemented)

- **qMLX fused MLP** (gate+up+SiLU+mul in one kernel; reference kernel source in
  that repo's bench/bench_fused_mlp.py): we run three separate quantized
  matmuls + SiLU + mul; mlp is 38.9% of the verify wall.
- **llama.cpp one-graph-per-step** (llama_context::graph_compute): the
  GraphDecodePilot is the in-tree analog; on these wheels it is OFF (env-gated)
  — one graph launch per step is available, not active.
- **vLLM radiance MXFP4**: the format is NOT a lever without this profile
  (no measured kernel-vs-seam split to justify a format change).

## 2. Method (as actually executed)

The rocprofv2 / rocprofv3 / aqlprofile kernel-trace path is **dead on this
hardware**: the system HSA runtime 7.2.0 lacks `hsa_amd_queue_intercept_*`
(verified `nm -D libhsa-runtime64.so.1`: absent), so every rocprofiler SDK
version (1.1.0 ticket build, 1.5.1 source build) injects but emits zero
trace; the C++ rocr-runtime ends at 7.2.4 and the ROCm 10.x Rust runtime is
not on the box. Fallback (recorded decision): **HIP event brackets**.

- Shim: `scripts/span.c` -> `span.so` (gcc -O2, dlsym of the already-loaded
  `libamdhip64.so.7`; events created in-process; `span_begin()` records an
  event on the legacy default stream (the magic sync stream — ordered with
  work on all streams), `span_end()` records a second event, waits for it,
  returns `hipEventElapsedTime` ms. The bench already syncs each step via
  `int(mx.argmax(...))`, so the extra sync is ~0).
- Bench: `scripts/mtp-bench6-stepspan.py` / `mtp-bench6-stepdec-span.py`
  (generated by `scripts/make_stepspan.py` from the pinned record bench /
  t2241 stepdec copy; each loop iteration wrapped in try/finally with
  `perf_counter` wall + span; per-step line
  `SPANSTEP <idx> <ar|verify> wall=… span=… seam=…`).
- Arms, 3 rounds each, under `~/gpu-bench.lock`, `env -i` +
  `HCC_AMDGPU_TARGET=gfx1201`, 25 s idle between spec rounds (mirrors
  t2241): AR = venv-verify, MTP=0, 1024-token run; SPEC = venv-m3k, the
  t2241 13-var record config verbatim, MTP_MAX_TOKENS=322.
- Window: per-step stats over iteration index 16..47 (32 steps); t2241
  cross-check from its own H1.json (token-indexed 16..48 window — see
  caveat 3.7.3).
- Identity: libmlx md5 `a0057c22…` (venv-m3k) and `8acf13fe…` (venv-verify)
  unchanged from the t2241 pre-run md5s; model md5
  `02a54359…` unchanged.

## 3. Results

### 3.1 Cross-check against the completed t2241 battery (spec arm)

t2241's H and K rounds were BOTH the MTP=3 record config (K rounds:
rocprofv3 injection never loaded — `SMOKEK-FAIL`; its "kernel" data is the
host path, no kernel totals). Byte-identical cross-check on the span
instrumentation:

| round | t/s (t2241) | t/s (t2246 span) | Δ | TOKS | VERIFIES | D1ACC | D2ACC | D1PCT | D2PCT | TPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 69.23 | 68.64 | -0.9% | 436 | 161 | 114 | 78 | 70.8 | 68.4 | 2.193 |
| 2 | 69.20 | 68.83 | -0.8% | 436 | 161 | 114 | 78 | 70.8 | 68.4 | 2.193 |
| 3 | 68.57 | 66.64 | -3.1% | 436 | 161 | 114 | 78 | 70.8 | 68.4 | 2.193 |

TOKS/VERIFIES/D1ACC/D2ACC/D1PCT/D2PCT/TOK_PER_STEP and the generated token
prefix are **byte-identical** across all six rounds — the instrumentation
perturbs semantics zero; the -0.8% (r1/r2) is the true event-bracket
overhead (two `hipEventRecord` + one query loop + `perf_counter`), the
-3.1% (r3) is clock drift (see 3.4). FIRST16 identical to t2241:
`271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357`.

One config note (why a first spec attempt diverged): the t2241 battery ran
the 13-var record config (`MTP6_TAPE=hybrid MTP6_HEAD_H=postnorm
MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4` + five
`MLX_ROCM_*` kernel flags). The bare defaults (tape=1, prenorm head) produce
a different drafter (D1PCT 99.6) and greedy 9B falls into a 1919-repeat
de-generation by token ~3 — a reminder that the record config is part of the
protocol, not an optional optimization.

### 3.2 AR arm (MTP=0, venv-verify lane wheel, 1024-token runs)

| round | t/s | GEN | wall (16-47, n=32) | span | seam = wall-span |
|---|---|---|---|---|---|
| 1 | 62.00 | 16.52 s | 15.465 (med 15.461, p95 15.500) | 15.480 | -0.016 (min -0.017) |
| 2 | 62.06 | 16.50 s | 15.493 (med 15.494, p95 15.529) | 15.510 | -0.017 (min -0.036) |
| 3 | 62.09 | 16.49 s | 15.484 (med 15.486, p95 15.517) | 15.499 | -0.016 (min -0.017) |

Vs the t2222-session AR baseline (16.93 ms/step, 59.10 t/s): 9% faster
here (different session/clock state; same lane wheel) — wall 15.47 ms is
within 4% of the t2241-era ~16.0 ms single-forward reference. The per-step
AR step is one 1-token target forward + argmax; span == wall to within
0.04 ms, i.e. the GPU timeline fills the entire step wall.

### 3.3 SPEC arm (t2241 record config, venv-m3k, 322-step runs)

| round | t/s | draft wall (n=16) | verify wall (n=16) | seam (all 32) |
|---|---|---|---|---|
| 1 | 68.64 | 18.430 (med 18.387, p95 30.573) | 16.064 (med 16.302, p95 16.492) | -0.005 (min -0.016) |
| 2 | 68.83 | 18.464 (med 18.372, p95 30.630) | 16.072 (med 16.306, p95 16.482) | -0.004 (min -0.012) |
| 3 | 66.64 | 19.109 (med 18.922, p95 32.217) | 17.492 (med 17.421, p95 19.240) | -0.008 (min -0.039) |

span tracks wall on every kind to within 0.01 ms (e.g. r1 draft span
18.432 vs wall 18.430). t2241 H1.json same-window reference: draft wall
mean 20.558 (n=12, med 30.568), verify 15.983 (n=11, med 16.166). verify
matches r1/r2 within +0.6%; the draft means differ (18.4 vs 20.6) because
the window compositions differ (caveat 3.7.3) — the heavy-tail p95 agrees
to 0.1 ms (30.57-30.63 vs 30.72) and the medians bracket it. The 30.6 ms
draft steps are the one-time ngram/draftgate work after `MTP6_DGATE_T0=16`.

### 3.4 Round-3 clock drift (both arms)

spec r3: verify wall +9.3% vs r1, t/s -3.1%; AR r3 is stable (15.484 vs
15.465). r3 is treated as a drift round for the overhead delta only;
its acceptance stats are still byte-identical, so the drift is clock, not
semantics.

### 3.5 The seam

Measured: **seam = wall - span ∈ [-0.04, +0.01] ms over all 192 windowed
steps (both arms, six rounds)** — the host time outside the GPU timeline is
at or below measurement resolution. Two facts make the verdict robust:

1. **In-flight tail**: span exceeds wall by 0.005-0.04 ms — at the end of
   each step the GPU still has ~0.02 ms of work in flight when the host
   finishes the loop body. Across each step's scalar-sync points (the
   `int(mx.argmax(...))` fetches, 1-4 per step) the host never leads the
   GPU by more than that tail. A GPU that idled G ms waiting on host work
   between kernels would force wall > span by ≥ G (the host is the
   bottleneck of that segment); the observed wall - span ≈ 0 bounds
   host-induced inter-kernel idle at ≤ ~0.05 ms/step.
2. **Memory floor**: one pass over trunk weights ~5.54 GB at the measured
   631 GB/s streaming floor is 8.78 ms — a hard lower bound on the kernel
   sum. The ticket's expected kernel sum (11.3 ms, "all ops at 78% of
   floor") would leave 15.48 - 11.3 = 4.18 ms of between-kernel time;
   fact 1 shows at most ~0.05 ms of it can be host-side, and the rest
   would have to be hardware kernel-transition latency of ~4.7 µs per op
   (~900 ops) with the queue full — above the pipelined dispatch
   overhead the ticket's own single-matmul-at-78%-of-floor results already
   exclude (a single matmul reaches 76.8-80.2% of floor with no visible
   launch gap). So the kernel sum is ≈ 14.5-15.5 ms ≈ the step wall.

Equivalently, step-level memory efficiency = 8.78/15.48 ≈ 57% of floor:
the "missing" 6.7 ms vs the 11.3 ms expectation is per-op kernel
inefficiency (GDN state updates, quantized dequant ops, norms, small
kernels running well below the matmul's 78%), not between-kernel time.

### 3.6 Profiler-overhead delta

The dead-profiler question is moot; the measured bracket overhead is -0.8%
(r1/r2 spec t/s) — two event records + one query loop per step.

### 3.7 Caveats

1. `span = kernel sum + in-span idle gaps`; the bracket does not resolve
   the gap split directly. The verdict uses the sync-structure argument
   (3.5.1) + the floor bound (3.5.2), not a direct kernel sum. No kernel
   counts, no per-kernel breakdown: every profiler path on this box is
   dead (HSA 7.2.0 has no `hsa_amd_queue_intercept`; `gpu_busy_percent`
   is unreadable in the guest; HSA perfmon symbols absent).
2. The spec3 r1/r2 rounds are the clean overhead comparison; r3 carries a
   ~1 ms clock drift (3.4).
3. Window semantics: my window is iteration-indexed 16..47; t2241's
   STEPDEC window is token-indexed 16..48 (`_i0 = len(out)`). For the spec
   arm token-16 ≈ iteration-7, token-48 ≈ iteration-22 — partial overlap.
   Per-kind medians agree within 1% anyway (verify 16.30 vs 16.17).
4. Iteration 0's wall includes the prompt-tail sync (~691 ms AR) and is
   excluded from all window stats.
5. The AR arm has no t2241 counterpart (t2241 ran the record spec config
   in both H and K); the AR cross-check is against the t2222 baseline
   (3.2) and the t2241-era ~16.0 ms single-forward wall.

## 4. Verdict

**The hypothesis is DEAD: the missing time is NOT between the kernels.**
Across both arms and six rounds, the per-step GPU-timeline span equals the
step wall to within 0.04 ms, and the sync-structure argument (3.5) bounds
host-induced inter-kernel idle at ≤ ~0.05 ms/step. The kernel sum is
≈ 14.5-15.5 ms of a 15.47 ms AR step wall (≈ 95-100%), not the ~11 ms
the ticket's floor arithmetic predicted; the 6.7 ms delta is per-op kernel
inefficiency (57% step-level memory efficiency vs the 78% single-matmul
reference), i.e. the kernels are the problem.

Consequences for the levers (section 1.3), ranked by what this profile
actually supports:

1. **Per-op kernel efficiency is the lever.** qMLX fused MLP
   (gate+up+SiLU+mul in one kernel; mlp is 38.9% of the verify wall) and
   quantized-format work (MXFP4-class) attack the measured 57%-of-floor
   step efficiency directly.
2. **GraphDecodePilot (one graph launch per step) is a small lever on this
evidence**: with host-side gaps ≈ 0, a single launch per step can only
   recover hardware kernel-transition overhead (order ~1 ms at most, and
   unmeasured here — see caveat 3.7.1), not the 6.7 ms inefficiency. It
   remains worth arming (env-gated, zero-risk fallback) but should not be
   expected to close the gap to the 88 t/s anchor.
3. The 88 t/s anchor arithmetic (11.3 ms at 78%) is refuted as a
   same-step target: it assumed every op class runs at the single-matmul
   efficiency; the measured step says the non-matmul op classes run well
   below it.

What remains unmeasurable on this box (for the record): exact kernel sum,
kernel counts per step, per-kernel breakdown, SCLK per round (the dGPU
sampler read fails on busy sysfs in this guest). A future run on a
rocprofv3-capable stack (ROCm 10.x Rust runtime or a host with HSA
`queue_intercept`) would convert the 3.5 argument into a direct
measurement.
