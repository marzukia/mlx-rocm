# REPORT-seam — decode-step seam measurement (t2231)

Claim under test: the kernels are roughly fine; the missing time is BETWEEN them,
not inside them. Verified arithmetic: AR reference 16.93 ms/step, 59.10 t/s
(t2222 session); one pass over trunk weights ~5.54 GB; in-session streaming floor
~631 GB/s → 8.8 ms floor; single large matmul in isolation hits 76.8-80.2% of
floor (t2199/t2207) → if every op ran at 78%, the step would be 8.8/0.78 =
11.3 ms = 88 t/s ≈ llama.cpp non-speculative anchor 87 t/s. EXPECTED: AR kernel
sum ~11 ms vs step wall ~17 ms. If the kernel sum is already ~17 ms the
hypothesis is DEAD — the kernels are the problem.

Token bases: AR = 1 tok/step vs record spec = 2.193 tok/step. Different bases:
stated whenever tok/s is compared.

## 1. Code-first: graph-decode state on the two wheels (no GPU)

Question: which is true NOW on the lane wheel (venv-verify, libmlx 8acf13fe) and
the record wheel (venv-m3k, libmlx a0057c22): (a) capture+replay armed (one
graph launch per step), (b) capture-then-relaunch-every-step, or (c) eager with
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

- The bench/serve path matters: `gpu_set_graph_decode_mode(bool)` (eval.cpp:738)
  is the only C entry that set the old dormant machine, and it is declared in
  NO header (grep over the whole vendor tree: the only occurrences are the
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
MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1 MTP_MAX_TOKENS=322).
**CONFIRMED on the box (2026-11-06), state (c) EAGER, one launch per op, both
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
  big=0, div=1)` — evidence the capture+replay machine arms when enabled).
- Predicted measurement signature of the eager state: hundreds-to-thousands of
  kernel launches per step (a graph-armed step would be ONE graph launch per
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

## 2. Method

- Box: neon, dGPU card0 (PCI 0000:03:00.0, device id 0x7550) only; card1
  (0000:75:00.0, 0x13c0 iGPU) never used; GPU under `env -i` with LD_PRELOAD
  of the system libhsa-runtime64.so.1 (/usr/lib/x86_64-linux-gnu) and
  HCC_AMDGPU_TARGET=gfx1201; serialized on
  /home/andryo/gpu.lock (holder PID recorded before each hold; stale-lock
  policy per brief). Profiler: rocprofv2 (ROCm 7.2.0 userland, /opt/rocm/bin;
  v2 binary is named `rocprofv2` on this box), `--kernel-trace`.
- Clean rounds: pinned mtp-bench6.py (md5 54ce16ac29e6c6aad2c48a233ff70d85),
  no profiler, MTP_MAX_TOKENS=322 per arm — step wall from these rounds.
- Profiled rounds: a per-step driver (scripts/seam_step.py) replicating the bench
  loop body exactly, warmup + ONE marker-bracketed step, under rocprof-v2
  kernel trace; per step: kernel-duration sum, kernel count, profiled wall.
  Profiler overhead = profiled step wall minus clean step wall (measured delta).
- SCLK sampled 25 s idle before the sample; per-arm P-state recorded;
  bench_identity.py pre and post per arm.

## 3. Results

(measurement pending — kernel sums, walls, seams, kernel counts, per-round
variance, profiler-overhead delta, identity blocks, SCLK/P-state, telemetry
caveats)

## 4. Verdict

(pending)
