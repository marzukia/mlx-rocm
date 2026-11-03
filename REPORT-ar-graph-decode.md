# REPORT-ar-graph-decode: graph-decode 2nd-capture fix, MTP=0 AR bench, verdict

Task: fix the known second-capture fault in the graph-decode path
(`MLX_ROCM_GRAPH_DECODE`), enable graph decode for the canonical AR bench
(MTP=0, fused-GDN), measure e2e performance, verify losslessness (ALLTOKS
bit-identical vs no-graph), run the MTP=4 guard, and deliver this report on
branch `ar-step-change-r2`. Goal: close the 12.3 ms/token AR gap vs
llama.cpp (86.9 t/s) by graph capture (eliminate ~517 eager kernel
launches/token).

**Verdict up front: the second-capture HSA fault is FIXED and the
capture-vs-replay op-sequence divergence is CLOSED, but graph decode (MTP=0)
is NOT lossless — every replay step collapses to a degenerate fixed point
(271 attractor), and it is also SLOWER than the eager baseline (67.8 vs
74.4 t/s). It is net-negative on both axes. The MTP=4 guard PASSES
(43.64 t/s ≥ 38.9, tokens correct). Per the negative-result rule, graph
decode stays DEFAULT OFF (zero behavior change when the env is unset). The
root cause of the replay collapse is traced to a per-step input/state
staging issue on the replay path (see §Diagnosis); it is not fixable
within this round's scope and is recorded as the blocker for the AR gap.**

## Setup (identity)
- Branch `ar-step-change-r2`. Guest-a: `100.78.68.72` (RX 9070 XT / gfx1201,
  ROCm 10.1 guest userland), GPU locked at `/home/andryo/gpu-bench.lock`.
- Standing config: `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1
  MLX_ROCM_GDN_FUSE=1`, sclk locked 2200 MHz, model
  `/mnt/models/qwen3-9b-mlx`.
- Bench: canonical `mtp-bench6.py` (md5 `452d363de7c6754de8333ae09999327f`),
  MTP=0, 400-token window, 3-run median, FIRST20/ALLTOKS anchor.
- Wheel: `mlx-0.32.3.dev20261103+75ca881-cp314` built on guest-a from
  `vendor-gd2` (fork pin per `vendor.json` + the graph-decode port), built
  with `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`,
  `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1;/opt/rocm-7.2.0"`
  (the `-7.2.0` prefix is required for `rocwmma`), installed into a clone
  venv `venv-gd2` (venv-canon untouched). The wheel carries env-gated
  diagnostics (`MLX_G1_OPDUMP` op-sequence dump, `MLX_G1_IDXDBG` gather-idx
  probe, `MLX_G1_H2DLOG` pre-launch H2D log, GPROBE_HID hidden-state digest
  in the probe script); all are OFF unless their env is set, so they do not
  affect the default-off or the A/G/MTP battery numbers.

## What was fixed (the 2nd-capture fault)
1. **Second-capture HSA fault — FIXED.** On the prior wheel, the 2nd
   stream-capture (the exec-cache: capture once per shape, then replay)
   produced HSA faults on the re-captured graph. On this wheel, 400-token
   MTP=0 graph runs complete with **0 HSA errors**. The capture/replay
   mechanism is stable.
2. **Capture-vs-replay op-sequence divergence — CLOSED.** `MLX_G1_OPDUMP`
   dumps the full op sequence (name + shape) for the capture step and the
   first replay step. They are **byte-identical: 875 ops, zero diff**. The
   replay graph executes exactly the op sequence the capture recorded. (On the
   prior wheel the sequences diverged; that is resolved on this build.)

So the two "known second-capture faults" (HSA fault + op divergence) are both
fixed. What remains is a **correctness** fault on replay, not a capture
fault.

## The remaining fault: replay collapses to a fixed point (NOT lossless)
- No-graph baseline (A.1/A.2/A.3): `271 1919 369 264 15352 11 70042 29006 ...`
  — correct, matches the canonical anchor exactly.
- Graph decode (G.1/G.2/G.3): `271 1919 369 271 271 271 271 ...` — the first
  three tokens match (prefill + 2 eager small steps + the capture step), then
  **every replay step collapses to the 271 attractor**. ALLTOKS is 271×397.
  NOT bit-identical to the no-graph baseline. **Losslessness FAILS.**

### Diagnosis (evidence chain)
- **Hidden-state digest (GPROBE_HID):** eager per-step hidden-state L2 norm
  varies (`1056 1184 1144 1384 1688 1296 1640 1696`); graph K3-OFF is
  **frozen at the capture step's value** (`1056 1184 1144 1144 1144 1144 ...`).
  The replay hidden state is byte-identical to the capture step → the replay
  is recomputing the capture step's exact computation (stale inputs/state), not
  the new step.
- **K3 idx redirect (confirmed, well-understood sub-bug):** in K3-on mode the
  `n_rows==1, out≤4KB` fast_rows gather is redirected to read its index from
  the 4 KB zero-slot, which is zero-cleared pre-launch → the gather always
  reads **row 0** (token 271) instead of the real idx buffer. This is what
  produces the 271 (not 264) attractor under K3-on, and it corrupts even the
  capture step (capture hidden norm 1360 ≠ eager 1144 under K3-on; correct at
  1144 under K3-off). The K3 idx redirect is therefore a real bug; the
  `MLX_GD_K3IDXOFF` escape hatch (default off for the redirect only, added this
  round) isolates it.
- **K3-off still collapses (to 264):** with the idx redirect off, the gather
  reads the real idx buffer, which **is** updated to the new token (probe
  confirms the device idx buffer holds the fresh value on every replay step).
  Yet the hidden state still freezes at the capture value. The 264 (not 271)
  attractor under K3-off shows the embedding path is correct but a **different
  per-step input/state is still stale on replay** — the replay graph is not
  consuming the newly-staged per-step data (KV/GDN-state/offset staging) even
  though the op sequence is identical and the idx buffer is fresh.
- **Pre-launch slot ordering (traced, not the sole cause):** the pre-launch
  delivers the recorded (capture-time) H2D memcpies after the fresh per-slot
  H2D, so a recorded H2D can clobber a freshly-staged slot value. Reversing
  that order was tested; it does not by itself cure the collapse (K3-off +
  NO_PREH2D still collapses), so the slot clobber is a contributing hazard, not
  the root cause.

**Root-cause summary (confidence: high on the symptom, medium on the exact
buffer):** on the replay path the recorded graph recomputes the capture step
byte-for-byte — its per-step inputs (beyond the idx buffer, which is
correctly staged) are still at capture-time values. The op sequence is
identical and the arena reuses the same VAs, so the freeze is a **value
staging** gap on the replay path (the new step's per-step data — the
growing KV/GDN state and their offsets — is not the value the replay graph
reads), not a shape/sequence or HSA fault. Closing it requires tracing the
exact replay staging of the per-step state buffers (KV/GDN `set_data`
re-alloc VA vs the graph's baked pointer), which is the next round's work.

## Battery (canonical, 3-run median)
| arm | median t/s | tokens | verdict |
|-----|-----------|--------|---------|
| A.1/A.2/A.3 (eager, no-graph) | **74.42** (74.42/75.26/74.41) | `271 1919 369 264 15352 11 70042 29006 ...` | baseline, correct |
| G.1/G.2/G.3 (graph decode) | **67.84** (67.79/67.84/68.00) | `271 1919 369 271 271 271 ...` (271×397) | **degenerate, NOT lossless, SLOWER** |
| MTP4.1 (MTP=4 guard) | **43.64** (≥ 38.9) | `271 1919 369 264 15352 11 70042 ...` (matches A) | **PASS**, correct |

- Losslessness (ALLTOKS bit-identical G vs A): **FAIL** (271 attractor).
- e2e: graph decode 67.84 t/s < eager 74.42 t/s → **−8.8%** (net-negative; the
  graph is not currently buying the launch-overhead win because the replay
  path is not the fast captured single-launch path in a correct form).
- MTP=4 guard: **PASS** (43.64 ≥ 38.9), tokens correct (matches baseline).

## Verdict & disposition
- The two known second-capture faults (HSA fault, op-sequence divergence) are
  **fixed** on this wheel.
- Graph decode MTP=0 is **net-negative on both axes** (slower 67.8 < 74.4, and
  degenerate/271 attractor, not lossless). Per the negative-result rule and the
  task's explicit fallback, **graph decode stays DEFAULT OFF**
  (`MLX_ROCM_GRAPH_DECODE` unset → eager path, zero behavior change).
- The 12.3 ms/token AR gap is **not closed** this round: the blocker is the
  replay per-step input/state staging gap traced in §Diagnosis, not a capture
  fault. That is the next round's work (trace the exact replay staging of the
  growing KV/GDN state buffers vs the graph's baked pointers).
- MTP=4 is unaffected and passes its guard.

## Reproduction
- Wheel: `/home/andryo/wheelout-gd2/mlx-0.32.3.dev20261103+75ca881-cp314-...whl`
  (venv-gd2). Install requires `rm -rf` of the old `mlx/`, `bin/`,
  `*.dist-info` dirs first (pip `--target` does not overwrite).
- Battery: `/home/andryo/gd2/{A.1,A.2,A.3,G.1,G.2,G.3,MTP4.1}.log`.
- Probes: `/home/andryo/arstep_gd_probe.py` (PYTOK + GPROBE_HID);
  `PROBE.OD3.log` (op-sequence dump), `PROBE.HL.log` (H2D log),
  `PROBE.XK.log` (idx K3 on/off).
- K3 idx redirect: `MLX_GD_K3IDXOFF=1` (isolate), `MLX_GD_K3OFF=1` (full off),
  pre-launch slot H2D: `MLX_G1_NO_PREH2D=1`.
