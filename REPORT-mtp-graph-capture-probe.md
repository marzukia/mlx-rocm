# MTP graph-capture feasibility probe (t1807, <user>@<tailnet-ip>, 2026-11-04)

## TL;DR

- **Q1 (capture feasibility): FAIL (structural).** No first-capture crash (0 HSA errors, rc=0 on all arms — the AR first-capture-crash class is not hit), but the S=4 verify forward is `big_shape` and is structurally forced eager by `gd_step_is_big` (the AR machinery is S=1-only), and the MTP step's allocation footprint catastrophically exhausts the 2 GB decode arena (37,072 arena-fallback lines, overflow #41000+). The machinery arms on the ~850–948-op forwards and captures only the micro steps (the 4-bit lm_head, 76–79 ops); those replay with stale buffers and the stream degenerates to a `271 0` attractor (D1ACC 26→0). The 64-token graph run is **not op-stable** (continuous micro re-capture + massive arena overflow + degenerate replay).
- **Lossless gate: FAIL.** Same-session A (eager) vs G (graph): A = 106 clean tokens (43.37 t/s, D1ACC=26), G = 64 degenerate tokens (23.92 t/s, D1ACC=0, TOK_PER_STEP=1.000); FIRST20 diverges at token 5 (A=15352, G=271); ALLTOKS md5 mismatch (f84f6856 vs cc24c7b5).
- **Q2 (replay latency): N/A (Q1 failed).** No valid graph-vs-eager speed comparison exists because the replay is degenerate. The observed G 23.92 vs A 43.37 t/s (0.55x) is the acceptance-floor degeneration (D1ACC=0), not a graph-overhead measurement; per-replay-step ms is not meaningful.
- **Best-case recovery estimate:** ~0.5–2% of the MTP step (≈0.4–1.8 ms out of a 76 ms step); realistically **negative**, bounded by the AR empirical result (graph slower than eager: 65.05 vs 72.11 t/s, −9%).
- **Recommendation: NO-GO** for a full MTP-capture implementation on the existing AR machinery. The blockers are structural (S=4 `big_shape` exclusion + decode-arena exhaustion), not the AR lane's stale-state class, and a full MTP capture would require a new S-batched (T>1) capture path plus a far larger arena or per-forward capture — a new implementation, not a reuse.

## Setup (identity)

- Commit under test: `390221f` (reports-only since `78ba063`; lever files byte-unchanged, verified via `git diff --stat 78ba063..390221f` = REPORT-int2w-step.md only). `78ba063` is an ancestor of `390221f`.
- Wheel: `venv-int2w`, `libmlx.so` md5 `0b8aa95df151a93404b53fee1e96d0de` — byte-identical to the t1796 78ba063 wheel → **reused, no rebuild**. Graph-decode machinery confirmed present in the .so (`graph_decode_mode` ×7, `BeginCapture` ×4, `gd_step_prepare` ×1, `decode_capture_begin` ×5); import gate OK (`gdn_fused_step` + `gdn_fused_step_tape` live).
- Model `<models-mount>/qwen3-9b-mlx`; sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` (the 02a54359 pin, verified pre- and post-session). Bench `mtp-bench6.py` md5 `3c7d13b56509634aae1708d4a4df7b16` (matches the t1796 canonical).
- Config of record: `MLX_MTP=4 MTP6_TAPE=hybrid MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1`, `LD_PRELOAD=libhsa-runtime64.so.1`, `HCC_AMDGPU_TARGET=gfx1201`, `MODELS=<models-mount>`. 64-token short run (the 43.6-class; the 512-verify protocol lands in the 40.3-class per the regime note — same-session A-vs-G is the valid gate).
- Capture gate: `MLX_ROCM_GRAPH_DECODE=1` (the master gate; note the secondary `MLX_GRAPH_DECODE` mode gate defaults on and is distinct — the first probe run set only the secondary and was therefore a clean eager baseline). `MLX_G1Y_DEBUG=1` for the instrumented run.

## Q1 evidence (capture feasibility)

- **First capture: no crash.** All arms (debug + clean G) complete rc=0 with **0 HSA errors** and no `previous error during capture` / `invalidated` / 906 / taint lines — the AR first-capture-crash class (in-window D2D / alloc / host-sync) is **not** triggered by the MTP bench.
- **The S=4 verify is structurally eager (the load-bearing blocker).** `GdPilotLogic::first_op` forces `Kind::Eager` and resets `next_capture`/`replay_armed` whenever `big_shape` is true, and `gd_step_is_big` classifies any first-op input of shape `[1, T]` with `T>1` as big. The MTP verify is a batched `[1, 4]` forward (`[1,4]` lm_head gather index visible in the `[g1] first-op` log), so it is `big_shape` and **never captured**. The arming streak (`eager_end`, op-count-based) still advances on the ~850–948-op forwards, but the capture-candidate step is always a `big_shape` step and runs eager. The instrumented log confirms the mechanism arms on the ~850-op forwards (`[g1] graph decode: armed (small steps 853/948 ops); capturing next`) and then captures only the micro steps (`[g1] graph decode: step captured+launched (ops=79/76, slot=1/0, active)` → `[g1] graph decode: micro step (ops=76/79); launched, re-capturing next`). The 76–79-op captured step is the 4-bit lm_head (gather `[248320, 64]` + dequant + matvec + argmax), not a full forward.
- **Decode-arena exhaustion (the second blocker).** The 64-token instrumented run logged **37,072 `arena fallback (overflow)`** lines, up to `arena overflow #41000`. The decode arena (`cap=2147483648` = 2 GB) is sized for a single AR token's allocations; the MTP step (S=4 verify + 3-step draft chain + 0–3 correction forwards) allocates far more, so the step's buffers repeatedly overflow out of the arena. A captured graph's baked pointers target arena VAs; when the buffers overflow at replay, the recorded nodes read freed/stale memory.
- **Degenerate collapse (stale-read class, not the AR period-4 attractor).** The clean (no-debug) G run's FIRST20 = `271 1919 369 264 271 1919 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0`, D1ACC=0, TOK_PER_STEP=1.000 — a `271 0` attractor (distinct from the AR lane's `264 15352 8214 5529` period-4 cycle). The first 4 tokens match the canonical `271 1919 369 264`, then the collapsed lm_head argmax (reading pre-capture hidden state) drives all drafts to reject (D1ACC 26→0). The identical collapse in both the instrumented and clean runs confirms it is intrinsic to the graph-decode MTP path, not a `MLX_G1Y_DEBUG` artifact.
- **64-token op-stability: FAIL.** The graph run does not reach a single stable capture: it continuously re-captures micro steps (`micro step …; launched, re-capturing next`) under sustained arena overflow, and the replay is degenerate (stale-read collapse), not a valid single-capture op-stable lossless replay.

## Lossless gate: FAIL (reported with reason)

- A (eager, same session): 106 tokens, clean stream, FIRST20 `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`, D1ACC=26 D2ACC=16 D3ACC=6, TOK_PER_STEP=2.500, 43.37 t/s.
- G (graph, same session): 64 tokens, degenerate `271 0` collapse, FIRST20 `271 1919 369 264 271 1919 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0 271 0`, D1ACC=0, TOK_PER_STEP=1.000, 23.92 t/s.
- FIRST20 diverges at token 5 (A=15352, G=271); ALLTOKS md5 mismatch (A `f84f6856…` vs G `cc24c7b5…`). Reason: G's captured micro steps replay stale buffers (arena overflow + S=1-only machinery on a batched step), so the MTP path degenerates to the 1-token/step floor and the stream is the degenerate `271 0` attractor, not A's accepted-MTP stream.

## Q2 (replay latency): N/A

- Q1 failed, so there is no valid captured-replay-vs-eager speed comparison. The observed G 23.92 vs A 43.37 t/s (0.55x) is the acceptance-floor degeneration (D1ACC 26→0), not a graph-overhead measurement. Per-replay-step ms is not meaningful for a degenerate replay; no 3-run median was taken (the Q2 3-run protocol is gated on Q1 passing).
- Eager-arm regime: the 64-token A arm = 43.37 t/s (the 43.6-class), consistent with the 40.30 512-verify-class per the regime note (shorter runs land higher; absolute t/s on this box is regime-dependent, same-session A-vs-G is the valid gate).

## Best-case recovery estimate (task item 4)

- Per-MTP-step time (eager, measured): 2.44 s / 32 verifies = **76 ms/step**.
- Per-op CPU enqueue overhead (measured, pipelined tight loop of 2000 tiny matmuls): **0.44 µs/op** (the async CPU enqueue; the serial launch+compute+sync is 107 µs but that is compute-dominated, not launch).
- Launch count per MTP step: ~850–4000 ops (draft chain 3 × ~130 [1-layer MTP module + lm_head] + S=4 verify ~930 + 0–3 correction forwards × ~930), from the `853/948`-op full forwards in the `[g1]` log and the 76–79-op lm_head micro steps.
- Best-case CPU enqueue overhead per step: 850–4000 × 0.44 µs = **0.37–1.76 ms** → **0.5–2.3% of the 76 ms step** (upper bound — the pipelined enqueue overlaps with GPU compute, so the non-overlapped, actually-recoverable fraction is smaller).
- AR empirical bound: graph capture of the single-token decode step is **net-negative on this box** (65.05 vs 72.11 t/s, −9%), i.e. the launch savings are smaller than the capture overhead (arena management, slot staging, graph launch). That directly bounds the realistic MTP recovery as negative-to-negligible.
- Conclusion: a captured MTP step, even in the best case, buys ~0.5–2% of the step (≈0.4–1.8 ms); realistically it is negative given the AR empirical result and the two structural blockers above.

## Blocker characterization + root cause (evidence chain)

1. **Structural exclusion (the load-bearing blocker).** `GdPilotLogic::first_op` (graph_decode_pilot_logic.h:47-52) forces `Kind::Eager` + resets `next_capture`/`replay_armed` on `big_shape`; `gd_step_is_big` (eval.cpp:277-290) classifies any first-op input `[1, T]` `T>1` as big. The MTP verify is `[1, 4]` → always eager. The machinery is S=1-only by design and cannot capture the batched verify.
2. **Decode-arena exhaustion.** The arena cap is 2 GB (gd_arena_cap default; `MLX_ROCM_GRAPH_DECODE_ARENA_MB` unset) sized for one AR token. The MTP step's allocation footprint (S=4 verify + 3-step draft + 0–3 corrections) overflows it (37,072 fallbacks, #41000+). Captured-node baked VAs target arena memory that is no longer valid at replay → stale reads.
3. **Stale-read collapse (distinct from the AR period-4 attractor).** The captured lm_head argmax reads the pre-capture hidden state → `271 0` attractor, D1ACC=0. This is a different signature from the AR lane's `264 15352 8214 5529` period-4 cycle (a recurrent-state staleness), consistent with a one-shot (non-recurrent) read of an overflowed buffer rather than the in-place-recurrent-state issue the AR lane fixed in the fused op.

## Recommendation: NO-GO for a full MTP-capture implementation on the existing AR machinery

- The blockers are structural and not tunable: (a) the S=4 verify is `big_shape` → eager (the machinery is S=1-only); (b) the 2 GB decode arena is exhausted by the MTP step's footprint. A full MTP capture would require a **new** S-batched (T>1) capture path and a far larger arena (or per-forward capture, which forfeits the single-exec replay model), i.e. a new implementation, not a reuse of the AR lane.
- The structural ground is small and likely negative even in the best case (≤2% of the step, and the AR empirical result is net-negative). Combined with the AR lane's own net-negative outcome (slower than eager on the box it was tuned for), the expected value of a full MTP-capture investment on this hardware/machinery is not positive.
- If the operator wants to proceed regardless, the minimal next probe is a **launch-bound vs compute-bound** characterization of the S=4 verify forward (a per-kernel profile of one verify forward) to confirm the ≤2% launch-overhead ceiling before any implementation investment.

## Checks

- GPU work under `flock -x ~/gpu.lock`, 25 s idle before and between runs, `fuser` checked before GPU work (no holders, no orphans pre- and post-session), lock released at end.
- `venv-canon` untouched (all arms on `venv-int2w`).
- Sidecar `model-mtp.safetensors` md5 `02a54359…` verified pre- and post-session (unchanged).
- No clock sysfs readback available at agent privilege on this bare-metal box (pp_dpm nodes unreadable); the same-session A-vs-G (the valid gate) and A = 43.37 (43.6-class, 40.30 512v-class per the regime note) bracket the regime.
