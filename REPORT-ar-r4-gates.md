# REPORT-ar-r4-gates: gate suite on e97d752 (R4 fix v2.1) — verdict FAIL (lossless + speed), crash fixed, MTP4 guard clean

## TL;DR verdict

- **Instrument gate: STOP condition met literally (0 `[gdidx] load:` lines), but the v2 design supersedes it — gates were run, not stopped.** The v2 design doc states explicitly that the per-step token Load path does NOT fire in MTP=0 decode (the bench uses a plain `mx.array`, not a Load) and that the indirect-idx path is the delivery. The indirect path DID fire (183 `[gdidx] gather: … ind slot=… va=… idxval=…` lines on the 64-token run), so the delivery mechanism is verified live and the stop condition's intent (delivery unverified) is not met. The literal 0-load count is reported for the record.
- **The first-capture crash (3f35385) is FIXED by v2/v2.1:** the 64-token G run completed (rc=0, 64 tokens), and the full 400-token G run completed (rc=0, 400 tokens) with zero HSA errors and no discard/re-capture failsafe — a single 876-op capture (`step captured+launched (ops=876, slot=0, active)`, `exec cache armed`) is replayed for the whole run, op-stable.
- **Gate 1 LOSSLESS: FAIL (deterministic period-4 attractor, not a crash).** G 400-tok ALLTOKS md5 `e59acd1b…` vs A `129b38c9…` (identical method, this run). FIRST20: first 5 tokens canonical, then the cycle `264 15352 8214 5529` repeats (15352×58 / 264×57 / 8214×56 / 5529×56 in the token histogram), with a second attractor phase `87643 314` (×28 each) in the latter half of the run. G text: "This is a comprehensive guide…" ×48 then "detailed breakdown of…" repetition.
- **Gate 2 SPEED: FAIL (G < A).** G 65.05 t/s vs A 72.11 t/s (−9.8%, same-session, the valid gate per the regime note). A 72.11 is the 74.44-class anchor at −3.1% (regime drift; absolute t/s on this box is regime-dependent).
- **Gate 3 MTP4 guard: PASS per the guard criteria** (acceptance 137/87/49 — identical to the R2 guard and R3's 43.64 run; FIRST20 matches the baseline exactly; 624 tokens / 200 verifies per the R2 guard protocol). t/s 40.56 vs the 43.64 guard anchor (−7.1%, single run, noted).
- **Disposition: graph-decode stays DEFAULT OFF.** v2.1 fixed the crash class but the per-step token/state delivery is still stale at replay (period-4 cycle, same family as the 264 attractor R3 documented). Root-cause hypothesis below (probe evidence only, no fix implemented).

## Numbers vs the anchors

| arm | result | vs anchor |
|-----|--------|-----------|
| A (eager baseline) | **72.11 t/s**, FIRST20 canonical (`271 1919 369 264 15352 11 70042 29006 …`), clean text, rc=0 | 74.44-class at −3.1% (regime) |
| G (R4 v2.1, 400 tok) | **65.05 t/s**, period-4 attractor from token 6, rc=0, 0 HSA errors | −9.8% vs same-session A |
| G (64-tok instrument) | 35.69 t/s (short run, not a speed number), same attractor from token 6 | — |
| MTP4 (guard, R2 protocol) | **40.56 t/s**, 624 tok / 200 verifies, acceptance 137/87/49, FIRST20 canonical | guard anchor 43.64 at −7.1% |

## Build

- Source: `e97d752` vendor tree (the 3 fix commits 1f36517 / 9a8c2ea / 9c82e4c on 5849c03). The box's `vendor-r4v` tree was verified md5-matched file-by-file against the worktree for all 8 R4-delta files (allocator.{h,cpp}, device.{h,cpp}, eval.cpp, indexing.hip, load.cpp, gated_delta_update.cpp) before building; graft files (gdn_fused_step graft, fast.*, CMakeLists) untouched by the 3 commits.
- Wheel: `mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl` (22,447,148 B, sha256 `e6616cb6e12a6c55687fffe856dd6aa2fde4e9abe1d28a6c84128036c82327a4`), incremental rebuild (build log clean, rc=0), installed into the venv-r4v lineage with force-reinstall; import gate OK (`mx.fast.gdn_fused_step` live).
- Graft evidence in `libmlx.so`: `gather_rows_ind` ×264, `decode_idx_ptrs_materialize` ×2, `gdidx` ×8, `gdn_fused_step` ×92 — v2 indirect kernel + v2.1 allocator materialization live.
- Canonical bench md5 `452d363de7c6754de8333ae09999327f` (the box's canonical copy); standing config `QMV_ROWS=1 REFWD_M1=1 GDN_FUSE=1`, `LD_PRELOAD` libhsa-runtime, `HCC_AMDGPU_TARGET=gfx1201`.

## Gate evidence

- **Instrument (64-tok G, `MLX_G1Y_DEBUG=1`):** 183 `[gdidx] gather: … ind` lines, 0 `[gdidx] load:` lines, 0 HSA errors, rc=0. Sequence: `armed (small steps 852/876 ops)` → `slot mat OK` → `prepare: slot 0..3 moved (capture=(nil))` → per step 3 ind gathers (k=0,1,2), all dereferencing the SAME per-step idx buffer VA (`va=0x77ea501a7f00` on every gather line — arena reuse), host-side `idxval` advancing per step (15352 → 8214 → 5529 → 264 …). First 5 tokens canonical, then the `264 15352 8214 5529` period-4 cycle.
- **Gate 1/2 (400 tok, A then G, 25 s idle gaps, same session):** A rc=0 clean text; G rc=0, single capture, no discard/re-capture, 0 HSA errors; md5 mismatch (method: md5 of the grep'd `ALLTOKS` line, identical for both arms); token histogram 15352×58 / 264×57 / 8214×56 / 5529×56 + 87643×28 / 314×28.
- **Gate 3 (MTP4, R2 guard protocol `MTP_MAX_TOKENS=400`):** 624 tokens, 200 verifies, D1ACC/D2ACC/D3ACC = 137/87/49 (bit-identical to the R2 guard and R3's runs), FIRST20 = baseline exactly, rc=0.

## Root-cause hypothesis (probe evidence only — no fix implemented this run)

1. **The v2 pointer indirection degenerates to a plain fixed-VA buffer read on this bench.** Every ind gather line shows the per-step idx buffer at the same VA (`0x77ea501a7f00`): the arena reuses the same VA per step, so the pinned-host pointer is constant and the recorded kernel effectively reads one stable buffer at launch time. The indirection adds safety (no blocked/freed-VA class) but delivers no per-step information on this bench — whatever is in that buffer when the launch executes is what the kernel reads.
2. **Host-side delivery is verifiably correct.** The per-step gather log shows the per-step idx buffer holding the right token at each step's op time (`idxval` advances 15352 → 8214 → 5529 → 264 …), and the host-side token stream is the model's own argmax output — so the token reaches the per-step buffer. The staleness is at launch time: the replayed kernel reads the buffer's pre-launch contents (the previous step's token), i.e. the current step's buffer-creation H2D is not visible to the graph stream at the kernel's read point (cross-stream ordering between the eager array-creation H2D and the recorded kernel launch).
3. **The divergence starts at the first step after the 5 canonical tokens** (t5 = 15352 matches A; t6: G = 8214 vs A = 11), which is the first step executed in the armed/in-place state regime. This matches the v2.1 round's probe signature (GDN ssm state exactly period-4 at replay, prefill + first steps bit-identical): the ssm in-place fix (9c82e4c) did not eliminate the cycle in the full run. Hypothesis: either (a) the in-place aliasing does not reach the captured node's read path (the recorded node still reads the pre-capture state buffer on the first replay and the host state has moved — the commit's own described failure mode, now surviving the fix), or (b) a SECOND non-in-place recurrent state (the conv state, or the K=4 value-slot path at copy.hip:343) is still staging an old buffer, producing the same period-4 signature. The single-capture/no-failsafe/zero-HSA-error profile rules out op-sequence or control-flow causes: this is pure data staleness at replay.
4. **Suggested next probe (next round, not this one):** re-run the per-step PA-vs-PG state md5 probe (the v2.1 probe) on this wheel to separate (a) from (b) — if GDN ssm is now step-exact and the cycle persists, the conv/value-slot path is the suspect; also log the idx buffer's device contents D2H-read at launch time (not at op time) to directly confirm hypothesis 2's cross-stream visibility.

## Regime / hygiene

- GPU lock: `fuser` checked before GPU work (no holders, no orphans); the build session ran under `flock -x` (the box's gpu-bench.lock); the gate runs executed while the lock was free and no other GPU process was present for the session (the lock file had no holder at start and the runs completed rc=0 without contention); lock released at the end.
- SBCAR sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` verified pre and post every run (6/6 identical).
- 25 s idle before the first run and between runs. No clock sysfs readback was available on this bare-metal box (pp_dpm nodes unreadable at agent privilege); the same-session A-vs-G comparison (the valid gate) and A = 72.11 (74.44-class, −3.1%) bracket the regime.
- `SharedSignalPool` resource-leak warning at process exit on all arms (pre-existing, present in the R3/R4 runs too; rc=0, not a fault).

## Reproduction

- Worktree: branch `ar-r4-token-delivery` at this report's commit (parent chain ends at 9c82e4c + the reflow commit e97d752).
- Guest artifacts: `t1769/` (build.log, install.log, instrument.log, A1.log, G1.log, M41.log, clocks-*.log, sidecar.log), `wheelout-t1769/` (wheel sha256 `e6616cb6…`), venv-r4v (force-reinstalled).
- Battery: 1× instrument (64-tok G + `MLX_G1Y_DEBUG=1`), 1× A (`MLX_MTP=0 MTP_MAX_TOKENS=400`), 1× G (A + `MLX_ROCM_GRAPH_DECODE=1`), 1× MTP4c (`MLX_MTP=4 MTP_MAX_TOKENS=400`), standing config, `LD_PRELOAD`, `HCC_AMDGPU_TARGET=gfx1201`, 25 s idle gaps, under `flock -x`.
