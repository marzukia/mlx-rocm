# REPORT-ar-r3: graph-decode R3 — per-step value/idx staging as a capturable kernel node; verdict

## TL;DR verdict

- **Gate 1 LOSSLESS: FAIL.** Graph-decode replay (G) is not bit-identical to the non-graph AR baseline (A): ALLTOKS md5 `d14f3eab…` vs `de160355…` (n=400, 3/3 runs each, each arm self-identical across its 3 runs). G collapses to the 264 attractor from step 4 (the capture step's token replayed forever).
- **Gate 2 AR > 74.4 t/s: FAIL.** G median **67.69** (67.71/67.69/67.69) vs the 74.4 gate (−8.8% vs A baseline 74.44), vs the 86.9 t/s mission anchor (−22.1%), vs the 84.8 non-graph AR best (−19.9%). R2 was 67.84 — R3 did not move the needle (−0.2%).
- **Gate 3 MTP=4 guard: PASS.** 43.69/43.60/43.61 (median 43.61) at the exact R2 guard protocol (MTP_MAX_TOKENS=400 → 200 verifies, 624 tokens, acceptance 137/87/49 — identical to R2's 43.64 run). Tokens: FIRST20 matches the baseline exactly.
- **Disposition: graph-decode stays DEFAULT OFF** (env unset → eager path, zero behavior change). R3 fixed the capturable-node half of the replay staging gap (no HSA errors, no non-capturable D2D in the graph, stable op sequence on post-first replays) but the per-step token never reaches the staged slot, so replay still recomputes the capture step's computation.

## Numbers vs the anchors

| arm | runs (t/s) | median | vs 74.4 | vs 86.9 (mission) | vs 84.8 (AR best) |
|-----|-----------|--------|---------|-------------------|-------------------|
| A (eager baseline) | 74.44 / 74.48 / 74.41 | **74.44** | +0.05 | −14.2% | −12.6% |
| G (graph decode) | 67.71 / 67.69 / 67.69 | **67.69** | −8.75% | −22.1% | −19.9% |
| MTP4 (guard, R2 protocol) | 43.69 / 43.60 / 43.61 | **43.61** | — | — | (guard: −0.07% vs R2 43.64) |

A arm = 74.44 reproduces the R2 baseline (74.42) and the canonical anchor regime on this box (the 11-01 84.8 band does not reproduce here; box regime unchanged vs R2).

## Losslessness evidence (bench protocol: canonical mtp-bench6.py md5 `452d363de7c6754de8333ae09999327f`, MTP=0, 400-tok window, 3-run median, standing config QMV_ROWS=1 REFWD_M1=1 GDN_FUSE=1)

- A.1/A.2/A.3 ALLTOKS md5 `de1603551b527577a9f3426bda6e4dd3` (n=400, identical across runs); FIRST20 `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` — matches the canonical anchor.
- G.1/G.2/G.3 ALLTOKS md5 `d14f3eab5ff57c376db3f5a81a32b8f7` (n=400, identical across runs); FIRST20 `271 1919 369 264 264 264 264 …` — 264×396 collapse. **NOT bit-identical to A. Gate 1 FAIL.**
- MTP4 first400 md5 `fc6aa5db…` (3/3 identical across runs; R2's MTP4 first400 also differed from A — S=4 batched verify is not bit-identical to AR by design; the guard compares FIRST20 + acceptance stats, both matching).

## What R3 did and what the evidence shows

The 5 R3 commits (c11d99e…e57d971) stage the per-step token idx at a persistent fixed-VA slot (own hipMalloc + pinned 16 B host), route ≤16 B per-step host Loads through the slot, pre-activate the staging mode at arm (no hip* inside the capture window), remove the K3 zero-slot idx redirect, and — the R3 delta — record the fresh-capture idx staging as a **capturable kernel node** (`stage_slot16_kernel` via `add_kernel_node`) instead of a D2D memcpy (not capturable on ROCm 10.1). The e57d971 commit fixed the template-mixup build break (the insertion had landed between the `template<…>` line and the kernel body); the reship + rebuild this round completed the build.

Mechanically, the fix works:
- 0 HSA errors across all G runs (the 2nd-capture fault stays fixed); the graph with the recorded `stage_slot16` kernel node captures and replays cleanly.
- The slot materializes pre-window (`[gdidx] slot mat OK`), the mode pre-activates at arm (2 arms = 2 captures, logged).

But the token delivery leg is dead:
- **`[gdidx] load:` fires 0 times** across the whole G run (1194 `[gdidx]` log lines; 2 arms, 0 slot arms, `take()=0` on every gather). The per-step `mx.array([[t]])` Load is evaluated in the previous step's tail — after the argmax, before the pilot arms — so the pre-activated staging mode never sees it, and the per-step Load writes its per-step buffer as before. The replayed `stage_slot16` node re-copies the capture-time idx buffer into the slot on every launch, and the embedding gather reads the slot → the capture step's token → the 264 attractor. The R2 value-staging gap persists: replay still does not consume the new step's per-step data.
- Secondary: the first replay after capture #1 diverges at op 84 (replay=QuantizedMatmul vs captured=Reshape — a first-launch-only op), is discarded by the pilot's divergence failsafe ("token lost, re-capturing"), and re-captures; post-re-capture replays are op-stable with zero further divergence.

Root-cause summary (confidence: high): the staging slot and its capturable staging node are correct, but the slot's write path (the Load small-path) is unreachable at the actual python eval order — the token array's Load lands one step earlier than the mode pre-activation. Next round: deliver the token at Load time for every ≤16 B Load occurring after an arm (independent of the pre-window mode), or pre-activate before the argmax that produces the next token.

## Build (step a: green wheel)

- Reship: the 6 changed vendor files (allocator.{h,cpp}, device.cpp, eval.cpp, indexing.hip, load.cpp) were already at their fixed md5s on the guest's `vendor-gd2` tree (t1509's reship had landed; only the rebuild was missing) — verified file-by-file md5 match against this worktree before building.
- Wheel: `mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl` (22,487,287 B, sha256 `39a072e1f6c3db27cd9ecd4ad0d7e57f1636367644123d33209a356b84c0b545`), clean build log (0 error/FAILED), built with `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1;/opt/rocm-7.2.0"`.
- Build log tail: `Created wheel for mlx: filename=mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl size=22487287 sha256=39a072e1…` / `Successfully built mlx`.
- Graft evidence in `libmlx.so`: `stage_slot16_kernel` device stub + host symbol present; `[gdidx]` staging strings present; `gdn_fused_step` graft live at import.
- venv-ar-r3: NEW venv (clone of the R2 venv's environment + pip bootstrap), old wheel artifacts removed before install; import OK, `mx.fast.gdn_fused_step = True`, model patch (FUSE_BRANCH_PATCH) live; venv-canon untouched.

## Regime / hygiene

- GPU locked (`flock -x` on the box's gpu-bench.lock) for the whole matrix; 25 s idle before the first run and between every run.
- Pre-run: the lock was held by a stale `sleep 7200` park + a queued probe — orphans of the dead t1509 run (its wrapper was SIGTERM'd and salvaged; the nohup children survived). The orphans were killed (verified ancestry first) rather than waiting out the park.
- SBCAR `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` verified pre and post the matrix (identical). Lock released cleanly at the end (fuser: no holders).
- Canonical bench md5 `452d363de7c6754de8333ae09999327f` recorded in the battery identity file.
- MTP4: the first (uncapped) MTP4 arm was an off-protocol run (default 1024 window → 1623 tokens, 37.62/32.74/32.04 t/s — longer runs average over KV growth; not comparable to R2's guard); the protocol-correct capped re-run (MTP_MAX_TOKENS=400, exactly R2's guard) is the one reported above.

## Reproduction

- Worktree: branch `ar-r3-value-staging` at `e57d971` (parent chain: 1f04a0d R2 → c11d99e → abc2fae → 095ea79 → dddd0bc → e57d971).
- Guest artifacts: `wheelout-gd3/` (wheel), `venv-ar-r3` (venv), `gd3-r3/` (battery logs A.1-3, G.1-3, MTP4.1-3, MTP4c.1-3, S1/S2 smoke, identity.txt), `build-gd3.log`.
- Battery: 3× A (`MLX_MTP=0 MTP_MAX_TOKENS=400`), 3× G (A + `MLX_ROCM_GRAPH_DECODE=1`), 3× MTP4c (`MLX_MTP=4 MTP_MAX_TOKENS=400`), standing config `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1`, `LD_PRELOAD` libhsa-runtime, `HCC_AMDGPU_TARGET=gfx1201`, 25 s idle gaps, under `flock -x`.
- Losslessness check: `grep ALLTOKS <arm>.i.log` → md5 of the token list; A arm must equal `de1603551b527577a9f3426bda6e4dd3` and G must equal it to pass (it is `d14f3eab…`).
