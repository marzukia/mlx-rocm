# REPORT-ar-r4: graph-decode R4 — instrument gate stop condition; G first-capture crash;
# verdict

## TL;DR verdict

- **WIP judgment (step 1): the pushed WIP `ad07e6b` is INCONSISTENT with the R4 design doc and was not used.** It re-adds the `set_seed`/`consume_seed` API and the end-of-record D2D/H2D delivery legs that the design (`.r4-design.md` @ `3f35385`) explicitly removes, plus a host-stage mechanism, capture-thread gating, and direct idx-buffer reads. The branch was reset to `3f35385` (the last coherent design-doc head); the WIP pivot is preserved on the mirror salvage branch (`fleetwork-salvage/1595-wip`) as unverified evidence for the next round.
- **Build (step 2): green.** Wheel (22,434,806 B, sha256 `4ff2369a1f64cc09bdf8d5d021d00134f0a50990b5be2b702bc7e01bb8ed0de0`), filename `mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl`; graft evidence in `libmlx.so` (`stage_slot16` x11, `gdidx` x10, `gdn_fused_step` x92, `persist=1` x1), import gate OK (`mx.fast.gdn_fused_step` + FUSE_BRANCH_PATCH live).
- **Instrument-first gate (step 3): STOP condition met.** The short 64-token G run showed **zero** `[gdidx] load:` lines (2 `[gdidx] gather:` lines only). Per protocol: stop and report. The G run then crashed on the first capture (see gate 1).
- **Gate 1 LOSSLESS: FAIL (hard crash, not a divergence).** G dies on the first capture: `hipLaunchKernel ... failed: operation failed due to a previous error during capture` at the bench argmax, ~25 s after arm. No token output, no md5 to compare.
- **Gate 2 speed: N/A** (G never completes; no t/s to report).
- **Gate 3 MTP4 guard: not run** (the same first-capture crash precedes it; the MTP4 arm shares the capture machinery).
- **A/B (5b1bb4b vs 3f35385, same guest, same ROCm 10.1 userland, same graft, same bench): 5b1bb4b (R3) green (rc=0, full run); 3f35385 (R4) crashes on the first capture.** The R4 delta is the cause. Root cause below (confidence: high).
- **Disposition: graph-decode default NOT flipped. Stays DEFAULT OFF** (env unset → eager path, zero behavior change).

## Numbers vs the anchors

| arm | result | notes |
|-----|--------|-------|
| A (eager baseline) | not re-run | R3 anchor 74.44 t/s stands (regime unchanged) |
| G (R4) | **crash, first capture** | `previous error during capture`; 0 tokens |
| G (R3 control) | **rc=0** | degenerate T=0 text; 0 HSA errors; 1 staged-node log |
| MTP4 (guard) | not run | same first-capture crash precedes it |

## Instrument-first gate evidence (short G run, 64 tokens, MLX_G1Y_DEBUG=1)

```
[g1] graph decode: armed (small steps 852/876 ops); capturing next
[gdidx] slot mat OK dev=0x7ca41aad1000 host=0x7ca8051bc000
[gdidx] arm: mode set; slot dev=0x7ca41aad1000 host=0x7ca8051bc000
[g1] prepare: slot 0..3 moved (capture=(nil))
[gdidx] gather: took=0 ixs.data=0x7ca41aad1000 src=0x7ca41aad2000
[gdidx] gather: staged node recorded, src=0x7ca41aad2000 seed=0x7cab288d0688 … mc=0
[gdidx] gather: took=0 ixs.data=0x7ca41aad1000 src=0x7ca41aad2000
RuntimeError: hipLaunchKernel( ... ) failed: operation failed due to a previous
error during capture
```

- `[gdidx] load:` count: **0** (the stop condition). The bench materializes the per-step idx array in the previous step's tail — the same reachability gap R3 documented — so the `load.cpp` persistent-flag slot path never fires on this bench. R4's core delivery mechanism is therefore unverified, and the staging path it replaced crashes.
- Both in-window seed gathers hit the **same** slot 0 / same global src buffer (`ixs.data=…1000`, `src=…2000`): the allocator holds a single global `decode_idx_src_va_`, and `fresh = (decode_idx_src_va_ == 0)`.

## Root cause (confidence: high)

In `indexing.hip` (`Gather::eval_gpu`, the in-window staging block), 3f35385 changed the replay branch from R3's `!fresh && !took && shadow != nullptr` (blocking H2D from the pinned shadow) to `!fresh && !took` (blocking D2D from the per-step idx buffer's device VA). Two compounding defects:

1. **The `fresh` flag is global.** The first in-window gather calls `decode_idx_set_src_va(src, …)`, so the *second* seed gather in the *same* capture sees `fresh == 0` and takes the "replay" branch — which the R4 comment assumes is "outside the capture window (replay steps are not captured)". Inside a capture window, that branch issues a **blocking `hipMemcpy` D2D**, which lands on the legacy stream and taints the capture (the 906 failure R4's own comments warn about: "a blocking hipMemcpy lands on the legacy stream and 906s inside a capture window"). The taint surfaces at the next launching op — the bench argmax — as `operation failed due to a previous error during capture`. The second gather's `staged node recorded` log is never reached: its execution takes the `!fresh` slot-read branch (no log) after the poisoning copy.
2. **The second gather reads the wrong slot even if it did not crash.** The `!fresh` branch unconditionally sets `h_indices[0] = ixs.data` (slot 0), but slot 0's staging node was seeded by the *first* seed's idx buffer — the second seed's token never reaches a slot (there is one global src/slot pair, and both gathers share it).

5b1bb4b (R3) is green on the identical bench because its in-window replay branch was guarded by `shadow != nullptr` (null for the bench's fresh per-step idx arrays, so the branch is skipped inside the window) and its second gather falls through to a direct read of the per-step idx buffer (stream-ordered after the recorded buffer-creation H2D in the same graph). The A/B control run (5b1bb4b + the same graft, wheel sha256 `ead7bebad1469e0064985833de0f93699eda3c0ea16e8e642e06a6b41348c872`) completed with rc=0, one staged-node log, and the `[gdidx] seed: D2D pre-launch … mc=0` end-record D2D — the R3 mechanism working as designed.

## Recommendation for the next round

- Make the staging state **per-gather or per-slot**, not a single global flag: the second in-window seed gather must take the fresh/staging path (or a direct buffer read), never the blocking-replay path inside the window.
- Or restore R3's `shadow != nullptr` guard on the in-window replay branch (skips the blocking copy inside the window; the second gather reads the per-step idx buffer directly — proven green by the A/B control) and keep the R4 blocking D2D only for genuinely outside-window replays.
- Fix slot assignment: one global src/slot pair cannot serve two in-window seeds.
- The `load.cpp` persistent-flag path remains unreachable on this bench (0 load lines, same gap R3 documented); it needs the Load-timing fix (deliver at Load time regardless of the pre-window mode) before it can be the delivery leg.
- The `.r4-design.md` "blocking D2D refresh outside the window" statement only holds for one seed gather per capture; the multi-seed in-window case should be stated explicitly.

## Build (step 2)

- Source: `3f35385` vendor snapshot over the guest's grafted tree (graft files `gdn_fused_step*.hip` + grafted `fast.*`/`python` untouched; the 5 R4-delta files allocator.{h,cpp}, device.{h,cpp}, indexing.hip verified md5-matched to `3f35385`). First build attempt failed twice for toolchain reasons (stale CMake cache in the copied build dir; grafted `fast.*` clobbered by the base snapshot and restored from the grafted clone), third attempt clean.
- Command: `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1;/opt/rocm-7.2.0" python3.14 -m pip wheel . --no-deps`.
- Graft evidence in `libmlx.so`: `stage_slot16` x11, `gdidx` x10, `gdn_fused_step` x92, `persist=1` x1. Import gate OK.
- venv: fresh `venv-r4v` (clone of the prior-round venv, old mlx artifacts removed, its pip shim's stale shebang fixed to the venv's own python before install). venv-canon untouched.
- A/B control wheel: 5b1bb4b + graft, sha256 `ead7beba…` (see root cause).

## Regime / hygiene

- GPU locked (`flock -x` on the box's gpu-bench.lock) for every run; 25 s idle before runs; `fuser` checked first; lock free at the end.
- Canonical bench md5 `452d363de7c6754de8333ae09999327f` recorded in the identity file.
- SBCAR `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` verified pre and post (identical).
- Standing config: `QMV_ROWS=1 REFWD_M1=1 GDN_FUSE=1`, `LD_PRELOAD` libhsa-runtime, `HCC_AMDGPU_TARGET=gfx1201`.
- Hygiene note: the prior-round venv's wheel was replaced while fixing the pip-shim misdirection (its original wheel artifact is not on disk and the build tree is gone; the prior-round numbers stand as recorded in REPORT-ar-r3.md). The R4 wheel was reinstalled in `venv-r4v` at the end.
- PR hygiene: persona scrub commit brought the PR diff to 0 hits (was 31, in prior-round scripts/reports: the guest home and the models mount were replaced with the `<guest-home>` and `<models-mount>` placeholders); authorship normalized (0 fallbacks — 7 lineage commits rewritten to the standing identity); `ruff check` + `ruff format --check` green; hermetic tests 25 passed / 1 skipped; size gate OK.

## Reproduction

- Worktree: branch `ar-r4-token-delivery` at this report's commit (parent chain ends at `3f35385` + the two hygiene commits).
- Guest artifacts: `wheelout-r4v/` (R4 wheel), `wheelout-r3chk/` (A/B control wheel), `venv-r4v`, `r4v-r4/` (R4 G run log), `r4v-r3chk/` (A/B control G run log), `build-r4v2/3.log`, `build-r3chk2.log`, `r4v-instrument.sh`.
- A/B protocol: same 64-token G run (canonical bench, `MLX_G1Y_DEBUG=1`), only the wheel differs (3f35385+graft vs 5b1bb4b+graft); R4 crashed on the first capture, the control completed (rc=0).
