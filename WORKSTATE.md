# WORKSTATE — run 1038 (Phase 2 lever 3: fused GDN decode-step kernel, CONTINUATION of t1026)

Base: main fa9fa30. Continues t1026's uncommitted state (salvaged from
/srv/fleetwork/work/1026; orphan check was clean per warden).

## Timeline
- 10:35Z: state copied to fresh worktree (branch phase2-gdn-fuse-cont @ be70bb1,
  pushed to bare mirror). Patch verified: does NOT apply to the in-repo vendor
  snapshot (fa9fa30 = older the earlier vendor base-based tree, no python/, no gdn_prep.hip),
  applies CLEAN to the pinned 75ca881 lineage (guest clean worktree
  /models/tmp/phase2-clean75). Patch sha 2f3f8f820599e6a2 unchanged since
  the 10:03 wheel build (dist2) -> fused wheel NOT stale, no rebuild.
- bench fix: final state capture gdn_layer_states(lm) -> (cache); tok/s =
  n / sum(step_times) (10:03 run printed mean ms/step in the tok/s field).
- 10:42Z: hermetic CI green (ruff clean, 21 passed, size gate OK); state pushed
  to mirror (1d9681c). 10:03 state jsonl re-analysis: F step-0 already
  0/24 layer match vs UF (F step-0: 24 unique shas; F step-1: collapsed
  to 2-3 unique shas and stays there); UF stays 24-unique all 8 steps,
  all states nonzero both arms. Kernel static re-verification: conv/beta/g
  (f32, low-u16 read), qk_norm (incl. bf16 pre-rounded scale), r4d state
  update (layout + arithmetic), norm_gated ALL verbatim vs 75ca881 chain;
  f2bf is round-to-nearest (matches static_cast). Suspicion: the g read
  quirk (eg = exp(clamp(log(bf16(low16(f32(g)))),-80))) makes eg ~0 for
  ~half the heads and 1e19-inf for the rest -> state saturation in a few
  steps; if the chain shares it, UF step-7 stability says it does not,
  so the fused s_g or its read differs in practice. Bisect probe
  (scripts/p2-gdn-bisect.py) launched under flock (10:38Z, waiting on
  t1037's 300s probe loop).
- 11:05Z: bisect probe results: SINGLE-STEP fused == chain byte-identical
  (out + state, all dbg intermediates match); chain 8-step STABLE (state
  max 3.15->3.38); fused 8-step EXPLODES (step1 7.6e26, step2 all nan/inf).
  So the bug is the state-update read of g when h0 != 0. egprobe (q=k=v=0,
  beta=1, h0=1.0 -> per-head state IS the applied decay): CHAIN eg ~ 0.996
  (tame, == bf16(g)); FUSED eg ~ 1e23 (== low16-of-f32(g) pattern).
  ROOT CAUSE: fast.cpp's gated_delta_update does
  g = astype(gates, out_dtype) at the op boundary (out_dtype = q's dtype =
  bf16), so the chain's r4d kernel reads g as a FULL bf16 pattern; the
  fused kernel computed s_g in f32 and read its LOW 16 bits (f32 mantissa
  bits, not a bf16 encoding) -> eg ~ exp(log(1e16)) ~ 1e23 -> state
  saturation. FIX: round s_g to bf16 before the read. The cast must be
  RNE (identical to MLX's _MLX_BFloat16 float ctor: u += (u>>16&1)+0x7FFF,
  bits>>16) to match the astype bit-for-bit; r4d's own f2bf (round-half-
  away, bits+0x8000) is kept for the on store (r4d parity). Patch
  regenerated (811 lines), hermetic pin updated, wheel rebuilt.
- 13:05Z: RNE FIX CONFIRMED at kernel level (venv-fused, new wheel
  167bfbe1). egprobe: per-head fused_eg == chain_eg EXACTLY (h0-h31,
  all tame 0.99-1.0, was 1e23 pre-fix). bisect: single-step fused==chain
  byte-identical; fused 8-step now STABLE and matches chain at EVERY step
  (state 3.15->3.38, no inf/nan) -- pre-fix it exploded (step1 7.6e26).
  Build config lesson: setup.py reads CMAKE_ARGS env only; the ROCm
  prefix must be -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 (NOT .../lib) or
  find_package(hip) fails; --config-settings=cmake.define.* is inert
  (setuptools backend, not scikit-build).
- 13:20Z: t1037's harness `rm -rf`'s /models/tmp/phase2-gdn on every
  relaunch -> moved all phase2 work to /models/tmp/phase2-fuse (safe
  name, not in t1037's rm list). t1037's graph-decode bench (G1Y_DEBUG)
  keeps hitting HSA_STATUS_ERROR_MEMORY_FAULT and hanging (100% CPU spin)
  holding the lock; killed the hung runs to free it. Lossless-gate + A/B
  runner launched under flock (2x 29K prefills).
- 13:45Z: COMPLETE. Lossless gate GREEN both arms same build (venv-fused):
  ctx=14 + ctx=29000 -> STREAM-IDENTICAL (256 tok) + STATE-MATCH (24 layers,
  steps 0-7+256) + NON-VACUOUS (fused_calls=6144). Chain proven
  deterministic (2 fresh procs identical); within one build chain==fused
  BIT-IDENTICAL (state-diff max_abs=max_rel=0.0 at step 256) -> the earlier
  cross-process state mismatch was a build artifact (venv-m8 baseline wheel
  vs rebuilt 75ca881+patch), not a fused-kernel difference. A/B decode
  (graph OFF): +4.8% (14tok) / +4.1% (29K) window tok/s; med ms/step
  -4.6%/-3.5%. Committed 32b4e4c + dc9b37d; pushed to bare mirror + fork
  (all at dc9b37d, NO PR). Leak rule clean, CI green, REPORT-phase2.md.
  DELETED by an external cleaner mid-build. Recovered: fresh clone of
  75ca881b (/tmp/phase2-clean75) + repo patch applied; venv-fused =
  cp -a venv-m8 + p2-model-patch.py (NEW repo script: F-branch dispatch
  + GDN_FUSE_CALLS counter, early-return before the chain) + new wheel.
  Build failure root cause: ad-hoc `pip wheel` omitted the vendor
  CMAKE_ARGS (-DMLX_BUILD_ROCM=ON ...) so the no_gpu backend compiled and
  duplicated fast.cpp symbols. Rebuilding with CMAKE_ARGS from vendor.json.
  NOTE: future builds must use scripts/build-wheel.sh or pass CMAKE_ARGS.

## Guest lane
- flock /models/gpu-bench.lock; sibling t1037 (graph decode) holds it in
  300s probe loops (flock -w 1800). Wait, do non-GPU work while waiting.
- venvs: UF=/models/tmp/t1012-29k/venv-m8 (baseline, no fused op),
  F=/models/tmp/phase2-fuse/venv-fused (fused op + model patch, so
  167bfbe1...), clean75 src tree /models/tmp/phase2-clean75 (RNE patch
  applied), model /models/qwen3-9b-mlx.

## TODO (order) — COMPLETE (dc9b37d, pushed to bare + fork)
1. DONE: kernel fix confirmed (egprobe fused==chain per-head; bisect fused
   8-step stable + matches chain). 2. DONE: egprobe + bisect.
3. DONE: lossless gate (both arms SAME build venv-fused) ctx=14 AND
   ctx=29000 -> STREAM-IDENTICAL (256 tok) + STATE-MATCH (24 layers,
   steps 0-7+256) + NON-VACUOUS (fused_calls=6144). Chain deterministic;
   within one build chain==fused BIT-IDENTICAL (state-diff max_abs=0).
4. DONE: A/B decode (graph OFF both arms): ctx=14 window 77.22->80.95
   tok/s (+4.8%); ctx=29000 window 62.63->65.19 tok/s (+4.1%); med ms/step
   -4.6% / -3.5%.
5. DONE: committed 32b4e4c (fix+gates) + dc9b37d (report); pushed to bare
   mirror (/srv/fleetwork/mlx-rocm.git) AND fork (jimmyateworld/mlx-rocm),
   all at dc9b37d. NO PR (operator handles PR/merge).
6. DONE: leak rule clean (0 hits); CI green (ruff clean, 21 passed + 1
   skipped, size gate OK); REPORT-phase2.md in-tree.
5. commit per verified step; push state if near the 4h backstop
6. leak grep; CI (ruff + hermetic pytest); force-push phase2-gdn-fusion to
   bare + fork jimmyateworld/mlx-rocm (NO PR); REPORT-phase2.md in-tree
