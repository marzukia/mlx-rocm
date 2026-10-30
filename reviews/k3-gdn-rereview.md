# Adversarial review: k3-gdn-port (head 1155097, diff main..1155097)

Reviewer: fresh K3-lane session (first review of this branch; prior an earlier run attempt
died empty-final, no prior verdict carried). Read-only except this doc.
Scope: R4D GDN linear-attention kernel port (chunk-scan + kkt_solve +
recurrent_update + wmma helper, gated_delta_update.cpp wiring) + crash fix
(allocator.cpp `shadow_chunked_copy` re-add of an earlier run/an earlier run chunked host-shadow
staging). Report under review: reports/k3-gdn-crash-fix.md.

Branch: 3 commits (2b28f2e R4D port, d9a5de7 WIP salvage, 1155097 fix+verify).
WIP commit is fine — squash-merge on land.

## Files

| file | lines | role |
|---|---|---|
| reports/k3-gdn-crash-fix.md | 145 | root cause + crash-mode + numerics |
| rocm/CMakeLists.txt | +4 | 3 .hip in HIP_SOURCES, gated_delta_update.cpp in target_sources |
| rocm/allocator.cpp | 76± | shadow_chunked_copy re-add |
| rocm/gated_delta_update.cpp | 303+ | GDN eval wiring + gates |
| rocm/primitives.cpp | 20± | stub removed (use_fallback/eval_gpu now in gated_delta_update.cpp) |
| r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip | 1019 (998 nonblank) | chunk-scan + gate_prep + extreme-decay repair |
| r4d_gdn_kkt_solve_k128_c64_bf16.hip | 358 | kkt solve (chunk preamble) |
| r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip | 140 | recurrent (T=1) kernel |
| r4d_gdn_wmma.h | 154 | gfx1201 wave32 bf16 WMMA fragment helpers |

Size gate: `scripts/size_ratchet.py` scans `.py` only; C++/HIP are exempt,
`size-ratchet.json` is `{}` — no ratchet issue from the 998-line .hip.

## 1. allocator.cpp — no-destroy design — VERDICT: PASS

- **Leak bound.** One stream per chunk per shadow copy, never destroyed.
  Bounded per copy (1.2 GB copy = 12 streams at the 100 MB default) but
  unbounded across process life (each flush of a dirty shadow re-leaks
  ceil(size/100 MB) streams). Streams + KFD queues are small (KB-scale);
  documented in the report with a phase-2 follow-up (shared persistent
  stream). Acceptable for a crash fix. **MINOR (M5).**
- **Thread-safety.** `shadow_chunked_copy` has no shared mutable state
  (chunk size is a function-local `static const`); `hipStreamCreate` is a
  thread-safe runtime API; each chunk uses its own stream.
  `flush_all_dirty_shadows` holds `mutex_` for the whole flush (as before);
  the other two sites have the same locking posture as the legacy
  `hipMemcpy` they replace. No new races. PASS.
- **Sync fallback on `hipStreamCreate` failure.** The failed chunk falls back
  to synchronous `hipMemcpy` for that chunk only and continues; a
  `hipMemcpyAsync` failure skips the sync and returns the error. Correct.
- **Chunk edge cases.** size=0 → loop skipped → `hipSuccess`. size<100 MB →
  single chunk, one stream. `n = min(chunk, size-off)` so `off+n ≤ size`, no
  overflow. knob=0 → `SIZE_MAX` → single chunk (one copy). Non-numeric env
  → strtol 0 → single copy. All fine.
- **knob=0 is "safely legacy" with one nuance.** It is not literally the
  legacy path: it is one async copy on a fresh stream + sync (blocking,
  single copy — same hang mode as pre-chunking, which is the point of the
  knob), and it still leaks one stream per call. The report's "pre-chunking
  behavior" wording is about copy count, not mechanism. **MINOR (M4).**
- **All three shadow sites covered** (ensure_host_shadow D2H,
  flush_host_shadow H2D, flush_all_dirty_shadows H2D), verified in diff.
- **Crash fix itself.** The an earlier run/an earlier run root cause (per-chunk
  `hipStreamDestroy` racing a never-promptly-retired KFD completion event)
  is removed by not destroying; the hang-avoidance property (fresh
  per-chunk stream + sync = per-chunk queue reset) is preserved. The
  load-gated evidence (16/16 concurrent crashes on the 04:11 wheel → 0/16 on
  the fix, plus 20× 300 MB D2H and 10× H2D hammer) is the right hammer for a
  load-gated bug. PASS.

## 2. GDN kernels — shape/dtype gates, bf16/f32, T=1 fast path — VERDICT: PASS w/ 1 MAJOR

**MAJOR (M1): the `use_fallback` gate and the kkt_solve launcher disagree on
GQA ratios 5 and 7.** `use_fallback` accepts any `Hv/Hk ≤ 8` (plus
`Hv % Hk == 0`), but `r4d_gdn_kkt_solve_k128_c64_bf16` switches on
`H/Hg ∈ {1,2,3,4,6,8}` and returns -3 for anything else. Ratio 5 or 7
therefore passes the gate, reaches `eval_gpu`, and throws
`runtime_error("kkt solve launch failed (-3)")` instead of taking the
op-loop fallback. The chunk_scan kernel is ratio-agnostic (plain integer
division), so the gap is purely the kkt_solve templated launch. The target
9B-class Qwen3 geometries (ratio 2/4) are unaffected, and the failure is
loud (exception, not silent miscompute), but the gate's contract is broken
for 5/7. Fix is one of: tighten the gate to the supported set, or add the
5/7 templates.

Verified-good:

- **Dtype gating is sound because of the framework, not the gate.**
  `fast.cpp::gated_delta_update` casts q/k/v/g/beta all to one promoted
  `out_dtype` and h0 to float32 (or f32 zeros); the state output is
  float32 by the `make_arrays` spec. So `eval_gpu`'s single
  `q.dtype() != bfloat16` check covers k/v/g/beta, and the out buffer is
  bf16 exactly when the kernel writes bf16. No partial-dtype-check bug.
- **Scalar-gate check matches the framework.** Fallback supports
  `g.ndim() ∈ {2,3}`; fused requires `g.ndim()==3 && g.shape(2)==Hv` —
  ndim-2 (global scalar) and [B,T,Dk] vector gates fall back at runtime via
  `fallback_(inputs)`. `Custom::fallback_` exists (fast_primitives.h:19).
- **Geometry gates.** Dk/Dv==128, Hv%Hk==0, ratio≤8 (see M1), has_mask,
  CPU device, `MLX_ROCM_GDN_FALLBACK=1` A/B knob. Kernel-side checks:
  recurrent `K!=128||V!=128 → -1, H%Hg → -2`; kkt_solve same plus the
  ratio switch; chunk_scan checks K/V/bt (no H%Hg check — caller-guaranteed,
  **MINOR M3**).
- **T=1 fast path (recurrent kernel).** One thread per (v row, k half);
  in-lane dot products closed by `__shfl_xor(dot, 1, 32)` — paired tids are
  consecutive, so the pair is always inside one 32-lane wave. g is the
  per-token multiplicative decay; `log` clamped at -80 so g=0 → e^-80, no
  NaN. State load/store are float4 at 16 B-aligned offsets (256 B-aligned
  base + 64-float strides). VS split (1/2/4) is safe: no cross-v-row
  traffic in the update. Numerics: T=1 fused vs fallback is exact (0.0),
  vs f32 ref 3.7e-3 — consistent.
- **kkt_solve.** Gram computed once per k head (not per v head), upper
  triangle oriented for 16-byte accumulator stores, blocked 16→32→64 inverse
  (fp32 forward substitution for the diagonal blocks, WMMA merges via the
  [[P,0],[R,Q]]^-1 identity). The padding-row NaN fix is correct: dead
  elements get exp(-INF)=0 (live i>j exponents are ≤ 0 since the cumsum is
  decreasing), and the comment documents a real measured bug (16 non-finite
  values on Qwen3.8-27B TP2). In-place inverse staging is barrier-ordered.
- **chunk_scan.** State held in WMMA accumulators across the chunk loop
  (no HBM round-trip for h/w/u/v_new); midpoint decay factorization with
  clamps at 80 (measured ceiling, PPL-verified 70/80/83 sweep); padding rows
  clamped to the last valid row (in-bounds; `lim = nval-1`), padding
  beta/gv/rs zeroed, O store guarded by `tok < nval`. The
  extreme-decay repair (span > 128) recomputes the whole sequence/head from
  h0 in f32, overwrites both o and ht, runs on the same stream after the
  scan (ordering guaranteed), with correct chunk-local cumsum bookkeeping
  (`previous = 0` at chunk starts).
- **wmma.h.** Hardware-probed fragment layouts, software bf16 conversion
  (round-half-away, documented 2^-16 divergence from RTNE), 8-byte-aligned
  paired loads. The repair kernel's RTNE bf16 packer differs from the main
  path's round-half-away only at exact 0x8000 mantissae — negligible
  (**MINOR M8**).
- **CMake wiring** correct (3 .hip in HIP_SOURCES, gated_delta_update.cpp in
  target_sources; wmma.h is an include-only header).
- **Scale contract.** `scale=1.0f` is passed because q/k arrive
  l2-normalised upstream and the MLX primitive applies no output scale —
  consistent with the fallback in fast.cpp (which also applies no scale)
  and with the exact T=1 numerics.

## 3. Re-add vs an earlier run/an earlier run/an earlier run large-copy hang; knob=0 — VERDICT: PASS

- The hang lineage was single large (≥300 MB) shadow copies stalling in
  `kfd_wait_on_events` with the GPU idle. The re-add keeps every staging
  copy ≤100 MB by default — the an earlier run per-chunk queue reset is the
  hang-avoidance property and it is preserved. The only behavior change vs
  an earlier run is the removed `hipStreamDestroy`, which fixes the segfault without
  touching the chunking. No new hang mode is introduced: each chunk is
  synced before the next, so no chunk inherits a queue with a pending
  un-retired event.
- knob=0 restores the single-copy (pre-chunking) behavior, i.e. the hang
  mode can return — that is the knob's documented purpose, and it is the
  safe default-off position. Default (100 MB) is the hang-avoiding
  position. PASS.
- Caveat (inherent, not a regression): the hang was *intermittent* in the
  an earlier run lineage, so 20 clean 300 MB copies is evidence, not proof, that the
  default chunk size is hang-free at, say, 2 GB. The 16/16→0/16 concurrent
  crash result is the load-bearing verification and it is the right one for
  the segfault.

## 4. Gaps the numerics don't cover — VERDICT: noted, non-blocking

Numerics ran B=1, Hk=4/Hv=8 (ratio 2), Dk=Dv=128, T∈{1,3} only:

- **No multi-chunk sequence (T>64)** — the chunk loop, cross-chunk state
  carry, and the prefetch path are untested at scale; the T=3 run exercises
  one partial chunk only.
- **No B>1** — batch offsets (`bos`, grid.z) untested.
- **No ratio 1/3/4/6/8** — only ratio 2's templated kkt_solve variant is
  exercised (M1's ratio 5/7 are both untested *and* broken).
- **No extreme-decay repair exercise** — needs a gate span >128; test data
  was ordinary decay. The repair is the one code path with no numeric
  coverage at all.
- **h0≠0 not stated** — if the runs used h0=0, the state-load path is only
  partially exercised (state writeback is covered).
- **No g=0 (log clamp) edge, no knob=0 / MLX_ROCM_D2H_CHUNK_MB env-path
  verification, no B=0/T=0 (throws, see M2).**
- The `MLX_GDN_DEBUG` instrumentation ships in the code (env-gated fprintf;
  no behavior change when unset — acceptable, noted).

## Adversarial pass

- Re-verified the three shadow call sites in the diff (all converted to
  `shadow_chunked_copy`). Re-verified the framework dtype casting in
  fast.cpp (q/k/v/g/beta → out_dtype, h0 → f32, state out → f32) — the
  single q-dtype check in eval_gpu is therefore complete, not partial.
- Checked `Custom::fallback_` (fast_primitives.h:19) — the runtime
  `fallback_(inputs)` call in eval_gpu is valid.
- Checked size gate scope (.py only) — no ratchet violation.
- Checked padding/clamp bounds in all three kernels (`lim`/`rows-1` = last
  valid row; in-bounds). Checked the kkt_solve NaN-fix predicate
  (`live && i > j && j < rows`). Checked the repair kernel's chunk-local
  cumsum reset and its same-stream ordering after the scan.
- Checked `__shfl_xor(…, 32)` wave pairing in recurrent and repair kernels
  (consecutive tids, width 32 — always intra-wave).
- No BLOCKER found. No silent-miscompute path found outside M1 (which
  throws loudly).
- Git-history corroboration of the report's root cause: bd061d6 (an earlier run)
  has `hipStreamDestroy(stream)` at line 920 inside
  `chunked_copy_stream_separated` (the segfaulting design); 85b521e
  (the re-land) has zero chunked-copy references (the masking re-land);
  the fix commit re-adds chunking without the destroy. The crash-mode
  table (load-gated 16/16 concurrent → 0/16 on the fix) is consistent
  with a stream-lifecycle race, not a deterministic code bug.
- M1 re-verified by grep: kkt_solve switch = {1,2,3,4,6,8} (lines
  348–353), gate = `Hv / Hk > 8` (gated_delta_update.cpp:120).

## Findings

| # | sev | finding |
|---|---|---|
| M1 | MAJOR | `use_fallback` accepts GQA ratios 5/7 (≤8) but kkt_solve's launcher switch only has {1,2,3,4,6,8} → ratio 5/7 throws `runtime_error` instead of falling back. Target 9B geometries (ratio 2/4) unaffected. Tighten the gate or add the templates. |
| M2 | MINOR | B=0 or T=0 (T≠1): grid dim 0 → launch error → `runtime_error` from eval_gpu, while the op-loop fallback handles both fine. Gate `B==0 || T==0` in use_fallback. |
| M3 | MINOR | chunk_scan entry lacks the `H % Hg` check kkt_solve has (returns -2). Caller-guaranteed via use_fallback; defensive gap only. |
| M4 | MINOR | knob=0 is "single copy" but not literally the legacy path: one async copy on a fresh stream + sync, still leaking one stream per call. Hang mode (single large copy) returns by design; wording in the report is about copy count. |
| M5 | MINOR | Stream leak is bounded per copy, unbounded across process life (e.g. 12 streams per 1.2 GB flush; 1000 flushes → 12k streams). KB-scale objects; documented + phase-2 follow-up. Acceptable for a crash fix. |
| M6 | MINOR | `mlx_gdn_gate_prep_c64` launches 32-thread blocks with no `threadIdx.x==0` guard — all 32 threads run the same 64-step loop and write the same values (32× redundant work). Cosmetic. |
| M7 | MINOR | Dangling doc reference: primitives.cpp, gated_delta_update.cpp, wmma.h and both kernel headers cite `reports/k3-gdn-port.md`; the committed report is `reports/k3-gdn-crash-fix.md`. |
| M8 | MINOR | Repair kernel packs bf16 with RTNE; main path uses round-half-away. Diverge only at exact 0x8000 mantissa (~2^-16 of values, ≤1 ulp). Negligible. |

Counts: 0 BLOCKER, 1 MAJOR, 7 MINOR.

## Verdict: **PASS**

The crash fix is correct and verified with the right hammer (load-gated
16/16 → 0/16), the hang-avoidance property of the an earlier run chunking is
preserved, and the GDN port is numerically verified (T=1 exact vs fallback,
T=3 within 1 bf16 ulp, f32-exact reference cross-check). The one MAJOR
(M1, ratio 5/7 gate/launcher mismatch) is a real gate-contract bug but
fails loudly and does not affect the target 9B-class Qwen3 geometries;
land and fix in a follow-up (tighten the gate to the supported set is the
one-line fix). The MINORs are hygiene.
