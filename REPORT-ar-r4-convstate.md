# AR R4 follow-up (t1788): F-branch GDN conv-state in-place fix + gate re-run

TL;DR:
- Fix implemented (one, minimal, mirrors the 9c82e4c-pattern ssm in-place fix from t1779): in `gdn_fused_step.hip`, a new `gdn_fused_conv_state_kernel` emits the conv state as the op's third output, and in `graph_decode_mode()` `conv_state_new.copy_shared_buffer(cs)` (same row_contiguous + nbytes guards as the ssm alias) aliases it onto the stable pre-captured conv-state buffer. In-place safety: the kernel runs after the main fused kernel (same-stream ordering) and stages the (potentially aliased) source rows into shared memory with a `__syncthreads()` barrier before any write. The conv update is a pure bf16 copy of the last 3 rows of [cs (3 rows), qkv (T rows)] — bit-identical to the python path.
- Model F-branch unpacks the third output and sets `cache[0] = conv_state_new` (the python-side `mx.contiguous` update runs unconditionally before the F-branch check and is shadowed by it — zero new captured ops vs. status quo).
- Gate re-run (new wheel 22,449,702 bytes on the t1779 base, 400-tok protocol, same session): Gate 1 LOSSLESS FAIL (G1 first token fork at idx 7, 10597 vs 29006; A1 bit-identical to the canonical/t1769/t1779 FIRST20), Gate 2 SPEED: G 65.55 vs A 71.96 = -8.9% (t1779: -8.7%), Gate 3 MTP4 guard: 3 runs 82/13/5 @ 31.37/27.59/30.18 t/s (guard 137/87/49 @ 43.64; measured on the divergent G trajectory, not directly comparable to the clean baseline). Stopped per brief.

## What changed (single minimal change, mirroring the ssm pattern)

- `gdn_fused_step.hip` (762 lines):
- `gdn_fused_conv_state_kernel`: copies the last 3 rows of the [cs; qkv] row stack into `new_cs[r] = row (TMAX+r)`. T=1: [cs1, cs2, v0]; T=2: [cs2, v0, v1]; T=4: [v0, v1, v2]. No arithmetic — pure bf16 copy.
- Shared-memory staging (768 elements/block, 3 per thread; Qwen3.5-9B: 3×8192/768 = 32 blocks) + `__syncthreads()` before the write: all reads from the (aliased) cs buffer complete before any write, so the in-place case (destination rows overlap source rows, e.g. T=1 writes row 1 while row 1 was read for new row 0) is safe.
- `eval_gpu`: third output `conv_state_new` (shape `cs.shape()`, bf16) + the in-place alias in `rocm::graph_decode_mode()`: `cs.flags().row_contiguous && cs.nbytes() == conv_state_new.nbytes()` → `conv_state_new.copy_shared_buffer(cs)` (exact mirror of the ssm alias).
- `fast.cpp` (box 75ca881 API): op outputs 2 → 3 (make_arrays shapes/types; fallback computes `new_cs` with the free `slice(a, start, stop, s)` — this mlx version has no `array::slice` member).
- `fast_primitives.h`: `output_shapes` → 3 (`inputs[7].shape()`).
- Model F-branch: `out, state, conv_state_new = mx.fast.gdn_fused_step(...)`, `cache[0] = conv_state_new.reshape(B, conv_kernel_size - 1, conv_dim)`.
- The graft API files (fast.cpp, fast.h, fast_primitives.h, python/src/fast.cpp) were already 3-output by design on the box from t1779 — this round's delta is the .hip change plus the 2→3 API wiring.
- Graft patch regenerated against a pristine 75ca881 worktree (the previous record had a stale-base artifact: a spurious `gdn_prep` deletion hunk and a drifted python context — the box base is 75ca881, not the local b0ee42a tree). Verified `git apply --check` clean on pristine 75ca881.

## Gate evidence (t1788 wheel, 2026-11-03 23:30–2026-11-04 00:30 UTC)

### Gate 1 — lossless: FAIL

A1 (eager, MTP=0, 400 tok): 71.96 t/s, FIRST20 = 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 — bit-identical to the canonical (t1769 == t1779).

G1 (graph, MTP=0, 400 tok, 25 s idle between arms, same session): 65.55 t/s, FIRST20 = 271 1919 369 264 15352 11 70042 10597 29006 314 1204 264 6278 20853 5568 13 357 6278 20853 369 — first token fork at idx 7 (10597 vs 29006). The t1779 hard period-4 attractor (264 15352 8214 5529 from step 5) is gone; G1 now drifts and loops loosely (PG probe: ...6278 20853 369 524 15756 264 328).

G1 debug run (MLX_GDN_DEBUG=1, 400 tok): the ssm alias FIRES (9528 lines = 24 layers × 397 steps) and the conv-state alias FIRES (9528 lines, cs==new VA on every step).

### Per-step state probe (PA3 vs PG3, 24 steps, md5 of ssm AND conv per layer)

- s0–s3: all 24 GDN layers bit-exact (ssm and conv).
- s4, s5: 21/24 layers' ssm dirty (g4+); g0, g1, g2 clean; ALL conv clean (h:0).
- s6: 21/24 ssm dirty; 21/24 conv dirty — conv turns dirty only at s6, exactly when the first dirty qkv row (s4) enters the 3-row window.
- s7: first token fork (29006 vs 10597); PG lags PA by one token through the rest of the trace.

Read: the conv state is EXONERATED (bit-exact at s4/s5 despite replay; its s6 dirt is purely downstream of the ssm divergence). GDN blocks 0–2 — which feed on pre-attention inputs — stay clean while blocks 3+ go dirty at s4, so the residual suspect is the ATTENTION/KV path (the probe skips KVCache entries): whatever the graph replays stale for the full-attention layer between GDN blocks 2 and 3 feeds the GDN ssm updates from block 3 on.

### Gate 2 — speed (record)

G 65.55 vs A 71.96 = -8.9% (t1779: -8.7%; t1769: -9.8%). The second (tiny) conv-state launch per GDN step costs ≈ nothing relative — both arms pay it.

### Gate 3 — MTP4 guard (3 runs, graph decode, MTP=4, 400 max)

82/13/5, 82/13/5, 82/13/5 @ 31.37 / 27.59 / 30.18 t/s (guard 137/87/49 @ 43.64; t1769: 40.56 = -7.1%). Measured on the divergent G trajectory (Gate 1 FAIL), so the acceptance/t-s numbers are not directly comparable to the clean baseline; recorded as evidence of the MTP path running stably (200 verifies, deterministic across runs).

## Stop condition

Met per brief: Gate 1 still fails after the conv-state fix; per-step state md5s captured; remaining suspect named (attention/KV path, first full-attention layer between GDN blocks 2 and 3). Stopping here.

## Artifacts

- `data/t1788/A1.log`, `G1.log`, `G1dbg.log` (400-tok gate arms + debug), `PA3.log`/`PG3.log` (24-step state probes), `MTP4.1-3.log` (guard runs). (G1dbg.log is a trimmed excerpt — head 3000 + tail 3000 of the 25027-line full log, 322387 B; the full log exceeded the 1 MB file-size cap.)
- Graft patch: `patches/gdn_fused_step/gdn_fused_step.patch` (regenerated vs pristine 75ca881; `git apply --check` clean), `.hip` copy in sync.
- Model diff: `patches/qwen35_fuse_branch/qwen35_fuse_branch.diff` (header counts repaired; context lines repaired; applies to the box model state).
