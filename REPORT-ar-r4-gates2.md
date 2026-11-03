# AR R4 follow-up (t1779): diagnostic probes + F-branch GDN ssm fix + gate re-run

TL;DR:
- Probe A (per-step PA vs PG GDN ssm state md5, v2.1 methodology, 24 steps): GDN ssm
  is STILL period-4 at replay (s0–s3 bit-exact, s4+ all 24 layers diverge; PG locks
  264 15352 8214 5529 from step 5). Root cause isolated by code reading: under
  GDN_FUSE=1 the LIVE decode op is `gdn_fused_step` (the F-branch,
  S<=4, B==1, state!=None), whose `eval_gpu` FRESH-allocates `state_new` on every
  step. The v2.1 in-place fix (9c82e4c) is in `gated_delta_update.cpp` — the
  DEFAULT-CHAIN op, which the F-branch returns before reaching. The v2.1 fix was a
  NO-OP in the gate config.
- Probe B (D2H-read the per-step idx buffer device contents at launch time, PG run,
  24 steps): every launch reads the CORRECT current token (24/24 match=1; the
  arena slot VA is stable 0x...a700 and the op-time `idxval` is correct). The
  cross-stream-visibility / 1-step-lag token hypothesis is FALSIFIED.
- Fix implemented (one, minimal, mirrors 9c82e4c's pattern): in
  `gdn_fused_step.hip` `GdnFusedStep::eval_gpu`, in `rocm::graph_decode_mode()`
  (with the same row_contiguous + nbytes guards as 9c82e4c),
  `state_new.copy_shared_buffer(st)` instead of the fresh malloc. In-place safety
  verified in the fused kernel: each thread loads its h[64] tile into registers
  BEFORE the row loop and writes `state_new` only AFTER it.
- Gate re-run (new wheel, 400-tok protocol, same session): Gate 1 LOSSLESS FAIL (G1
  still period-4: first 5 tokens match A1, attractor through idx 228, escape at
  229 into a 29006-family loop; A1 tokens bit-identical to t1769), Gate 2 SPEED
  FAIL (G 65.26 vs A 71.50 = -8.7%). The fix FIRES (9528 in-place aliases,
  st==state_new) and CHANGES the ssm dynamics (pre-fix PG ssm state was a strict
  period-4 cycle; post-fix it is evolving/chained) — the remaining attractor
  points to the conv state (python-side, non-in-place in both paths) as the dominant residual. Stopped per brief.

## Probe evidence

### Probe A — ssm state step-exactness (t1779, 2026-07-09 22:30 UTC, t1771 wheel
e6616cb6... installed in venv-r4v, sidecar 02a54359 pinned)

Protocol: v2.1 methodology (t1688): A = 24 eager decode steps; G = 24 decode steps
after capture (capture on step 3), same session, canonical env (QMV_ROWS=1
REFWD_M1=1 GDN_FUSE=1 MTP=0). Per-step md5 of 24 GDN-layer ssm state samples
(layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30).

Tokens (first 24 post-capture decode):
- PA: 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 1379 6150 6365 314
  (== the t1769 gate A1 FIRST20 exactly: 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357
    20853 369 34383 279 — the probe reproduces the gate A arm)
- PG: 271 1919 369 264 15352 8214 5529 264 15352 8214 5529 264 15352 8214 5529 ...
  (== the t1769 gate G1 FIRST20 exactly: the period-4 attractor 264 15352 8214 5529, locked from step 5)

Both arms share the first 5 tokens (271 1919 369 264 15352) and then diverge.

State comparison (PA vs PG, per step, 24 layers each):

```
s0  MATCH        s8  DIFF (24/24 layers)
s1  MATCH        s9  DIFF
s2  MATCH        s10 DIFF
s3  MATCH        s11 DIFF
s4  DIFF         ... s12–s23 all DIFF (24/24 layers)
s5  DIFF
s6  DIFF
s7  DIFF
```

Same signature as the pre-fix v2.1 probe: first steps bit-exact, then period-4.
CONCLUSION: the GDN ssm state is still non-in-place in the graph decode path.

### Why the v2.1 fix did not apply (code reading, t1771 wheel)

- qwen3_5.py GatedDeltaNet `__call__`: with GDN_FUSE=1 and S<=4 and B==1 and
  state not None, the F-branch dispatches `mx.fast.gdn_fused_step(...)` and
  returns BEFORE the default chain (`gated_delta_update(...)`).
- `GdnFusedStep::eval_gpu` (gdn_fused_step.hip): `state_new.set_data(malloc_async
  (state_new.nbytes()))` — FRESH per-step allocation; no graph_decode_mode branch.
- `GatedDeltaUpdate::eval_gpu` (gated_delta_update.cpp) — the file 9c82e4c fixed
  with the in-place alias — is NOT the op executed in the gate config (S<=4,
  GDN_FUSE=1). It is only reached by the default chain (prefill T>4 / S>4 paths).

So in the gate config the recorded GDN step node reads the pre-capture ssm state buffer and writes a fresh arena buffer forever — the exact v2.1 stale-read class, now in the fused op.

### Probe B — per-step idx buffer device contents at launch time

Protocol: PG run (capture on step 3), MLX_G1Y_DEBUG=1, per step: D2H-copy the
idx buffer slot device contents immediately before the step (recorded op reads the
slot at launch; the slot is what the indirect path reads), compare with the
token that step is supposed to process.

```
[probeB] s0 tok=271  readback=271  match=1
[probeB] s1 tok=1919 readback=1919 match=1
[gdidx] arm: mode set; slot dev=0x72174429b000 host=0x721b2d5b8000
[probeB] s2 tok=369  readback=369  match=1      (slot materialized, window)
[probeB] s3 tok=264  readback=264  match=1
[gdidx] gather: k=0..2 ... va=0x7216c769a700 inwin=0 idxval=264
[probeB] s4 tok=15352 readback=15352 match=1
[probeB] s5 tok=8214  readback=8214  match=1
[probeB] s6 tok=5529  readback=5529  match=1
[probeB] s7 tok=264   readback=264   match=1
[probeB] s8..s23: match=1 every step (24/24 total)
```

The per-step token buffer's device contents equal the current token at launch for
EVERY replay step; the op-time `idxval` reads (from within the recorded gather)
also match (264/15352/8214/5529 cycling correctly). The slot VA is stable across steps (0x7216c769a700). CONCLUSION: the 1-step-lag / cross-stream-visibility token hypothesis is falsified — token delivery via the indirect path is fresh at
launch; the period-4 attractor is not caused by stale token delivery.

## Fix (t1779, mirrors 9c82e4c)

`gdn_fused_step.hip`, `GdnFusedStep::eval_gpu` (the live F-branch op):

```cpp
  if (rocm::graph_decode_mode() && st.flags().row_contiguous &&
      state_new.nbytes() == st.nbytes()) {
    // in-place ssm state update in graph decode mode: the recurrent kernel
    // loads its h tile into registers before the row loop and writes
    // state_new after it (verified in-place safe), so the output state
    // aliases the input state buffer: at replay the recorded node's read
    // and write hit the SAME stable VA.
    state_new.copy_shared_buffer(st);
  } else {
    state_new.set_data(
        mlx::core::rocm::malloc_async(state_new.nbytes(), encoder));
  }
```

Same guards as 9c82e4c (graph_decode_mode + row_contiguous + equal nbytes).
In-place safety: the fused kernel's phase-4 loop is verbatim r4d — `h[64]` is
loaded ONCE before the t-loop, updated row-by-row in registers, written to
`state_new` after the loop; each thread reads only its own (hv,row,k0) slice
before writing it.

Not touched: `gdn_fused_step_tape.hip` (MLX_ROCM_GDN_TAPE=1 default-off path),
the conv-state update (python-side `mx.contiguous`, see residual suspect below).

## Gate re-run (protocol per t1771/t1769, 400 tokens, same session)

New wheel: sha256 25ad0a0b731e1197 (mlx-0.32.3.dev20261103+75ca881), LIBMD5 790dac8a
(the t1769/t1771 library md5 was b4d206b4). Graft evidence intact (gdidx=8,
gather_rows_ind=264, gdn_fused_step=92) + fix marker present (GDNF-DBG=1).

### Gate 1 (lossless): FAIL

- A1 (eager, t1779 wheel): 71.50 t/s; FIRST20 = 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568
    13 357 20853 369 34383 279 — bit-identical tokens to t1769 A1 (72.11 t/s; -0.8% speed = run variance). Eager path
    unaffected by the fix.
- G1 (graph, t1779 wheel): 65.26 t/s; FIRST20 = 271 1919 369 264 15352 8214 5529 264 15352 8214 5529 264 15352 8214
    5529 264 15352 8214 5529 264 — STILL the period-4 attractor. ALLTOKS: period-4 through idx 228, escapes at idx 229
    into a 29006/314/10288-family attractor (t1769 G1: cycle through idx 229, escape at 230 into the same family). md5
    mismatch A vs G persists.

### Gate 2 (speed): FAIL

- G 65.26 t/s vs A 71.50 t/s = -8.7% same-session (t1769: -9.8%). No change.

### Diagnostic evidence that the F-branch fix fires and works

- G1 re-run with MLX_GDN_DEBUG=1: **9528 GDNF-DBG lines** (st == state_new, same VA on every line; 24 stable per-
    layer state VAs, 397 steps each — the host state arrays are pinned after capture). The alias fires on every replay-
    step enqueue, including at capture.
- None of the 24 aliased state VAs appears in any [g1t] arena trace line (g1t=0) — the traced arena ops do not
    reuse the state VAs.
- Post-fix probe (PA2/PG2, 24 steps, v2.1 methodology):
  - PG2 ssm state is EVOLVING: all of s4..s23 pairwise-unique (cyc4=False, frozen=False).
  - Pre-fix PG (t1771 wheel) ssm state was a STRICT period-4 cycle (s_k == s_{k+4} for all k; cyc4=True).
  - => the in-place alias CHANGED the replayed ssm dynamics from a frozen-read 4-cycle to a properly chained,
    evolving state. The v2.1-pattern mechanism works on the live F-branch op.
  - PG2 tokens are nonetheless still period-4-locked (== G1 attractor), while PA2 == A1 tokens (24/24).
- Provenance note: PA (t1771-wheel eager) vs PA2 (t1779-wheel eager) tokens are identical (24/24; both == t1769 A1
    FIRST20) but the ssm state md5s differ from s0 (24/24 layers). The t1779 wheel was built from vendor-r4v HEAD
    75ca881 ("t761: fix gdn_prep launcher"), which postdates the t1769/t1771 build tree; the F-branch fix is mode-gated
    and code-identical on the eager path, so this is a pre-existing tree/build difference (likely the gdn_prep launcher fix itself), not a regression from this change.

### Interpretation

- The F-branch GDN ssm state was one of the non-in-place recurrent states in the gate config (the v2.1 fix never
    reached it); it is now in-place and chains at replay.
- The period-4 attractor persists => the DOMINANT remaining non-in-place recurrent state is the conv state:
    `cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])` (python-side, shared by both paths) is a recorded copy whose
    source chain (the concat/conv kernels) reads the pre-capture conv-state buffer VA at replay — the same stale-read
    class, and it cannot be made in-place from python. (The K=4 value-slot path at copy.hip:343 remains a secondary
    suspect.)
- Gate 1 FAIL and Gate 2 FAIL persist. Per the run brief: report the numbers and stop.

## Next-round suggestion

1. Conv-state in-place: extend the F-branch fused op (or a small new op) to also emit the conv-state shift (the
    3-row slide of [cs0, cs1, cs2, v0]) and, in graph_decode_mode, write it in-place into the conv-state buffer
    (mirroring this fix's pattern); update qwen3_5.py to take the conv state from the op output instead of
    `mx.contiguous`. Re-run Gate 1 + Gate 2.
2. If the attractor persists: instrument the recorded GDN node's actual read VA vs the aliased write VA at replay
    (op-arg tracing) to rule out a recorded-argument VA mismatch, and check the K=4 value-slot path at copy.hip:343.
3. Gate 3 (MTP4) re-run with the new wheel (the F-branch op serves S=4 verify; the in-place branch is mode-gated
    and the guard runs under graph decode — acceptance 137/87/49 should be re-verified).

## Residual suspect if the gate still FAILs

The conv state update (`cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])`,
python-side, shared by both paths) is a recorded copy whose source chain
(concat/conv kernels) reads the pre-capture conv-state buffer VA at replay — the
same stale-read class. It cannot be made in-place from python; fixing it
requires either an op-side change (fused op also emits the conv-state shift, aliased in graph decode mode) or an arena-slot-reuse argument. If Gate 1 still shows the period-4 attractor after this fix, that is the next suspect (along
with the K=4 value-slot path at copy.hip:343).
