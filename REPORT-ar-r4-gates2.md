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
- Gate re-run: see below (Gate 1 + Gate 2, 400-tok protocol, same session, t1769
  scripts).

## Probe evidence

### Probe A — ssm state step-exactness (t1779, 2026-07-09 22:30 UTC, t1771 wheel
e6616cb6... installed in venv-r4v, sidecar 02a54359 pinned)

Protocol: v2.1 methodology (t1688): A = 24 eager decode steps; G = 24 decode steps
after capture (capture on step 3), same session, canonical env (QMV_ROWS=1
REFWD_M1=1 GDN_FUSE=1 MTP=0). Per-step md5 of 24 GDN-layer ssm state samples
(layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30).

Tokens (first 24 post-capture decode):
- PA: 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 1379 6150 6365 314
  (v2.1-probe-protocol eager trajectory; first 5 tokens == the gate A-arm prefix
  271 1919 369 264 15352; token 6 onward differs from the gate A arm
  (8214 5529 11 ...) because the probe prefills differently than the mtp-bench6
  S=4-chunk A arm — the within-run PA-vs-PG comparison below is the relevant one)
- PG: 271 1919 369 264 15352 8214 5529 264 15352 8214 5529 264 15352 8214 5529 ...
  (== the gate G1 period-4 attractor 264 15352 8214 5529, locked from step 5)

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

So in the gate config the recorded GDN step node reads the pre-capture ssm state
buffer and writes a fresh arena buffer forever — the exact v2.1 stale-read class,
now in the fused op.

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
also match (264/15352/8214/5529 cycling correctly). The slot VA is stable across
steps (0x7216c769a700). CONCLUSION: the 1-step-lag / cross-stream-visibility
token hypothesis is falsified — token delivery via the indirect path is fresh at
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

[awaiting results]

## Residual suspect if the gate still FAILs

The conv state update (`cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])`,
python-side, shared by both paths) is a recorded copy whose source chain
(concat/conv kernels) reads the pre-capture conv-state buffer VA at replay — the
same stale-read class. It cannot be made in-place from python; fixing it
requires either an op-side change (fused op also emits the conv-state shift,
aliased in graph decode mode) or an arena-slot-reuse argument. If Gate 1 still
shows the period-4 attractor after this fix, that is the next suspect (along
with the K=4 value-slot path at copy.hip:343).
