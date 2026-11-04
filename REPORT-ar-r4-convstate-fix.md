# t1788 — R4 GDN FUSE round: conv-state in-place commit + gate re-run (LOSSLESS FAIL, residual suspect isolated)

> **PROVENANCE NOTE (t1816):** the original t1788 `-fix` report (374 lines as
> listed in the 928b30f PR) was not preserved in the final 928b30f commit — the
> branch's last force-push (22:18:42Z) predates the round, and the file is
> absent from every reachable object (exhaustive t1816 search: all loose
> objects + packs in the fleet mirror, all worktree reflogs, pre-amend commits
> 13647d7/531ef31/4c86685/cd6b90f/3f7866e, box filesystem, both forks, GitHub
> API file list). This file is a faithful **reconstruction** of it from the
> committed evidence: every number below reproduces from `data/t1788/` and
> `data/t1779/` logs, the box wheel build log, and the t1788 session handoff
> (274-turn single session, 23:14:55Z–01:09:11Z, 3 compactions, clean exit).
> The companion summary `REPORT-ar-r4-convstate.md` (in 928b30f) cross-checks
> this report line by line.

## 1. Round purpose

Round N of the R4 GDN-FUSE diagnostic sequence. The t1779 kernel left the
conv-state commit dangling (the updated conv state was never aliased back onto
the captured `conv_state` buffer in graph mode). This round: one minimal fix
mirroring the 9c82e4c-pattern ssm in-place fix from t1779, rebuild the wheel,
re-run the canonical Gate 1/2/3 battery, per-step state probes, MTP4 guard.

## 2. Environment (pinned)

| item | value |
|---|---|
| box | guest-a, RX 9070 XT (gfx1201), ROCm 10.1.0 guest userland |
| model | /mnt/models/qwen3-9b-mlx/model-mtp.safetensors, md5 `02a54359` (sidecar pinned pre/post, no drift) |
| wheel source | 75ca881 + full R4 graft (82 files, 6406+/548-) |
| wheel | `mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl` (22,449,702 B, on the t1779 base), sha256 `eee04542d620b2c025f48359aebb3d52d1be6d998664859728338eaae750f9cd`, BUILD_RC=0 |
| lib symbols | `gdn_fused_step` 5, `gdn_fused_conv_state` 4, `GdnFusedStepTape` 4, `gdn_prep` 7 |
| model file | venv `qwen3_5.py` re-patched (t1779 revert + t1788 apply) |
| protocol | 400 tok, same session: A1 (eager) → 25 s idle → G1 (graph); MTP4 ×3; probe A/G pairs |

## 3. The fix (one, minimal)

- `gdn_fused_step.hip`: a new `gdn_fused_conv_state_kernel` emits the conv
  state as the op's **third output**. `new_cs[r] = row (TMAX+r)`: T=1 →
  [cs1, cs2, v0]; T=2 → [cs2, v0, v1]; T=4 → [v0, v1, v2]. No arithmetic —
  pure bf16 copy of the last 3 rows of [cs (3 rows); qkv (T rows)],
  bit-identical to the python path.
- In-place safety: shared-memory staging (768 elements/block, 3 per thread;
  Qwen3.5-9B: 3×8192/768 = 32 blocks) + `__syncthreads()` before the write, so
  all reads from the (aliased) cs buffer complete before any write. The kernel
  runs after the main fused kernel (same-stream ordering).
- `graph_decode_mode()`: `cs.flags().row_contiguous && cs.nbytes() ==
  conv_state_new.nbytes()` → `conv_state_new.copy_shared_buffer(cs)` — the exact
  mirror of the ssm alias.
- `fast.cpp` (box 75ca881 API): op outputs 2 → 3 (make_arrays shapes/types);
  the fallback computes `new_cs` with the free `slice(a, start, stop, s)`
  (this mlx version has no `array::slice` member).
- `fast_primitives.h`: `output_shapes` → 3 (`inputs[7].shape()`).
- Model F-branch: `out, state, conv_state_new = mx.fast.gdn_fused_step(...)`,
  `cache[0] = conv_state_new.reshape(B, conv_kernel_size - 1, conv_dim)`; the
  python-side `mx.contiguous` update runs unconditionally before the F-branch
  check and is shadowed by it — zero new captured ops vs. status quo.
- The graft API files were already 3-output-capable by design on the box from
  t1779; this round's delta is the .hip change plus the 2→3 API wiring.

## 4. Gate 1 — LOSSLESS FAIL

A1 (eager, MTP=0, 400 tok): **71.96 t/s**, FIRST20 =
`271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`
— bit-identical to the canonical (t1769 == t1779).

G1 (graph, MTP=0, 400 tok, 25 s idle between arms, same session): **65.55 t/s**,
FIRST20 = `271 1919 369 264 15352 11 70042 10597 29006 314 1204 264 6278 20853 5568 13 357 6278 20853 369`
— **first token fork at idx 7 (10597 vs 29006)**. The t1779 hard period-4
attractor (`264 15352 8214 5529` from step 5, t1779 G1 FIRST20 locked on it) is
gone; G1 now drifts and loops loosely (PG probe: `...6278 20853 369 524 15756 264 328`).

G1 debug run (MLX_GDN_DEBUG=1, 400 tok): the **ssm alias FIRES (9528 lines =
24 layers × 397 steps)** and the **conv-state alias FIRES (9528 lines, cs==new
VA on every step)** — both fast-path commits ran the whole run; the divergence
is not a fallback/route mismatch.

## 5. Per-step state probe (PA3 vs PG3, 24 steps, md5 of ssm AND conv per layer)

- **s0–s3: all 24 GDN layers bit-exact** (ssm and conv).
- **s4, s5: 21/24 layers' ssm dirty (g4+); g0, g1, g2 clean; ALL conv clean.**
- **s6: 21/24 ssm dirty; 21/24 conv dirty — conv turns dirty only at s6,
  exactly when the first dirty qkv row (s4) enters the 3-row window.**
- **s7: first token fork (29006 vs 10597); PG lags PA by one token through the
  rest of the trace.**

Read: the **conv state is EXONERATED** (bit-exact at s4/s5 despite replay; its
s6 dirt is purely downstream of the ssm divergence). GDN blocks 0–2 — which
feed on pre-attention inputs — stay clean while blocks 3+ go dirty at s4, so
the residual suspect is the **ATTENTION/KV path** (the probe skips KVCache
entries): whatever the graph replays stale for the full-attention layer between
GDN blocks 2 and 3 feeds the GDN ssm updates from block 3 on.

## 6. Gate 2 — speed (record)

G 65.55 vs A 71.96 = **-8.9%** (t1779: -8.7%; t1769: -9.8%). The second (tiny)
conv-state launch per GDN step costs ≈ nothing relative — both arms pay it.

## 7. Gate 3 — MTP4 guard (3 runs, graph decode, MTP=4, 400 max)

**82/13/5, 82/13/5, 82/13/5 @ 31.37 / 27.59 / 30.18 t/s** (guard 137/87/49 @
43.64; t1769: 40.56 = -7.1%). Measured on the divergent G trajectory (Gate 1
FAIL), so the acceptance/t-s numbers are not directly comparable to the clean
baseline; recorded as evidence of the MTP path running stably (200 verifies,
deterministic across runs).

## 8. t1779 comparison

| gate | t1779 | t1788 |
|---|---|---|
| G1 t/s | 65.26 | 65.55 |
| G1 behaviour | hard period-4 attractor from step 5 | idx-7 fork, drift, loose loop |
| A1 t/s (bit-identical FIRST20, canonical) | 71.50 | 71.96 |
| G2 gap | -8.7% | -8.9% |
| MTP4 | 82/13/5 | 82/13/5 |

## 9. Stop condition + next steps

Met per brief: Gate 1 still fails after the conv-state fix; per-step state md5s
captured; remaining suspect named (attention/KV path, first full-attention layer
between GDN blocks 2 and 3). Stopping here.

Kernel change itself is correct as implemented (in-place commit + alias verified
in the debug run). The next round's probe set should extend to KV cache rows
and the full-attention output around step 4.

## 10. Graft deliverable note (added by t1816 reconstruction)

The t1788 handoff recorded the graft as "regenerated vs pristine 75ca881,
git apply --check clean". The delivered `gdn_fused_step.patch` in 928b30f
carried a residual spurious `gdn_prep`/`gdn_norm_gated` deletion hunk and
lacked the `gdn_fused_step_tape.hip` creation its own CMake hunk references —
flagged in the t1808 fleet review. Fixed in t1816: graft regenerated as
`diff(75ca881 -> box wheel tree)` with the 411512b explicit T-admission gate
restored, `git apply --check` on pristine 75ca881, 7/7 md5-identical to the
box wheel source tree; the wheel was rebuilt from the grafted tree
(BUILD_RC=0, sha256 `caf6299a...`) and the Gate 1 spot-check A1 returned
FIRST20 bit-identical to the committed t1788 A1 at 70.20 t/s (sidecar pin
02a54359 pre/post; lib symbols all present).

## 11. Artifact manifest

- `data/t1788/`: A1.log, G1.log, G1dbg.log (400-tok gate arms + debug),
  PA3.log/PG3.log (24-step state probes), MTP4.1-3.log (guard runs),
  sidecar.log, clocks-*.log
- `data/t1779/`: A1.log, G1.log, probe logs (baseline)
- box: `~/t1788/` (apply/build/gate scripts + full logs), `~/wheelout-t1788/`
  (wheel + sha256), `~/vendor-r4v` (build tree, 82-file graft)
- `REPORT-ar-r4-convstate.md` (summary, in 928b30f)
