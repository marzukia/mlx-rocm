# NOTES-r5 — R5 design probe working notes

Purpose: externalize all analysis for REPORT-r5.md. One section per file
read; the report is assembled from this file at the end.

Base: branch mtp-r4b-fused-verify @ 09ddc98 (PR #14-era, 48.93 t/s stack).
Work branch: r5-correction-fusion.
GPU lock check at run start: fuser exit 1 (no holder) = lock FREE.

## 1. Measured situation (source: REPORT-mtp-traffic-audit.md, t1582/PR #14, commit 4cd7083 on branch mtp-traffic-audit)

Anchor mix: 161 verifies, 126 corrections = 78.3% of steps rejected.
Per-phase bytes (measured tensor bytes, median of 3):

| phase | target reads/step | target GiB/step | other GiB/step | total |
|---|---|---|---|---|
| verify (S=4, rows-fused M=4, one weight pass) | 1.000 | 4.16 | 0 | 4.16 |
| commit (tape rebind + KV trim, no forward) | 0 | 0 | 0 | 0.00 |
| correction (S=1 per rejected step) | 0.783 | 3.26 | 0 | 3.26 |
| draft (3x sidecar + 3x 4-bit head) | 0 | 0 | 1.97 | 1.97 |
| TOTAL | 1.783 | 7.42 | 1.97 | 9.39 |

- Target per forward = 4.16 GiB (trunk 3.63 GiB + 4-bit lm_head 0.533 GiB).
- Sidecar mtp module = 0.125 GiB; int2 draft head 0.296 GiB ABSENT in R4b wheel.
- KV reads ~0.07 GiB/step, negligible.
- Total 9.39 GiB/step <= ~11 GiB anchor budget => NOT weight-traffic-bound;
  remaining gap to 144.3 t/s is compute/schedule-side.
- R5 removes the correction read: 9.39 -> 6.13 GiB/step.

Per-phase wall (phase harness 64v, median of 3) — KEY for schedule analysis:
- draft 39.3 ms/step (also FORCES the lazily-evaluated correction forward)
- verify 0.5 ms (lazy graph build; S=4 GEMM forced by first argmax)
- commit 26.7 ms (S=4 GEMM + tape rebind)
- correction 0.4 ms (bookkeeping; GEMM billed to next draft)
- Anchor: 65.2 ms/step at 48.93 t/s (10.49 s / 161 steps).
- CRITICAL OBSERVATION: the correction forward is ALREADY LAZY — it executes inside the NEXT step's draft phase (the "draft 39.3 ms" includes it). So
  "one S=1 target re-read per rejected step" runs temporally in the draft
  window of the following step. Any fusion design must slot into that window or the verify window, not the (near-free) correction bookkeeping slot.
- Dispatch counters: GDN_TAPE_CALLS = 24 x (verifies + corrections) => the
  fused/tape GDN op path runs in BOTH verify and correction forwards.
- FIRST20 / accepts bit-stable across runs (lossless gate anchor).

## 2. R4b report (reports/mtp-r4b-fused-verify.md)

Facts that matter for the design:
- Anchor 48.93 t/s = +6.2% over OFF (46.07); lever = MLX_ROCM_QMV_ROWS=1 (env gate;
  the wheel change is belt-and-braces for bias-free calls).
- rows-fused M=4 = ONE weight read shared across the 4 verify rows; output vs
  per-row maxdiff 1.2e-4 (bf16, NOT bit-identical); acceptance drift is the
  expected FP-order effect (D3 -2.7 pts, D2 +3.4, tok/step 2.385 -> 2.410).
- FIRST20 identical across arms/runs: 271 1919 369 264 15352 ... (lossless anchor).
- mx.eval is a side effect: the S=4 verify GEMM is forced by the first argmax in
  the commit section and billed to commit. verify timer = lazy graph build only.
- Phase 64v ON totals (ms, /64 => per-step): draft 2515.2 (39.3), verify 35.3
  (0.55), commit 1709.2 (26.7), correction 25.5 (0.4). 512v: draft 14789.8
  (28.9), verify 278.1 (0.54), commit 25189.8 (49.2), correction 187.3 (0.37).
- Standing env: MTP6_TAPE=hybrid GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1. Wheels: venv-canon (md5 b1de59fd...) vs R4b wheel
  (md5 3d19e4f3...).
- Implied per-step schedule (64v): 39.3 + 0.55 + 26.7 + 0.4 = 66.9 ms/step, matching
  the audit's ~66.9. The S=4 GEMM weight read (4.16 GiB) happens inside the
  26.7 ms commit window; the correction S=1 read (3.26 GiB) happens inside the
  39.3 ms draft window of the NEXT step (audit: "GEMM billed to next draft").

## 3. Code map (files/functions)

Bench (repo, the only files R5 touches):
- `bench/mtp-bench6.py` (499 L, canonical) and `bench/mtp-bench6-phase.py`
  (515 L, instrumented copy): mode-4 branch = the spec loop. Draft phase
  (single-token step, `if draft1 is None:`), verify (batched S=4
  `lm(mx.array([[prev_token, d1, d2, d3]]), ...)`), then the
  `commit(krow, prefix)` / `reject(krow, lk, hb, ktrim, prefix, ctoken)`
  closures. reject() today = tape-row rebind (or canonical restore) +
  `mtp_cache[0].trim(ktrim)` + ONE S=1 correction forward `lm([[ctoken]])`,
  then `l, h` from the correction (hybrid) or the batched row (tape=1).
  The correction forward is LAZY (MLX eval is a side effect): it executes in the NEXT step's draft phase (the 39.3 ms draft includes it).
- Wheel model file (guest venv, NOT in repo): `mlx_lm/models/qwen3_5.py`
  (patched; venv-r4v md5 3503e632, = /tmp/qwen3_5.py 7763fbbb plus the
  F-branch 3-output `gdn_fused_step` conv_state_new update):
  - `MTPModule.__call__(hidden_states, embedding, cache)`: pre-norm both,
    `fc(concat(emb_norm, h_norm))`, one `MTPDecoderLayer` (own attention +
    MLP; attention KV = `mtp_cache`), final norm. Draft call i processes
    token d_{i-1} (d0 = t) and predicts position k+i+1.
  - `GatedDeltaNet.__call__` TAPE_BRANCH_PATCH: for S<=4 with state set,
    `mx.fast.gdn_fused_step_tape` emits `(out, state, state_tape,
    conv_state_tape)`; `self.last_tape` = per-row states; `GDN_TAPE_CALLS`
    counter. `last_tape` rebind in commit() = lazy cache-leaf swap + KV
    `c.trim(3 - krow)`; `cache.lengths` is None in this model (no fixup).
- Vendor wheel: `mlx/backend/rocm/gdn_fused_step_tape.hip` (per-row state
  tapes), `quantized/qmm.hip` wide_ok multi-row (MLX_ROCM_QMV_ROWS, M=4
  one shared weight read).

State-staging order per accepted-pair step (hybrid, the baseline):
1. draft: 3 mtp calls (sidecar 0.125 GiB + 4-bit head 3 x 0.533 GiB =
   1.97 GiB), mtp_cache +3 entries; no target forward.
2. verify: S=4 forward built lazily (0.5 ms graph build); one 4.16 GiB
   weight pass (rows-fused M=4), per-row GDN tapes emitted.
3. commit: `argmax(l0)` FORCES the S=4 GEMM (the 26.7 ms window); accept /
   reject; tape-row select (free) + KV/mtp trims (free).
4. correction: hybrid runs the S=1 re-forward here (3.26 GiB/step avg);
   executes in the NEXT draft window (that is why draft = 39.3 ms).

## 4. Option (a) PIGGYBACK (correction token = row 0 of the next verify) — LOCKED

On reject at row r: ctoken = argmax(l_r) (free, from the batched S=4);
tape-select row r (state -> after the r+1-token prefix, free); KV trim
(3 - r); mtp trim (3 - r) — keeps history [t, d1, ..., d_{r-1}], the
mtp-(r+1)-th-call-of-chain shape, exactly the clean-flow analogue (in the
clean flow the chain anchored at position p attends [t, d1, d2]; here it
attends [t, ..., d_{r-1}] with ctoken in slot r+1, the same relative
structure). Then NO correction forward: the draft phase re-anchors directly:
draft1 = mtp(h_r, emb(ctoken)), draft2 = mtp(dh1, emb(d1)), draft3 =
mtp(dh2, emb(d2)) (h_r = the rejected row's pre-norm hidden, already in
the S=4 graph — zero extra target work). Next verify = [ctoken, d1, d2,
d3] (S=4, one rows-fused pass); t2 = argmax(l0) is the free next token;
acceptance logic is byte-identical to today. EOS: ctoken-EOS breaks
before the reject call (as today); a later all-reject makes t2 the new
correction token (recursion works). The S=1 re-read disappears:
9.39 -> 6.13 GiB/step. Fallback: if the tape is unavailable (fresh-cache
STATE_NONE_GUARD path) use the existing correction-forward path — never triggered here because prefill precedes every verify. This is exactly the llama.cpp 29c720e structure (O(1) seq_rm snapshot select + correction as row 0 of the next verify) and the radiance structure (zero re-forward, on-GPU state shift). Verified by the gap-dive audit (section B): HOLDS.

## 5. Option (b) ROLLBACK-REDRAFT

Rolling the state back and re-running the draft from the accepted prefix
is option (a) modulo bookkeeping (the tape IS the rollback; the redraft is
the re-anchored draft above). The only real alternative in this family is the canonical savestate re-forward of the accepted prefix — that is the pre-tape path (28.3 GiB/step era), strictly worse. No distinct value.

## 6. Option (c) CONFIDENCE SKIP

Skip the correction forward when the argmax margin is high: the state must advance by exactly ctoken regardless of margin (a margin gate would change the token stream / introduce a host sync — lossless-gate risk) and offers
no structural saving: the forward is either done or folded. Folding = (a).
Rejected.

## 7. Option (d) / other findings

- S=5 verify with the correction as a 5th row: the QMV wide path is M<=4
  only and the tape op is S<=4; S=5 falls back to the slow path and breaks
  the rows-fused M=4 lever. Rejected.
- "Run the correction during the commit window": the correction depends on the reject decision (argmax(l0..)), which is only known inside the commit
  window; the next draft window is the earliest it can execute — which is
  exactly what the hybrid already does. The fix is to not execute it at all, i.e. (a).
- No wheel/model change is required: the tape op, the tape rebind,
  `mtp_cache.trim` and the QMV rows-fused path all exist in venv-r4v
  (0.32.3.dev20261104+75ca881, libmlx md5 8acf13fec6fc1636026d8e1dc69cd497;
  probe: gdn_fused_step_tape present, QMV_ROWS env accepted, gfx1201
  bound, 32 CUs). R5 is a bench-loop change only.

## 8. Recommendation + hook points

Implement (a) as a new arm `MTP6_TAPE=r5` in BOTH bench files
(mtp-bench6.py canonical + mtp-bench6-phase.py instrumented), minimal diff:
1. `tape_mode` assert accepts "r5"; r5 sets MLX_ROCM_GDN_TAPE=1 +
   MLX_ROCM_GDN_FUSE=1 (like 1/hybrid).
2. Loop state: `pending_c` (deferred correction token), `h_r` (rejected row
   pre-norm hidden).
3. `commit()` returns bool (tape row committed vs canonical fallback).
4. `reject()`: r5 + tape-committed -> NO correction forward; `pending_c =
   ctoken`, `h_r = hb` (no n_refwd bump). Fallback (no tape) -> existing
   correction-forward path, unchanged.
5. Draft step: `if pending_c is not None:` t = pending_c (already appended
   to out at the reject), first draft call uses h_r; else the existing
   `argmax(l)` free-token path, unchanged. r5-pending step skips the
   t==EOS break (ctoken was checked EOS at the reject).
6. INFO line: add N_CDEFERRED (count of deferred corrections); REFWDS must
   be 0 for r5 (except a first-step fallback, which does not occur).
Expected: draft 39.3 -> ~22-26 ms/step (the 3.26 GiB read removed); total
65.2 -> ~45-50 ms/step; t/s 48.93 -> ~60-65. Acceptance may drift (the
mtp re-anchoring changes draft tokens — the R4b-class FP/structure effect);
FIRST20 vs the 390221f baseline is the lossless gate (report on FAIL).

## 9. Measurement plan

- Box: neon (lock FREE at run start, fuser rc=1), flock on ~/gpu.lock
  (`exec 9>~/gpu.lock; flock -w 300 9`), 25 s idle before the first run
  and between runs, venv-r4v, model /mnt/models/qwen3-9b-mlx (mtp safetensors md5 02a54359cda8b247f76b277dd9bc4d4c, verified).
- Standing env both arms: GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1; arm switch MTP6_TAPE=hybrid (B, the 48.93
  baseline) vs MTP6_TAPE=r5 (R5). 3 repeats per arm, median.
- Protocols: anchor MTP_MAX_TOKENS=322 (322-iter / 161-verify regime,
  the R4b anchor protocol); phase 64v (MTP_MAX_TOKENS=128) for the
  per-phase wall breakdown (draft/verify/commit/correction).
- Same-session B-vs-R5 relative gate (regime-dependent absolutes per the
  clock-regime standing); log SCLK per run; record libmlx + model md5.
- Lossless gate: FIRST20 / accepts bit-stable vs the 390221f baseline;
  report only on FAIL. Acceptance drift expected (structure effect);
  record the D1/D2/D3 + tok/step movement per the R4b precedent.
