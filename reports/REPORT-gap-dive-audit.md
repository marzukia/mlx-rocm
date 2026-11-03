# REPORT — gap-dive audit: per-phase reconciliation of the two MTP gap dives

Provenance: adversarial read-only audit of the two gap-analysis dives

- `mtp-gap-llamacpp` (dbc8092): `reports/REPORT-llamacpp-mtp-gap.md`, static
  comparison vs the t1436 llama.cpp `draft-mtp` anchor (144.3 t/s, `n_max=3`).
- `mtp-gap-radiance` (8cfc59a): `REPORT-radiance-mtp-gap.md`, static
  comparison vs the t1439 vllm-radiance K=3 anchor (142.31 t/s).

Evidence base: raw t1436 session (`.fleetwork/handoff.md` +
`REPORT-llamacpp-anchor.md`), raw t1439 session
(`reports/vllm-radiance-anchor-9b.md`), t1462/t1463 dive-session handoffs,
the traffic audit (`mtp-traffic-audit`, 4cd7083), the R4b report, and
llama.cpp source at the exact pinned commit 29c720e (checked out locally,
read-only). The radiance/vLLM source is guest-only (not in this repo); its
citations are verified against the t1463 session record, not the source.
No GPU work. No per-phase runtime numbers exist for the llama.cpp side from
the raw sessions (t1451 profile crash; unrefetched); the llama.cpp numbers
are wall-level + static structure.

## A. Per-phase reconciliation — verdict: ACCOUNTING-HOLE (both sides)

Nothing in the raw data decomposes 144.3 t/s into draft / verify / commit /
correction phases. Evidence:

- The llama.cpp dive says it outright (report lines 3-5): "no per-phase
  runtime numbers exist yet for either side... the llama.cpp breakdown is a
  static estimate, not a measurement".
- The t1436 raw session captures: per-step `accepted N/3` lines (n_steps=161,
  sum_accepted=346, mean_accepted=2.14907; acc.sh output), the verbose
  per-candidate draft lines, and summary `Generation: 144.1 t/s`. It does not
  capture the `t_draft_us` / `t_accept_us` counters even though they exist in
  the code (verified at 29c720e: `common/speculative.cpp:200-201`, printed by
  `common_speculative_print_stats` at :3126). So the llama.cpp side has no
  per-phase split in raw data — its "6.93 ms/token" is wall-level only.
- Our side now has phase walls (traffic audit, 4cd7083, anchor 65.2
  ms/verify pair): draft 39.3, verify 0.5, commit 26.7, correction 0.4 ms;
  bytes: verify 4.16, correction 3.26 (0.783/step), draft 1.97, total 9.39
  GiB/step.

Phase mapping (our current vs llama.cpp static):

- 3x draft (39.3 ms, 1.97 GiB) vs 3x MTP-layer forward + 4-bit shared head +
  host-side greedy sampling (a few tens of us per 3 draft tokens, per dive
  line 5): we carry the draft cost on-device; llama's sampling is host-side
  (D2H + argmax x4 + draft sampler) — not a lever for us.
- verify S=4 (26.7 ms counted in commit; 4.16 GiB) vs one `llama_process`
  per S=4 batch = ONE submission of a 32-layer compute graph: we do 24 GDN
  launches + 8 SDPA + QMM dispatch per forward (the named-ladder GDN merge
  addresses the launch count).
- correction S=1 (0.4 ms wall, 3.26 GiB re-read @0.783) vs none (O(1)
  snapshot rollback + correction token folded into the next verify, see B):
  R5 target.
- commit (tape-row select) vs O(1) `seq_rm` via per-token snapshot index
  (see B): par.

Unexplained remainder: llama.cpp does a full spec step (3.15 tokens, 21.8 ms)
where we do 65.2 ms. None of llama.cpp's 21.8 ms is attributable per-phase
from raw data. What the static structure shows: (i) we pay a correction
S=1 re-read (3.26 GiB/step) that neither reference pays; (ii) our S=4 verify
runs 26.7 ms vs the 4.16-GiB byte floor at 546 GB/s = 7.6 ms (~3.5x floor,
effective ~146 GB/s); (iii) draft 39.3 ms vs an estimated 2-4 ms on the
llama.cpp side. The llama.cpp per-phase split remains an open validation
item (dive §5.2 names exactly what to capture: `t_draft_us`/`t_accept_us` +
metrics pre/post decode in one lock window). Clock-regime caveat [217]: the
t1436 window was 2200 MHz locked; our 65.2 ms/step figures are from the
R4b/audit era on the same guest.

## B. The correction question — verdict: HOLDS (R5 = both references' design)

llama.cpp, verified at pinned commit 29c720e:

- Partial acceptance: `slot.mem.seq_rm(slot.id, pos_next, -1)`
  (`tools/server/server-context.cpp:5034`) dispatches to
  `llama_memory_recurrent::seq_rm` — "partial rollback via per-token
  snapshot indices (limited to the most recent ubatch snapshots)",
  implemented as `set_rs_idx` + cell invalidation
  (`llama-memory-recurrent.cpp:166-216`). Zero target re-forward; the
  recurrent state is selected, not recomputed. The snapshot plane is
  pre-allocated `mem_size x (1 + n_rs_seq)` rows (`:102`) — the recurrent
  state is written 4x per forward (all positions kept); the cost is state
  WRITE traffic, not re-forward FLOPs.
- The correction token is folded into the next step: the next draft starts
  with `id_last = slot.sampled` (`server-context.cpp:3897`); checkpoints
  `spec_ckpt.update_pos/update_dft` (`:3880`) track the accepted prefix. The
  bonus/commit token is row 0 of the next verify — no separate re-forward.
- Draft context is trimmed per chain via `llama_memory_seq_rm` + checkpoint
  restore (`server-context.cpp:3931-3940`).
- Provenance check: the t1436 "llama-cli" binary at this commit
  (`tools/cli`) embeds an in-thread llama-server (`tools/cli/cli-server.h`:
  "spawn llama-server in a thread"); the raw log's slot lines
  ("launching slot ... speculative:true") confirm the run executed the
  server-context code path, so the dive's `server-context.cpp` citations
  are valid for the anchor. The dive's "server path" label is accurate.

radiance/vLLM (citations per t1463 session record; source is guest-only):

- Zero re-forward: the one S=4 verify pass writes GDN candidate state
  columns (`mamba_cache_mode=align`) + all KV;
  `postprocess_mamba_align_gpu` shifts the GDN state to the accepted column
  on-GPU (`vllm/v1/worker/gpu_model_runner.py:1586-1613`); drafter KV is
  trimmed with `seq_lens -= num_rejected_tokens_gpu` computed on-device
  (`llm_base_proposer.py:674-678`). No host-side sync.

R5 coverage: R5 (fold the correction token into the next verify S=4;
eliminate the S=1 re-read) is EXACTLY the structure both references use
(correction token = row 0 of next verify; recurrent state =
snapshot/candidate-row selection; zero rejection re-read). Neither reference
eliminates or merges any per-rejection target read that R5 does not already
cover. No finding beyond confirming R5. One nuance for R5: our bench
already runs B = [commit, d1, d2, d3], so post-R5 our per-pair structure
(3x draft + 1x verify S=4 + 0 correction) is structurally identical to both
references.

## C. Levers missed vs verified

Verified / no lever:

1. K/M > 4 per step: NONE. llama.cpp `n_max=3` (t1436 raw: `n_max=3,
   n_min=0`); radiance K=3 is the best arm (K=4 = 95.05 < 142.31, t1439
   report); we are at K=3. HOLDS.
2. Eager claim: VERIFIED. The t1436 build (`-DGGML_HIP=ON`, no graph flag)
   links the native ggml-hip backend, which has no graph-capture path at
   29c720e (graph-capture macros `GGML_CUDA_USE_GRAPHS` / `GGML_HIP_GRAPHS`
   live in ggml-cuda only; no capture in the ggml-hip sources, no capture
   lines in the raw log). So 144.3 is EAGER: each `llama_process` submits a
   per-batch ggml compute graph, and the whole S=4 verify is ONE submission
   of a 32-layer graph. Consequences: (a) our graph-capture lever is NOT
   "catching up to the anchor" — the anchor has no graph; hitting 144.3
   with a graph would be a structural advantage over llama.cpp; (b) the 3x
   wall gap is therefore NOT explained by graph-vs-eager structure — it is
   kernel/scheduling efficiency. Radiance DOES use CUDA graphs
   (cudagraphs [1,2,4,8], per the dive), so graph capture = parity with
   radiance, advantage over llama.cpp.
3. llama.cpp per-row lm_head (4x per-row passes) vs our rows-msg single
   pass (`qmm.hip:4170-4217`): favors us. HOLDS.
4. llama.cpp 4x snapshot state writes per forward (dive line 4) vs our 1x
   tape: favors us. HOLDS.
5. llama.cpp catch-up forward: `common_speculative_process` runs after EVERY
   target batch, "for now, always re-evaluate for simplicity" (their own
   TODO, `server-context.cpp:4755-4758`) — llama.cpp pays ~1 MTP forward +
   head per step for draft-KV alignment and SKIPS none. We skip it (no
   trim, no catch-up) and acceptance parity holds (3.19 vs 3.15 tok/step),
   so the skip is free. The named-ladder "draft-KV trim" should keep the
   skip (trim without catch-up), or measure the catch-up variant per dive
   §5.4 — the acceptance delta is the open question, not the speed.
6. Radiance items (int2 draft head, dynamic draft + n-gram, GDN 24-layer
   merge, CUDA graphs, draft-KV trim) all map onto the named ladder. The
   int2 head kernel ALREADY exists in the vendor tree on
   `mtp-r4b-fused-verify` (`vendor/.../ops/lm_head_int2.hip`, added in the
   R4b work) but is NOT compiled into the R4b wheel (traffic audit:
   "INT2=off in all runs"). The ladder item is closer to done than it
   looks: it is a wheel build + A/B, not new kernel work.

New observations (not on the ladder):

- llama.cpp's whole verify = 1 compute-graph submission of 32 layers; our
  forward = 24 GDN launches + 8 SDPA + QMM dispatch. The single-fused-step
  gap is broader than the 24-layer GDN merge: llama.cpp's per-batch
  cgraph also collapses inter-op scheduling. Graph capture (named ladder)
  is the lever that matches this; the GDN merge is a partial address.
- llama.cpp's host-side sampling is not a lever for us (we are on-device);
  it is a small cost on their side (a few tens of us per step).
- Radiance K=4 < K=3 (95.05 vs 142.31) independently confirms K=3 is the
  operating point (vLLM's own "multiple forwards on same MTP layer may
  result in lower acceptance rate" warning fired at K=4). We are at K=3.
- Radiance's dynamic draft gate at K=3 is capped at <=3 forwards per step
  (dive §4: "the dynamic controller is a gate, not an extension"), so its
  value here is skipping low-draft forwards plus the free-width n-gram
  fill; the int2 head is the bigger draft-phase lever.

## D. Stale premises (superseded by the t1585 traffic audit, 4cd7083)

| # | stale claim (dive) | superseded by |
|---|---|---|
| 1 | both: "our MTP ~32.7 ms per 2.15 tokens, ~66 t/s" | R4b 48.93 t/s, 65.2 ms/verify pair, 3.19 tok/pair |
| 2 | radiance: "2-4 re-forwarded passes, ~18 GiB, ~35 ms/step" | eliminated by tape; now 0.783 S=1/step (3.26 GiB) |
| 3 | radiance: "bf16 lm_head x3 = 6.1 GiB, ~11 ms/step" | 4-bit head: 3 x 0.533 = 1.60 GiB (1.97 incl. sidecar) |
| 4 | radiance §3: "per-step budget ~10-11 GiB" | 9.39 GiB/step; pre-tape 28.3 (+6.7 regime = the old 34) |
| 5 | radiance: "MTP layer ~0.47 GiB bf16" | 0.125 GiB sidecar (4-bit class, 1 GDN layer + norm) |
| 6 | llama.cpp: "2-4 extra re-forwards ... dominant term" | eliminated; residual gap is compute/schedule-bound |
| 7 | both: no per-phase runtime numbers (dive line 3-5) | OUR side now has phase walls; llama.cpp unmeasured (t1451) |
| 8 | "MTP layer 0.47 GiB" in both dives' GiB arithmetic | 0.125 GiB (item 5) — all 0.47-based arithmetic is stale |
| 9 | radiance: "4-bit target ~5.5 GiB" (bf16-head assumption) | measured 4.16 GiB incl. 4-bit head (audit) |

Unit mislabels (both dives; detail in E): "13.6 µs/token" and "11.5
µs/token" (1000x off — ms); "32.7 ms per 2.15-accepted-token step" and
"14.9 ms per 2.15-token step" (these are ms-per-2.15-tokens; per spec step
of 3.15 tokens: 48.0 vs 21.8 ms).

## E. Number audit

Reproduced from raw data (OK unless flagged):

- t1436: 144.3 = median(144.5, 144.3, 144.1) OK; verbose run 144.1 OK;
  AR 86.9 = median(86.9, 86.9, 86.7) OK; wall 5.033/6.831 s OK; 512/6.831 =
  74.95 and 512/5.033 = 101.73 OK; 161 spec steps, 346 accepted,
  mean 2.14907 OK (raw acc.sh output); 3.15 tok/step = 1 + 346/161 OK.
- t1439: AR 63.32 = (63.32 + 63.31)/2 OK; K=3 142.31 = (142.09 + 142.53)/2
  OK; K=4 95.05 = (94.59 + 95.51)/2 OK.
- Radiance dive: "15.8 ms" = 1000/63.32 OK (the one correctly-labelled ms in
  the table); "2.03 GiB" bf16 head = 248320 x 4096 x 2 B OK; "0.27 GiB"
  int2 head is radiance's own number (vs our measured 0.296 GiB — same
  class, 9% apart, OK); "2.2 GiB draft" = 3 x (0.47 + 0.27) OK;
  "7.5-8 GiB verify" = 5.5 + 2.03 + GDN OK (but built on the stale 5.5
  assumption, D#9).
- R4b anchor row: d1 = 118, d2 = 74, d3 = 35, tok_step = 2.410 OK (1 +
  227/161); d1-accept = 35 + 43 = 126 OK; 126/161 = 78.3% OK; D2PCT
  100 x 74/118 = 62.7 OK; D3PCT 100 x 35/74 = 47.3 OK; full-accept 35,
  d1-reject 43, d2-reject 44, d3-reject 39; tokens = 35 x 4 + 43 x 2 +
  44 x 3 + 39 x 4 = 514 OK; 514/48.93 = 10.51 s ~ 10.49 s wall OK.
- Deltas: 48.93/46.07 = +6.2% OK; 56.08/48.60 = +15.4% OK;
  74.32/71.12 = +4.5% OK.
- Traffic audit: 9.39 = 4.16 + 3.26 + 1.97 OK; 0.783 = 126/161 OK;
  9.39 - 3.26 = 6.13 (R5 end state) OK; phase walls 2516/64 = 39.3 and
  1709.2/64 = 26.7 OK; 6888 launches = 24 x 287 OK.

Findings (not reproducible / mislabelled):

1. Unit mislabels (both dives, confirmed 1000x): "13.6 µs/token" (73.45
   t/s) and "11.5 µs/token" (86.9 t/s) — these are ms (1000/73.45 = 13.61;
   1000/86.9 = 11.51).
2. Per-step vs per-token conflation: "32.7 ms per 2.15-accepted-token step"
   (ours) and "14.9 ms per 2.15-token step" (llama.cpp) are
   ms-per-2.15-tokens; per spec step (3.15 tokens) the true figures are
   48.0 vs 21.8 ms. The "32.7 vs 14.9" framing in both dives understates
   the per-step gap (it is 2.2x, not 2.2x-per-2.15-tokens vs 1.4x).
3. Radiance dive: "same acceptance profile (2.15 tok/step on both sides)"
   — the raw t1439 report contains NO acceptance measurement for radiance;
   2.15 is an unverified assumption imported from our side. The t1463 brief
   repeats it as a "known cost fact". Unsupported.
4. R4b 64v acceptance row (D1PCT 79.7 / D2PCT 56.2 / D3PCT 20.3 /
   tok_step 2.56) is internally inconsistent: 1 + (d1 + d2 + d3)/64 = 2.56
   requires d1 + d2 + d3 = 99.8, but the PCT chain gives 51.0 + 28.7 + 5.8
   = 85.5 (tok_step 2.34); and 50.02 t/s x 4.285 s = 214.3 tokens vs 227.8
   from the counters (~6% off). The anchor row (item above) IS consistent;
   the 64v row appears to be a transcription error. Impact limited: 64v is
   the phase-protocol arm, not the anchor. Raw logs for the 64v arm are not
   in the repo, so this cannot be adjudicated further — flag for the next
   run.
5. Radiance dive regime note: "33-35 t/s ~ 90+ ms/step at 2.5 tok/step" —
   1000/33 = 30.3 ms/token vs 90/2.5 = 36 ms/token, a ~20% internal
   spread in a two-word estimate. Minor.
6. Precedent: the D1 95->72.7 incident in this lane (unit misread) matches
   the pattern in findings 1-2; the R4b report's own anchor row
   (D1PCT 73.3 ON = 73.3 OFF) shows the correct reproduction.

## Missed-lever ranking (expected ms per verify-pair, anchor regime)

Estimates use the audit's effective-bandwidth method (65.2 ms/pair;
effective ~146 GB/s on the 9.39 GiB/step mix):

| # | lever | expected value | basis |
|---|---|---|---|
| 1 | R5 (kill S=1 re-read) | ~22-30 ms (3.26 GiB in the draft phase) | both refs confirm (B); 9.39 -> 6.13 GiB |
| 2 | int2 draft head | ~5-8 ms (0.71 GiB; radiance 4.2x faster) | kernel in vendor tree; wheel build + A/B |
| 3 | MTP graph capture | ~5-15 ms if launch-bound | anchor EAGER (C#2): parity with radiance, edge over llama.cpp |
| 4 | GDN 24-layer merge | ~2-5 ms (48 launches/forward removed) | radiance log; partial address of #3's overhead |
| 5 | dynamic gate + n-gram fill | ~0-3 ms (skips 2.09 GiB low-confidence drafts) | capped at <=3 forwards at K=3 |
| 6 | draft-KV trim | ~0 direct; correctness/robustness | keep the catch-up skip (C#5) |

## Updated ladder recommendation

R5 (unchanged, #1 — largest, and both references confirm the end state) ->
int2 draft head (promoted to #2: the kernel already exists in the vendor
tree on `mtp-r4b-fused-verify`; the remaining work is a wheel build + A/B,
so it is the cheapest win on the ladder) -> GDN 24-layer merge + MTP graph
capture (combined as one "schedule/launch reduction" step; they address the
same overhead, and the graph adds the structural edge over the eager
anchor) -> dynamic draft gate + n-gram fill -> draft-KV trim (hygiene,
last).

Two validation items before/with R5 A/B, in the same lock window: (a) the
llama.cpp per-phase capture the t1451 crash left undone (t_draft_us /
t_accept_us + metrics pre/post decode — this fills the A accounting hole on
their side); (b) re-capture the 64v acceptance counters (E#4) so the phase
protocol is adjudicated from raw logs, not the inconsistent report row.
