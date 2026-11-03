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

| our phase | llama.cpp analogue | status |
|---|---|---|
| 3x draft (39.3 ms, 1.97 GiB) | 3x MTP-layer forward + 4-bit shared head + host-side greedy sampling (a few tens of us per 3 draft tokens, per dive line 5) | we carry the draft cost on-device; llama's sampling is host-side (D2H + argmax x4 + draft sampler) — not a lever for us, we are already on-device |
| verify S=4 (26.7 ms counted in commit; 4.16 GiB) | one `llama_process` per S=4 batch = ONE submission of a 32-layer compute graph | we do 24 GDN launches + 8 SDPA + QMM dispatch per forward (24 GDN-layer launch merge on the named ladder addresses the launch count) |
| correction S=1 (0.4 ms wall, 3.26 GiB re-read @0.783) | none: O(1) snapshot rollback + correction token folded into next verify (see B) | R5 target |
| commit (tape-row select) | O(1) `seq_rm` via per-token snapshot index (see B) | par |

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
