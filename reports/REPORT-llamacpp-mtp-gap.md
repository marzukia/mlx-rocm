# REPORT-llamacpp-mtp-gap — static gap analysis: llama.cpp draft-mtp vs our mlx-rocm MTP=4

Date: 2026-11-02 (task t1462). **Static code comparison only — no GPU benches were run**
(the t1451 per-phase profile run crashed on HBM-full at load, so **no per-phase runtime
numbers exist yet** for either side; §5 names the measurements a follow-up must take).

## 1. Pinned provenance

| Side | Artifact | Pin |
|---|---|---|
| llama.cpp | reference tree on guest `100.78.68.72:~/work-t1436/llama.cpp` | commit `29c720eaf72fcf8e4e61a56a88b994d92668261d` |
| ours | `bench/mtp-bench6.py` (repo) | md5 `452d363de7c6754de8333ae09999327f` |
| ours | `mlx_lm/models/qwen3_5.py` (venv-canon) | md5 `7763fbbb498129ac2750811be9154b5b` |
| ours | `mlx_lm/models/qwen3_next.py` (venv-canon) | md5 `e0d911fbcaebd3814430a14bd4a647af` |
| ours | `mlx_lm/utils.py` (venv-canon) | md5 `68eb4485fadaf3bc9198ea5b079a0e95` |
| ours | `mlx/lib/libmlx.so` (venv-canon, 0.32.3.dev20261102+75ca881) | md5 `b1de59fda0e4fc607d620e287deeaaf8` |
| ours | `vendor-canon/mlx/backend/rocm/quantized/qmm.hip` | md5 `1ca31eea8ee07bb82370678cfa40010a` |

Citation convention: llama.cpp paths are relative to the llama.cpp root at the pinned
commit; our Python paths are in the venv-canon site-packages (as pinned above); our C++
paths are in the vendor-canon tree.

Known end-to-end facts carried in from prior tasks (NOT re-measured here): our MTP=4
verify ≈ 32.7 ms per 2.15-accepted-token step; our AR (MLX_MTP=0) 13.6 µs/token
(73.45 t/s); llama.cpp verify ≈ 14.9 ms per 2.15-token step, AR 11.5 µs/token
(86.9 t/s), draft-mtp end-to-end 144.3 t/s. Standard config for both: 512-token prompt,
greedy, ctx 1024, all-GPU; ours with rows-ON (`MLX_ROCM_QMV_ROWS=1`),
`MLX_ROCM_REFWD_M1=1`, `MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1`.

## 2. Protocol as executed (one step, n_max = 3)

**llama.cpp (draft-mtp, server path, RS-type memory — the anchor config):**
1. pre-decode: trim draft ctx to checkpoint position, then 3× draft forward — one MTP
   block + shared `lm_head` per token, growing draft KV (re-add only the new token);
   greedy top-1 draft sampling from host-side sampler;
   (`common/speculative.cpp:1751-1917`, trim `tools/server/server-context.cpp:3931-3934`).
2. verify: single 4-token target forward — 32 trunk layers (24 GDN recurrent + 8 full
   attention), one `llama_process`;
   (`tools/server/server-context.cpp:601-641`, trunk `src/models/qwen35.cpp:131-346`).
3. catch-up: one MTP-block 4-token forward (+ shared `lm_head`) right after the verify
   batch, re-anchored to the target's pre-norm h rows, so the draft KV stays aligned
   with the accepted prefix
   (`tools/server/server-context.cpp:4758` → `common/speculative.cpp:1637-1735`).
4. accept: greedy per-position target-sample match, break on first mismatch, bonus
   token if all 3 matched (`common/sampling.cpp:690-720`); partial acceptance →
   `mem.seq_rm` on both contexts — recurrent state rollback is an O(1) snapshot-index
   op, KV cells just invalidated, **no re-forward**
   (`tools/server/server-context.cpp:4930-5033` → `src/llama-memory-recurrent.cpp:166-216`).

**Ours (mtp-bench6, MLX_MTP=4):**
1. draft: 3× `lm.mtp(h, emb, mtp_cache)` + 3× `lm.lm_head(dh)` + 3×
   `embed_tokens` + 3× on-device `mx.argmax` (`bench/mtp-bench6.py:136-148`); h stays
   on-device. MTP module = 1 layer: prenorm(h)+prenorm(emb) → concat → fc → gated
   attention (Qwen3NextAttention) + residual → postnorm → MLP → norm
   (`qwen3_5.py:53-80`, `qwen3_5.py:25-45`, `qwen3_next.py:74-151`).
2. verify: `save_gdn` (zero-copy object-ref snapshot of 24 GDN states), single S=4
   batched target forward (`bench/mtp-bench6.py:290-320`; GDN via
   `gdn_fused_step_tape` T≤4 `qwen3_5.py:236-271`; full attention via SDPA
   `qwen3_next.py:146`), 4× on-device `mx.argmax`.
3. accept: greedy per-position argmax match, break on mismatch, bonus if all 3
   (`bench/mtp-bench6.py:306-336`). Partial rejection → `restore_gdn` (object-ref
   swap, zero-copy), `KVCache.trim`, then **2–4 extra T=1 target re-forwards**
   re-running the accepted prefix + bonus to re-derive the hidden state
   (`bench/mtp-bench6.py:320-335`).

## 3. Divergence table — ranked by static cost impact (decode step)

| # | Phase | llama.cpp (29c720e) | Ours (pinned above) | Divergence | Static cost impact (decode step) |
|---|---|---|---|---|---|
| 1 | **Rollback on partial acceptance** | O(1): `seq_rm` → recurrent `set_rs_idx` snapshot-plane index op; KV cells invalidated; zero re-forward (`server-context.cpp:5033`, `llama-memory-recurrent.cpp:184,200-216`; planes pre-allocated `(1+n_rs_seq)` at `llama-memory-recurrent.cpp:102`, `llama-model.cpp:3226-3236`) | 2–4 extra **full target T=1 forwards** (32 trunk layers + `lm_head` each) + state save/restore bookkeeping (`mtp-bench6.py:320-335`, `save/restore_gdn` `:80-92`) | **Structural.** With 2.15/3 mean acceptance a substantial fraction of steps enter the partial branch, each costing 2–4 full extra AR forwards; llama pays ~nothing. This is the single largest mechanism-level gap and the most likely dominant term in the 32.7 vs 14.9 ms/step delta |
| 2 | **MTP draft-KV alignment** | Trim draft KV to checkpoint pos every step (`server-context.cpp:3931-3934`) + post-verify 4-token MTP catch-up incl. `lm_head` (`server-context.cpp:4758`, `speculative.cpp:1637-1735`); draft attention always sees the true prefix (full prompt catch-up at prefill) | `mtp_cache` is **never trimmed or catch-up'd** — stale draft rows persist after rejection (bench header, `mtp-bench6.py:34-50`); first draft sees an empty MTP prefix (context only via h) | Two-sided: we skip the catch-up forward (~1/32 of a trunk + one M=4 `lm_head` QMM) but draft quality/acceptance may degrade vs their aligned KV. Net effect on tokens/s is unknown — must be measured (acceptance rates on identical prompts). Also a correctness-adjacent divergence: our stale rows are attended by the next draft chain |
| 3 | **`lm_head` M=4 QMM dispatch (verify)** | per-row: 4 separate single-row passes over the ~620 MB q4 weight (4× weight traffic) | rows-msg kernel: one weight pass covering all 4 rows (`qmm.hip:4170-4217`, `rows_ok = M∈[2,4] ∧ N≥16384`; `lm_head` N=248320 qualifies) | **Favors us**: one vs four full weight reads per verify step on the largest weight tensor in the model |
| 4 | **GDN state management** | `(1+3)`-plane snapshot state in HBM (4× footprint, `llama-memory-recurrent.cpp:102`); per-cell snapshot write on every forward (4 state writes per verify ubatch); rollback = index op | single functional state array per forward (new output array, `mlx/fast.cpp:1436-1505`); per-step save = 24 object-ref snapshots (zero-copy, valid because the kernel allocates fresh state buffers) | **Favors us** on HBM traffic (1 vs 4 state writes per verify) and enables the (expensive, #1) rollback; theirs costs 4× the static state footprint |
| 5 | **Draft h-transport + sampling** | per draft token: device→host h copy (`llama_get_embeddings_nextn_ith`, `speculative.cpp:1837`) + host-side sampler chain + host argmax over 248 320 logits, then host→device re-feed | all on-device: h stays in HBM, `mx.argmax` on GPU (`mtp-bench6.py:141-148`) | Favors us; small (a few tens of µs of D2H/H2D + CPU sampler per 3 draft tokens) |
| 6 | **MTP module weights** | MTP block quantized with the model (4-bit), shared 4-bit `lm_head` | MTP module bf16 sidecar (`model-mtp.safetensors`, 15 tensors, loaded via the `model*.safetensors` glob merge, `utils.py:449-456`, keys kept at `qwen3_5.py:502-504`); shared `lm_head` 4-bit | Favors us slightly (less weight traffic in the 3 draft forwards + any catch-up we add); 15 tensors: fc + attn q/k/v/o + mlp gate/up/down + norms + shared head-norm |
| 7 | **Prefill phase** | MTP catch-up also runs over every prompt batch → one M=512 MTP-block + `lm_head` M=512 forward at prefill (`server-context.cpp:4758`, fires post every target batch incl. prompt) | no MTP work at prefill; `mtp_cache` starts empty | Favors us (one-time, not in the decode step), but couples with #2: their first draft attends the full prompt prefix, ours attends none |
| 8 | **Verify GDN kernel** | `ggml_gated_delta_net` in-place per-token sequential recurrence, 24 GDN layers process the 4 positions one at a time in-kernel (`qwen35.cpp:131-346`) | `gdn_fused_step_tape` single T=4 kernel call, state updated in-loop, 24 layers (`qwen3_5.py:236-271`) | Structurally equivalent (one in-place recurrence over the 4 positions); kernel micro-architecture differs — timing must be measured, not inferred |
| 9 | **Acceptance logic** | greedy per-position target-sample match, break on first mismatch, bonus if all 3, EOG guard (`sampling.cpp:690-720`) | greedy per-position argmax match, break on mismatch/EOS, bonus if all 3 (`mtp-bench6.py:306-336`) | **Equivalent** — no divergence (modulo #2's effect on draft quality) |

## 4. Confirmed-equivalent (do not re-test in follow-up)

- MTP module architecture: concat(e_norm, h_norm) → fc → gated attention
  (q*2+gate split, q/k norms, rope, sigmoid gate) → residual → postnorm → SiLU MLP →
  residual → shared-head-norm → shared `lm_head` — identical dataflow on both sides
  (`qwen35.cpp:494-663` vs `qwen3_5.py:25-80` + `qwen3_next.py:74-151`).
- h semantics: draft1 from target pre-final-norm h (ours `PRENORM_PATCH`
  `qwen3_5.py:444-451`; theirs `embd_state` pairing, `speculative.cpp:1678-1694`);
  draft2/3 from the MTP module's own post-norm output before the head
  (`speculative.cpp:1837` vs the `dh` return of `MTPModule.__call__` `qwen3_5.py:65-79`).
- Draft ctx memory: 1-layer plain attention KV on both sides (theirs filtered to
  MTP-only layers, `llama-model.cpp:3299-3305`; ours a single `KVCache`,
  `mtp-bench6.py:107`).
- Target attention-KV rollback: `KVCache.trim` (ours) vs `seq_rm` cell invalidation
  (theirs) — both effectively O(1) metadata ops.

## 5. Runtime measurements a follow-up MUST take (t1451 profile crashed — HBM full at load)

The 32.7 vs 14.9 ms/step delta is currently a single unexplained number. To attribute
it, a follow-up (GPU, guest-a, same pinned wheel/model/sidecar as §1) must measure,
per side, **per decode step**:

1. **Our side — phase split of the MTP=4 step** (the t1451 crash was HBM-full at
   *load*; rerun with a clean GPU — check `rocm-smi --showmeminfo vram` pre-run and
   release the `gpu-bench.lock` — and if load OOMs, drop ctx to 512 for the profile
   arm only and note the deviation):
   - (a) draft phase wall: 3× `mtp()` + 3× `lm_head` (M=1, REFWD_M1 LX path) + 3×
     argmax;
   - (b) verify S=4 wall, per-op: `gdn_fused_step_tape` ×24 (GDN_TAPE_CALLS counter
     already exists, `qwen3_5.py:81`), SDPA S=4 ×8, QMM dispatch (rows-msg vs per-row
     at N≥16384, `qmm.hip:4170-4217`);
   - (c) **branch frequency + per-branch cost**: counts of full-accept / d1-reject /
     d2-reject / d3-reject steps (D1PCT/D2PCT/D3PCT already printed) and the wall of
     the 2–4 extra T=1 re-forwards in each partial branch — this isolates divergence #1
     from #3/#8;
   - (d) `save_gdn`/`restore_gdn` + `KVCache.trim` cost (expected ~0; confirm).
2. **llama.cpp side — same phase split** with its built-in timers:
   `common_speculative` prints `t_draft_us`/`t_accept_us`
   (`speculative.cpp:3040,3068`, `common_speculative_print_stats` `:3126`); server
   per-decode metrics (`metrics_pre_decode`/`metrics_post_decode`,
   `server-context.cpp:5099-5113`) for the verify; and a manual split of the catch-up
   (`speculative.cpp:1637-1735`) if the built-in timers fold it into draft time.
3. **Acceptance parity**: D1PCT/D2PCT/D3PCT + mean tokens/step on **identical
   prompts** (same 512 prompt, greedy, both stacks) — this is the only way to
   separate divergence #2 (draft quality from stale vs aligned MTP KV) from the
   mechanism costs (#1, #3, #8). If our acceptance rate is materially lower, the
   144.3 vs our end-to-end gap is partly quality, not speed.
4. **Optional ablation (decisive for #2)**: our bench with `mtp_cache.trim(n)` on
   rejection + a 4-token MTP catch-up after verify (mirroring
   `server-context.cpp:3931-3934,4758`) — measure acceptance-rate delta and the
   catch-up's per-step cost.
5. **HBM traffic spot-check** (only if (1)–(3) leave the gap unexplained): GDN state
   write traffic 1-plane (ours) vs 4-plane (theirs) and `lm_head` weight traffic
   (rows single-pass vs per-row ×4) via `rocprof` on a 32-step window.

No GPU benches were performed for this report; every cost claim in §3 is a static
(code-structure / memory-traffic) inference and is flagged as such.
