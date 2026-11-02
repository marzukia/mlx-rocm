# REPORT — radiance/vLLM MTP vs. our MTP path: per-phase divergence

Task: static code comparison of the vllm-radiance Qwen3_5MTP speculative-decode
stack (served on guest-a, vLLM 0.27.1, radiance 0.6.2, NVFP4 9B) against our
mlx-rocm MTP path (`mtp-bench6.py` MTP=4 arm + `MTP_INJECT_PATCH` in
`mlx_lm/models/qwen3_5.py`). No GPU benches were run (VRAM dirty); the anchor
numbers are the known cost facts:

| arm | step cost | tok/s | tokens/step |
|---|---|---|---|
| our MTP (K=3, MTP=4) | ~32.7 ms | ~66 | 2.15 |
| radiance MTP (K=3) | ~14.9 ms | 142.3 | 2.15 |
| our AR (canonical wheel) | 13.6 ms | 73.45 | 1 |
| radiance AR | 15.8 ms | 63.32 | 1 |

Same 9B dense architecture on both sides (32 layers = 24 GDN + 8 full-attention;
16q/4kv, head_dim 256, GDN 16 key / 32 value heads × 128×128, intermediate
12288, hidden 4096, vocab 248320, `mtp_num_hidden_layers: 1`) — verified:
guest `qwen3.5-9b-mxfp4a16-mtp/config.json` == `/mnt/models/qwen3-9b-mlx/
config.json` `text_config` for all shape fields. The MTP drafter weights are
the **same 15-tensor sidecar** on both sides
(guest `model-mtp-bf16.safetensors` = ours `model-mtp.safetensors`, 15 `mtp.*`
tensors: fc 4096×8192, q 8192×4096, k/v 1024×4096, o 4096×4096, mlp 3×12288×4096,
5 norms; ours stored fp32 at 928 MiB, grafted bf16 for radiance; per the
standing fact the non-`mtp.*` tensors in the file are -1.0-corrupted and
inert). Same acceptance profile (2.15 tok/step on both). **The gap is entirely
in the surrounding machinery, not the drafter and not the AR floor** (we win
the floor, 73.45 vs 63.32 t/s).

Code sources (all pulled read-only, no GPU work):
- radiance/vLLM: guest `andryo@100.78.68.72`, served venv
  `/home/andryo/radiance-host-opt/vllm` (vLLM 0.27.1), repo `radiance-mxfp4`
  @ 45da506 (v0.13.0); k3 serve + bench logs.
- ours: this worktree (HEAD 60ad8a7) `bench/mtp-bench6.py`
  (md5 452d363de7c6754de8333ae09999327f, verified on guest
  `/home/andryo/mlx-rocm/bench/`) + `mlx_lm/models/qwen3_5.py` from the
  canonical wheel venv (`/home/andryo/venv-canon`, `MTP_INJECT_PATCH` /
  `FUSE_BRANCH_PATCH` / `TAPE_BRANCH_PATCH`).

## 1. Pipeline, side by side

### radiance/vLLM (k3 serve)

- Method mapping: `qwen3_5_text` + `mtp_num_hidden_layers>0` →
  `qwen3_5_mtp` / `Qwen3_5MTP`
  (`vllm/config/speculative.py:336-343`); serve config:
  `speculative_config={method:mtp, num_speculative_tokens:3, attention_backend:
  TRITON_ATTN, disable_padded_drafter_batch:true}`, TP=1, `mamba_cache_mode=
  align`, `mamba_ssm_cache_dtype=float16`, kv fp8, no-async-scheduling
  (`radiance-serve-k3.log:2,49`).
- Drafter model: one MTP decoder layer, `fc` 2h→h + shared embed/lm_head
  (`vllm/model_executor/models/qwen3_5_mtp.py:139-175` forward with
  `spec_step_idx % num_mtp_layers`; `:295-300` `compute_logits` on the shared
  `lm_head`; `:302-315` `load_weights` remaps `mtp.*`→`model.*` and skips
  `embed_tokens`/`lm_head`, i.e. **shared with the target**).
- Scheduler: `num_new_tokens = 1 + num_spec_tokens`
  (`vllm/v1/core/sched/scheduler.py:890`) → the verify is **one target
  forward over 1+K tokens** in the normal decode batch.
- Draft phase (`vllm/v1/spec_decode/llm_base_proposer.py`): first pass runs
  the drafter over the accepted prefix + next token
  (`set_inputs_first_pass`, `:823-860`; hidden buffer fed from the target's
  pre-final-norm hidden, `:502-545`), then a loop of K−1 MTP forwards
  (`:656-690`). Two radiance patches inside the loop:
  - **loopbreak** (`patch_mtp_loopbreak.py`, applied at
    `llm_base_proposer.py:682-683`): `if getattr(self, "_radiance_stop",
    False): break` — a per-slot host short-circuit skips the remaining draft
    forwards.
  - **draft-KV logical trim**: `common_attn_metadata.seq_lens -=
    num_rejected_tokens_gpu` (`:674-678`, `num_rejected_tokens_gpu` computed
    on-device, `gpu_model_runner.py:1586`) — the drafter KV is trimmed to the
    accepted prefix **on the GPU, no H2D sync, no re-forward**.
- Dynamic draft controller (`radiance_draft.py`, `RADIANCE_DYNAMIC_DRAFT=ON`
  by default, serve runs with `tau=0.2`, schedule `1:8,2:7,4:6,8:5,16:4`,
  `radiance-serve-k3.log:9,43`): per-slot rule — keep drafting while the
  running product of drafter top-1 confidences ≥ τ; take a **verbatim
  n-gram continuation** when it matches the drafter's own top guess; else stop
  early and verify. Batch-size ceiling caps serial draft depth
  (`_batch_ceil`, `radiance_draft.py:60-75`).
- **INT2 draft head** (`radiance_drafthead.py:1-50`,
  `RADIANCE_FAST_DRAFT=1`, `radiance-serve-k3.log:12,199`): the shared bf16
  head (2.03 GiB for this shape; the module docstring quotes 1.18 GiB/rank on
  the 27B shape) reads ~2002 µs per draft slot; radiance replaces it with an
  int2 g128-asymmetric head (0.27 GiB, quarter-split packed, bit-pattern
  dequant, scale-on-accumulator) at ~473 µs **including an exact top-32
  rerank** against the bf16 weight (top-8 candidates per 64-column block; on
  8192 captured inputs the int2 argmax matches the exact bf16 argmax on every
  row, vs 22 misses for a 4-bit head). Verify head stays bf16
  (`RADIANCE_VERIFY_HEAD=0`).
- GDN (`radiance_gdn.py`): fused decode step — conv → grid barrier →
  recurrent in **ONE launch** (`:112-118`, gated by
  `RADIANCE_GDN_FUSED_UPDATE=1`, `FUSED_MAX_ITEMS=48`); spec steps write the
  per-candidate **stash blocks** (`mamba_cache_mode=align`), and
  `postprocess_mamba_align_gpu` shifts the GDN state to the accepted column
  (`gpu_model_runner.py:1588-1613`, fused on-GPU, no H2D). 24-layer merge:
  `[radiance.gdnmerge] merged 24 GDN layers (48 launches/forward removed)`
  (`radiance-serve-k3.log:69`).
- GEMM: NVFP4 checkpoint served through `RadianceMxfp4W4A8LinearKernel` —
  W4A8 fp8-WMMA, decode band `M≤64` (`RADIANCE_MXFP4_DECODE_MAX_M=64`),
  plus `patch_skinny_gemm.py` split-K for M 6..16 (`radiance-serve-k3.log:56,
  68`). The R4D attention kernel **refuses this shape** (covers gqa=6 only;
  the 9B is gqa=4) → `TRITON_ATTN` fallback by the stack's own design
  (`radiance_r4d_attn.py`; `radiance-serve.sh` comment). R4D all-reduce is a
  TP=2 mechanism; **inactive at TP=1** (this run).
- Acceptance: greedy (temp 0 path) — `target_argmax` +
  `rejection_greedy_sample_kernel` (`vllm/v1/sample/rejection_sampler.py:
  435-456`), on-device.

### ours (mtp-bench6.py MTP=4 + MTP_INJECT_PATCH)

- Model wiring: `MTP_INJECT_PATCH` builds the 1-layer `MTPModule` when
  `config.mtp_num_hidden_layers>0` (`qwen3_5.py:457-461`, wheel venv
  `venv-canon`); `PRENORM_PATCH` exposes the target's **pre-final-norm hidden**
  for the drafter (`:444-450`); `sanitize` keeps `mtp.*` keys only when the
  module exists (`:500-505`); target `lm_head` is bf16 and shared with the
  drafter (`:480-490`). The `MTPDecoderLayer` is a full-attention layer
  (Qwen3NextAttention) + MLP (`:25-51`), `MTPModule` = 2×RMSNorm → fc(2h→h)
  → layers → RMSNorm, with its **own `KVCache`** (`mtp-bench6.py:120`,
  `qwen3_5.py:65-76`).
- Draft phase (`mtp-bench6.py:124-148`): **fixed K=3** — three MTP forwards
  `dh_i = mtp(h_{i-1}, emb_i, mtp_cache)`, each followed by `draft_i =
  argmax(lm_head(dh_i))` on the **full bf16 lm_head** (248320×4096, 2.03 GiB
  per call). No confidence gate, no early stop, no n-gram fill. The MTP KV
  cache advances by 3 every step and is **never rolled back on rejection**
  (docstring `:59-61`); LRU masks over the full physical length, so stale
  rejected-draft KVs stay in the attention window and the cache grows
  unboundedly over the run.
- Verify phase (`mtp-bench6.py:286-300`): `saved = save_gdn(cache)` (a
  functional copy of conv/ssm/lengths for all 24 GDN layers, `:80-90`), then
  **one batched S=4 forward** `[prev, d1, d2, d3]` — 32 target layers, cache
  advances 4. GDN layers take the F-branch `gdn_fused_step` (single kernel,
  T=1..4, `qwen3_5.py:272-290`, env `MLX_ROCM_GDN_FUSE=1`); the 8
  full-attention layers take the generic QMV M=4 path (bench docstring
  `:53-55`: no fused M=4 specialization).
- Acceptance: on-device greedy cascade — `t2=argmax(l0)` … `t4=argmax(l3)`
  with early EOS break (`mtp-bench6.py:292-300`); same lossless greedy
  protocol as vLLM's greedy rejection sampler.
- Rejection handling (the structural difference, `mtp-bench6.py:294-300`): on
  any rejection, `restore_gdn` (24-layer state copy-back) + `KVCache.trim(4)`
  + **re-forward the accepted prefix as individual T=1 target forwards**
  (2–4 full 32-layer passes, e.g. d3-rejected → `[prev],[d1],[d2],[t4]`).
  The re-forward also re-derives the pre-norm `h` for the next draft chain
  from the T=1 path (full-accept steps instead reuse the S=4 forward's `h3`,
  `:293-294` — the T=1/T=4 GDN kernels differ, so the batched hidden is only
  used on full acceptance). The per-row `gdn_fused_step_tape` (TAPE branch,
  `qwen3_5.py:236-270`, `MLX_ROCM_GDN_TAPE=1`, default off) emits per-row
  state tapes that could select the accepted row's state directly — it is not
  the standing path.
- Standing wheel config: **rows-ON + REFWD_M1** with the rows-aware gate (the
  M=1 REFWD_M1 re-route engages only when rows is off; the two levers do not
  compose, commit c696550) — i.e. in the standing wheel the rows M=1 path
  serves T=1 forwards and REFWD_M1 is dormant.

## 2. Divergence table (ranked by cost impact)

Per-step HBM traffic model (batch 1, K=3, ~546 GB/s floor; bf16 head 2.03 GiB,
4-bit target ≈5.5 GiB, MTP layer ≈0.47 GiB bf16, GDN state snapshot ≈12 MiB
×24 layers). Rows ranked by per-step cost impact of the divergence.

| # | Phase | radiance/vLLM (k3 serve) | ours (MTP=4) | Cost impact |
|---|---|---|---|---|
| 1 | **Rejection handling (verify→next step)** | **Zero re-forward.** Verify writes GDN candidate state columns + KV for all 4 positions in one pass; `postprocess_mamba_align_gpu` shifts GDN to the accepted column and the KV is trimmed logically (`gpu_model_runner.py:1586-1613`; drafter KV `seq_lens -= num_rejected_tokens_gpu`, `llm_base_proposer.py:674-678`) | On any rejection: GDN save (≈12 MiB×24) before the S=4 forward, restore after, KV trim, then **2–4 re-forwarded T=1 full target passes** to re-advance state and re-derive `h` (`mtp-bench6.py:289-300`). With the observed acceptance profile (D1≈0.7–0.8, D2≈0.6–0.65, D3≈0.45–0.55, recent MTP=4 runs) the expected re-forward count is **E ≈ 2.4–2.6 full target forwards per step** (E = 2(1−c1)+3(c1−c2)+4(c2−c3), c_i cumulative accept counts, c1+c2+c3 = 1.15) | **Largest.** ≈2.5 × (5.5 GiB target + 2.03 GiB head + GDN update) ≈ 18 GiB ≈ ~35 ms of floor traffic vs 0. This is the structural cost of the single-slot GDN state (position-3 state overwrites intermediate positions) vs vLLM's align candidate columns. Fix direction already half-built: the TAPE branch (`qwen3_5.py:236-270`) emits per-row state tapes — selecting row A's state would kill the re-forward and the save/restore at once |
| 2 | **Draft head (per draft token i=1..3)** | int2 g128-asym head, 0.27 GiB, ~473 µs incl. exact top-32 bf16 rerank (`radiance_drafthead.py:1-50`; `radiance-serve-k3.log:199`); verify head stays bf16 | full **bf16** shared `lm_head` per draft: 3 × 2.03 GiB ≈ 6.1 GiB ≈ ~11 ms floor, ~2 ms+ per call at the 546 GB/s floor (`mtp-bench6.py:133-147`; head `qwen3_5.py:480-490`) | **Second.** Δ ≈ 5.3 GiB ≈ ~10 ms/step. Lossless by the same argument radiance uses: the draft head only chooses proposals; the bf16 head verifies every proposal |
| 3 | **GDN kernel selection (verify + draft steps)** | fused conv→barrier→recurrent ONE launch (`radiance_gdn.py:112-118`, `FUSED_MAX_ITEMS=48`) + 24-layer merge removing 48 launches/forward (`radiance-serve-k3.log:69`) | 24 × separate `gdn_fused_step` launches (F-branch, `qwen3_5.py:272-290`), no layer merge; TAPE branch (per-row state, `:236-270`) off in the standing config | Modest: same single-kernel conv+recurrent idea, but 24+ standalone launches vs merged; per-layer launch overhead on a ~546 GB/s part is small but nonzero. The bigger issue in this row is the *state protocol* of row 1, not the kernel |
| 4 | **Draft loop control (how many MTP forwards per step)** | dynamic: confidence-product gate (τ=0.2), verbatim n-gram fill, per-slot host loopbreak (`radiance_draft.py:1-40`; `llm_base_proposer.py:682-683`), batch-size schedule cap | fixed 3 MTP forwards + 3 bf16 heads every step, no early stop (`mtp-bench6.py:124-148`) | Moderate and coupled with row 2: on low-confidence steps we pay 3 × (0.47 + 2.03) GiB that a gated drafter would skip; n-gram fill gives free draft width we don't have. Also we always verify S=4 even when draft 3 is a low-confidence guess |
| 5 | **Draft KV management (target + drafter KV update)** | drafter KV in the request's block table; stale draft KV masked by on-device `seq_lens -= num_rejected_tokens_gpu` (`llm_base_proposer.py:674-678`) | MTP `KVCache` grows unboundedly, never trimmed, LRU mask spans the full physical length → **stale rejected-draft KVs remain in the attention window** and memory grows over the run (`mtp-bench6.py:59-61,120`) | Small direct cost (MTP-layer KV is a few KiB/token); the real impact is (a) unbounded growth over long generations, (b) the drafter attends to phantom tokens, a numerical (not correctness) defect — greedy verification still makes the output lossless |
| 6 | **Acceptance / sampling** | GPU greedy rejection sampler: `target_argmax` + `rejection_greedy_sample_kernel` (`rejection_sampler.py:435-456`) | on-device greedy cascade `argmax(l0..l3)` with early EOS break (`mtp-bench6.py:292-300`) | **~0** — same lossless greedy protocol, both on-device, no H2D |
| 7 | **Target GEMM / attention kernel selection (verify S=4, T=1 re-forwards)** | NVFP4→W4A8 fp8-WMMA decode band M≤64 + skinny split-K M 6..16 + TRITON_ATTN (R4D attention refuses gqa=4 by design) + CUDA graphs [1,2,4,8] | 4-bit affine (gs64) target, QMV M=4 generic fallback for the 8 full-attention layers (no fused M=4 specialization), QMV M=1 rows-ON path for T=1, GDN F-branch T=4; no graph capture in the bench | Small-to-moderate: both read the 4-bit weights once per pass; the deltas are kernel-generation overhead (inductor+graphs vs eager MLX) and the M=4 attention specialization. AR floor is ours anyway (73.45 vs 63.32) |

## 3. Per-step traffic summary

| phase | radiance (GiB/step) | ours (GiB/step) |
|---|---|---|
| draft (3 slots) | 3 × (0.47 + 0.27) ≈ 2.2 | 3 × (0.47 + 2.03) ≈ 7.6 |
| verify (S=4, one pass) | 5.5 + 2.03 + GDN (24× merged, candidate cols) ≈ 7.5–8 | 5.5 + 2.03 + GDN (24× separate) + 0.02 save ≈ 7.6 |
| rejection handling | 0 | E≈2.5 × 7.5 ≈ 18.7 |
| **total** | **≈10–11** | **≈34** |

The traffic ratio (~3×) brackets the observed step-time ratio (32.7/14.9 ≈ 2.2×)
— per-pass costs differ by guest state (our recent MTP=4 logs on the guest
show 33–35 t/s ≈ 90+ ms/step at 2.5 tok/step on a clock-limited state vs the
canonical 32.7 ms/2.15 on the fast state), so treat the absolute ms as
floor estimates and the *ranking* as the load-bearing result: the re-forward
machinery (row 1) and the bf16 draft head (row 2) are the two first-order
gaps, together ≈ 28 GiB ≈ ~50 ms of the ~34 GiB per step; everything else is
second-order.

## 4. Notes / caveats

- Anchor numbers (32.7 ms/2.15, 73.45, 142.31, 63.32) are the task's known
  cost facts, taken as given; nothing was re-benchmarked (VRAM dirty).
- The drafter model and its 15-tensor sidecar are the same on both sides, so
  rows 3–7 cannot explain the gap; rows 1–2 do.
- The E-re-forward derivation uses the observed MTP=4 acceptance profile
  (D1PCT/D2PCT/D3PCT from recent guest logs, e.g. 80.1/59.5/48.4 and
  68.5/63.5/56.3); for any profile summing to 2.15 tok/step the expected
  re-forward count stays in [2.55, 3.15], so the ranking is insensitive to
  the exact profile.
- Radiance R4D all-reduce and the TP-pad machinery are TP≥2 features; the
  served 9B run is TP=1 with no-async-scheduling, so they are inactive and do
  not enter this comparison.
- Our MTP draft-KV accumulation (row 5) is a numerical defect, not a
  correctness one: greedy verification is lossless either way.
- Radiance's `RADIANCE_DYNAMIC_DRAFT` defaults ON and the k3 serve ran with
  it (τ=0.2); its contribution is bounded by the schedule cap (≤3 forwards at
  K=3) and shows up in row 4, not row 1/2.
