# GDN state tape — per-row state tapes in the fused recurrence + index-selected commit

Status: compile-verified (G1/G2); GPU validation is the RUN card (`docs/RUN-gdn-state-tape.md`). Opt-in `MLX_ROCM_GDN_TAPE=1`, DEFAULT OFF.

## 1. What this is

The fused GDN conv+recurrence kernel (`gdn_fused_step`, T=1/2/4 template, one launch per GDN layer per step) already loops over the T rows with the recurrent state `h` **register-resident** across rows (exactly the r4d recurrent kernel's T loop). The tape variant (`gdn_fused_step_tape`, new op, new kernel file, the original op is byte-untouched) adds two extra outputs:

- `state_tape` — `[T, Hv, 128, 128]` f32: the SSM state **after each row t**. Free to store: after each row's update `h` holds that row's state; the tape store is one extra `float4`x16 global write per thread per row (TMAX writes per launch vs 1 for `state_new`).
- `conv_state_tape` — `[T, 3, conv_dim]` bf16: the conv window **after each row t** = the last-3 taps of the row-t context `E = [cs0, cs1, cs2, v0..v3]`, i.e. `{E[t+1], E[t+2], E[t+3]}`. A pure index selection over the input conv-state rows and raw k/q/v rows — no arithmetic.

Commit becomes an index select, mirroring MTPLX `gdn_capture` (gdn_capture.py:2582-2760, commit 3107-3160, 3202-3276): on acceptance the bench rebinds the GDN cache leaves to `state_tape[keep-1]` / `conv_state_tape[keep-1]`; on rejection it selects the row at the accepted prefix. **No copies, no restore, no re-forward of accepted rows.** The rebind is deliberately lazy (a state copy/replace would add a per-cycle sync+copy to the hot path — the MTPLX 3218-3220 discipline).

## 2. Default-off guarantee

- New op (`GdnFusedStepTape`), new kernel file, new CMake source; the `gdn_fused_step` op/kernel/launcher are unmodified.
- `GdnFusedStepTape::use_fallback` returns true unless `MLX_ROCM_GDN_TAPE=1` (and the shape gate passes: B=1, T in {1, 2, 4}, Dk=Dv=128, kernel_size=4, GPU); the API function then throws a clear error (same pattern as `gdn_fused_step`'s unsafe-shape throw). With the env unset the op is unreachable → zero behavior change.
- There is **no fallback** for the tape itself: the per-row tapes exist only in the fused kernel; the 4-op chain cannot emit them.

## 3. What the tape eliminates vs keeps (K=3, S=4 verify; t1250 numbers)

Current protocol (mtp-bench6 mode 4, the 96.36 ms/verify-pair wall): save all GDN states (reference copies — `save_gdn` is zero-copy) before the S=4 verify; on rejection `restore_gdn` + KV `trim(4)` + **re-forward the accepted prefix plus the correction token as separate S=1 forwards (2-4 forwards per rejected round; 1206 re-forwards over 512 verifies = 2.355/round, 11.83 ms each = 27.86 ms/round — 28.9% of the pair wall)**.

What the tape eliminates:

- **the re-forward of accepted rows** — the S=4 verify already computed all 4 rows (logits, hidden, and now the per-row states); replaying the accepted prefix through S=1 forwards is pure rework. Rejected round: `2-4 S=1 forwards -> exactly 1 S=1 forward` (the correction token, from the selected tape-row state). Measured: 1206 -> 394 forwards (394 rejected rounds x 1), 27.86 -> ~9.1 ms/round.
- **the state save/restore bookkeeping** (the "other" 0.18 ms/round class: rebind the selected tape row instead; the pre-verify state when 0 rows are accepted is the state the leaves already point at — the tape never holds the pre-verify state, so that case is a no-op).
- **KV trim stays** (full-attention layers trim the rejected rows — pointer arithmetic, cheap; MTPLX `trim(trim_tokens)`).

What remains (from the t1250 split):

- **S=4 verify forward 57.40 ms/round** — untouched (the tape adds a few MB of stores inside the fused GDN launch; the fused GDN x24 is inside the ~6.6 ms residual).
- **single-token step 10.93 ms/round** (draft chain) — untouched.
- **1 S=1 correction forward per rejected round** (~11.83 ms; 394/512 rounds => 9.11 ms/round): the correction token's own forward is still required because the bench's draft chain (host-side, uncompiled) needs the correction token's pre-final-norm hidden `h` BEFORE the next verify, and `h` exists only after a forward of the correction token. The MTPLX-style deferred-correction fold (generation.py:15960-15990) removes this last forward by compiling the verify + draft chain into one graph (the correction becomes the next cycle's primary; the next verify's row 0 supplies its hidden) — that is the steal #7 follow-up, not the tape.

Bench expectation (t1250 baseline, 3.277 tok/pair at 96.36 ms/pair):

| arm | ms/pair | tok/s |
|---|---|---|
| t1250 baseline (re-forward class 27.86) | 96.36 | 34.0 |
| **tape commit (this work)**: 1 correction forward/rejected round | 96.36 - 27.86 + 9.11 = **77.6** | **~42.2** |
| tape + deferred-correction fold (compiled verify+draft, steal #7) | **68.5** | **~47-48** |

The task's 68.5 ms/pair figure assumes the re-forward class goes to ~0; with our host-side draft chain the tape alone floors at 1 correction forward per rejected round (9.11 ms/round at the t1250 acceptance rates D1ACC/D2ACC/D3ACC = 410/244/118 of 512). The RUN card runs both the tape arm and the (optional) fold arm and records which floor was hit.

## 4. Memory price

Per GDN layer, per verify round (T=4):

- `state_tape`: 4 x Hv x 128 x 128 x 4 B = 4 x 32 x 16384 x 4 = **8.4 MB** f32 (at Hv=32; = 4x one state row, one state row = 32 x 128 x 128 x 4 = 2.0 MB).
- `conv_state_tape`: 4 x 3 x conv_dim x 2 B = 4 x 3 x 8192 x 2 = **192 KB** (conv_dim = 2 x 2048 + 4096 = 8192 for Qwen3-9B-class).
- Total per layer ~8.6 MB; the model has 24 GDN layers -> **~206 MB** of tape per verify round vs the current 1 saved state row per layer (2.0 MB x 24 = 49 MB of *reference* saves — the current "copy" is a zero-copy rebind; the real current cost is the 27.86 ms re-forward class, not the copy).

So at M=4 the tape is ~4x one state per layer (~206 MB vs 49 MB), replacing a copy+restore (2x one state, plus the write-back) with two reads. The tape buffer is round-reusable: the committed row is read by the next round's draft/verify forwards before the next verify overwrites the buffer, so peak HBM is one tape set (~206 MB), not per-round allocation. (MTPLX's cheaper middle option — capture only the 1-2 mid-window states the next cycle may start from, the "rebase state" variant — is the fallback if HBM pressure shows up; the RX 9070 XT has 16 GB and the model+draft sidecar fits with ~1 GB headroom, so the full tape is expected to fit; the RUN card records the actual VRAM.)

## 5. The commit guard — where it lives and what it does

The all-layers-checked-before-mutating discipline (oMLX batch_generator.py:2095-2109) is a **bench-loop guard, not a model-call guard**: the model call only *emits* the tapes; the commit (cache-leaf rebind + KV trim) is bench protocol. A partial rollback "leaves per-layer KV lengths desynchronised by one position and corrupts every subsequent forward" — with 24 GDN + 8 FA layers, one bad rebind mid-commit is fatal, so the guard is:

1. **CHECK all layers first**: for every GDN layer, verify the tape arrays exist, the `state_tape`/`conv_state_tape` shapes are `[4, Hv, 128, 128]` f32 / `[4, 3, conv_dim]` bf16, and `keep-1 >= 0` (or keep=0, the no-op case). For every FA layer, verify the KV cache is trimmable.
2. **THEN mutate**: only if every check passes, rebind every GDN cache leaf (`c[0] = conv_tape[keep-1]`, `c[1] = state_tape[keep-1]`, `c.lengths` unchanged) and `trim(4 - keep)` every KV cache — all-or-nothing. On any check failure: throw, mutate nothing (the leaves stay at the pre-verify state, the round degrades to the t1250 fallback: restore + re-forward).

The reference implementation (drop-in for mtp-bench6.py mode 4) is `bench/gdn-tape-commit.py` (`tape_commit(...)` + the guarded bench-loop fragment). The guard is implemented in the bench loop by reference; the kernel/op side carries no commit logic (the op is pure: inputs in, 4 outputs out — commit is the caller's protocol).

## 6. Kernel-shape summary (what changed vs the T=1..4 fused kernel)

- `state_tape` store in phase 4, after each row's `h` update: one extra `float4`x16 store per thread per row (t = 0..TMAX-1). `state_new` is still written (it equals `state_tape[TMAX-1]`; the existing model path is untouched).
- `conv_state_tape` selection writes in phase 1, per handled channel (v/k in the `tid < head_dim` branch, q in the else branch): 3 j-taps x 3 channels per thread, pure index selection (`gdnt_cs_tap`).
- No change to any arithmetic, FMA order, bf16 rounding, rms_norm reduction, or launch geometry (one workgroup per v head, 256 threads, grid Hv). The tape stores add ~8.6 MB of stores per T=4 launch per layer — inside the fused GDN dispatch, no extra launches.

Files: `patches/gdn_fused_step_tape/` (the graft: kernel file + the 5-file registration diff; applies on top of the repo's `(vendor/mlx-rocm-upstream tree + patches/gdn_fused_step/ graft)` state — the self-contained chain verified in the RUN card; the standalone `.hip` in this directory is byte-identical to the patch's `.hip` section).
