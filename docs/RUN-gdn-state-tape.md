# RUN card — GDN state tape (MLX_ROCM_GDN_TAPE=1)

Lane: guest `<guest>` (RX 9070 XT / gfx1201, ROCm 10.1 userland). **Check VRAM first** (`/opt/rocm/bin/rocm-smi --showmeminfo vram`); the GPU lane belongs to the lane-holder — do not fight for it; park the model sidecars to `*.disabled` for the duration (t1228 convention) and restore them after.

Wheel: the compile-verified wheel built from `<host-user>/vendor-tape` (base = the repo's self-contained graft chain: the `vendor/mlx-rocm-upstream/` tree + the `patches/gdn_fused_step/` graft + the `patches/gdn_fused_step_tape/` graft — chain verification note below). Build log: `<host-user>/tape-build.log`; wheel: `<host-user>/wheelout-tape/`. Install into a FRESH venv (do not touch t1256's venv-fused2), with the `<model-mount>/qwen3-9b-mlx` model + the MTP sidecar (`model-mtp.safetensors` sha256 `edef9ec6…df8` — restore the parked sidecars to their pinned names for the run, park them back after).

## Graft chain verification (self-contained, in-repo)

Fresh worktree of `vendor/mlx-rocm-upstream/`, step by step:

1. `patch -p1 --dry-run < patches/gdn_fused_step/gdn_fused_step.patch` — all hunks pass, then apply. (Host git 2.53 note: its xdiff engine rejects the gfs patch's context-free deletion hunk and anchors its context-free additions at EOF, so `git apply --check` cannot validate the gfs step on THIS git version against any base; the 2026-11-01 guest-side verification used the guest's git. The tape patch is fully context-carrying and IS `git apply --check`-validated.)
2. `git apply --check patches/gdn_fused_step_tape/gdn_fused_step_tape.patch` — passes (verified 2026-11-02 after the D2 re-cut against this exact state).

Provenance: the `75ca881` single-commit snapshot is not present in the public fork refs; the repo's vendor tree carries the reconstructed graft-base regions both grafts reference (marked in-file; see `vendor/mlx-rocm-upstream/README.md`). The tape patch's `.hip` section is byte-identical to the standalone `patches/gdn_fused_step_tape/gdn_fused_step_tape.hip` (the 1-character `static_cast<float>(` fix line included).

## G1/G2 (done at build time, recorded here for the GPU run)

- G1: clean build, 0 errors (build log tail + the warning list are in REPORT.md §G1).
- G2: `nm -D`/`nm` on the built `libmlx.so` shows the new symbols (GdnFusedStepTape::eval_gpu / use_fallback, gdn_fused_step_tape, the kernel mangled name, the python binding) — see REPORT.md §G2.

## Probe P1 — tape invariance (the t1223 probe pattern)

Fork the same GDN state into two arms, compare bit-exactly:

1. **Tape vs sequential T=1**: prefill to a fixed state (the m7-state-probe pattern: prefill + N AR steps, snapshot the GDN leaves `cache[i][0]/cache[i][1]` + `lengths`). Arm A: one `mx.fast.gdn_fused_step_tape(...)` call with T=4 rows (q/k/v/b/a/z sliced from a 4-row input, `MLX_ROCM_GDN_TAPE=1`); arm B: four sequential `mx.fast.gdn_fused_step` T=1 calls, each feeding arm B's previous state + conv state as its input (the model's conv-state update is the same COPY op both arms use).
2. Check, per GDN layer, with `maxdiff == 0` (bit-exact, f32 state and bf16 conv — exact comparison, no tolerance):
   - `state_tape[t] == arm B state after step t` for t = 0..3 (the **invariance**: the register-resident T=4 loop must equal the sequential T=1 chain row for row — this is the same guarantee the T=1..4 kernel has, now extended to the tape rows).
   - `state_new == state_tape[3]` (same array content; the op writes both).
   - `result` (T=4 tape op) == `cat` of the four T=1 arm B results.
   - `conv_state_tape[t] ==` the conv window after step t (arm B's conv state after step t, i.e. what the model's conv-state COPY op produces after consuming row t).
   - The tape op vs the plain `gdn_fused_step` T=4 op on the same inputs: `result` and `state_new` bit-exact (the tape stores must not perturb anything — they are stores, not math).
3. Also run at T=2 (the S=2 arm) — same checks, t = 0..1.
4. Record: `P1=maxdiff=0.0` per arm/layer + the layer count checked.

## Probe P2 — the commit round-trip (bit-exact commit)

Same prefill state, one S=4 verify-style forward with the tape op on a real layer stack is out of scope for the probe (that is the A/B below); instead, at the op level: take arm A's `state_tape[k]` / `conv_state_tape[k]` (k in {0,1,2,3}), feed them as the NEXT step's inputs to a T=1 `gdn_fused_step` call (one more row), and compare against arm B run to row k then one more T=1 step: next-step `state_new` and `result` must be bit-exact. This proves the selected tape row is a valid continuation state (the commit is a pure rebind to a correct state).

## Lossless A/B — K=3 (mtp-bench6 mode 4)

Same run env as t1250 (sidecars parked, same model, `MTP_MAX_TOKENS=1024`, `MLX_MTP=4`):

- **Arm B (baseline)**: `MLX_ROCM_GDN_TAPE=0` (unset) + `MLX_ROCM_GDN_FUSE=1` → the t1250 protocol (save/restore + re-forward cascade). Expected: tok/pair = 3.277 ± noise, FIRST20 = the t1250/t1225/t1226 token head (deterministic stream).
- **Arm A (tape)**: `MLX_ROCM_GDN_TAPE=1` + `MLX_ROCM_GDN_FUSE=1` + the tape commit in the bench loop (`bench/gdn-tape-commit.py` fragment: the tape op in the verify forward via the model-side dispatch, the guarded commit, the single correction forward on rejection). Expected: `ALLTOKS` **byte-identical to arm B** (greedy; the tape op is bit-identical to the fused op per P1, so the verify rows' logits are unchanged; only the state bookkeeping differs, and commit-to-tape-row must equal restore-to-saved-state per P1/P2), re-forward count **1206 -> ~394** (the probe prints the count; the instrumented bench `mtp-bench6-instr.py` checkpoint for the re-forward class).
- Lossless gate: `ALLTOKS(A) == ALLTOKS(B)` byte-exact + per-round state-sha match on the first 32 rounds (the t1223 state-probe discipline).

## K=3 bench expectation

| arm | ms/verify-pair | tok/s | re-forwards (1024 toks) |
|---|---|---|---|
| baseline (t1250) | 96.36 | 34.0 | 1206 (2.355/pair) |
| **tape commit** | **~77.6** (96.36 - 27.86 + 9.11) | **~42** | ~394 (1.0/rejected round) |
| tape + deferred-correction fold (steal #7, optional) | **~68.5** | **~47-48** | ~0 |

Record the measured wall/verify (50-rep, warm), tok/pair, the re-forward count, D1ACC/D2ACC/D3ACC (must equal the baseline — the acceptance distribution is unchanged: the tape changes bookkeeping only), VRAM peak (the tape price: ~206 MB vs baseline — record `rocm-smi --showmeminfo vram` before/after model load during the run), and FIRST20. The tape arm is a PASS at tok/pair >= 3.9 (= 96.36/77.6 x 3.277 floor minus bench noise) with byte-identical ALLTOKS; a FAIL is any token divergence, any state-sha mismatch, or tok/pair < 3.5.

## Protocol change the tape implies (bench-protocol note)

The commit logic moves **into the bench loop** (it never lives in the model call — the model just emits the tapes):

1. **Verify**: the S=4 verify forward uses the tape op (model-side dispatch, `MLX_ROCM_GDN_TAPE=1`; the model-side patch mirrors the `FUSE_BRANCH_PATCH` dispatch in `patches/qwen35_fuse_branch/qwen35_fuse_branch.diff`, selecting `gdn_fused_step_tape` when the tape env is on — same shape gate, S in {1, 2, 4}; S=3 falls through to the F-branch). The 4 per-round tape buffers (24 GDN layers) are round-reusable (allocated once, reused every round; the committed row outlives the round by construction — it is read by the next round's forwards before the next verify overwrites it).
2. **Commit (guarded, `bench/gdn-tape-commit.py`)**: all-layers-checked-before-mutating (check every GDN layer's tape shapes/dtypes and every FA layer's trimmability FIRST; then rebind all GDN leaves + trim all KV, all-or-nothing; on any failure throw and fall back to the t1250 restore+re-forward for that round). On full acceptance: rebind to `state_tape[3]`/`conv_state_tape[3]` (== `state_new` — the rebind is a no-op content-wise but keeps one code path). On partial acceptance (k in {1,2,3} rows accepted): rebind to row `k-1`, KV `trim(4-k)`. On 0 accepted: no rebind (the leaves already hold the pre-verify state), KV `trim(4)`.
3. **Rejection**: ONE S=1 forward of the correction token from the committed state (replaces the 2-4 S=1 re-forwards); its `h`/`l` feed the next round's draft chain exactly as today. The tape changes nothing about the single-token step, the draft chain, or the acceptance test (greedy argmax cascade).
4. **Draft (MTP) cache**: same protocol as today (NOT rolled back on rejection — the committed-timeline discipline, steal #4, is the follow-up; it does not gate this run).

## Env / cleanup

- `MLX_ROCM_GDN_TAPE=1` (tape; DEFAULT OFF), `MLX_ROCM_GDN_FUSE=1` (the fused step the tape op is a variant of), `MLX_MTP=4`.
- Restore the parked sidecars to their original names after the run; delete nothing in `<host-user>/vendor-1c` or `venv-fused2` (t1256's); the tape wheel stays in `<host-user>/wheelout-tape/` (NOT installed in this run — install into the fresh venv for the A/B, then leave the venv in place and report its path).
