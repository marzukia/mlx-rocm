# int2w wide draft head: step-level MTP A/B (t1796, guest-a, 2026-11-04)

## TL;DR

- Arm A (4-bit head, R4b config of record): **40.30 t/s** (2.438 tok/step, D1 72.9%).
- Arm B (int2w wide draft head, `MTP6_DRAFT_HEAD=int2w`): **27.07 t/s** (1.000 tok/step, 0% acceptance).
- **Ladder: B = 0.67x of A (27.07 / 40.30, -32.7%).** The wide per-row op is 0.60x the 4-bit head per row (573.8 us vs 955 us), but the int2 g128 draft never accepts: the draft contributes nothing and its per-row cost plus the top-32 re-rank overhead put the arm below the 4-bit arm. **NO ADOPT** for int2w as the MTP draft head on this model/wheel.
- Lossless gate: **FAIL** (reported with reason, per protocol): A streams 1628 tokens, B streams 1024; FIRST20 identical, first divergence at generated token 32. B diverges because its draft head produces 0% accepted tokens, so B's MTP path degenerates to 1 token/step and its stream is the plain-greedy-equivalent stream, not A's.
- Root cause (measured, three independent ways — P1/P2/P3 below): the int2 quantization itself is a poor approximation of this 4-bit lm_head, not a kernel bug. A float-level manual int2 GEMV computed from the exact int2 values matches the 4-bit verifier's argmax **0/64** steps (both zero-point sign conventions tested), and the wide op's top-32 candidate set covers the 4-bit argmax **0/64** steps. The MTP state is healthy (P2: 4-bit d1 = 56.2% under the B-state).

## Setup and identity

- Commit under test: `78ba063` (v3c, the 570.8 us gate PASS wheel).
- Wheel: `venv-int2w`, `libmlx.so` md5 `0b8aa95df151a93404b53fee1e96d0de`, byte-identical to `~/wheelout-int2vgpr3c/` (the 78ba063 wheel, built Nov 3 23:13). Vendor tree at the box (`~/vendor-int2wbuild`) md5-matches 78ba063 for all lever files (`lm_head_int2w.hip` `19144e91...`, `lm_head_int2.hip` `5948fdce...`, `qmm.hip` `275755b5...`, `fast.cpp` `d8a04ae1...`, `fast.h` `74060c67...`, `fast_primitives.h` `756a1d9f...`).
- Wheel identity by behavior (tight warm loop, `r4c_xcheck.py`, the v3 gate instrument): `wide_loop=573.8us`, `head_loop=955us`, `int2_loop=18976us`, `wide_vs_head=0.60`, `ratio=19.87`, `sel_match=0/256` (the documented structural artifact: the wide op is a per-row M=1 selection; full-rerank selection is done host-side by the bench).
- Import gate: `mx.fast.lm_head_int2_wide` present, `gdn_fused_step` + `gdn_fused_step_tape` present.
- mlx_lm: the venv-int2w `qwen3_5.py` was the unpatched 0.32.2 (strips `mtp.*` weights, no `lm.mtp`); replaced with the proven patched copy from venv-r4c (md5 `7763fbbb...`, builds `self.mtp` from `mtp_num_hidden_layers`; the only differing file in the package). Backed up at `~/t1796/qwen3_5.py.orig`.
- Model: `/mnt/models/qwen3-9b-mlx`, sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` (unchanged pre/post). lm_head is a 4-bit `QuantizedLinear`, `group_size=64`, packed `uint32 [248320, 512]` (8 nibbles/int32 = 4096 wide), scales/biases `[248320, 64]`.
- Bench: `bench/mtp-bench6.py` at 78ba063, md5 `3c7d13b56509634aae1708d4a4df7b16` (the 452d363 box copy is the pre-graft canonical; 8e1cb13 keeps the canonical path as the default-when-unset, with int2w an explicit opt-in).
- GPU: RX 9070 XT gfx1201, idle 2200 MHz sclk (locked high), 31 C, 0% use at session start; both GPU locks free; no orphans; all GPU work under `flock ~/gpu.lock`.

## Protocol (r4c config of record, the 48.93 anchor config)

- Common: `MLX_MTP=4 MTP6_TAPE=hybrid MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1`, `LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`, `MODELS=/mnt/models`.
- Arm A: no `MTP6_DRAFT_HEAD` (4-bit qmm draft head, R4b config of record). Arm B: `MTP6_DRAFT_HEAD=int2w` (int2w wide draft head, per-row op + top-32 4-bit re-rank).
- 3 runs per arm, medians, 25 s idle before each run, `flock ~/gpu.lock` around each run, sclk sampled every 2 s under load.
- Lossless gate: `ALLTOKS` md5, A vs B, per run pair (report on FAIL only); acceptance (tok/step) reported for both arms.
- MAX_TOKENS: bench default (1024) = 512 verify steps / 1024 tokens (the 512-verify protocol). Regime note: the 48.93-class number is the 322-verify / 384-token EOS-truncated anchor regime; the 512-verify protocol lands in the 40.3-class regime (A = 40.30 here, consistent with the R3 512v 40.38 / r4c 512v 40.31 lane).

## Results

| arm | runs (t/s) | median | tok/step | D1PCT | tokens | FIRST20 |
|---|---|---|---|---|---|---|
| A (4-bit head) | 40.09 / 40.30 / 40.34 | **40.30** | 2.438 | 72.9 | 1628 | `271 1919 369 264 15352 11 70042 29006 314 1204 ...` |
| B (int2w head) | 27.07 / 27.07 / 26.89 | **27.07** | 1.000 | 0.0 | 1024 | identical to A |

- **Ladder: 27.07 / 40.30 = 0.6717 -> 0.67x (delta -32.7%).**
- B's 1.000 tok/step = 0% draft acceptance: every MTP step emits only the verify token. The 27.07 t/s is the plain-AR-equivalent throughput under the MTP=4 machinery (vs A's 40.30 with 2.438 tok/step). B is not "27 t/s fast int2" — it is the acceptance floor.

## Lossless gate: FAIL (reported with reason)

- A: 1628 tokens, B: 1024 tokens (md5 mismatch by length). FIRST20 identical. First divergence at generated token 32.
- Reason: B's draft head never accepts (P1/P3 below), so B's stream is the 1-token/step greedy-equivalent stream while A's stream benefits from 2.438 tok/step accepted MTP tokens. The divergence is not a verifier/model discrepancy (P2 shows the 4-bit verifier and MTP state are healthy under B-state); it is the int2 draft approximation itself.

## Diagnosis (P1/P2/P3, sequential greedy on 64 AR hidden states, same wheel/model)

- **P1 (head pipeline):** the int2 draft head — both the `lm_head_int2_wide` op and the scalar `lm_head_int2` op, with the bench's 4-bit re-rank — matches the 4-bit verifier's argmax **0/64** steps.
- **P2 (MTP state):** with the 4-bit draft head running under the B-state (MTP=4, hybrid tape), d1 acceptance = 56.2%, dw_acc = 0, d4mism = 32/32 in the int2 arm. The MTP sidecar state is healthy; the failure is isolated to the int2 draft head.
- **P3 (isolation, the load-bearing probe):** manual int2 GEMV computed in float from the exact int2 values (per 256-row chunks; both zero-point conventions, `s*q - z*s` = the kernel algebra per r4c's "Quantization semantics" section, and `s*q + z*s`) matches the 4-bit verifier's argmax **0/64** steps, and the wide op's top-32 candidate set covers the 4-bit argmax **0/64** steps. Three independent implementations (manual float minus, manual float plus, kernel wide + scalar) agree: the int2 g128 (2-bit, group-128, per-group `s = max((hi-lo)/3, 1e-8)`, `z = clamp(round(-lo/s), 0, 3)`) approximation of this 4-bit lm_head produces logit rankings that do not agree with the 4-bit verifier on any of 64 greedy steps. The kernel is not the broken stage; the 2-bit quantization is too coarse for top-1 selection on this head (r4c measured max abs logit diff 1.935 int2 vs 4-bit — consistent with a top-1 argmax that is wrong essentially always).

## Cross-reference with r4c (the prior int2 lane)

- r4c's int2 ON arm (27.32 t/s, 0.56x the 48.93 anchor) **lost the speed gate** — no prior lane ever showed int2 winning on a built wheel (the 1.2-1.4x was an expectation predicated on a vectorized int2-dequant GEMM the port did not have).
- r4c's ON-arm D2=62% acceptance figure was produced under the **pre-fix reference quantizer** (the `scale = 4*s, zs = (4+z)*s` revision whose per-group transform distorted the logit ranking); r4c explicitly marked those phase rows "PENDING re-measure" after correcting the quantizer to the kernel's algebra. This A/B is that re-measure under the corrected quantizer: **0% acceptance, 27.07 t/s**.
- The v3 wide kernel's gate (570.8 us, 0.60x the 4-bit anchor, bit-identical to the scalar int2 op) is a kernel-speed result; it says nothing about draft acceptance, and this A/B shows the acceptance gap dominates the per-row speed win: 0.67x end-to-end at the 512-verify protocol.

## Verdict

**NO ADOPT** for `MTP6_DRAFT_HEAD=int2w` as the MTP draft head on qwen3-9b-mlx (4-bit lm_head, group 64) on the 78ba063 wheel: 0.67x the 4-bit arm (27.07 vs 40.30 t/s, 512-verify protocol), 0% draft acceptance, lossless gate FAIL. The int2w lever stays an explicit opt-in, off by default (canonical path when unset). For int2 draft to compete, the 2-bit approximation would need per-row top-1 agreement with the verifier (group-size or level-count increase, or a verifier-matched quantizer), which is a quantizer change, not a kernel change — the wide kernel already runs 0.60x the 4-bit head per row.

## Checks

- GPU work under `flock ~/gpu.lock`, 25 s idle between runs, no orphans.
- `venv-canon` untouched (A/B ran on `venv-int2w`).
- Sidecar md5 re-verified post-session: `02a54359...` (unchanged).
