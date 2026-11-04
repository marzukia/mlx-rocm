# int2w wide draft head: step-level MTP A/B (t1796, guest-a, 2026-11-04)

## TL;DR

- Arm A (4-bit head, R4b config of record): **pending** (runs in progress).
- Arm B (int2w wide draft head, `MTP6_DRAFT_HEAD=int2w`): **pending**.
- The wide per-row op on the 78ba063 wheel: 573.8 us vs 955 us 4-bit (0.60x, i.e. faster
  than the 4-bit head per row) and 18.98 ms scalar int2 — wheel identity re-verified on
  the box before the A/B.
- Lossless gate: pending.

## Setup and identity

- Commit under test: `78ba063` (v3c, the 570.8 us gate PASS wheel).
- Wheel: `venv-int2w`, `libmlx.so` md5 `0b8aa95df151a93404b53fee1e96d0de`, byte-identical
  to `~/wheelout-int2vgpr3c/` (the 78ba063 wheel, built Nov 3 23:13). Vendor tree at the
  box (`~/vendor-int2wbuild`) md5-matches 78ba063 for all lever files (`lm_head_int2w.hip`
  `19144e91...`, `lm_head_int2.hip` `5948fdce...`, `qmm.hip` `275755b5...`, `fast.cpp`
  `d8a04ae1...`, `fast.h` `74060c67...`, `fast_primitives.h` `756a1d9f...`).
- Wheel identity by behavior (tight warm loop, `r4c_xcheck.py`, the v3 gate instrument):
  `wide_loop=573.8us`, `head_loop=955us`, `int2_loop=18976us`, `wide_vs_head=0.60`,
  `ratio=19.87`, `sel_match=0/256` (the documented structural artifact: the wide op is a
  per-row M=1 selection; full-rerank selection is done host-side by the bench).
- Import gate: `mx.fast.lm_head_int2_wide` present, `gdn_fused_step` +
  `gdn_fused_step_tape` present.
- mlx_lm: the venv-int2w `qwen3_5.py` was the unpatched 0.32.2 (strips `mtp.*` weights,
  no `lm.mtp`); replaced with the proven patched copy from venv-r4c (md5 `7763fbbb...`,
  builds `self.mtp` from `mtp_num_hidden_layers`; the only differing file in the package).
  Backed up at `~/t1796/qwen3_5.py.orig`.
- Model: `/mnt/models/qwen3-9b-mlx`, sidecar `model-mtp.safetensors` md5
  `02a54359cda8b247f76b277dd9bc4d4c` (unchanged pre/post).
- Bench: `bench/mtp-bench6.py` at 78ba063, md5 `3c7d13b56509634aae1708d4a4df7b16`
  (the 452d363 box copy is the pre-graft canonical; 8e1cb13 keeps the canonical path as
  the default-when-unset, with int2w an explicit opt-in).
- GPU: RX 9070 XT gfx1201, idle 2200 MHz sclk (locked high), 31 C, 0% use at session
  start; both GPU locks free; no orphans; all GPU work under `flock ~/gpu.lock`.

## Protocol (r4c config of record, the 48.93 anchor config)

- Common: `MLX_MTP=4 MTP6_TAPE=hybrid MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1`, `LD_PRELOAD=libhsa-runtime64`,
  `HCC_AMDGPU_TARGET=gfx1201`, `MODELS=/mnt/models`.
- Arm A: no `MTP6_DRAFT_HEAD` (4-bit/bf16 head). Arm B: `MTP6_DRAFT_HEAD=int2w`.
- 3 runs per arm, medians, 25 s idle before each run, `flock ~/gpu.lock` around each run,
  sclk sampled every 2 s under load.
- Lossless gate: `ALLTOKS` md5, A vs B, per run pair (report on FAIL only); acceptance
  (tok/step) reported for both arms.
- MAX_TOKENS: bench default (1024); the 48.93 run stopped at 384 tok (EOS at ~iter 322).
  The "512 verifies" figure in the task brief is the task-side expectation; the
  reproducible r4c anchor is 322 verifies / 384 tokens, and that is what an on-target
  Arm A should reproduce.

## Results

_pending (session running on guest-a)_
