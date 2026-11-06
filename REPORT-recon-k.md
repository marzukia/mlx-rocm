# REPORT-recon-k: RECONCILIATION-322 (COR verbatim) + the D3 fact + K sweep

- Date: 2026-11-06 (00:47-01:10 UTC), one box, single lock holds, 25 s idle between arms
- Box: bare-metal ROCm gfx1201 box, work card verified by PCI identity: `GPU_PCI=0000:03:00.0 GPU_DEV=0x7550 ROCSMI_IDX=0 GPU_NAME=R907XTGL`; SCLK sampler reading (between-arm, shared sampler, rocm-smi -d 0): 41 MHz (idle-point sampling; the sampler is live and identity-verified, readings at idle only - no sustained load sample this run)
- Wheel: venv-verify, libmlx md5 8acf13fec6fc1636026d8e1dc69cd497 (verified pre AND post, unmodified, not rebuilt)
- Model: <models-mount>/qwen3-9b-mlx, sidecar md5 02a54359cda8b247f76b277dd9bc4d4c (verified pre AND post, unmodified)
- Bench: m3k/bench/mtp-bench6.py md5 54ce16ac29e6c6aad2c48a233ff70d85 (stage 1, K1, K3 arms)
- Protocol: the 27-token canon prompt (the bench default, verbatim), MTP_MAX_TOKENS=322 cap (iteration cap; generation runs to natural stop - 516 tokens for the K=3 COR arm, the same 516/161-verify profile as t2213's 69.43 record battery), greedy, 3 rounds x arm interleaved
- Exact env as run (stage 1, the 13-var COR verbatim set + box plumbing):

  ```
  MODELS=<models-mount>
  LD_LIBRARY_PATH=<venv-verify>/lib/python3.14/site-packages/mlx/lib
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
  HCC_AMDGPU_TARGET=gfx1201
  MLX_MTP=4
  MTP6_TAPE=hybrid
  MLX_ROCM_GDN_FUSE=1
  MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1
  MLX_ROCM_QMM_M3=1
  MTP6_HEAD_H=postnorm
  MLX_ROCM_SDPA_S4VA=1
  MLX_ROCM_REFWD_M1=1
  MLX_QMV_MR_LX=0
  MTP_MAX_TOKENS=322
  ```

The bench sets `MLX_ROCM_GDN_TAPE=1` and `MLX_ROCM_GDN_T3=1` internally in tape mode (engaged, confirmed by GDN_TAPE_CALLS=6792 - identical to t2213's record arm).

## STAGE 1 - RECONCILIATION-322: the 13-var COR verbatim, one round

The 13-var COR ran verbatim on the lane wheel with the tape branch engaged - no fused-GDN shape error (the t2212 shape error did not reproduce on the m3k bench copy; see the K=2 note in stage 3 for the one real wheel limitation found this run).

| metric | value |
|---|---|
| tok/s | **47.37** (516 tok / 10.89 s) |
| ms/step (decode wall / verifies) | 67.6 ms |
| tokens/step | 2.447 |
| verifies | 161 |
| D1/D2/D3 acceptance | 119 / 75 / 39 |
| D1% / D2% (of D1) / D3% (of D2) | 73.9 / 63.0 / 52.0 |
| TAPEMODE / GDN_TAPE_CALLS / REFWDS / CDEFERRED | hybrid / 6792 / 122 / 0 |
| HEAD / HEAD_H | int2 / postnorm |
| FIRST20 | 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 |
| ALLTOKS md5 | 0930e7fc5f79317cdfa33b266a10cd16 |

**Reconciliation verdict:** the absolute basis on the lane wheel (8acf13fe) is **47.37 tok/s at COR verbatim = -31.8% vs the 69.43 record** (record wheel, no longer on the box). This confirms the t2212 A/B absolute basis (~48.7, labeled deviation config) as the correct order of magnitude: the between-arms A/B delta stands, and the absolute gap to the record is a wheel-level regression, not a config artifact - every K in stage 3 sits 20-32% under the record, including the K=1 arm. vs t2212 A/B basis (48.68 D / 49.62 N, the labeled deviation config): 47.37 is -2.7%/-4.7% (cross-session, different K - not a like-for-like comparison; the within-session K sweep is the valid basis).

## STAGE 2 - THE D3 FACT

**D3 at COR verbatim = 39 accepted (24.2% of all verifies; 52.0% of the D2-accepted steps).** D3 is NOT ~0 at COR verbatim: the third draft is accepted at a 52% conditional rate and contributes 39/161 = +0.242 to tokens/step (2.447 = 1 + 233/161).

The t2212 A/B "D3 = 0" is structural, not behavioral: t2212's A/B ran MLX_MTP=3, which on this bench family is **K=2 - two drafts, S=3 verify - there is no third draft position** (this run reproduced t2212's arm bit-identically: same 445 tokens, md5 c360b537, 123/82/0). The "K=3 pays a third draft forward every step for nothing" reading was a mapping error between bench versions (K=3 is MLX_MTP=4 here, not =3). Accordingly, the "K=2 is the answer regardless of timing" inference is NOT supported by the data: D3 earns its keep on acceptance. Whether K=3 still wins end-to-end is a timing question - and stage 3 answers it: even with D3 accepted, K=3 is the SLOWEST arm.

## STAGE 3 - K SWEEP

**MLX_MTP <-> K mapping, established empirically** (smoke: monkeypatched MTPModule.__call__ count over MTP_MAX_TOKENS=12 runs on this bench/wheel):

| MLX_MTP | mtp calls / verifies | K | verify |
|---|---|---|---|
| 1 | 6 / 6 | 1 | two T=1 forwards |
| 2 | 6 / 6 | 1 | ONE batched S=2 forward |
| 3 | (smoke crashed on the wheel's clean tape rejection; K2fix arm below ran clean) | 2 | ONE batched S=3 forward |
| 4 | 18 / 6 | 3 | ONE batched S=4 forward |

**K=3 = MLX_MTP=4 on this bench.** K=1 arms use MLX_MTP=2 (the batched S=2 verify; mode 1 is the legacy 2xT=1 variant, also K=1 per the smoke).

Base = the COR-verbatim base of stage 1 (QMV_ROWS=1, QMV_ROWS_MIN_N=1, QMM_M3=1, HEAD_H=postnorm, SDPA_S4VA=1, REFWD_M1=1, QMV_MR_LX=0, MTP_MAX_TOKENS=322, LD_PRELOAD, HCC), DEFAULT dispatch (NO_MR unset - not in the COR set). The K-dependent parts: the tape/fuse vars exist only for bench modes 3/4 (absent for K=1 - K-dependent exclusion, not a config choice); the K=2 COR-base config (S=3, TMAX=3 tape) is rejected by the 8acf13fe wheel with the clean error `gdn_fused_step_tape: tape unavailable for this shape (require MLX_ROCM_GDN_TAPE=1 ... B=1, T<=4, Dk=Dv=128, kernel_size=4)` (the TMAX=3 instantiation's use_fallback rejection - note t2213's 6792-tape record arm was the S=4/TMAX=4 path). Per the protocol, **K=2 ran on the t2212 A/B config (MTP6_TAPE=0 + model-default GDN path) - labeled deviation**. It ran on the m3k bench with the exact t2212 one-line plumbing fix (GDN_FUSE moved inside the tape check): the patched copy's md5 686c9f6e3105abb9e904263cd42d16c5 is byte-identical to t2212's bench copy md5 686c9f6e, and the arm reproduced t2212's A/B token stream bit-identically (md5 c360b537).

3 rounds interleaved (K1, K2, K3 per round), per-arm values:

| arm | round | tok/s | ms/step | tok/step | D1 | D2 | D3 |
|---|---|---|---|---|---|---|---|
| K=1 (MLX_MTP=2) | 1 | 55.19 | 36.2 | 1.683 | 110 (68.3%) | - | - |
| K=1 | 2 | 55.12 | 36.3 | 1.683 | 110 (68.3%) | - | - |
| K=1 | 3 | 55.00 | 36.3 | 1.683 | 110 (68.3%) | - | - |
| K=2 (MLX_MTP=3, TAPE=0 labeled deviation) | 1 | 51.24 | 54.0 | 2.273 | 123 (76.4%) | 82 (66.7% of D1) | 0 (structural) |
| K=2 | 2 | 51.24 | 54.0 | 2.273 | 123 (76.4%) | 82 (66.7%) | 0 |
| K=2 | 3 | 51.13 | 54.1 | 2.273 | 123 (76.4%) | 82 (66.7%) | 0 |
| K=3 (MLX_MTP=4, COR verbatim) | 1 | 47.16 | 68.0 | 2.447 | 119 (73.9%) | 75 (63.0%) | 39 (52.0% of D2) |
| K=3 | 2 | 46.93 | 68.3 | 2.447 | 119 (73.9%) | 75 (63.0%) | 39 (52.0%) |
| K=3 | 3 | 46.95 | 68.3 | 2.447 | 119 (73.9%) | 75 (63.0%) | 39 (52.0%) |
| **median** | | **K1 55.12 / K2 51.24 / K3 46.95** | | | | | |

Trajectories are bit-identical across rounds within each arm (ALLTOKS md5: K1 2785318b, K2 c360b537, K3 0930e7fc); wall spread is 0.2-0.6% per arm.

**RANKING (tok/s only - never cross-K ms/step):**

1. **K=1: 55.12 tok/s** (1.683 tok/step) - wins by +17.4% vs K=3, +7.6% vs K=2
2. K=2: 51.24 tok/s (2.273 tok/step, labeled deviation base)
3. K=3: 46.95 tok/s (2.447 tok/step, COR verbatim)

Mechanism: the 8acf13fe wheel's per-step verify cost grows with K (36.2 / 54.0 / 68.0 ms per verify-step) faster than acceptance grows (1.683 / 2.273 / 2.447 tok/step). D3 acceptance (52% conditional) is real but buys +0.174 tok/step at +14 ms/step - on this wheel the third draft costs more than it returns. The K=2 arm's base is the labeled deviation (S=3 tape unavailable on this wheel), so K=2-vs-K=3 is not fully same-base; K=1 vs K=3 is COR-base-clean (K=1's non-tape is a K-dependent exclusion). Note the acceptance pattern: K=1 D1 = 68.3% vs K=3 D1 = 73.9% - the K=3 first draft is stronger (postnorm head, S=4 context), but the verify-step cost dominates.

Cross-check vs t2212 (cross-session, same wheel 8acf13fe, same K=2 deviation config, bit-identical tokens): this run's K=2 = 51.24 vs t2212's D 48.68 / N 49.62 (+5.2% / +3.3%) - the COR tuning knobs (QMV_ROWS=1, QMM_M3=1, SDPA_S4VA=1, REFWD_M1=1, MR_LX=0) are worth ~+3-5% on the K=2 base. The NO_MR A/B delta (+1.93%) is smaller than this cross-session spread, so the NO_MR question cannot be settled on these numbers.

## STAGE 4 - NO_MR

NOT decided here. Numbers reported only: lane-wheel e2e absolute basis K1 55.12 / K2 51.24 / K3 46.95 tok/s (vs record 69.43, missing wheel: -20.4% / -26.3% / -32.4%); t2212 A/B basis (same wheel, K=2 deviation config) 48.68 (DEFAULT) / 49.62 (NO_MR=1), +1.93% - the default-flip call is deferred to after this reconciliation, at whatever threshold the e2e actually supports.

## Gotchas logged

1. The m3k bench sets `MLX_ROCM_GDN_FUSE=1` for ALL of modes 3/4 (incl. MTP6_TAPE=0) - running mode 3 with tape=0 on the 8acf13fe wheel dies in `gdn_fused_step` (fallback disabled for the shape). t2212's bench copy is byte-identical to the m3k copy plus the one-line move of GDN_FUSE inside the tape check (md5 686c9f6e, confirmed this run).
2. Box HSA plumbing: use `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` (the /opt/rocm/lib copy is a different runtime (1.18.70200) and fails the import with `libamdhip64.so.7: undefined symbol hsa_amd_external_semaphore_handle_close`); prior lane runners all used the /usr/lib copy.
3. The 8acf13fe wheel rejects the S=3 (TMAX=3) tape instantiation with a clean error; the S=4 (TMAX=4, 6792-call) path is the one that runs. The K=2 COR-base arm is unrunnable on this wheel until the TMAX=3 path is fixed.
