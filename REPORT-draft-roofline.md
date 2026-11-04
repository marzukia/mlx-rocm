# Roofline probe: the K=3 MTP draft step, per-op (B config, tape commit)

Box-side bench (bench/roofline-draft-perop.py), one K=3 draft step = the hybrid S=1 correction trunk forward (full 4-bit trunk re-read, 1 row, the next chain's pre-norm h) + the draft chain (3x MTP module calls + 3x shared 4-bit lm_head argmax + 3x embed_tokens; the MTP sidecar's own KV cache advances by 3 rows), B config (GDN_TAPE=1 GDN_FUSE=1 QMV_ROWS=1 QMV_ROWS_MIN_N=1 REFWD_M1=1), model qwen3-9b-mlx (4-bit trunk + bf16 MTP sidecar, md5 02a54359cda8b247f76b277dd9bc4d4c verified), CTX=539 (prompt + 512 AR warmup), 25 s idle before each pass, SCLK sampled 2200 MHz before/after both passes (DPM force high, clocks not changed), median of 5 wall reps with state save/restore.

## TL;DR

- Fused draft-step wall: 23.36 ms (r4v, BEFORE) / 21.36 ms (kfix, AFTER) — the kernel fix cut 2.01 ms (8.6%): the S=1 correction forward 12.56 -> 11.96 ms, the chain 10.93 -> 9.57 ms.
- The draft step moves 8.06 GiB and runs at 371 GB/s = 67.9% of the 546 GB/s floor on r4v (406 GB/s = 74.3% on kfix). It is bandwidth-dominated, not launch-dominated: sum-of-parts 23.74 ms vs wall 23.36 ms (1.6% above, i.e. launch/serialization ≈ 0 within rep variance).
- The 1-row trunk weight read (the "3.6 GiB" claim) is already at 63.6% of the floor on r4v (66.8% on kfix) — inside the 50-60%+ roofline band, so direction (a) has only ~4.5 ms of headroom to the 100% roof; the bigger lever is the 4.73 GiB head traffic of the chain (direction (b)), which is at 85-97% of the floor and is the only place a cheaper head actually saves ms.
- The 58% claim cross-checks against the box's tape-phase production run (draft phase 60.3% of gen wall, chain 25.7 ms/step on that wheel); on this r4v wheel the chain is 10.93 ms, so the draft step is 34% of the spec-step wall (68.5 ms = verify 45.07 + draft 23.36 + commit) and 52% of the verify wall. The 58% figure is a slower-wheel regime, flagged below.

## Per-op table (one K=3 draft step; "n x" = instances per step; M=1 unless noted)

BEFORE — venv-r4v (libmlx md5 8acf13fec6fc1636026d8e1dc69cd497):

| op class (n x) | ms | GiB moved | GB/s eff | % of 546 |
|---|---|---|---|---|
| qmv_gdn (24) | 3.161 | 0.798 | 272.1 | 49.8% |
| gdn_fused tape op T=1 (24) | 1.417 | 0.097 | 73.3 | 13.4% |
| attn M=1 (8) | 1.420 | 0.046 | 34.5 | 6.3% |
| mlp (32) | 6.707 | 2.390 | 383.0 | 70.1% |
| norms in+post (32) | 1.515 | 0.0005 | 0.35 | 0.06% |
| lm_head 4-bit M=1 (4 x: 1 correction + 3 draft argmax) | 3.902 | 2.011 | 554.4 | 101.5% (at roof; read-only stream can exceed D2D floor) |
| mtp_head bf16 module (3 x) | 5.462 | 2.715 | 534.6 | 97.9% |
| embed (3 x) | 0.157 | 0.00002 | 0.16 | 0.03% |
| sum-of-parts | 23.74 | 8.059 | 365.0 | 66.9% |
| fused wall (median of 5) | 23.36 | 8.059 | 370.9 | 67.9% |
| sub-wall: S=1 correction forward | 12.56 | 4.06 | 347.0 | 63.6% |
| sub-wall: K=3 chain | 10.93 | 4.729 | 465.1 | 85.2% |

AFTER — venv-kfix (parallel kernel-fix ticket, libmlx md5 69d73260670b59dfbc4ca626a0a6cd93):

| op class (n x) | ms | GiB moved | GB/s eff | % of 546 |
|---|---|---|---|---|
| qmv_gdn (24) | 2.491 | 0.798 | 345.2 | 63.2% |
| gdn_fused tape op T=1 (24) | 1.406 | 0.097 | 73.8 | 13.5% |
| attn M=1 (8) | 1.388 | 0.046 | 35.3 | 6.5% |
| mlp (32) | 6.174 | 2.390 | 416.1 | 76.2% |
| norms in+post (32) | 1.526 | 0.0005 | 0.34 | 0.06% |
| lm_head 4-bit M=1 (4 x) | 3.829 | 2.011 | 564.9 | 103.5% |
| mtp_head bf16 module (3 x) | 5.455 | 2.715 | 535.3 | 98.0% |
| embed (3 x) | 0.157 | 0.00002 | 0.16 | 0.03% |
| sum-of-parts | 22.43 | 8.059 | 386.4 | 70.8% |
| fused wall (median of 5) | 21.36 | 8.059 | 405.7 | 74.3% |
| sub-wall: S=1 correction forward | 11.96 | 4.06 | 364.5 | 66.8% |
| sub-wall: K=3 chain | 9.57 | 4.729 | 531.2 | 97.3% |

Kernel-delta (after - before): qmv_gdn -0.670 ms, mlp -0.531 ms, lm_head -0.073 ms, attn -0.032 ms, gdn_fused -0.010 ms, chain -1.363 ms, wall -2.008 ms (-8.6%). Deterministic parity check: both wheels produce the identical AR trajectory (CTX=539, LAST=188912, TOK=561), so the delta is kernel, not model drift.

## Bytes cross-check (brief: "the draft step should move ~3.6 GiB of trunk weights per full forward")

Measured 4-bit trunk weight bytes: layers packed 3.222 GiB + layers f32 scales 0.201 GiB + head packed 0.474 GiB + head scales 0.030 GiB. Layers+head packed-only = 3.695 GiB; with scales = 3.927 GiB. The 3.6 GiB claim falls inside the accounting band (layers+head packed-only, 2.8% under) — cross-check PASSES. Per-full-forward the trunk read is 3.70-3.93 GiB depending on scale inclusion. The verify probe's attn table under-counted the KV read by the nkv=8 factor (0.033 GiB at M=4 should be 0.266 GiB); corrected in this table, flagged for the verify probe.

## Fix directions (estimates on r4v base; bench-only, retest criteria only)

(a) Roofline the 1-row trunk read: the S=1 correction forward moves 4.06 GiB (3.93 GiB trunk 4-bit weights + GDN state 0.097 + KV 0.019 + activations) in 12.56 ms = 347 GB/s = 63.6% of the floor — already at the 60% edge of the 50-60% band. Roofline: 3.93 GiB at 50% (273 GB/s) = 15.4 ms; at 60% (327.6) = 12.9 ms; at 100% (546) = 8.0 ms. Fixable headroom to the 100% roof: ~4.5 ms (12.56 -> 8.0 on r4v; 11.96 -> 8.0 on kfix, ~3.9 ms). The rows GEMV (QMV_ROWS=1) is the mechanism; the kfix wheel is already 66.8% here, so remaining (a) work is the last 33%.

(b) Cheap head: the K=3 chain is 10.93 ms (r4v) / 9.57 ms (kfix) at 85.2% / 97.3% of the floor — bandwidth-bound, so only byte reduction saves ms. Ranked options:
- 4-bit MTP module (a 152 MB 4-bit MTP sidecar exists on the box, currently disabled): mtp_head 2.715 -> 0.84 GiB, 5.46 -> ~1.6 ms, saves ~3.9 ms. Retest only (would flip the B config's MTP_4BIT switch).
- int2 shared head (mtp-bench6 int2w sidecar): the 3x 4-bit head argmax 1.51 GiB -> 0.94 GiB, saves ~2.2 ms. The old-regime NO-ADOPT verdict is explained by the old kernel floor; the chain now runs at 85-97% of floor so the verdict may flip. Retest criterion: same SCLK-locked regime on the kfix wheel, B vs B+int2w, adopt only if the chain drops >= 2 ms (or tok/s +4%) at acceptance-neutral.
- Batch the 3 draft argmax into one M=3 head call (rows kernel with QMV_ROWS=1 reads the 0.503 GiB head once): 3x 0.9756 ms -> ~1.0-2.0 ms, saves ~0.5-2.0 ms at zero accuracy cost. Mechanism note: the M=1 row is at 530 GB/s but the M=4 verify head is only 190 GB/s, so the M=3 cost is between; measure, do not assume.
- Combined (4-bit MTP + int2 head + batched argmax): chain 10.93 -> ~3 ms, draft step 23.36 -> ~15.5 ms. A llama-style tiny dedicated draft head (replacing the whole MTP module + shared head with a <0.2 GiB head) is the larger-accuracy-cost version of the same direction; gate on acceptance.
- GDN state advance (1.42 ms, 13.4% of floor) and M=1 attention (1.42 ms, 6.3% — SDPA is cheap at 1 query row, unlike the M=4 verify SDPA) are small; not worth a fix pass.

Launch/other: sum-of-parts 23.74 vs wall 23.36 ms — launch/serialization is 0.4 ms above wall, i.e. ~0 within the 0.05-0.11 ms rep spread; the step is purely bandwidth.

## 58% cross-check (flagged deviation)

Measured draft step 23.36 ms (r4v) vs verify wall 45.07 ms (t1854, same wheel): the draft step is 34% of the all-accepted spec-step wall (~68.5 ms) and 52% of the verify wall. The brief's ~58% matches the box's tape-phase production run (draft phase 60.3% of generation wall, chain 25.7 ms/step on that wheel) — a slower-wheel regime than this r4v wheel (chain 10.93 ms). Both figures are real; the 58% is regime-specific.

## Provenance

- BEFORE: venv-r4v, libmlx md5 8acf13fec6fc1636026d8e1dc69cd497 (verified at run start); AFTER: venv-kfix, libmlx md5 69d73260670b59dfbc4ca626a0a6cd93 (built by the parallel kernel-fix ticket).
- Model: <models-mount>/qwen3-9b-mlx 4-bit + MTP sidecar md5 02a54359cda8b247f76b277dd9bc4d4c; env B config per brief; GPU locked (flock, looped, never killed holders), SCLK 2200 MHz / DPM force high sampled before and after both passes.
- Harness: bench/roofline-draft-perop.py (per-op class timing, save/restore state between reps, 3 reps median per class, 5 reps median wall, 25 s idle after the 512-token AR warmup); the wall goes through the model's S=1 tape-branch forward (GDN_TAPE=1, S=1 <= 4, state non-None); the gdn_fused class replays the tape op directly, which is why the TAPE/FUSE dispatch-counter deltas print 0 (they count model dispatches only, not the direct class replay).
- Check line: uv run ruff check . and uv run ruff format --check . pass (99 files); scripts/line_wrap_check.py absent at this snapshot (same as the verify report) — manual equivalent check (all non-empty report lines are single paragraphs, no mid-paragraph hard wraps) passes; persona-leak grep over the diff (the run brief's pattern set: fleet-IP block, guest-identity prefixes, home/model mount paths, junkyard scripts, mrzk.io mail): zero hits.
