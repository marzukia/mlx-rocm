# REPORT-sdpa-probe — SDPA/bandwidth probe (MTP multiplier #2)

Base r5-correction-fusion @ 4b9285b; branch sdpa-bandwidth-probe. Measured 2026-11-04 09:56-10:23 UTC on the neon GPU lane (bare metal, RX 9070 XT, gfx1201, SCLK locked 2200 MHz).

## TL;DR

- The S=4 verify step's SDPA dispatch on the wheel of record (venv-r4v, libmlx md5 8acf13fec6fc1636026d8e1dc69cd497) is the scalar flash prefill kernel (kernel_sdpa_flash_opt): the vector decode kernel is gated qL<4, the WMMA flash on qL>4, so the K=3 verify batch (qL=4) falls through to the serial key scan. Measured: SDPA at qL=4 takes 2.96 ms per layer-call at kL=768 vs 0.045 ms for qL=1 - 65x the decode cost at 1.1 GB/s effective (0.2% of the 546 GB/s floor). 8 attention layers (24 GDN + 8 full-attention verified) account for 19.96 ms of the 44.85 ms fused S=4 verify wall (44%).
- Candidate (a) - the env-gated S4VA graft already present in the BOX vendor tree (vendor-canon working tree; MLX_ROCM_SDPA_S4VA=1 sends qL=4 rows to the 1-pass vector kernel, grid (H, qL, B), 1024 threads, parallel key scan) - was adopted: a fresh wheel built from that tree (libmlx md5 0970c43ceb39b10466a74b728b82d088) cut the verify wall 44.85 -> 22.09 ms (-50.7%) and took SDPA qL=4 to 0.054 ms (54.7x), 59.3 GB/s effective (10.9% of floor).
- Gate: PASS on both prongs - verify step -50.7% (>=30% cut) and +18.1% t/s on the clean same-wheel A/B (>=+15%). Lossless: FIRST20 identical across all arms and bit-identical to the R5 reference; acceptance profile in the same order with no systematic degradation; isolated SDPA diff S4VA-vs-flash = 1 bf16 ulp.
- Verdict: ADOPT S4VA (env-gated, default off in the wheel; engage with MLX_ROCM_SDPA_S4VA=1). The remaining verify wall (22.1 ms) is now GDN+MLP bound (qmv_gdn 6.0 + gdn_fused 6.2 + mlp 11.7 + norms 1.5 ms), so the next lever is the M=4 QMM/GEMM class - the exact class t1858 abandoned; this report isolates the SDPA share so a future GEMM A/B is not confounded.

## Step 1 - per-op probe of the S=4 verify step

- Probe: bench/sdpa-verify-perop.py (this branch; extends the t1858-salvage roofline-verify-perop.py with an isolated SDPA microbench and per-layer attention walls; the salvage file was absent from the r5 tree, recovered from the f51c0c7 salvage and 9e4e289 - identical copies). ctx=539 (prompt 27 + 512 AR), KV rows 768.
- Fused S=4 verify wall 44.85 ms (reps 44.81-44.93) - reproduces the t1854 roofline 45.07 ms.
- Per-layer class walls (x layer count): qmv_gdn 0.304 x 24 = 7.29 ms; gdn_fused 0.272 x 24 = 6.52 ms; attention 2.490 x 8 = 19.96 ms; mlp 0.479 x 32 = 15.32 ms; norms 0.046 x 32 = 1.49 ms; lm_head 0.001 ms. Sum of parts 50.6 ms vs 44.85 ms fused (replay overhead).
- Isolated SDPA microbench at the model's real attention shapes (16 q-heads, 4 kv-heads, D=256, kL=768, causal): qL=1 0.045 ms (vector decode kernel); qL=4 2.959 ms (scalar flash fallback); qL=5 0.959 ms (WMMA flash). Effective bandwidth at qL=4: 1.09 GB/s = 0.2% of the 546 GB/s floor - the 1000-2000x-over-roofline attention the ticket named.
- Kernel identification: code read of the vendor SDPA dispatch (scaled_dot_product_attention.cpp, byte-identical in the r5 tree and vendor-r4v) + the env strings of the r4v wheel (MLX_ROCM_SDPA_S4VA absent; vendor-r4v gate stock qL<4) + the 65x behavior delta. GDN-hybrid structure verified: NL=32, NG=24, NA=8 (full_attention_interval=4).

## Step 2 - candidate build

- Candidate wheel: fresh build from the BOX canonical vendor tree (vendor-canon working tree @ 75ca881 + committed graft state: m8 runtime-causal vector kernel + MLX_ROCM_SDPA_S4VA gate), pip-wheel recipe with the box userland ROCm prefix (the box's ROCm 10.x userland now lives at the stock prefix; core-10.1 layout is gone). Wheel md5 d2380a9502af9a5945b907df456836ea, libmlx md5 0970c43ceb39b10466a74b728b82d088 (S4VA gate string present). Installed into a fresh venv cloned from the config-of-record venv (only the mlx package replaced); the config-of-record venv was not modified (md5 verified pre and post).
- Isolated correctness: seeded identical q/k/v, S4VA off vs on, output maxdiff 0.00390625 = 1 bf16 ulp.

## Step 3 - same-tree A/B (anchor 322, 161 verifies, 3x3 interleaved, medians, SCLK 2200 MHz in 19/19 under-load samples)

- Arm A = base (wheel of record, MTP6_TAPE=r5, S4VA absent); arm C = fresh control (candidate wheel, S4VA off); arm B = candidate (candidate wheel + MLX_ROCM_SDPA_S4VA=1). Bench = the R5-report mtp-bench6.py (md5 943a028bd25763b4e886cb7cf41e43f8), sidecar model md5 02a54359cda8b247f76b277dd9bc4d4c verified pre and post.

| metric | A (base) | C (new control) | B (candidate) | B/C | B/A |
|---|---|---|---|---|---|
| TOKS_S (medians of 3) | 38.36 (30.09/38.37/38.36) | 43.91 (43.90/43.91/43.97) | 51.92 (51.88/51.92/51.96) | +18.1% | +35.2% |
| GEN s | 9.72 | 8.64 | 7.36 | -14.8% | -24.3% |
| TOKS | 373 | 380 | 382 | - | - |
| VERIFIES | 161 | 161 | 161 | - | - |
| D1/D2/D3 pct | 66.5/62.6/55.2 | 70.8/61.4/48.6 | 68.9/59.5/65.2 | -1.9/-1.9/+16.6 pts | - |
| tok/step | 2.311 | 2.354 | 2.366 | +0.5% | +2.3% |

- Per-op deltas (probe, candidate wheel vs base wheel): attention 19.96 -> 2.30 ms per 8 layers (-88.5%); SDPA qL=4 2.959 -> 0.054 ms (54.7x); the non-SDPA classes also differ (qmv_gdn 7.29 -> 6.02, mlp 15.32 -> 11.71) - wheel drift from the other grafts in the 82-file delta, which is exactly why arm C (same build, S4VA off) exists. B-vs-C (+18.1% t/s) is the clean graft attribution; B-vs-A (+35.2%) is the headline against the config-of-record.
- The verify wall itself: 44.85 -> 22.09 ms (-50.7%), measured by the probe on the two wheels at identical ctx.

## Lossless gate (PASS)

- FIRST20, all 9 runs: 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 - bit-identical across all arms and matching the R5-report reference prefix. PASS.
- First divergence from base: C at token 76 (wheel drift, m8/QMM/GDN grafts), B at token 32 (the S4VA 1-ulp argmax flip); generated text remains coherent and structurally identical in form across arms.
- Acceptance profile: B 68.9/59.5/65.2 (tok/step 2.366) vs C 70.8/61.4/48.6 (2.354) vs A 66.5/62.6/55.2 (2.311) - same order and magnitude, no systematic degradation; the D3 swing is the single-batch consequence of the argmax flip (43 vs 34 of 161).

## Gate and verdict

- Verify step cut: -50.7% (gate: >=30%) - PASS. t/s on the clean same-tree A/B: +18.1% (gate: >=+15%) - PASS. Lossless: PASS.
- Verdict: ADOPT. Engage the graft with MLX_ROCM_SDPA_S4VA=1 in the candidate wheel; the graft is default-off, so the no-graft behavior of that wheel is the stock path. Follow-up: the remaining 22.1 ms verify wall is GDN+MLP bound; an M=4 GEMM A/B is the next lever and must be same-tree only (t1858 toolchain-confound rule stands).

## Provenance

- Box: the GPU lane; lock queued and acquired iter=1 (10:17:41 UTC) behind two other tenants' queued jobs; 25 s idle before each run; SCLK sampled every 10 s - 2200 MHz in all 19 under-load samples.
- Wheels: arm A libmlx md5 8acf13fec6fc1636026d8e1dc69cd497 (the config-of-record, unmodified, verified pre and post); arm B/C libmlx md5 0970c43ceb39b10466a74b728b82d088 (fresh build, wheel md5 d2380a9502af9a5945b907df456836ea). Sidecar model md5 02a54359cda8b247f76b277dd9bc4d4c pre and post.
