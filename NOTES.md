# NOTES — SDPA/bandwidth probe (t1858 MTP multiplier #2)

Base: r5-correction-fusion @ 4b9285b. Same-tree A/B only; anchor 322 (MTP_MAX_TOKENS=322, 161 verifies); 3x3 interleaved; medians.

## Box state

- [x] flock + fuser check (gpu.lock unheld at 09:5xZ; fuser empty)
- [x] venv-r4v libmlx md5 = 8acf13fec6fc1636026d8e1dc69cd497 (config-of-record, unmodified)
- [x] mtp sidecar model-mtp.safetensors md5 = 02a54359cda8b247f76b277dd9bc4d4c
- [ ] SCLK sample at idle

Box state facts:
- venv-r4v wheel = mlx-0.32.3.dev20261104+75ca881 (lib .so dated 2026-11-04 03:05Z), built from vendor-r4v source: SDPA hip gate = stock `qL < 4` (no S4VA; strings grep count 0).
- BOX canonical tree = ~/vendor-canon working tree @ 75ca881 + uncommitted grafts (82 files, +5968): m8 runtime-causal SDPA vector + MLX_ROCM_SDPA_S4VA env gate. SDPA .cpp dispatch md5 identical to r5 repo vendor (3bcfed3b).
- Model: 32 layers = 24 linear_attention (GDN) + 8 full_attention (interval 4); 16 q-heads, 4 kv-heads, D=256, 4-bit affine quant.
- No rocprof on box; kernel ID = code read + behavior A/B + SDPA microbench timing (qL=1 vector vs qL=4 fallback vs qL=5 WMMA).
- Disk 2.4G free; build tree ~0.7-1.5G. Wheel build in flight from copy of vendor-canon (pip wheel, CMAKE_PREFIX_PATH=/opt/rocm/core-10.1, no ccache, 16 cores).

## Step 1: per-op probe of S=4 verify (base wheel venv-r4v, S4VA absent)

Probe: bench/sdpa-verify-perop.py (this branch; t1858 salvage + SDPA microbench section; mask="causal" API fix). ctx=539 (prompt 27 + AR 512), kL cache rows 768.

- [x] GDN-hybrid verified: NL=32 NG=24 NA=8 (full_attention_interval=4; 16 q-heads, 4 kv-heads, D=256)
- FUSED_VERIFY_MS = 44.85 (all 44.81-44.93) — matches ticket 45.07 (t1854)
- Per-layer class walls (ms): qmv_gdn 0.304 (24 layers = 7.29), gdn_fused 0.272 (24 = 6.52), attn 2.490 (8 = 19.96), mlp 0.479 (32 = 15.32), norms 0.046 (32 = 1.49), lm_head 0.001
- SDPA micro at real shapes (H=16 HKV=4 D=256 kL=768 causal): qL=1 (vector) 0.045 ms; qL=4 (scalar flash fallback) 2.959 ms = 65.6x the per-call decode cost; qL=5 (WMMA flash) 0.959 ms
- Effective BW at qL=4: 1.09 GB/s = 0.2% of the 546 GB/s floor; attention block = 19.96 ms of 44.85 ms fused verify (44%), per-layer attn wall 2.47-2.54 ms flat across the 8 layers

## Step 2: candidates

- (a) S4VA vector kernel at qL=4: wheel built from BOX vendor-canon tree (md5 d2380a95..., libmlx 0970c43c..., S4VA gate present), fresh venv-sdpa-cand (clone of venv-r4v + wheel swap); recipe = pip wheel, CMAKE_PREFIX_PATH=/opt/rocm (box userland moved: /opt/rocm/core-10.1 gone, ROCm 10.x at /opt/rocm), LD_LIBRARY_PATH=/opt/rocm/lib, no ccache
- (b) GDN-hybrid check: 24 GDN + 8 attention (verified)
- (c) layout/dispatch alternatives at S=4

S4VA candidate probe (venv-sdpa-cand, MLX_ROCM_SDPA_S4VA=1, same ctx=539):
- FUSED_VERIFY_MS = 22.09 (all 22.06-22.13) vs base 44.85 = -50.7% verify wall
- per-layer attn wall 2.49 -> 0.287 ms (-88.5%); SDPA micro qL=4 2.959 -> 0.054 ms (54.7x); eff BW 59.3 GB/s = 10.9% of floor
- other classes on the fresh wheel: qmv_gdn 7.29->6.02, gdn_fused 6.52->6.19, mlp 15.32->11.71, norms 1.49->1.47 (wheel drift vs r4v — why the same-wheel control arm C exists)
- SCLK idle 2200 MHz locked (probe log)

## Step 3: A/B (flock) — 322 anchor, 3x3 interleaved, medians, 2200 MHz locked (19/19 sclk)

Arms: A = base (venv-r4v, MTP6_TAPE=r5, S4VA absent); C = fresh control (venv-sdpa-cand, same build, S4VA off); B = candidate (venv-sdpa-cand + MLX_ROCM_SDPA_S4VA=1). bench6 md5 943a028b (R5-report script), sidecar md5 verified pre/post.

- A: 38.37 / 38.36 / 30.09 -> median 38.36 t/s; GEN 9.72 s; 373 toks; VERIFIES 161; D1/D2/D3 66.5/62.6/55.2; TPS 2.311
- C: 43.90 / 43.91 / 43.97 -> median 43.91 t/s; GEN 8.64 s; 380 toks; D1/D2/D3 70.8/61.4/48.6; TPS 2.354
- B: 51.88 / 51.92 / 51.96 -> median 51.92 t/s; GEN 7.36 s; 382 toks; D1/D2/D3 68.9/59.5/65.2; TPS 2.366
- B vs C (same-wheel graft A/B): +18.1% t/s (gate >=+15% PASS); B vs A (headline, wheel drift included): +35.2%
- per-op deltas (probe, new wheel vs base wheel): qmv_gdn 7.29->6.02 ms, gdn_fused 6.52->6.19, attn 19.96->2.30 (-88.5%), mlp 15.32->11.71, norms 1.49->1.47; SDPA qL=4 2.959->0.054 ms (54.7x), eff BW 1.1->59.3 GB/s (0.2%->10.9% of floor)
- verify wall: 44.85 -> 22.09 ms = -50.7% (gate >=30% cut PASS)
- lossless: FIRST20 identical all arms (== R5 reference FIRST20); B-vs-C first token divergence at 32 (1-ulp SDPA argmax flip, text coherent), C-vs-A at 76 (wheel drift: m8/QMM grafts); acceptance profiles same order (D1 67-71%, D2 59-63%), no systematic degradation
- SDPA isolated sanity (seeded random q/k/v, same inputs): S4VA vs scalar-flash out maxdiff = 0.00390625 = 1 bf16 ulp

## Step 4: deliverable

- REPORT-sdpa-probe.md (TL;DR + verdict ADOPT S4VA; gate PASS on both prongs)
- branch sdpa-bandwidth-probe pushed (bare + GitHub), PR #23 opened, body cites the ticket
