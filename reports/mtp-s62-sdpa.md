# m4: direct SDPA A/B at layer 3 — the m3 root cause, isolated

Date: 2026-10-29 (UTC). Run on guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest,
venv-d, mlx 0.32.3.dev20261028+75ca881, model `/models/qwen3-9b-mlx`).
Script: `scripts/diag-s62-sdpa.py` (MODE=harvest + MODE=ab). Guest log
`/models/tmp/run-785/diag-m4-harvest.log`, captures
`/models/tmp/run-785/cap-m4-sdpa.pkl`, kernel traces
`/models/tmp/run-785/prof-*.csv_kernel_trace.csv`.

## What m4 does

The m3 per-layer diff localized the first diverging COMPUTATION to the
layer-3 attention (SDPA) output for token 3545: row 1 of a 2-row causal
SDPA (variant A, T=2 verify) vs a 1-row unmasked SDPA (variant B, T=1 AR),
over bit-identical q/k/v. m4 isolates WHERE in the SDPA path the T-dependence
lives (mask handling vs row-count tiling vs kernel dispatch) and records the
exact kernel name + parameter delta (the fix pointer).

The m3 pkl (`cap-m3-s62.pkl`) holds the per-layer STATE (gdn conv/ssm, attn
k/v/offset) but NOT the layer-3 SDPA q inputs or the SDPA output. So m4
re-ran the m3 flow once (deterministic; re-armed at the S62 arm: prev=279
draft=3545 n_out=61, act=ACC), hooked `mx.fast.scaled_dot_product_attention`
during variant A's T=2 forward, and captured the FIRST hooked SDPA call
(layer 3 — the first full-attention layer; layers 0-2 are GDN). The captured
k/v was verified bit-exact against the m3 pkl's layer-3 cache state
(0/91136 both). The captured tensors (q_2row, k_89, v_89, sdpa_out, scale,
mask, dtypes) were saved to `cap-m4-sdpa.pkl`. The A/B test (MODE=ab) then
runs the direct SDPA calls on the captured tensors (no model).

## The exact SDPA call the model makes

From `qwen3_next.Qwen3NextAttention` + `mlx_lm.models.base
.scaled_dot_product_attention` + `cache.py create_attention_mask`:

- q layout `[B, n_h=16, S, h_dim=256]`; k/v layout `[B, n_kv=4, S, h_dim=256]`
- GQA 16 q heads / 4 kv heads (4:1), handled natively by
  `mx.fast.scaled_dot_product_attention`
- scale = head_dim^-0.5 = 256^-0.5 = 0.0625
- mask = the string `"causal"` for N>1 (T=2), `None` for N==1 (T=1)
  (`cache.py create_attention_mask(N, offset, return_array, window_size)`:
  window_size None, N=2 -> "causal"; N=1 -> None)
- the non-quantized path (KVCache has no `bits`) is taken, so the call is
  exactly `mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask=...)`
- all dtypes bfloat16 (q, k, v, out)

Captured layer-3 SDPA: `qshape=(1,16,2,256) kshape=(1,4,89,256)
vshape=(1,4,89,256) scale=0.0625 mask='causal'`.

## The A/B table (direct SDPA on the captured tensors)

| # | call | mask | S | B | vs | result |
|---|------|------|---|---|----|--------|
| 1 | A: re-run S=2 causal | "causal" | 2 | 1 | captured SDPA out | **MATCH** (n=8192) — sanity PASS |
| 2 | B1: S=1 unmasked row[279], 88 keys | None | 1 | 1 | A row 0 | **MATCH** (n=4096) — control PASS |
| 3 | B2: S=1 unmasked row[3545], 89 keys | None | 1 | 1 | A row 1 | **MATCH** (n=4096) — m3 divergence NOT reproduced |

B2 is the correct 1-row causal equivalent of A row 1: both see the same 89
keys (87 cached + k279 + k3545); the only difference is A computes 3545 as row
1 of a 2-row causal-masked call, B2 as a 1-row unmasked call. **They are
bit-exact.** So the SDPA output for token 3545 is the SAME in A and B — the
m3 "divergence" is NOT reproduced by direct SDPA.

## Bisection (single-variable A/Bs)

| # | variable | A | B | result | reading |
|---|----------|---|---|--------|---------|
| a | mask off both | unmasked S=2 row1 | unmasked S=1 (B2) | **MATCH** | unmasked path is row-count-invariant |
| b | mask representation | full-True array S=2 row1 | "causal" string S=2 row1 | **DIFF** | the mask OBJECT (array vs string) changes the kernel + result, even for the row that is unmasked in both |
| c | row-0 content | zero-row0 "causal" S=2 row1 | A row1 | **MATCH** | row 0's content does not matter (row 1 is independent of row 0) |
| d | seq-len padding | "causal" S=4 (2 valid) row1 | A row1 (S=2) | **DIFF** | padding the sequence length (S=4 vs S=2) changes the kernel + result, attend set identical |
| e | batch | batched B=2,S=1 row1 | B2 (B=1,S=1) | **DIFF** | batching changes the grid (Z=2 vs Z=1), same 1-pass kernel |

## Kernel dispatch ID (rocprofv3 --kernel-trace, gfx1201)

The SDPA kernel is T-dependent — it dispatches on the mask representation, the
sequence length S, and the batch B:

| spec | mask | S | B | SDPA kernel | grid | workgroup | LDS | VGPR |
|------|------|---|---|-----------|------|-----------|-----|------|
| causal_S2 | "causal" | 2 | 1 | `kernel_sdpav_1pass<bf16, true, 256>` | (16384,2,1) | (1024,1,1) | 4608B | 40 |
| unmasked_S1 | None | 1 | 1 | `kernel_sdpav_1pass<bf16, false, 256>` | (16384,1,1) | (1024,1,1) | 4608B | 48 |
| batched_S1 | None | 1 | 2 | `kernel_sdpav_1pass<bf16, false, 256>` | (16384,1,2) | (1024,1,1) | 4608B | 48 |
| causal_S4 | "causal" | 4 | 1 | `kernel_sdpa_flash_opt<bf16, true, 256, 128, 64>` | (2048,1,1) | (128,1,1) | 65536B | 248 |
| fulltrue_S2 | all-True array | 2 | 1 | **non-flash 2-pass**: `Cijk_Alik_..._MT16x16x128` (QK^T GEMM) + `Cijk_Ailk_..._MT16x16x32` (weighted-sum GEMM) + `softmax_kernel<bf16,float,256,4>` + `binary_sv<Multiply>` + `ternary_g<Select>` + `fillBufferUnAligned` | GEMM1 (192,1,1) / GEMM2 (512,1,1) / softmax (8192,1,1) | (32,1,1) / (256,1,1) | GEMM1 12800B / GEMM2 6656B / softmax 512B | 128 / 128 / 16 |

The parameter delta (the fix pointer):

- **mask representation**: the string `"causal"` dispatches the fast
  `kernel_sdpav_1pass` (1-pass, masked=true); an explicit boolean array mask
  dispatches the slow non-flash 2-pass path (two `Cijk` GEMMs + `softmax` +
  `select` + `multiply`). Same S=2, same attend set for row 1, different
  kernel, different floating-point result (bisection b = DIFF).
- **sequence length**: S=2 with `"causal"` dispatches `kernel_sdpav_1pass`;
  S=4 with `"causal"` dispatches `kernel_sdpa_flash_opt` (a different flash
  kernel, 65536B LDS, 248 VGPR). Same attend set for row 1, different kernel,
  different result (bisection d = DIFF).
- **batch**: B=1 vs B=2 both dispatch `kernel_sdpav_1pass` (same kernel), but
  the grid Z differs (1 vs 2) — one workgroup-tile per row/batch. Bisection e
  = DIFF (the batched result differs from the single-row result).

## Data-dependent vs structural verdict

- **The m3 divergence (2-row causal vs 1-row unmasked) is NOT in the SDPA.**
  B2 (1-row unmasked, 89 keys) == A row 1 (2-row causal, row 1), bit-exact, on
  the captured data. The synthetic test (fresh random q/k/v, same shapes) also
  gives causal S=2 row1 == unmasked S=1 (MATCH). So the m3 "divergence" is NOT
  reproduced by direct SDPA on either the captured or the random data. The SDPA
  is NOT the source of the m3 divergence.
- **The SDPA kernel dispatch IS T-dependent (structural).** The mask
  representation (string vs explicit array) and the sequence length S change the
  kernel dispatch (bisection b, d = DIFF; the kernel table above). This is a
  real, structural T-dependence in the ROCm SDPA — but it is NOT the m3
  divergence (the m3 divergence is NOT reproduced).

## Verdict

- **The m3 report's attribution to the SDPA is imprecise.** The direct A/B
  shows the SDPA output for token 3545 is the SAME in A and B (B2 == A row 1,
  bit-exact, on both captured and random data). So the m3 divergence (the
  layer-3 block output / GDN-4 conv state) is NOT caused by the SDPA.
- **The SDPA kernel dispatch is T-dependent (structural).** The mask
  representation (string "causal" vs explicit boolean array) and the sequence
  length S (S=2 vs S=4) dispatch different kernels with different
  floating-point results. This is the fix pointer for any future
  mask/seq-len-dependent SDPA discrepancy:
  - string "causal" -> `kernel_sdpav_1pass<bf16, true, 256>` (1-pass)
  - explicit array mask -> non-flash 2-pass (Cijk GEMM x2 + softmax + select + multiply)
  - S=4 "causal" -> `kernel_sdpa_flash_opt<bf16, true, 256, 128, 64>` (flash)
- **The m3 divergence must be re-localized.** Since the SDPA is exonerated
  (B2 == A row 1), the m3 divergence (the layer-3 block output / GDN-4 conv
  state) is NOT in the SDPA. The m3 chain of evidence (GDN 4 conv state
  diverges) is solid, so the layer-3 block output differs — but the SDPA
  (the core attention computation) is the same. The divergence is therefore in
  the attention layer's non-SDPA part (the gate / o_proj) or the MLP, OR the
  m3 chain of evidence has a flaw (e.g. the GDN-2 output — the layer-3 block
  input — is NOT bit-exact even though the GDN-2 state is). The next step
  (m5) should directly A/B the layer-3 block's non-SDPA components (gate,
  o_proj, MLP) on the captured tensors to re-localize the m3 divergence.

## Artifacts

- `scripts/diag-s62-sdpa.py` (MODE=harvest + MODE=ab)
- `reports/mtp-s62-sdpa.md` (this file)
- guest: `/models/tmp/run-785/cap-m4-sdpa.pkl`,
  `/models/tmp/run-785/diag-m4-harvest.log`,
  `/models/tmp/run-785/prof-*.csv_kernel_trace.csv`
