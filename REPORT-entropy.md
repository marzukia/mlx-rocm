# Lever 1 — lossless compressed 4-bit weight stream + in-CU decode (ent4)

Re-dispatch of run-1039 (ended empty). Goal: shrink the 4-bit weight stream on the
qmv path and reconstruct it bit-exactly on the GPU, so dequant is unchanged and
the lever is lossless by construction.

## Result (the build VM, RX 9070 XT / gfx1201, ROCm 10.1, Qwen3 9B 4-bit)

| metric | value |
|--------|-------|
| weight bytes (250 U32 tensors) | 4.476 GB → **4.185 GB** (×1.0696, **6.5 % saved**) |
| measured E[L] | **3.713 bits/code** (per-tensor shared canonical Huffman) |
| lossless (CPU verifier, 250/250) | bit-identical, bad=0 |
| lossless (GPU kernel gate, 3 tensors incl. K=12288 + full embed N=248320) | bad=0 |
| lossless (full-stack A/B output text) | **identical** (baseline == entropy) |
| decode overhead (A/B, 3-sample median) | baseline 65.77 vs entropy 65.70 tok/s → **~0.1 % (neutral)** |

**Net verdict: a win.** 6.5 % of the weight stream is saved in memory (the weight
buffer is allocated at the compressed `comp_bytes`; the dense form lives only in
one reused decode scratch), decode is lossless, and the decode cost is
negligible (~0.1 %), not the 20–35 ms/token a per-code 16-way prefix loop would
have cost.

## Why the decode is fast (vs the 20–35 ms prediction)

The decoder uses a **precomputed 15-bit → symbol LUT** (32 KB), built host-side
at load and uploaded once. Per code it is one LUT lookup + `pos +=
code_len[sym]` — O(1), no loop. The 32 KB LUT lives in L1/L2, so the ~9e9
lookups/token run at L1/L2 aggregate bandwidth (~0.1 ms/token total), not global
memory. The LUT also sidesteps a gfx1201/hipcc codegen bug: the nested loop that
assembles the canonical code order from `code_len` miscompiles on device
(order[0] came out 1 instead of 11; negative shifts produced codes like
1073741824). A single lookup has no such loop to miscompile.

## Format (see docs/entropy4.md)

- Per-tensor **shared canonical Huffman** over the 16-symbol 4-bit alphabet
  (a per-group codebook costs ~1 bit/code in table overhead and cancels the win).
- Bitstream packed MSB-first, row-major; u32 cumulative row-start table (4 B/row,
  O(1) direct row addressing); 4096-row chunks byte-aligned (≤7 pad bits).
- Weight buffer allocated at `comp_bytes`; decode kernel (one thread/row) packs
  2 codes/byte into a persistent `[N, K/2]` scratch; `qmv_wide_kernel` reads the
  scratch as `w` — identical FMA order, identical codes.
- Gated by `MLX_ROCM_ENTROPY=1` + `MLX_ROCM_ENTROPY_FILE`; default path and CI
  untouched.

## Work items

1. **Offline compressor + full-stream verifier** — green (250/250 bit-identical).
2. **In-CU decoder** (LUT) — green (kernel gate bad=0).
3. **Lossless gate** — green (CPU + GPU + full-stack output identical).
4. **A/B bench** — green (baseline 65.77 vs entropy 65.70 tok/s; ~0.1 % overhead).

## Notes

- The offline compression and verification are fully host-side (Python + C), so
  they are hermetic and CPU-only.
- The vendor patch (`scripts/apply-ent4-patch.py`) is idempotent across the
  pre-ent4 / LUT-only / complete states.
