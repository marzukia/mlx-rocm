# ent4 — lossless compressed 4-bit weight stream (Phase 3, lever 1)

Lossless entropy coding of the packed 4-bit weight codes. Decode reconstructs the
exact 4-bit code values, so the existing dequant (scale/bias from the untouched
safetensors side data) produces **bit-identical** results. Lossless by
construction; the only failure mode is a corrupt stream (checksum/row-end checks
catch it).

## Why Huffman, not ANS

16-symbol alphabet. ANS buys ≤0.1 bits/code (3.70 → ~3.71) over Huffman but
adds per-symbol state + renormalization that serializes in-CU decoding.
Canonical Huffman decodes in one prefix pass with a 16-entry codebook in
registers. The gap is <3% of the total win; the decoder ALU is what this lever
bets on. Per-tensor **shared** codebook: a per-group (64-code) codebook costs
8+ bytes / 64 codes = 1 bit/code of table overhead and cancels the win.

## Measured entropy (run-1023 per-tensor histograms)

- per-tensor shared E[L] = **3.713 bits/code** (range 3.62–3.75)
- global codebook: 3.737; raw: 4.0

## File format (`*.ent4`)

One file per model, side-by-side with the safetensors. Scale/bias stay raw in
the safetensors (unchanged path).

```
offset  size    field
0       16      magic "MLXENT4V0" + 7 zero bytes
16      2       version (u16 LE, = 0)
18      2       n_tensors (u16 LE)
20      ...     per tensor, in order:
                4   n_rows (u32)
                4   k (u32)            # codes per row (K; 4096 or 12288)
                4   comp_bytes (u32)   # stream bytes for this tensor
                4   total_bits (u32)   # sum of REAL row bits (end-check base)
                1   plain (u8)         # 1 = canonical 4-bit (above entropy gate)
                16  code lengths (u8 x16, symbol-major, 0 = absent)
                4n  row starts (u32 x n, absolute stream bit offset per row)
                comp_bytes  compressed bit stream (byte-aligned per 4096-row chunk)
```

**Per-tensor entropy gate** (decision C; `--gate 3.9` default, `--gate 0` =
compress-all legacy): a tensor with measured `E >= gate` is written *plain* —
a canonical 4-bit encoding (identity symbols, all lengths 4, stream = raw
packed bytes high-nibble-first, `row_start[r] = r*k*4`, `comp_bytes = n*k/2`,
`total_bits = n*k*4`). The decoder works unchanged: a length-4 canonical code
is symbol identity, the 11-bit window always contains the 4 code bits, and
the LUT covers every 11-bit suffix. On the final qwen3-9b-mlx dataset the
gate is a no-op (all 250 tensors E ≤ 3.75 → all compressed); it is the
safety mechanism for models where a class runs above 3.9.

Side file `<name>.ent4.index.json`: `{"format":"ent4","version":0,
"tensors":[names], "gate":3.9, "tensor_info":[{name,class,plain,E,n,k,
comp_bytes}]}` (tensor_info is diagnostics; the loader only needs `tensors`).

### Stream layout

- Codes are encoded **row-major** (row 0..n-1, code 0..k-1), MSB-first.
- The 16 canonical code values are assigned by sorting symbols by
  (length, symbol) — the standard canonical form, so the decoder derives the
  same code values from the 16 lengths alone.
- Every **4096-row chunk** is byte-aligned (0–7 padding bits at the chunk end,
  zero). Row starts are absolute bit offsets into the whole stream and already
  account for chunk padding, so a block reads one u32 and seeks — no prefix
  sum.
- Row-end integrity: after decoding row r, the bit cursor must land within 7
  bits before row r+1's start (chunk padding), and the final row's cursor must
  land within 7 bits × n_chunks of `total_bits`.
- **Stream-end padding**: consumers (the C verifier, the GPU payload, the
  standalone kernel gate) zero-pad the per-tensor stream by 3 extra bytes —
  the 15-bit window decoder reads a 4-byte window and the last code's window
  can extend up to 3 bytes past the content end (run-1044 8ec0580).

### Why a u32 row-start table

Per-row u16 lengths overflow (a down_proj row is 12288 codes × ~3.7 bits ≈
45k bits > 65535). A u32 cumulative table is 4 B/row = 7.7 MB for the whole
model (≈0.007 bits/code — negligible) and gives O(1) direct row addressing.

## Byte budget (per token, run-1023 accounting)

| item | current | compressed |
|------|---------|-----------|
| 4-bit codes (4.476 GB @ 4.0 bits) | 4.476 | 4.155 (3.713 bits) |
| scale/bias | 0.560 | 0.560 |
| bf16 other (off qmv path) | 0.913 | 0.913 |
| row-start table | 0 | 0.008 |
| **total** | **5.95** | **5.63** |

Projection @93% (595.2 GB/s): 5.95 → 100.0 tok/s; 5.63 → **105.8 tok/s**
(full stream). qmv-path only (codes + scale/bias = 4.72 vs 5.04 GB): 118.2 →
**126.2 tok/s**.

**Measured A/B (the build VM, 3-sample median, total incl. prefill): baseline 65.77
vs entropy 65.70 tok/s → ~0.1 % overhead (neutral).** The projection is
optimistic: it assumes the whole 0.32 GB/token saving is on the memory-bound
critical path. In practice the LUT decode (~0.1 ms/token) and the
non-bandwidth-bound portions offset most of the saving, so the lever is
speed-neutral; its value is the 6.5 % memory reduction (weight buffer at
`comp_bytes`).

## Decoder

- **Reference** (CPU, C): `scripts/ent4_verify.c` — sequential decode, 16-way
  prefix match (only the top `lengths[s]` bits of the 15-bit window are
  compared, since the lower bits belong to the next code), row-end checks.
- **GPU** (work item 2): a separate decode kernel (one thread per row) walks
  the bitstream and packs 2 codes/byte into a persistent dense scratch
  (`[N, K/2]`); the existing `qmv_wide_kernel` then reads the scratch as `w`
  — identical FMA order, identical code values. Decode is O(1) per code via a
  **precomputed 15-bit → symbol LUT** (32 KB, built host-side at load and
  uploaded once). This is both fast (single lookup, no per-code loop) and
  robust: it sidesteps a gfx1201/hipcc codegen bug where the nested loop
  that assembles the canonical code order from `code_len` miscompiles on
  device (verified by isolating each step). `MLX_ROCM_ENTROPY=1` selects the
  path; default path and CI are untouched.
- **Memory**: the weight buffer is allocated at the *compressed* size
  (`comp_bytes`); the dense form lives only in the persistent decode scratch.
  Net in-memory saving = `4.476 − 4.185 − max_scratch` GB (the scratch is one
  reused buffer of size `max(N·K/2)` over the qmv tensors, ~tens of MB).

## Verification

`scripts/ent4_verify.py` decodes every tensor from the `.ent4` file and
bit-compares against the original U32 tensors in the safetensors. Any
mismatch, bad code, or row-end misalignment is a hard failure.
