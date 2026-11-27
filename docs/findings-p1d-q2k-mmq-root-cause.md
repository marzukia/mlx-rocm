# P1-d Q2_K mmq M=1 GEMV — root-cause card (Leg 1: analysis only)

Pinned defects (rel error vs kqfg ground truth; target ≤ 1e-3):

- **G1** rel err 1.65 / 1.47 — the Q2_K mmq path returns a wrong-magnitude result.
- **Fragment hole** — the F2 probe (offset encoded into the codes) shows output
  rows 0-7 → data, rows 8-15 → nothing (-5), rows 16-23 → 16-23: exactly the
  upper half of every 16-row C tile is zero.

Both defects are in the **Q2_K mmq M=1 GEMV path** (gfx1201 / ROCm 10.1,
`AMD_WMMA_AVAILABLE`). Ground truth is the kqfg dequant-at-GEMM kernel
(`kquant_file_gemm.hip`), which is bit-exact vs python-gguf 0.19.

Nothing in this leg is built or run on the GPU. Every line reference is to the
in-tree vendor source under
`vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/`.

---

## 1. Defect 1 — the [K, N/256] file is misread as [K/256, N]

### Ground truth (the file layout)

`kquant_file_gemm.hip:318` (kqfg_gemm_kernel):

```cpp
const uint8_t* blk = w + ((size_t)uf * (size_t)NBS + (size_t)cf) * (size_t)bb;
const float wv = deq_elem<ktype>(blk, tid);          // tid = 0..255 = n-offset
...
out[(m0+i)*NBS*256 + c*256 + tid] = acc[i];          // n = c*256 + tid
```

`uf = k` (the full K loop, 0..K-1), `cf = c` (the N super-block column,
0..NBS-1), `bb = 84` for Q2_K. So the weight file is a flat array of 84-byte
blocks indexed **`k * NBS + c`** — i.e. **[K rows, N/256 super-block columns]**.
Each block holds **256 N values (n = c*256 + 0..255) for ONE K row (k)**. The
256 elements of a block are 16 scale-groups × 16 N values; the 2-bit code for
element `e` is `blk[16 + 32*(e/128) + (e%32)] >> (2*((e/32)&3))`, the scale is
`sc[e/16]`, and `dm`/`minv` are per block.

So: **one 84-byte block = 1 K row × 256 N values.** There is no block that
holds 256 K values for 1 N row.

### The misread

`qmm.hip:152-156` (the mmq_q2k_file launcher):

```cpp
// the file weight buffer is [K/256, N] (K-major blocks); the mmq loader
// addresses kbx0 + i*stride_row_x in the upstream [N, K/256] convention.
// stride_row_x = 1 (N-row stride) and stride_kbx = N (K-block stride)
// resolve block (kb0, n) to kb0*N + n.
args.stride_kbx = N;
```

The comment asserts the file is **[K/256, N]** (K-major, one block per
(K/256-row, N-row), 256 K values per block). **That is the opposite of the
ground-truth [K, N/256] layout.** With `stride_row_x = 1` and `stride_kbx = N`,
`mmq.cuh:1039` passes

```cpp
load_tiles(x, x_gate, tile_x, offset_x + kb0*stride_kbx, tile_x_max_i, stride_row_x);
//                                  ^ kbx0 = offset_x + kb0*N              ^ stride = 1
```

and `mmq-load-tiles.cuh:621` reads

```cpp
const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;   // = x + kb0*N + i
```

— **one block per tile row (N row i)**, then reads all 256 codes from that single
block and interprets them as **256 K values for N row i**. But in the real
[K, N/256] file, the block at index `kb0*N + i` is the block for **K row
(kb0*N+i)/... , N super-block (kb0*N+i)%NBS** — it holds **256 N values for one
K row**, which the loader misreads as 256 K values for one N row. This is a
full K↔N transpose of the weight tile, which is exactly the G1 wrong-magnitude
signature.

(For the degenerate K=256 case the two layouts coincide — `1 == K/256` — which
is why the earlier K=256 probes looked clean and this was missed.)

### Named fix

**Transpose the loader to the [K, N/256] convention.** For tile row `i`
(global N row `n = n0 + i`, where `n0 = it*I*stride_row_x`) and K element `k`
(0..255, global K = `kb0*256 + k`):

```
block  = (kb0*256 + k) * NBS + (n / 256)        // ONE block per K element
elem   = n % 256                                 // the N offset within the block
code   = blk[16 + 32*(elem/128) + (elem%32)] >> (2*((elem/32)&3))
scale  = sc[(elem)/16] * dm                    // the 16-N-value group for this N row
```

i.e. **256 different blocks per tile row** (stride `NBS`, not `N`), each
contributing **one K value** (element `elem = n%256`), not one block per tile
row contributing 256 K values. Concretely:

- The per-`l` loop (`mmq-load-tiles.cuh:628-640`) must read a **different block
  per K element** `k` (block index `k*NBS + c`), extract the single code for
  element `elem`, and stage it as **one 2-bit code per sram element** (low 2
  bits, upper 26 bits zero) — **not** the dp4a 4-code scatter
  `(x_ql_0 >> (2*l)) & 0x03030303` that the current code writes (that scatter is
  correct for the dp4a path, where 1 byte = 1 K value and 4 bytes = 4 K values,
  but wrong for the WMMA path, where the 16x16x16 iu8 A/B fragment reads the 4
  bytes of a u32 as 4 distinct K positions).
- The scale/dm read (`mmq-load-tiles.cuh:642-648`) must come from the **per-k
  block** (scale group `elem/16`, dm of that block), not from one block shared
  by all 256 K values.
- The launcher (`qmm.hip:152-156`) must set the file strides for the [K, N/256]
  layout: the block index is `k*NBS + c`, so stepping K by 1 steps the block
  index by `NBS` (not `N`), and stepping the N row changes only `elem = n%256`
  and `c = n/256` (a `NBS` step only when crossing a 256-N super-block boundary).
  `stride_kbx` becomes `256*NBS` (the block stride for a 256-deep K tile), and
  the loader must receive the global N-row offset `n0` so it can compute
  `elem = (n0+i)%256` and `c = (n0+i)/256`.

---

## 2. Defect 2 — the 8/16-row WMMA fragment hole

### Evidence

- F2 probe: C rows 0-7 → data, rows 8-15 → nothing, rows 16-23 → 16-23. Exactly
  the upper half of every 16-row C tile is zero — a clean 8/16 hole, not
  scrambled data. (Scrambled sram from defect 1 would give non-zero garbage on
  all rows; the clean half-zero signature points at the C fragment, not the data.)
- The 93-fix C/D layout (`mma.cuh`, tile<16,16> `get_i`/`get_j`):
  `get_i(l) = 8*(lane>>4) + l` ("m = 8h + e"), `get_j(l) = lane & 15`
  ("n = lane%16"), with the comment **"verified gfx1201 C/D layout
  (r4d_gdn_wmma.h, probed on-card)"** — i.e. it was verified on-card **for the
  bf16/f16 wmma (the GDN kernel), not for the iu8 builtin.**
- `try3.s` (compiled probe of `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`):
  the real instruction is `v_wmma_i32_16x16x16_iu8 v[7:14], v[0:1], v[2:3],
  v[7:14] neg_lo:[1,1,0]` — C = 8 VGPRs (V8i), A/B = 2 VGPRs each (V2i),
  confirming the operand shapes but **not** the per-lane (row, col, k) mapping.
- The rocwmma library (`wmma_impl.hpp:981`) is a thin wrapper over the same
  builtin; it does not expose the gfx12 iu8 per-lane fragment mapping, so the
  iu8 C layout is **not** independently confirmed by the checked-out source.

### Root cause (best available; the exact iu8 mapping needs an on-card probe)

The iu8 16x16x16 WMMA **C-fragment writeback reuses the 93-fix layout
(`m = 8*(lane>>4) + e`, `n = lane%16`) that was verified on-card only for
bf16/f16.** The F2 signature (rows 8-15 = the `h = lane>>4 == 1` lanes'
registers = zero) indicates the iu8 C layout does **not** populate the h=1
lanes' 8 registers the way the 93 layout assumes. The writeback
(`mmq-vec-dot.cuh`, Q2_K mma path, the
`sum[(j0/tile_C::J + n)*tile_C::ne + l] += Cd.x[l]*...` line) therefore writes
zero into C rows 8-15.

The A/B staging (tile<16,4> `get_j`: `l==0 ? h : 2+h`; `load_ldmatrix` 16x4 int
→ `t.x[0] = sram[row][k0+h]`, `t.x[1] = sram[row][k0+2+h]`) is symmetric for A
and B and gives a valid partial dot over 2 K positions per word-pair (4 per
`k01` step, 64 steps = 256 K), so the hole is **not** in the A/B K mapping —
it is in the C writeback layout (and/or a mismatch between the iu8 C layout and
the 93 layout). **Confirming the exact iu8 C (and A/B) per-lane mapping
requires an on-card probe (Leg 2).**

### Named fix

**Probe the iu8 16x16x16 WMMA fragment layout on-card and correct the C
writeback (and A/B staging if the probe shows they also differ) to match the
actual iu8 layout.** Concretely: build a minimal probe kernel that writes the
V8i C (and reads the V2i A/B) of
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` to global memory with a
known A/B (identity / one-hot) and dump the 32-lane × 8 register pattern; compare
to the 93 layout (`m = 8h + e`, `n = lane%16`); then fix
`mma.cuh` tile<16,16> `get_i`/`get_j` (and `mmq-vec-dot.cuh` Q2_K mma writeback)
to the verified iu8 mapping. The bf16/f16 GDN path must be left untouched
(no-regression guardrail — it shares the 93 layout and is already correct).

---

## 3. Exact edit list

All paths relative to `vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/`.

### Defect 1 (layout transpose)

1. **`mmq/mmq-load-tiles.cuh:621`** — replace the one-block-per-row read
   `const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;` with a
   per-K-element block read. The per-`l` loop (628-640) becomes: for K element
   `k` (global `kb0*256 + k`), read block `x + (kb0*256 + k)*NBS + c` (where
   `c = (n0+i)/256`), extract the single 2-bit code for element
   `elem = (n0+i)%256`, and stage `x_qs[i*sram_stride + k] = code` (one code per
   element, low 2 bits) instead of `(x_ql_0 >> (2*l)) & 0x03030303`.
2. **`mmq/mmq-load-tiles.cuh:642-648`** — read the scale from the **per-k block**
   (`scales[elem/16]`) and its `dm`, instead of one block's `scales[kqsx]` shared
   by all 256 K values.
3. **`mmq/mmq.cuh:1039`** — pass the global N-row offset `n0` (and `NBS`) to
   `load_tiles` so the loader can compute `elem = (n0+i)%256` and
   `c = (n0+i)/256`. (The `load_tiles` signature at `mmq.cuh:15` gains the
   `n0`/`NBS` parameters; the declaration at 15 and the Q2_K definition at
   621-648 both change.)
4. **`qmm.hip:152-156`** — fix the launcher strides for the [K, N/256] layout:
   `stride_kbx = 256*NBS` (the block stride for a 256-deep K tile; not `N`), and
   correct the comment. `stride_row_x` no longer indexes the block directly (the
   N row changes `elem`/`c`, not the per-row block stride).

### Defect 2 (WMMA fragment hole)

5. **`mmq/mma.cuh` tile<16,16> `get_i`/`get_j`** (the 93-fix block) — after the
   on-card iu8 probe, correct `get_i`/`get_j` to the verified iu8 C mapping
   (only if the probe shows the iu8 C layout differs from `m = 8h + e`,
   `n = lane%16`). Keep the bf16/f16 GDN path on the 93 layout.
6. **`mmq/mmq-vec-dot.cuh` Q2_K mma writeback** (the
   `sum[(j0/tile_C::J + n)*tile_C::ne + l] += ...` block) — align to the
   corrected C layout if #5 changes it.

### Coupling

7. The defect-1 sram repack (one code per element) and the defect-2 C-layout
   fix are independent, but **both must land before the F2/G1 probes are
   re-run**; verify the A/B staging K mapping still lines up with the new sram
   pack (the 16x4 `get_j` `h`/`2+h` pattern reads `sram[row][k0+h]` /
   `sram[row][k0+2+h]`, which is unchanged by the defect-1 repack since it is
   index-based, not value-based).

---

## 4. Risk notes

- **Shared mma.cuh with the KQFG M=2 path and the bf16/f16 GDN kernel.** The
  16x16 int iu8 mma wrapper and the 93 C layout are shared. Any change to
  tile<16,16> `get_i`/`get_j` or the iu8 wrapper risks regressing the already-
  correct bf16/f16 GDN path and the KQFG M=2 path. Guardrail: keep the bf16/f16
  path on the 93 layout; change only the iu8 mapping; re-run the GDN regression
  after any `mma.cuh` edit.
- **The dp4a path (non-WMMA) depends on the 4-code scatter.** The defect-1 fix
  changes the sram pack to one-code-per-element for the WMMA branch. The dp4a
  branch (`#else` at `mmq-load-tiles.cuh:639`) must keep the 4-code scatter.
  Keep the two packs in separate `#if` branches (the existing
  `AMD_MFMA_AVAILABLE || TURING_MMA_AVAILABLE || AMD_WMMA_AVAILABLE` vs `#else`
  split) and do not merge them.
- **`stride_kbx` / `stride_row_x` are shared args.** They are read by the general
  mmq dense kernel (`mmq.cuh:1039`, 1193, 1292, 1376). Changing their meaning
  for Q2_K must not change them for the other quant types (Q4_0/Q8_0/etc.) that
  use the upstream [N, K/256] convention (`stride_kbx = 1`). Scope the [K, N/256]
  strides to the Q2_K file path only.
- **K=256 degeneracy.** At K=256 the two layouts coincide (`1 == K/256`); the fix
  must still be correct there (it is — `NBS` indexing is valid for any K), but the
  earlier "clean" K=256 probes do not de-risk the general case.
- **The iu8 C layout is unconfirmed.** The defect-2 fix is contingent on the
  on-card probe (Leg 2). Do not ship a C-layout change based on the static
  analysis alone.

---

## 5. Leg-2 steps (build + GPU, on guest-a)

1. **Probe the iu8 fragment layout (defect 2 first — it is the cheaper oracle).**
   Build a minimal kernel that runs
   `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` with a one-hot / identity
   A and B and writes the V8i C (all 32 lanes × 8) plus the V2i A/B to global
   memory. Dump the pattern; confirm or refute the 93 layout
   (`m = 8h + e`, `n = lane%16`) for iu8. This names the exact C (and A/B)
   mapping and whether #5/#6 are needed.
2. **Apply the defect-1 transpose** (edits #1-#4): per-K-element block read,
   one-code-per-element sram pack, per-k scale/dm, [K, N/256] launcher strides.
   Build the wheel on guest-a (`MLX_VENDOR_DIR=<vendor clone> bash
   scripts/build-wheel.sh`).
3. **Apply the defect-2 C-layout fix** (edits #5/#6) per the probe result.
4. **Re-run the F2 probe** (offset-encoded codes) — expect rows 0-15 all → data
   (the 8/16 hole closed).
5. **Re-run the G1 rel-error probe** vs the kqfg ground truth — expect rel err
   ≤ 1e-3.
6. **No-regression:** re-run the bf16/f16 GDN kernel and the KQFG M=2 path
   (they share `mma.cuh`); confirm bit-exact / unchanged.
7. **Hermetic gate:** `uv run pytest -m "not gpu"` + `uv run python
   scripts/size_ratchet.py --base HEAD` (the `load_tiles` signature change and the
   loader rewrite may touch the size gate — split if a file crosses 800 non-blank
   lines; `--update` only after a split).
