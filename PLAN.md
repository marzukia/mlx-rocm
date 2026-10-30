# PLAN — mlx-rocm

Living plan. Full research report: [docs/plan.md](docs/plan.md) (an earlier research pass, 2026-10-25).

## Phase 3 — Lever 1: lossless compressed 4bit weight stream + in-CU decoder (run-1039 re-dispatch, 2026-10-31)

Goal: byte-identical greedy ALLTOKS vs uncompressed AR at ctx=14 and ctx=29K; A/B bench for GB/token + tok/s. Baseline anchor: 5.95 GB/token, 100.0 tok/s @93% (595.2 GB/s of 640).

### Format spec draft v0 (commit gate: step 0)

- **Unit:** per-tensor. One **shared canonical Huffman codebook** over the 16 4bit code values (signed int4), table = 16 lengths x 4 bits = 8 bytes + 4-byte checksum, stored once per tensor (negligible: 250 tensors).
- **Huffman vs ANS — Huffman. Justification:** with a 16-symbol alphabet ANS buys <=0.1 bits/code (3.74 vs 3.71) at the cost of per-symbol state + renormalization that serializes in-CU decoding; canonical Huffman decodes in one prefix pass with a 16-entry table in registers — the 0.1 bit gap is <3% of the total win and the decoder ALU is what this lever bets on.
- **Measured E[L] (run-1023 per-tensor histograms):** per-tensor shared = 3.713 bits/code (range 3.62–3.75); global = 3.737; raw = 4.0. A per-group (64-code) codebook would need 8+ bytes of table per 64 codes = 1 bit/code overhead — it cancels the win, hence shared-per-tensor.
- **Group size:** 64 (== quantization group; one decoded group == one qmv K-tile FMA chunk). Groups are NOT byte-aligned in the stream (no 15-bit/chunk padding). Row starts = per-row **u32 cumulative row-start table** (absolute stream bit offset per row; 4B/row, 7.7MB model total — a block reads one u32, no prefix sum; max value 4.7e9 bits < 2^32). Overhead ≈ 0.007 bits/code — negligible. (Model facts: all 4bit tensors K=4096 except down_proj K=12288; U32 [N, K/8] packed layout; 250 tensors.)
- **Losslessness by construction:** decode reconstructs the exact 4bit code values; the existing dequant (scale/bias from the safetensors side data, untouched) runs on the decoded tile → bit-identical results. Greedy ALLTOKS gate at ctx=14 and ctx=29K vs uncompressed AR (fresh process per path, discarded warm pass, steady window skip=56).

### Per-token byte budget (run-1023 accounting, re-anchored at E=3.713)

| item | current (GB/token) | compressed (GB/token) |
|------|-------------------|-----------------------|
| 4bit codes (4.476 @ 4.0 bits) | 4.476 | 4.155 (3.713 bits) |
| scale/bias | 0.560 | 0.560 |
| bf16 other (off qmv path) | 0.913 | 0.913 |
| row-length table | 0 | 0.008 |
| **total** | **5.95** | **5.63** |

Projection @93% (595.2 GB/s): 5.95 → 100.0; 5.63 → **105.8 tok/s** (full stream). qmv-path only (codes+scale/bias = 4.72 vs 5.04 GB): 118.2 → **126.2 tok/s**. (run-1023 anchored at raw H=3.706 → 108.0; per-tensor E[L]=3.713 lands at 105.8.)

### Kernel design
- qmv wide path (per WORKSTATE: row=blockIdx.y, grid=(N/8, M)): each (m, n) block owns one compressed weight row (≤5.7KB at K=12288). Block start: read its row's start bit offset from the L2-resident table; load the row's compressed bytes into LDS; cooperative sequential decode (warp lanes split each 64-code group 2-codes/lane, shfl-passed bit positions, 16-entry canonical codebook in registers) → decoded 4bit codes in LDS (≤6KB at K=12288; both fit the 64KB LDS budget); the EXISTING FMA loop then reads the decoded uint4 tile from LDS instead of global — identical FMA order, identical code values → lossless.
- Fallback: separate decode pass writing unpacked per-layer scratch (≤3.2MB/layer, L2-resident) for small tiles or if inline decode ALU is measured to hurt; qmv then reads the scratch.
- Env flag `MLX_ROCM_ENTROPY=1` selects the compressed path; default path + CI untouched. Compressed streams shipped as a side file (`*.ent4`) next to the safetensors; loader picks it up when the flag is set. Scale/bias tensors stay raw (unchanged).

### Work items (in order, commit after each verified step)
1. Offline compressor (CPU): safetensors → per-tensor codebook + compressed stream + side table; in-tree format spec doc (docs/entropy4.md). FIRST commit.
2. Loader + in-CU decoder in the 4bit qmv path (vendor patch), `MLX_ROCM_ENTROPY=1`.
3. Lossless gate: ALLTOKS byte-identical @ ctx=14 and ctx=29K vs uncompressed AR.
4. A/B bench under flock: GB/token + tok/s + delta vs 5.95/100.0.
5. Leak rule on new/changed files (repo public). 6. CI green. 7. Push branch `phase3-entropy-compress` to mirror + fork (no PR). 8. EXIT after push; REPORT-entropy.md in-tree.

## Definition of minimum

qmlx-serve runs a 9B-class dense Qwen3 on the build VM (RX 9070 XT / gfx1201, ROCm 10.1),
correctly, at usable decode speed. MTP, disk KV, 27B+/9B-hybrid = Phase 2.

**ROCm version (single source of truth, 2026-10-26):** the verified guest toolchain on
the build VM is **ROCm 10.1** (core at `/opt/rocm/core-10.1`,
`CMAKE_PREFIX_PATH=/opt/rocm/core-10.1`, `amdrocm-*-dev10.1` packages; see
docs/quickstart.md). Nothing else is verified: 6.4 was the early probe validation and
7.7 the original target — both are historical, not the verified toolchain.

## Current state (2026-10-25)

- Research done (an earlier research pass): the backend is ~90% done in an open community
  MLX ROCm backend PR; the vendored tree is a forward-merge of that base.
  Effort: 2.5-5 person-weeks for the minimum (Phase 0+1).
- The build VM is UP (RX 9070 XT; userland ROCm 10.1 — the only verified toolchain).
  The MVP is being built on the build VM (smoke unblocked).
- Repo bootstrapped: hermetic CI (lint/format/pytest/size gate), vendor pin, build +
  smoke scripts. GPU validation still to be done on the build VM.

## Phases

### Phase 0 — Adopt + build + smoke (0.5-1 person-week)
1. Clone the pinned primary (vendor.json), build the wheel:
   `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS"`.
2. On the build VM: `mx.rocm.is_available()` true, matmul/softmax correct, Qwen3-0.6B (bf16 + 8bit)
   generates sane text. (`scripts/smoke.py`)
- **Exit criteria:** is_available()==True; numerics correct; 0.6B sane.
- **Gate:** if primary (the vendor ref) does not build clean on ROCm 10.1, fall back to
  the vendor ref and record why (docs/plan.md open question 1).

### Phase 1 — Validate gfx1201 + qmlx-serve minimum (1-2 person-weeks)
1. QMM 4bit on gfx1201 (Rdna4 tier + WMMA dense qmm) vs CPU numerics.
2. SDPA decode (vector) + prefill (WMMA flash); verify prefill uses the WMMA path.
3. fp8 GEMM: test the known gfx1201 fp8 prefill bug; ship `MLX_ROCM_GEMM_BF16=1` as the
   default until proven clean (docs/plan.md).
4. Allocator on a discrete GPU: host_shadow / pinned host_stage path (not APU fine-grained).
5. Run qmlx-serve with a 9B-class dense Qwen3; check the `is_metal` error-classification
   path in qmlx `vllm_mlx/engine_core.py:675` handles ROCm/hip runtime errors.
- **Exit criteria:** 9B Qwen3 served on gfx1201; no NaNs; no OOM→SEGV single-process.
  (`scripts/smoke-serve.sh`)

### Phase 2 — Full qMLX stack (deferred; ~2-4 person-weeks)
GatedDeltaNet (9B hybrid), MoE (A10B/27B+), MTP spec decode (blocked on the M3
stream-TLS fix), disk KV cache validation. Details in docs/plan.md §Phase 2.

## Open questions (operator's call)

1. Base: the vendor ref (preferred) vs the vendor ref (fallback) — resolved by Phase 0 build.
2. Wheel vs source: source-on-the build VM for now; PyPI wheel (`mlx-rocm` name slot already held
   by us) is a Phase 2+ concern.
3. fp8 default: `MLX_ROCM_GEMM_BF16=1` until the gfx1201 fp8 prefill is proven clean on 10.1.
4. MTP: Phase 2, strictly (M3 stream-TLS fix blocks it either way).
