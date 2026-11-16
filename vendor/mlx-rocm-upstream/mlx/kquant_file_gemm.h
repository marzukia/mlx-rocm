// File-layout k-quant ops — the dequant-at-GEMM family.
//
// W1 build: consumes the reference GGUF's raw 256-element super-block
// bytes in the FILE's grouping (super-block = 256 consecutive output
// values at fixed input k; the w array shape is (K, N/256, block_bytes))
// and dequantizes INSIDE the op — no f32/bf16 weight materialization,
// no f32 weight cache. The dequant math is the single-sourced,
// G1-verified per-class k-quant math (Q4_K: kquant_q4k_math.hpp;
// Q5_K/Q6_K/Q8_0: the WS3-verified ggml 6023692 ports); the file-layout
// part is the access/iteration pattern (the 32-unit 4096-element window
// even/odd unscramble, if present in the file), not the math.
//
// Layout parameters (supplied from the pinned per-tensor tables; format
// semantics, not per-file constants in the kernels):
//   ktype:      8 = Q8_0 (8 x 34 B 32-elem blocks per 256), 12 = Q4_K
//               (144 B), 13 = Q5_K (176 B), 14 = Q6_K (210 B);
//   mode:       0 = identity (super-blocks in canonical engine order),
//               1 = mid_transpose (the 32-unit 4096-element window holds
//               [unit 0][even engine units 2,4..,M][odd 1,3..,M-1][last]
//               in the file; restored by the canonical inverse);
//   axis:       0 = the window runs over 32 rows along d0 (K), 1 = over
//               32 super-block columns along d1 (N);
//   group/keep_ends/pairs: window shape (32/1/15 for this file family;
//               carried as data so the ops stay generic).
#pragma once

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

// y = x @ W, with x (M, K) float32/bfloat16, w (K, N/256, block_bytes)
// uint8 (the file's raw super-block stream), W (K, N) dequantized in-kernel.
// Output (M, N) in x's dtype. f32 accumulation; per-element dequant is
// bit-exact vs the pinned GGUF oracle (G1).
MLX_API array kquant_file_gemm(
    const array& x,
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    StreamOrDevice s = StreamOrDevice());

// y = gather of W's columns: idx (B) integer, w (K, N/256, block_bytes)
// uint8, W (K, N) dequantized in-kernel; output (B, K) — y[b, j] =
// W[j, idx[b]] (the row of W^T), in out_dtype (float32 or bfloat16).
// The embedding-lookup sibling of the GEMM op (same dequant
// family; a lookup is not a GEMM, so its own op — no weight
// materialization: one output element per dequantized weight element).
MLX_API array kquant_file_gather(
    const array& idx,
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    Dtype out_dtype,
    StreamOrDevice s = StreamOrDevice());

// Standalone dequantization in the file layout (the kernel's dequant path,
// exposed for G1 bit-exact verification): w (K, N/256, block_bytes) uint8
// -> (K, N) float32, in the file's [K rows][N/256 super-blocks] element
// order, with the window map applied.
MLX_API array kquant_file_dequant(
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    StreamOrDevice s = StreamOrDevice());

}  // namespace mlx::core
