// Copyright © 2023-2024 Apple Inc.
#pragma once

#include "mlx/io.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

extern "C" {
#include <gguflib.h>
}

namespace mlx::core {

Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor);

// WS3: K-quant (Q2_K/Q5_K/Q6_K) tensor stored as the raw compressed byte
// stream (uint8 {nblocks, block_bytes}); no re-quantization, no CPU
// dequant. Dequantized on the GPU at GEMM time via
// fast::kquant_dequantize (mx.rocm.kq_dequant; ktype = the GGUF type
// code: Q2_K = 10, Q5_K = 13, Q6_K = 14).
void gguf_load_kquant_raw(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor);

} // namespace mlx::core
