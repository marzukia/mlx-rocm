// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/fast_primitives.h"
#include "mlx/kquant_file_gemm.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

// Linker anchor for the file-layout k-quant free functions (W1): they live
// in the HIP kernel static archive and are referenced only by the Python
// extension (resolved dynamically). The asm "r" operands force the symbol
// addresses into registers (the plain volatile-store form gets optimized
// away under -O3 + hidden visibility), pulling the archive member into
// libmlx.so so the extension resolves at load.
__attribute__((used)) void kquant_file_link_anchor() {
  const void* a = (const void*)&kquant_file_gemm;
  const void* b = (const void*)&kquant_file_gather;
  const void* c = (const void*)&kquant_file_dequant;
  __asm__ volatile("" : : "r"(a), "r"(b), "r"(c) :);
}

namespace {

inline array ensure_row_contiguous(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array
ensure_contiguous(const array& x, rocm::CommandEncoder& enc, const Stream& s) {
  if (x.flags().row_contiguous || x.flags().col_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

} // namespace

// Note: affine_quantize, affine_dequantize, fp_quantize, fp_dequantize
// are implemented in affine_quantize.hip and fp_quantize.hip
// ConvertFP8 is implemented in convert_fp8.hip

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(mlx::core::rocm::malloc_async(w.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(mlx::core::rocm::malloc_async(wq.nbytes(), enc));
    scales.set_data(mlx::core::rocm::malloc_async(scales.nbytes(), enc));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(mlx::core::rocm::malloc_async(biases.nbytes(), enc));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

// Note: ConvertFP8::eval_gpu is implemented in convert_fp8.hip

} // namespace mlx::core
