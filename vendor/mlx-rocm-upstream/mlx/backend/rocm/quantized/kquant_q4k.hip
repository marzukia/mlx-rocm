// kquant_q4k.hip — Q4_K dequantization for the ROCm backend.
//
// One thread per element (grid-stride). The shared block math
// (kquant_q4k_math.hpp) is the single source of truth, shared with the
// hermetic CPU G1 test (tests/kquant_q4k/).
// `w` is the raw 144-byte-block stream (uint8, row-contiguous; shape
// (nb, 144) for dequantize or (N, K/256, 144) for quantized matmul);
// the output is the dequantized tensor (nb*256 elements), in the dtype of
// the output array (float32 for the standalone op, x.dtype() for GEMM).
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/quantized/kquant_q4k_math.hpp"
#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

using namespace mlx::core::rocm;
using namespace mlx::core::rocm::kquant;

namespace {

template <typename OutT>
__global__ void q4k_dequant_kernel(
    const uint8_t* __restrict__ w,
    OutT* __restrict__ out,
    int64_t nblocks) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < nblocks * kQ4KBlockElems; i += stride) {
    int64_t b = i / kQ4KBlockElems;
    int e = (int)(i - b * kQ4KBlockElems);
    float v = dequant_elem(w + b * kQ4KBlockBytes, e);
    out[i] = static_cast<OutT>(v);
  }
}

} // namespace

void kquant_dequantize(
    const array& w,
    array& w_out,
    int group_size,
    int bits,
    CommandEncoder& encoder,
    const Stream& s) {
  if (bits != 4 || group_size != 256) {
    throw std::runtime_error(
        "[kquant] dequantize currently implements bits=4 (Q4_K) only; "
        "Q2_K/Q5_K/Q6_K land in WS3");
  }
  if (w.shape(-1) != kQ4KBlockBytes) {
    throw std::runtime_error(
        "[kquant] w last dim must be 144 (one Q4_K block)");
  }
  int64_t nb = w.size() / kQ4KBlockBytes;
  auto block = dim3(256);
  int64_t grid = std::min((nb * kQ4KBlockElems + 255) / 256, (int64_t)4096);
  auto launch = [=](hipStream_t stream) {
    void* outp = gpu_ptr<void>(w_out);
    if (w_out.dtype() == float32) {
      q4k_dequant_kernel<float><<<dim3(grid), block, 0u, stream>>>(
          gpu_ptr<const uint8_t>(w),
          static_cast<float*>(outp),
          nb);
    } else if (w_out.dtype() == bfloat16) {
      q4k_dequant_kernel<__hip_bfloat16><<<dim3(grid), block, 0u, stream>>>(
          gpu_ptr<const uint8_t>(w),
          static_cast<__hip_bfloat16*>(outp),
          nb);
    } else {
      q4k_dequant_kernel<__half><<<dim3(grid), block, 0u, stream>>>(
          gpu_ptr<const uint8_t>(w),
          static_cast<__half*>(outp),
          nb);
    }
  };
  encoder.set_input_array(w);
  encoder.set_output_array(w_out);
  encoder.launch_kernel([=](hipStream_t stream) { launch(stream); });
}

} // namespace mlx::core
