// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cuda.h"
#include "mlx/fast.h"

#ifdef MLX_USE_ROCM
#include "mlx/backend/rocm/rocm.h"
#endif

namespace mlx::core {

namespace cu {

bool is_available() {
  return false;
}

} // namespace cu

namespace fast {

CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
#ifdef MLX_USE_ROCM
  if (rocm::is_available()) {
    // No CUDA, but the ROCm custom-kernel path (hipRTC under
    // backend/rocm/custom_kernel.cpp) compiles the same CUDA-style kernel
    // bodies. Route instead of failing so callers' "cuda" kernels run.
    return hip_kernel(
        name,
        input_names,
        output_names,
        source,
        header,
        ensure_row_contiguous,
        shared_memory,
        {});
  }
#endif
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

std::vector<array> precompiled_cuda_kernel(
    const std::string&,
    const std::string&,
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    const std::vector<ScalarArg>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    int shared_memory,
    std::optional<float> init_value,
    bool ensure_row_contiguous,
    StreamOrDevice) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

} // namespace fast

} // namespace mlx::core
