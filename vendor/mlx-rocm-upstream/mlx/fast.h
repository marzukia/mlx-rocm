// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

/** K2: fused residual add + RMSNorm: w * rmsnorm(x + residual).
 *
 * Output dtype is result_type(x, weight); the residual is astype'd to it. */
MLX_API array rms_norm_residual(
    const array& x,
    const array& residual,
    const array& weight,
    float eps,
    StreamOrDevice s = {});

/** K2: batched in-place KV append: writes keys and values into the
 * pre-allocated cache buffers at `offset` in one kernel. Returns the
 * updated (in-place on the GPU fast path) cache arrays; the fallback path
 * (CPU stream, non-row-contiguous inputs) returns fresh buffers. */
MLX_API std::vector<array> kv_append(
    const array& keys,
    const array& values,
    const array& cache_keys,
    const array& cache_values,
    int64_t offset,
    StreamOrDevice s = {});

/** Fused cross entropy with class indices as targets. */
MLX_API array
cross_entropy(const array& logits, const array& targets, StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    bool force_fused = false,
    StreamOrDevice s = {});

MLX_API std::vector<array> gated_delta_update(
    const array& queries,
    const array& keys,
    const array& values,
    const array& gates,
    const array& beta_,
    const std::optional<array>& initial_state = std::nullopt,
    const std::optional<array>& mask = std::nullopt,
    StreamOrDevice s = {});

MLX_API std::vector<array> gdn_fused_step(
    const array& q_raw,
    const array& k_raw,
    const array& v_raw,
    const array& b,
    const array& a,
    const array& A_log,
    const array& dt_bias,
    const array& conv_state,
    const array& conv_weight,
    const array& z,
    const array& state,
    const array& norm_weight,
    float inv_scale,
    float prep_eps,
    float norm_eps,
    StreamOrDevice s = {});

MLX_API std::vector<array> gdn_fused_step_tape(
    const array& q_raw,
    const array& k_raw,
    const array& v_raw,
    const array& b,
    const array& a,
    const array& A_log,
    const array& dt_bias,
    const array& conv_state,
    const array& conv_weight,
    const array& z,
    const array& state,
    const array& norm_weight,
    float inv_scale,
    float prep_eps,
    float norm_eps,
    StreamOrDevice s = {});


// lm_head_int2: int2 g128-asymmetric dequant GEMV (M=1) + in-kernel per-64-column top-8 candidate emission (backend/rocm/lm_head_int2.hip). x [1, K] bf16, wq [N, K/4] u8 (quarter-split 2-bit planes), scale [N, NG] f32 = s, zs [N, NG] f32 = z*s, sx [NG] f32 = sum of x per 128-group. Returns {logits [N] f32, cand_val [N/64, 8] f32, cand_col [N/64, 8] i32}.
MLX_API std::vector<array> lm_head_int2(
    const array& x,
    const array& wq,
    const array& scale,
    const array& zs,
    const array& sx,
    StreamOrDevice s = {});


MLX_API std::vector<array> gdn_prep(
    const array& q_raw,
    const array& k_raw,
    const array& v_raw,
    const array& b,
    const array& a,
    const array& A_log,
    const array& dt_bias,
    const array& conv_state,
    const array& conv_weight,
    float inv_scale,
    float eps,
    StreamOrDevice s = {});

MLX_API std::vector<array> gdn_norm_gated(
    const array& out,
    const array& z,
    const array& weight,
    float eps,
    StreamOrDevice s = {});


using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false,
    const CompileOptions& compile_options = {});

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

// ROCm (hipRTC) custom kernel. CUDA-style kernel bodies (threadIdx, blockIdx,
// __global__ __shared__ etc.) compile unchanged under HIP; on ROCm-only builds
// cuda_kernel() forwards here.
MLX_API CustomKernelFunction hip_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0,
    std::vector<std::pair<int, int>> output_input_aliases = {});

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
