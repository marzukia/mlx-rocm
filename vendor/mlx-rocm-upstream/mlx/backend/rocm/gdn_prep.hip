// Copyright © 2025 Apple Inc.
//
// Tail fusion (Lever 2): fuse the per-token GDN tail (conv1d + silu + q/k
// norm + beta + g, and the final norm_gated = rms_norm + precise_swiglu)
// into the fewest kernels the bit-exact constraint allows.
//
// Three kernels replace the ~14-16 separate tail launches per GDN layer:
//   1. gdn_prep_kernel:     conv1d + silu + beta + g
//   2. gdn_qk_norm_kernel:  q_norm + k_norm (32 rms_norms, weight=None)
//   3. gdn_norm_gated_kernel: rms_norm + precise_swiglu (32 heads)
//
// Bit-exactness: the elementwise ops reuse the exact Op structs (Sigmoid,
// Exp, LogAddExp, Multiply) the unfused kernels use; the conv1d keeps the
// 4-value sequential k-order sum; the rms_norm keeps the exact 256-thread
// block reduction from rms_norm_row.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/unary_ops.hpp"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// ---- warp reduce (copied from rms_norm.hip, same shfl_xor tree) ----
__device__ float warp_reduce_sum_rms(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// ---- rms_norm row body (copied verbatim from rms_norm.hip) ----
// `out` must be block-wide visible (shared or global); the caller is
// responsible for a __syncthreads() before reading it if it is shared.
template <typename T, int BLOCK_DIM, int N_READS = 4>
__device__ void gdn_rms_norm_row(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  float normalizer = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]);
      normalizer += t * t;
    }
  }
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  float warp_sum = warp_reduce_sum_rms(normalizer);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    normalizer = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE)
        ? shared_sum[lane]
        : 0;
    normalizer = warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      T normalized = static_cast<T>(static_cast<float>(x[idx]) * normalizer);
      T wi = (w_stride == 0) ? w[0] : w[idx * w_stride];
      out[idx] = wi * normalized;
    }
  }
}

// ---- Kernel 1: conv1d + silu + beta + g ----
// conv1d: 4-value sequential k-order sum (matches depthwise_conv1d_kernel),
// cast to T, then silu = T * sigmoid(T) (Multiply + Sigmoid structs).
// beta = sigmoid(b); g = exp(-exp(A_log) * logaddexp(a+dt_bias, 0)) in f32,
// cast to T (matches compute_g + the astype in fast.cpp).
template <typename T>
__global__ void gdn_prep_kernel(
    const T* __restrict__ q_raw,
    const T* __restrict__ k_raw,
    const T* __restrict__ v_raw,
    const T* __restrict__ b,
    const T* __restrict__ a,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const T* __restrict__ conv_state,
    const T* __restrict__ conv_weight,
    T* __restrict__ q_conv,
    T* __restrict__ k_conv,
    T* __restrict__ v_conv,
    T* __restrict__ beta_out,
    float* __restrict__ g_out,
    int key_dim,
    int value_dim,
    int conv_dim,
    int kernel_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < conv_dim) {
    float acc = 0.0f;
    acc += static_cast<float>(conv_state[0 * conv_dim + c]) *
        static_cast<float>(conv_weight[c * kernel_size + 0]);
    acc += static_cast<float>(conv_state[1 * conv_dim + c]) *
        static_cast<float>(conv_weight[c * kernel_size + 1]);
    acc += static_cast<float>(conv_state[2 * conv_dim + c]) *
        static_cast<float>(conv_weight[c * kernel_size + 2]);
    float in3;
    if (c < key_dim) {
      in3 = static_cast<float>(q_raw[c]);
    } else if (c < 2 * key_dim) {
      in3 = static_cast<float>(k_raw[c - key_dim]);
    } else {
      in3 = static_cast<float>(v_raw[c - 2 * key_dim]);
    }
    acc += in3 * static_cast<float>(conv_weight[c * kernel_size + 3]);
    T conv_bf = static_cast<T>(acc);
    T silu = Multiply{}(conv_bf, Sigmoid{}(conv_bf));
    if (c < key_dim) {
      q_conv[c] = silu;
    } else if (c < 2 * key_dim) {
      k_conv[c - key_dim] = silu;
    } else {
      v_conv[c - 2 * key_dim] = silu;
    }
  }
  if (blockIdx.x == 0) {
    int t = threadIdx.x;
    if (t < value_dim / 128) {  // num_v_heads
      beta_out[t] = Sigmoid{}(b[t]);
      // Bit-exact compute_g chain (every intermediate rounding matches the
      // unfused op sequence):
      //   sum_bf = (bf16)(f32(a) + f32(dt_bias))            [Add bf16]
      //   sp_bf  = (bf16)(max + log1p(exp(min - max)))     [LogAddExp bf16]
      //   e_al   = expf(A_log)                             [Exp f32]
      //   g      = expf(-(e_al * f32(sp_bf)))             [f32; stays f32]
      // g stays f32: the GDN recurrent kernel reads it as u16 pairs, exactly
      // as the unfused path feeds it (f32 g, no astype).
      T sum_bf = static_cast<T>(static_cast<float>(a[t]) +
                               static_cast<float>(dt_bias[t]));
      float fx = static_cast<float>(sum_bf);
      float mxv = fmaxf(fx, 0.0f);
      float mnv = fminf(fx, 0.0f);
      float sp_f = mxv + log1pf(expf(mnv - mxv));
      float sp = static_cast<float>(static_cast<T>(sp_f));
      float e_al = expf(A_log[t]);
      float arg = -e_al * sp;
      g_out[t] = expf(arg);
    }
  }
}

// ---- Kernel 2: q_norm + k_norm (32 rms_norms, weight=None) ----
// Block 0..Hk-1: q head; block Hk..2*Hk-1: k head. Each block does one
// rms_norm (exact block reduction, w=1) into shared, then the scalar
// multiply (inv_scale**2 for q, inv_scale for k) in f32, cast to T.
template <typename T>
__global__ void gdn_qk_norm_kernel(
    const T* __restrict__ q_conv,
    const T* __restrict__ k_conv,
    T* __restrict__ q_normed,
    T* __restrict__ k_normed,
    float inv_scale,
    float rms_eps,
    int num_heads,
    int head_dim) {
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  __shared__ T rms_out[128];
  int block = blockIdx.x;
  bool is_q = block < num_heads;
  int head = is_q ? block : block - num_heads;
  const T* in = is_q ? (q_conv + (int64_t)head * head_dim)
                    : (k_conv + (int64_t)head * head_dim);
  T* out = is_q ? (q_normed + (int64_t)head * head_dim)
               : (k_normed + (int64_t)head * head_dim);
  float scale = is_q ? (inv_scale * inv_scale) : inv_scale;
  // The unfused path scales with a python float * bf16 array: the scalar is
  // cast to bf16 first, then the multiply is (bf16)(f32 * f32). Pre-round
  // the scale to bf16 to match bit-exactly (inv_scale is not exact in bf16).
  float scale_f = static_cast<float>(static_cast<T>(scale));
  T one = static_cast<T>(1.0f);
  T* w = &one;  // weight=None -> w[0]=1, w_stride=0
  gdn_rms_norm_row<T, BLOCK_DIM, N_READS>(
      in, w, rms_out, rms_eps, head_dim, 0);
  __syncthreads();
  for (int i = threadIdx.x; i < head_dim; i += BLOCK_DIM) {
    float v = static_cast<float>(rms_out[i]) * scale_f;
    out[i] = static_cast<T>(v);
  }
}

// ---- Kernel 3: norm_gated = rms_norm + precise_swiglu (32 heads) ----
// Each block (one v head): rms_norm over 128 (with weight, exact block
// reduction) into shared -> x_normed (T); then precise_swiglu:
//   gate = silu(z.astype(f32)) in f32; x = x_normed.astype(f32);
//   result = (gate * x).astype(T).
template <typename T>
__global__ void gdn_norm_gated_kernel(
    const T* __restrict__ out,
    const T* __restrict__ z,
    const T* __restrict__ weight,
    T* __restrict__ result,
    float eps,
    int head_dim) {
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  __shared__ T x_normed[128];
  int head = blockIdx.x;
  const T* in = out + (int64_t)head * head_dim;
  const T* zg = z + (int64_t)head * head_dim;
  T* res = result + (int64_t)head * head_dim;
  gdn_rms_norm_row<T, BLOCK_DIM, N_READS>(
      in, weight, x_normed, eps, head_dim, 1);
  __syncthreads();
  for (int i = threadIdx.x; i < head_dim; i += BLOCK_DIM) {
    float x_f = static_cast<float>(x_normed[i]);
    float z_f = static_cast<float>(zg[i]);
    float sig = Sigmoid{}(z_f);
    float gate = z_f * sig;  // silu in f32
    float r = gate * x_f;    // f32 multiply
    res[i] = static_cast<T>(r);
  }
}

}  // namespace rocm

// ---- host launchers ----
int launch_gdn_prep(
    const void* q_raw, const void* k_raw, const void* v_raw,
    const void* b, const void* a, const float* A_log, const float* dt_bias,
    const void* conv_state, const void* conv_weight,
    void* q_conv, void* k_conv, void* v_conv, void* beta_out, void* g_out,
    int key_dim, int value_dim, int conv_dim, int kernel_size,
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid((conv_dim + 255) / 256);
  rocm::gdn_prep_kernel<T><<<grid, block, 0, stream>>>(
      (const T*)q_raw, (const T*)k_raw, (const T*)v_raw, (const T*)b,
      (const T*)a, A_log, dt_bias, (const T*)conv_state, (const T*)conv_weight,
      (T*)q_conv, (T*)k_conv, (T*)v_conv, (T*)beta_out, (float*)g_out,
      key_dim, value_dim, conv_dim, kernel_size);
  return 0;
}

int launch_gdn_qk_norm(
    const void* q_conv, const void* k_conv, void* q_normed, void* k_normed,
    float inv_scale, float rms_eps, int num_heads, int head_dim,
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid(2 * num_heads);
  rocm::gdn_qk_norm_kernel<T><<<grid, block, 0, stream>>>(
      (const T*)q_conv, (const T*)k_conv, (T*)q_normed, (T*)k_normed,
      inv_scale, rms_eps, num_heads, head_dim);
  return 0;
}

int launch_gdn_norm_gated(
    const void* out, const void* z, const void* weight, void* result,
    float eps, int head_dim, int num_heads, hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid(num_heads);
  rocm::gdn_norm_gated_kernel<T><<<grid, block, 0, stream>>>(
      (const T*)out, (const T*)z, (const T*)weight, (T*)result, eps, head_dim);
  return 0;
}

}  // namespace mlx::core

// ---- eval_gpu implementations ----
namespace mlx::core::fast {

void GdnPrep::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& bb = inputs[3];
  auto& aa = inputs[4];
  auto& al = inputs[5];
  auto& db = inputs[6];
  auto& cs = inputs[7];
  auto& cw = inputs[8];
  auto& q_normed = outputs[0];
  auto& k_normed = outputs[1];
  auto& v_conv = outputs[2];
  auto& beta = outputs[3];
  auto& g = outputs[4];

  // Temp buffers for the silu'd conv output (read by the qk_norm kernel).
  array q_conv({(int)q.size()}, bfloat16, nullptr, {});
  q_conv.set_data(mlx::core::rocm::malloc_async(q_conv.nbytes(), encoder));
  encoder.add_temporary(q_conv);
  array k_conv({(int)k.size()}, bfloat16, nullptr, {});
  k_conv.set_data(mlx::core::rocm::malloc_async(k_conv.nbytes(), encoder));
  encoder.add_temporary(k_conv);

  for (auto& o : outputs) {
    o.set_data(mlx::core::rocm::malloc_async(o.nbytes(), encoder));
  }

  int key_dim = (int)q.size();
  int value_dim = (int)v.size();
  int conv_dim = key_dim + (int)k.size() + value_dim;

  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_gdn_prep(
        gpu_ptr<void>(q), gpu_ptr<void>(k),
        gpu_ptr<void>(v), gpu_ptr<void>(bb),
        gpu_ptr<void>(aa), gpu_ptr<float>(al),
        gpu_ptr<float>(db), gpu_ptr<void>(cs),
        gpu_ptr<void>(cw), gpu_ptr<void>(q_conv),
        gpu_ptr<void>(k_conv), gpu_ptr<void>(v_conv),
        gpu_ptr<void>(beta), gpu_ptr<void>(g), key_dim,
        value_dim, conv_dim, 4, stream);
    if (rc != 0) {
      throw std::runtime_error("gdn_prep: conv+silu+beta+g launch failed");
    }
  });

  encoder.launch_kernel([&](hipStream_t stream) {
    float inv_scale = inv_scale_;
    float rms_eps = eps_ * inv_scale * inv_scale;
    int num_heads = key_dim / 128;
    int head_dim = 128;
    int rc = launch_gdn_qk_norm(
        gpu_ptr<void>(q_conv), gpu_ptr<void>(k_conv),
        gpu_ptr<void>(q_normed), gpu_ptr<void>(k_normed),
        inv_scale, rms_eps, num_heads, head_dim, stream);
    if (rc != 0) {
      throw std::runtime_error("gdn_prep: qk_norm launch failed");
    }
  });
}

void GdnNormGated::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  auto& o = inputs[0];
  auto& zg = inputs[1];
  auto& w = inputs[2];
  auto& result = outputs[0];

  result.set_data(mlx::core::rocm::malloc_async(result.nbytes(), encoder));

  int head_dim = (int)w.size();
  int num_heads = (int)o.size() / head_dim;

  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_gdn_norm_gated(
        gpu_ptr<void>(o), gpu_ptr<void>(zg),
        gpu_ptr<void>(w), gpu_ptr<void>(result), eps_,
        head_dim, num_heads, stream);
    if (rc != 0) {
      throw std::runtime_error("gdn_norm_gated: launch failed");
    }
  });
}

}  // namespace mlx::core::fast
