// Copyright © 2025 Apple Inc.  Fused GDN T=1..4 decode/verify-step kernel (Phase 2, MLX_ROCM_GDN_FUSE=1).  One launch per GDN layer per step, replacing the 4-kernel default tail chain (gdn_prep conv+silu+beta+g -> gdn_prep qk_norm -> r4d recurrent state update -> gdn_norm_gated) at B=1, T=1..4, Dk=Dv=128.  T=2/T=4 (the S=2/S=4 spec-decode verify rows) extension: the recurrent state update loads h0 ONCE and updates the two rows sequentially in registers — exactly the r4d recurrent kernel's T loop (r4d_gdn_recurrent_update, `for (t = 0; t < T; ++t)` with h resident), so T=2 is bit-identical to two T=1 steps and to the default T=2 recurrent path (the ROCm gated_delta_update op routes T <= 8 to this same recurrent kernel). The conv1d window slides per row, mirroring the default `nn.silu(self.conv1d(conv_input))` (conv_input = [conv_state(3 rows), v0, v1]; row t = sum_j E[t+j]*w[j], E = [cs0, cs1, cs2, v0, v1]): row 0 = cs0*w0 + cs1*w1 + cs2*w2 + v0*w3   (== the T=1 row, unchanged) row 1 = cs1*w0 + cs2*w1 + v0*w2  + v1*w3 (row 1 uses the PREVIOUS row's v, not the conv state — the causal-conv semantics of the default path; the T=1 path must NOT be changed).  T=4 (K=3 MTP verify, rows 2,3) extends the same sliding window to the full 4-row context: E = [cs0, cs1, cs2, v0..v3], row t = sum_j E[t+j]*w[j]: row 2 = cs2*w0 + v0*w1 + v1*w2 + v2*w3 row 3 = v0*w0 + v1*w1 + v2*w2 + v3*w3 The window tap expressions are written in the generalized form (tap j = conv_state[j-t] while j-t in [0,2), else raw row t+1-j); with TMAX=2 (or 1) the loop unrolls to constants and every ternary folds EXACTLY to the T=2/T=1 straight-line windows, so the T=1/T=2 codegen is unchanged. h stays register-resident across all TMAX rows (r4d's T loop).  Losslessness: every phase below is a verbatim replication of the deployed 4-kernel chain's arithmetic, same FMA order, same per-op bf16 roundings, same 256-thread rms_norm block reduction. Fusion removes only the memory round-trips (intermediates live in shared) and the 3 extra launches; no math changes. The conv1d state update stays a separate COPY op in the model (not part of this op), so cache semantics are untouched.  Geometry (fixed by the compiled path, mirrored from r4d VS=1): one workgroup per v head (32 blocks x 256 threads at Hv=32). The state update replicates the r4d_gdn_recurrent_update_kernel VS=1 layout (NR=128 rows, two threads per v row, lane<->lane^1 shfl close, 64 contiguous-k floats per thread in float4 x 16). Two v heads share one k/q head (Hv/Hk=2); the conv + qk_norm for the shared head is computed by both v-head blocks redundantly but identically (pure function of the same inputs, same order -> bit-equal).  g is computed f32 (as the chain's gdn_prep outputs it) but the chain's r4d kernel receives it as BF16: fast.cpp's gated_delta_update does g = astype(gates, out_dtype) at the op boundary (out_dtype = q's dtype = bf16). The kernel's u16 read is therefore the full bf16 pattern; we replicate exactly by rounding s_g to bf16 (round-to-nearest, as the astype) before the read.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/device/unary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <fstream>

namespace mlx::core {

namespace rocm {

// ---- warp reduce (copied from gdn_prep.hip, which copied it from rms_norm.hip, same shfl tree) ----
__device__ float gdnf_warp_reduce_sum_rms(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// ---- rms_norm row body (copied verbatim from gdn_prep.hip, which copied it verbatim from rms_norm.hip): the exact 256-thread block reduction. ---- `out` must be block-wide visible (shared or global); the caller is responsible for a __syncthreads() before reading it if it is shared.
template <typename T, int BLOCK_DIM, int N_READS = 4>
__device__ void gdnf_rms_norm_row(
    const T* x,
    const T* w,
    T* out,
    float* shared_sum,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  float normalizer = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size;
       i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]);
      normalizer += t * t;
    }
  }
  float warp_sum = gdnf_warp_reduce_sum_rms(normalizer);
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
    normalizer = gdnf_warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);
  for (int i = threadIdx.x * N_READS; i < axis_size;
       i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      T normalized =
          static_cast<T>(static_cast<float>(x[idx]) * normalizer);
      T wi = (w_stride == 0) ? w[0] : w[idx * w_stride];
      out[idx] = wi * normalized;
    }
  }
}

// u16 -> f32 (identical to r4d's bf2f).
__device__ __forceinline__ float gdnf_bf2f16(unsigned short h) {
  return __builtin_bit_cast(float, (unsigned)h << 16);
}
// f2bf: round-half-away (identical to r4d's f2bf; used for the on store).
__device__ __forceinline__ unsigned short gdnf_f2bf(float f) {
  return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
// f2bf RNE: identical to MLX's _MLX_BFloat16 float ctor (astype f32->bf16), which is what the chain's astype(gates, out_dtype) does before r4d reads g.
__device__ __forceinline__ unsigned short gdnf_f2bf_rne(float f) {
  unsigned u = __builtin_bit_cast(unsigned, f);
  u += (u >> 16 & 1u) + 0x7FFFu;
  return (unsigned short)(u >> 16);
}

// ---- The fused decode/verify step: one block per v head, 256 threads. ---- T=1 is the original fused decode step (row-0 conv window: cs0, cs1, cs2, v0 — bit-identical to the original T=1 kernel). T=2/T=4 run the verify rows sequentially in ONE t-loop: phases 1..5 per row, with the phase-4 recurrent update's h resident in registers across rows (exactly r4d's T loop, so T=2/T=4 are bit-identical to the sequential T=1 steps).
template <typename T, int TMAX>
__global__ __launch_bounds__(256) void gdn_fused_step_kernel(
    const T* __restrict__ q_raw,
    const T* __restrict__ k_raw,
    const T* __restrict__ v_raw,
    const T* __restrict__ b,
    const T* __restrict__ a,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const T* __restrict__ conv_state,
    const T* __restrict__ conv_weight,
    const T* __restrict__ z,
    const float* __restrict__ state,
    const T* __restrict__ norm_weight,
    T* __restrict__ result,
    float* __restrict__ state_new,
    int key_dim,
    int value_dim,
    int conv_dim,
    int kernel_size,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    float inv_scale,
    float rms_eps,
    float norm_eps,
    unsigned char* dbg) {
  const int hv = blockIdx.x;
  const int hk = hv / (num_v_heads / num_k_heads);
  const int tid = threadIdx.x;

  __shared__ T s_qconv[128];
  __shared__ T s_kconv[128];
  __shared__ T s_vconv[128];
  __shared__ T s_qn_rms[128];
  __shared__ T s_kn_rms[128];
  __shared__ __attribute__((aligned(16))) T s_qn[128];
  __shared__ __attribute__((aligned(16))) T s_kn[128];
  __shared__ T s_out[128];
  __shared__ T s_xn[128];
  __shared__ T s_beta[4];
  __shared__ T s_one;
  __shared__ float s_g[4];
  // One reduction buffer PER rms_norm call: sequential calls in one block must not share storage (the tail of call N reads shared_sum[0] while the head of call N+1 writes shared_sum[0..7]; the reference kernels only ever call once per block, so a function-local buffer would race).
  __shared__ float s_ss_q[9];
  __shared__ float s_ss_k[9];
  __shared__ float s_ss_n[9];

  s_one = static_cast<T>(1.0f);

  // Phase-4 geometry (verbatim r4d VS=1): h is loaded ONCE, before the t loop, and updated row-by-row in registers; state_new is written after the loop — exactly r4d_gdn_recurrent_update_kernel's structure.
  const int row = tid >> 1, half = tid & 1, k0 = half * 64;
  float h[64];
  {
    const float* p =
        state + ((size_t)hv * head_dim + row) * head_dim + k0;
#pragma unroll
    for (int j = 0; j < 64; j += 4)
      *(float4*)&h[j] = *(const float4*)(p + j);
  }

  for (int t = 0; t < TMAX; ++t) {
    // ---------------- Phase 1: conv1d + silu ---------------- Verbatim gdn_prep_kernel conv body: 4-value sequential k-order sum in f32, cast to T, silu = Multiply(conv_bf, Sigmoid(conv_bf)) in T. The conv window slides per row (E = [cs0, cs1, cs2, v0..v3]; row t = sum_j E[t+j]*w[j]); written generalized over t in [0, TMAX): with TMAX=2/1 the loop unrolls and every ternary folds to the original T=2/T=1 windows (bit-identical codegen). Channels: q c = hk*128+i ; k c = key_dim + hk*128+i ; v c = 2*key_dim + hv*128+i (value_dim/128 = Hv).
    if (tid < head_dim) {
      int cv = 2 * key_dim + hv * head_dim + tid;
      const int vch = hv * head_dim + tid;
      float acc = 0.0f;
      acc += static_cast<float>(
                 (t <= 2) ? conv_state[t * conv_dim + cv]
                         : v_raw[(t - 3) * value_dim + vch]) *
          static_cast<float>(conv_weight[cv * kernel_size + 0]);
      acc += static_cast<float>(
                 (t <= 1) ? conv_state[(t + 1) * conv_dim + cv]
                         : v_raw[(t - 2) * value_dim + vch]) *
          static_cast<float>(conv_weight[cv * kernel_size + 1]);
      acc += static_cast<float>(
                 (t == 0) ? conv_state[2 * conv_dim + cv]
                         : v_raw[(t - 1) * value_dim + vch]) *
          static_cast<float>(conv_weight[cv * kernel_size + 2]);
      acc += static_cast<float>(v_raw[t * value_dim + vch]) *
          static_cast<float>(conv_weight[cv * kernel_size + 3]);
      T conv_bf = static_cast<T>(acc);
      s_vconv[tid] = Multiply{}(conv_bf, Sigmoid{}(conv_bf));

      int ck = key_dim + hk * head_dim + tid;
      const int kch = hk * head_dim + tid;
      float acc2 = 0.0f;
      acc2 += static_cast<float>(
                  (t <= 2) ? conv_state[t * conv_dim + ck]
                          : k_raw[(t - 3) * key_dim + kch]) *
          static_cast<float>(conv_weight[ck * kernel_size + 0]);
      acc2 += static_cast<float>(
                  (t <= 1) ? conv_state[(t + 1) * conv_dim + ck]
                          : k_raw[(t - 2) * key_dim + kch]) *
          static_cast<float>(conv_weight[ck * kernel_size + 1]);
      acc2 += static_cast<float>(
                  (t == 0) ? conv_state[2 * conv_dim + ck]
                          : k_raw[(t - 1) * key_dim + kch]) *
          static_cast<float>(conv_weight[ck * kernel_size + 2]);
      acc2 += static_cast<float>(k_raw[t * key_dim + kch]) *
          static_cast<float>(conv_weight[ck * kernel_size + 3]);
      T conv_bf2 = static_cast<T>(acc2);
      s_kconv[tid] = Multiply{}(conv_bf2, Sigmoid{}(conv_bf2));
    } else {
      int qi = tid - head_dim;
      const int qch = hk * head_dim + qi;
      int cq = hk * head_dim + qi;
      float acc = 0.0f;
      acc += static_cast<float>(
                 (t <= 2) ? conv_state[t * conv_dim + cq]
                         : q_raw[(t - 3) * key_dim + qch]) *
          static_cast<float>(conv_weight[cq * kernel_size + 0]);
      acc += static_cast<float>(
                 (t <= 1) ? conv_state[(t + 1) * conv_dim + cq]
                         : q_raw[(t - 2) * key_dim + qch]) *
          static_cast<float>(conv_weight[cq * kernel_size + 1]);
      acc += static_cast<float>(
                 (t == 0) ? conv_state[2 * conv_dim + cq]
                         : q_raw[(t - 1) * key_dim + qch]) *
          static_cast<float>(conv_weight[cq * kernel_size + 2]);
      acc += static_cast<float>(q_raw[t * key_dim + qch]) *
          static_cast<float>(conv_weight[cq * kernel_size + 3]);
      T conv_bf = static_cast<T>(acc);
      s_qconv[qi] = Multiply{}(conv_bf, Sigmoid{}(conv_bf));
    }
    __syncthreads();

    // ---------------- Phase 2: beta + g ---------------- Verbatim gdn_prep_kernel gate body (thread 0 there == thread 0 here), per row. g stays f32 (the state update reads its low u16, as the chain does). s_beta/s_g are sized [4] (the TMAX=4 row count); the TMAX=1/2 instantiations use only [0](+[1]) exactly as before.
    if (tid == 0) {
      s_beta[t] = Sigmoid{}(b[t * num_v_heads + hv]);
      T sum_bf = static_cast<T>(static_cast<float>(a[t * num_v_heads + hv]) +
                               static_cast<float>(dt_bias[hv]));
      float fx = static_cast<float>(sum_bf);
      float mxv = fmaxf(fx, 0.0f);
      float mnv = fminf(fx, 0.0f);
      float sp_f = mxv + log1pf(expf(mnv - mxv));
      float sp = static_cast<float>(static_cast<T>(sp_f));
      float e_al = expf(A_log[hv]);
      float arg = -e_al * sp;
      s_g[t] = expf(arg);
    }
    __syncthreads();

    // ---------------- Phase 3: q/k rms_norm + scale ---------------- Verbatim gdn_qk_norm_kernel body (one head per call; the chain runs 32 heads in 32 blocks, this runs q then k sequentially in one block -- the per-head reduction is the identical 256-thread computation).
    {
      float scale = inv_scale * inv_scale;
      float scale_f =
          static_cast<float>(static_cast<T>(scale));  // pre-round to bf16
      gdnf_rms_norm_row<T, 256, 4>(
          s_qconv, &s_one, s_qn_rms, s_ss_q, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 256) {
        float v = static_cast<float>(s_qn_rms[i]) * scale_f;
        s_qn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();
    {
      float scale = inv_scale;
      float scale_f = static_cast<float>(static_cast<T>(scale));
      gdnf_rms_norm_row<T, 256, 4>(
          s_kconv, &s_one, s_kn_rms, s_ss_k, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 256) {
        float v = static_cast<float>(s_kn_rms[i]) * scale_f;
        s_kn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();

    // ---------------- Phase 4: recurrent state update (one row) ---------- Verbatim r4d_gdn_recurrent_update_kernel VS=1 per-row body (B=1, row = tid>>1, half = tid&1, k0 = half*64), with h resident across rows. q/k are the phase-3 normalized rows (bf16, read as u16 exactly as the chain reads q_normed/k_normed); v is the phase-1 silu'd conv (bf16); g is the phase-2 value rounded to bf16 (the chain's r4d kernel receives g as bf16 -- fast.cpp astype's it at the op boundary) and read as that u16; beta is bf16.
    float qq[64], kk[64];
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
      const uint4 wq = *(const uint4*)(
          reinterpret_cast<const unsigned short*>(s_qn) + k0 + j);
      const uint4 wk = *(const uint4*)(
          reinterpret_cast<const unsigned short*>(s_kn) + k0 + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        qq[j + e] = gdnf_bf2f16(sq[e]);
        kk[j + e] = gdnf_bf2f16(sk[e]);
      }
    }

    // The chain's r4d kernel receives g as BF16: fast.cpp's gated_delta_update casts g = astype(gates, out_dtype) at the op boundary (out_dtype = q's dtype = bf16), so the kernel's u16 read is the full bf16 pattern, not the low 16 bits of the f32 mantissa. Replicate exactly: round s_g to bf16, then read that pattern.
    const float eg = __expf(
        fmaxf(__logf(gdnf_bf2f16(gdnf_f2bf_rne(s_g[t]))), -80.0f));
    const float bt = gdnf_bf2f16(
        *reinterpret_cast<unsigned short*>(&s_beta[t]));
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      h[j] *= eg;
      dot += h[j] * kk[j];
    }
    dot += __shfl_xor(dot, 1, 32);

    const float u = (gdnf_bf2f16(
                        *reinterpret_cast<const unsigned short*>(
                            &s_vconv[row])) -
                    dot) *
        bt;

    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      h[j] += u * kk[j];
      on += h[j] * qq[j];
    }
    on += __shfl_xor(on, 1, 32);

    if (half == 0) {
      unsigned short out_bf = gdnf_f2bf(on);
      s_out[row] = *reinterpret_cast<T*>(&out_bf);
    }
    __syncthreads();

    // ---------------- Phase 5: norm_gated = rms_norm + precise_swiglu ---- Verbatim gdn_norm_gated_kernel body (weight = RMSNorm weight, w_stride=1; silu in f32; (gate * x) cast to T), per row.
    gdnf_rms_norm_row<T, 256, 4>(
        s_out, norm_weight, s_xn, s_ss_n, norm_eps, head_dim, 1);
    __syncthreads();
    for (int i = tid; i < head_dim; i += 256) {
      float x_f = static_cast<float>(s_xn[i]);
      float z_f = static_cast<float>(
          z[(t * num_v_heads + hv) * head_dim + i]);
      float sig = Sigmoid{}(z_f);
      float gate = z_f * sig;  // silu in f32
      float r = gate * x_f;    // f32 multiply
      result[(t * num_v_heads + hv) * head_dim + i] = static_cast<T>(r);
    }
    __syncthreads();
  }

  // Final state store (after the last row's update).
  {
    float* p = state_new + ((size_t)hv * head_dim + row) * head_dim + k0;
#pragma unroll
    for (int j = 0; j < 64; j += 4)
      *(float4*)(p + j) = *(const float4*)&h[j];
  }

  if (dbg != nullptr) {
    // Debug capture (P2_GDN_DEBUG=1): the LAST row's phase-1..4 intermediates (shared holds the last row's values after the loop).
    unsigned char* p = dbg + (size_t)hv * 1042;
    for (int i = tid; i < head_dim; i += 256) {
      memcpy(p + 2 * i, &s_qn[i], 2);
      memcpy(p + 256 + 2 * i, &s_kn[i], 2);
      memcpy(p + 512 + 2 * i, &s_vconv[i], 2);
      memcpy(p + 774 + 2 * i, &s_out[i], 2);
    }
    if (tid == 0) {
      memcpy(p + 768, &s_beta[TMAX - 1], 2);
      memcpy(p + 770, &s_g[TMAX - 1], 4);
      // self-check: expected on[0] from the kernel operands
      float dk = 0.0f;
      for (int j = 0; j < 128; ++j)
        dk += gdnf_bf2f16(reinterpret_cast<const unsigned short*>(&s_kn[j])[0]) *
            gdnf_bf2f16(reinterpret_cast<const unsigned short*>(&s_qn[j])[0]);
      float u0 = gdnf_bf2f16(*reinterpret_cast<unsigned short*>(&s_vconv[0]));
      float bt0 = gdnf_bf2f16(*reinterpret_cast<unsigned short*>(&s_beta[TMAX - 1]));
      float expect0 = u0 * bt0 * dk;
      float chk[3] = {expect0, dk, u0};
      memcpy(p + 1030, chk, 12);
    }
    __syncthreads();
  }
}
}  // namespace rocm

// ---- host launcher ---- Instantiates TMAX in {1, 2, 4}, selected at launch from T (T=1 decode, T=2 S=2 verify, T=4 S=4 verify).
template <int TMAX>
int launch_gdn_fused_step(
    const void* q_raw,
    const void* k_raw,
    const void* v_raw,
    const void* b,
    const void* a,
    const float* A_log,
    const float* dt_bias,
    const void* conv_state,
    const void* conv_weight,
    const void* z,
    const float* state,
    const void* norm_weight,
    void* result,
    void* state_new,
    int key_dim,
    int value_dim,
    int conv_dim,
    int kernel_size,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    float inv_scale,
    float rms_eps,
    float norm_eps,
    unsigned char* dbg,
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid(num_v_heads);
  rocm::gdn_fused_step_kernel<T, TMAX><<<grid, block, 0, stream>>>(
      (const T*)q_raw, (const T*)k_raw, (const T*)v_raw, (const T*)b,
      (const T*)a, A_log, dt_bias, (const T*)conv_state,
      (const T*)conv_weight, (const T*)z, state, (const T*)norm_weight,
      (T*)result, (float*)state_new, key_dim, value_dim, conv_dim,
      kernel_size, num_v_heads, num_k_heads, head_dim, inv_scale, rms_eps,
      norm_eps, dbg);
  return (int)hipGetLastError();
}

}  // namespace mlx::core

// ---- eval_gpu implementation ----
namespace mlx::core::fast {

void GdnFusedStep::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  // The F-branch passes q3/k3/v3 as row-strided SLICES of the [1, T, 8192] qkv buffer (row stride 8192 != key_dim 2048). The kernel reads them packed (t * key_dim + ch), so T=1 (row 0 only, offset 0) looks right while T=2 misreads row 1 as row 0's k-channel region. Force contiguity only where the kernel needs it: T=1 decode reads row 0 at offset 0 either way, so the T=1 path takes the input references with zero extra host work (per-call array copies measurably regressed the latency-bound AR loop by ~0.6%).
  std::unique_ptr<array> tmps[7];
  const array *pq = &inputs[0], *pk = &inputs[1], *pv = &inputs[2],
              *pb = &inputs[3], *pa = &inputs[4], *pc = &inputs[7],
              *pz = &inputs[9];
  auto maybe_copy = [&](const array& x, const array** slot, int i) {
    if (!x.flags().contiguous) {
      tmps[i] = std::unique_ptr<array>(
          new array(mlx::core::contiguous_copy_gpu(x, s)));
      encoder.add_temporary(*tmps[i]);
      *slot = tmps[i].get();
    }
  };
  maybe_copy(inputs[3], &pb, 0);
  maybe_copy(inputs[4], &pa, 1);
  maybe_copy(inputs[7], &pc, 2);
  maybe_copy(inputs[9], &pz, 3);
  if ((int)inputs[0].shape(1) > 1) {
    maybe_copy(inputs[0], &pq, 4);
    maybe_copy(inputs[1], &pk, 5);
    maybe_copy(inputs[2], &pv, 6);
  }
  auto& q = *pq;
  auto& k = *pk;
  auto& v = *pv;
  auto& bb = *pb;
  auto& aa = *pa;
  auto& cs = *pc;
  auto& zg = *pz;
  auto& al = inputs[5];
  auto& db = inputs[6];
  auto& cw = inputs[8];
  auto& st = inputs[10];
  auto& nw = inputs[11];
  auto& result = outputs[0];
  auto& state_new = outputs[1];

  result.set_data(mlx::core::rocm::malloc_async(result.nbytes(), encoder));
  state_new.set_data(
      mlx::core::rocm::malloc_async(state_new.nbytes(), encoder));

  const char* dbg_env = std::getenv("P2_GDN_DEBUG");
  array dbg_arr({32 * 1042}, uint8, nullptr, {});
  if (dbg_env != nullptr && dbg_env[0] == '1') {
    dbg_arr = array({(int)(32 * 1042)}, uint8, nullptr, {});
    dbg_arr.set_data(
        mlx::core::rocm::malloc_async(dbg_arr.nbytes(), encoder));
  }

  // Shape-based dims (B=1 guaranteed by use_fallback; T is 1, 2 or 4): q [1, T, key_dim]  k [1, T, key_dim]  v [1, T, value_dim] z [1, T, Hv, 128]  nw [128]  cs [3, conv_dim]  cw [conv_dim, 4]
  int Tlen = (int)q.shape(1);
  int key_dim = (int)q.shape(2);
  int value_dim = (int)v.shape(2);
  int conv_dim = 2 * key_dim + value_dim;
  int head_dim = (int)nw.size();
  int num_v_heads = (int)zg.size() / head_dim / Tlen;
  int num_k_heads = key_dim / head_dim;

  // TMAX is a template constant so the t loop unrolls and the conv-window ternaries fold: the T=1 decode path compiles to exactly the original straight-line T=1 kernel (a runtime Tlen bound left a ~0.3% regression in the latency-bound AR loop), and TMAX=2 folds to the T=2 windows. T=4 (S=4 MTP verify) instantiates the same unrolled structure over 4 rows; h stays register-resident across all rows (r4d's T loop).
  auto launch_one = [&](hipStream_t stream, int tmax) {
    return (tmax == 1)
        ? launch_gdn_fused_step<1>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new), key_dim,
              value_dim, conv_dim, 4, num_v_heads, num_k_heads, head_dim,
              inv_scale_, eps_ * inv_scale_ * inv_scale_, norm_eps_,
              (dbg_env != nullptr && dbg_env[0] == '1')
                  ? dbg_arr.data<unsigned char>()
                  : nullptr,
              stream)
        : (tmax == 2)
        ? launch_gdn_fused_step<2>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new), key_dim,
              value_dim, conv_dim, 4, num_v_heads, num_k_heads, head_dim,
              inv_scale_, eps_ * inv_scale_ * inv_scale_, norm_eps_,
              (dbg_env != nullptr && dbg_env[0] == '1')
                  ? dbg_arr.data<unsigned char>()
                  : nullptr,
              stream)
        : launch_gdn_fused_step<4>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new), key_dim,
              value_dim, conv_dim, 4, num_v_heads, num_k_heads, head_dim,
              inv_scale_, eps_ * inv_scale_ * inv_scale_, norm_eps_,
              (dbg_env != nullptr && dbg_env[0] == '1')
                  ? dbg_arr.data<unsigned char>()
                  : nullptr,
              stream);
  };
  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_one(stream, Tlen);
    if (rc != 0) {
      throw std::runtime_error("gdn_fused_step: launch failed");
    }
  });

  if (dbg_env != nullptr && dbg_env[0] == '1') {
    auto* dbgp = dbg_arr.data<unsigned char>();
    encoder.synchronize();
    std::vector<unsigned char> host(dbg_arr.nbytes());
    hipMemcpy(host.data(), dbgp, host.size(), hipMemcpyDeviceToHost);
    auto f = std::ofstream("/tmp/p2_gdn_dbg.bin", std::ios::binary);
    f.write((const char*)host.data(), host.size());
  }
}

}  // namespace mlx::core::fast
