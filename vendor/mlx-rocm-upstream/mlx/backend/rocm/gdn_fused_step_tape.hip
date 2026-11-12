// Copyright © 2025 Apple Inc.
//
// Fused GDN T=1..4 decode/verify-step kernel WITH PER-ROW STATE TAPES (gdn_fused_step_tape, MLX_ROCM_GDN_TAPE=1; DEFAULT OFF — the gdn_fused_step op and its codegen are untouched, so MLX_ROCM_GDN_TAPE=0 is a zero-behavior-change path).
//
// This is the state-tape variant of gdn_fused_step (the T=1..4 fused conv+recurrence kernel, one launch per GDN layer per step, Dk=Dv=128). It computes EXACTLY the same result + final state (same arithmetic, same FMA order, same per-op bf16 roundings as gdn_fused_step — the tape stores add no math), plus two extra outputs:
//
// state_tape   [T, Hv, 128, 128] f32  — SSM state AFTER each row t conv_state_tape [T, 3, conv_dim] bf16 — conv window AFTER each row t (row t's new window = the last 3 taps of the row-t context E = [cs0, cs1, cs2, v0..v3]: {E[t+1], E[t+2], E[t+3]})
//
// h is register-resident across the TMAX-row t loop (exactly r4d's T loop), so the per-row state is FREE to store: after each row's update h holds that row's state and the tape store is one extra float4x16 global write per thread (TMAX writes per launch, vs 1 for state_new). The per-row conv window is a pure selection over the input conv_state rows and the raw k/q/v rows (no arithmetic), stored per handled channel in phase 1 (3 j taps x 3 channels per thread).
//
// Tape purpose: commit = an index select. The S=4 verify (T=4) produces all 4 rows' states in one launch; on acceptance the bench rebinds the GDN cache leaves to tape row (keep-1) (LAZY rebind — no sync, no copy: a state copy/replace would add a per-cycle sync+copy to the hot path), on rejection it selects the row at the accepted prefix (or leaves the leaves at the pre-verify state when 0 rows were accepted — the tape never holds the pre-verify state, which the leaves already point at). No restore, no re-forward of accepted rows. See docs/gdn-state-tape.md.
//
// Cost: one extra state-row store per row per v head (TMAX x Hv x 128 x 128 x 4 bytes of f32 stores = 8.4 MB at T=4/Hv=32) + the conv-tape selection writes (TMAX x 3 x conv_dim x 2 bytes). The tape buffer is round-reusable: the committed row is read by the next round's forwards before the next verify overwrites the buffer.
//
// Geometry and losslessness: identical to gdn_fused_step (one workgroup per v head, 32 blocks x 256 threads at Hv=32; the conv + qk_norm for the shared k/q head is computed by both v-head blocks redundantly but identically; g is computed f32 and rounded to bf16 before the read, exactly the chain's astype).

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/device/unary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// ---- warp reduce (copied from gdn_fused_step.hip) ----
__device__ float gdnt_warp_reduce_sum_rms(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// ---- rms_norm row body (copied verbatim from gdn_fused_step.hip, which copied it from rms_norm.hip): the exact 256-thread block reduction. ----
template <typename T, int BLOCK_DIM, int N_READS = 4>
__device__ void gdnt_rms_norm_row(
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
  float warp_sum = gdnt_warp_reduce_sum_rms(normalizer);
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
    normalizer = gdnt_warp_reduce_sum_rms(normalizer);
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

// u16 -> f32 (identical to the chain's bf2f).
__device__ __forceinline__ float gdnt_bf2f16(unsigned short h) {
  return __builtin_bit_cast(float, (unsigned)h << 16);
}
// f2bf: round-half-away (identical to the chain's f2bf; on store).
__device__ __forceinline__ unsigned short gdnt_f2bf(float f) {
  return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
// f2bf RNE: identical to MLX's _MLX_BFloat16 float ctor (astype f32->bf16), which is what the chain's astype(gates, out_dtype) does before the r4d kernel reads g.
__device__ __forceinline__ unsigned short gdnt_f2bf_rne(float f) {
  unsigned u = __builtin_bit_cast(unsigned, f);
  u += (u >> 16 & 1u) + 0x7FFFu;
  return (unsigned short)(u >> 16);
}

// Conv-window tap selection (NO arithmetic — a pure index select over the input conv_state rows and the raw rows, mirroring the phase-1 window taps): tap `idx` (0-based into E = [cs0, cs1, cs2, raw0..raw3]) of the channels: `raw_ch` into the raw array, `cs_ch` into the conv_state (they differ for the k/v channel blocks: conv channel = raw channel + the block's conv offset, e.g. 2*key_dim for v).
template <typename T>
__device__ __forceinline__ T gdnt_cs_tap(
    const T* cs,
    const T* raw,
    int raw_dim,
    int raw_ch,
    int cs_ch,
    int idx,
    int conv_dim) {
  return (idx <= 2)
      ? cs[idx * conv_dim + cs_ch]
      : raw[(idx - 3) * raw_dim + raw_ch];
}

// ---- The fused decode/verify step + per-row state tapes: one block per v head, 256 threads, TMAX in {1, 2, 3, 4} (T=3 only under the MLX_ROCM_GDN_T3 env gate). Identical to gdn_fused_step except: (a) the per-row SSM state tape store in phase 4 (h is the row's post-update state, register-resident), and (b) the per-row conv state tape writes in phase 1 (the next row-window selection). No other codegen difference; the tape stores add no math. ----
template <typename T, int TMAX>
__global__ __launch_bounds__(256) void gdn_fused_step_tape_kernel(
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
    float* __restrict__ state_tape,
    T* __restrict__ conv_state_tape,
    int key_dim,
    int value_dim,
    int conv_dim,
    int kernel_size,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    float inv_scale,
    float rms_eps,
    float norm_eps) {
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
  // One reduction buffer PER rms_norm call (sequential calls in one block must not share storage).
  __shared__ float s_ss_q[9];
  __shared__ float s_ss_k[9];
  __shared__ float s_ss_n[9];

  s_one = static_cast<T>(1.0f);

  // Phase-4 geometry (verbatim r4d VS=1): h is loaded ONCE, before the t loop, and updated row-by-row in registers.
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
    // ---------------- Phase 1: conv1d + silu (+ conv state tape) ------- Verbatim gdn_prep_kernel conv body: 4-value sequential k-order sum in f32, cast to T, silu = Multiply(conv_bf, Sigmoid(conv_bf)) in T. The conv window slides per row (E = [cs0, cs1, cs2, v0..v3]; row t = sum_j E[t+j]*w[j]).
    //
    // TAPE: the conv window AFTER row t is the last-3 of the row-t context, {E[t+1], E[t+2], E[t+3]} — a pure selection (gdnt_cs_tap), no arithmetic. Stored per handled channel (v/k in this branch, q in the else branch).
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
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int idx = t + 1 + j;
        conv_state_tape[((size_t)t * 3 + j) * conv_dim + cv] =
            gdnt_cs_tap(conv_state, v_raw, value_dim, vch, cv, idx, conv_dim);
      }

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
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int idx = t + 1 + j;
        conv_state_tape[((size_t)t * 3 + j) * conv_dim + ck] =
            gdnt_cs_tap(conv_state, k_raw, key_dim, kch, ck, idx, conv_dim);
      }
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
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int idx = t + 1 + j;
        conv_state_tape[((size_t)t * 3 + j) * conv_dim + cq] =
            gdnt_cs_tap(conv_state, q_raw, key_dim, qch, cq, idx, conv_dim);
      }
    }
    __syncthreads();

    // ---------------- Phase 2: beta + g ----------------
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

    // ---------------- Phase 3: q/k rms_norm + scale ----------------
    {
      float scale = inv_scale * inv_scale;
      float scale_f =
          static_cast<float>(static_cast<T>(scale));  // pre-round to bf16
      gdnt_rms_norm_row<T, 256, 4>(
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
      gdnt_rms_norm_row<T, 256, 4>(
          s_kconv, &s_one, s_kn_rms, s_ss_k, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 256) {
        float v = static_cast<float>(s_kn_rms[i]) * scale_f;
        s_kn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();

    // ---------------- Phase 4: recurrent state update (one row) + per-row SSM state tape ----------------
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
        qq[j + e] = gdnt_bf2f16(sq[e]);
        kk[j + e] = gdnt_bf2f16(sk[e]);
      }
    }

    // g is rounded to bf16 before the read (the chain's astype at the op boundary), exactly as gdn_fused_step.
    const float eg = __expf(
        fmaxf(__logf(gdnt_bf2f16(gdnt_f2bf_rne(s_g[t]))), -80.0f));
    const float bt = gdnt_bf2f16(
        *reinterpret_cast<unsigned short*>(&s_beta[t]));
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      h[j] *= eg;
      dot += h[j] * kk[j];
    }
    dot += __shfl_xor(dot, 1, 32);

    const float u = (gdnt_bf2f16(
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
      unsigned short out_bf = gdnt_f2bf(on);
      s_out[row] = *reinterpret_cast<T*>(&out_bf);
    }
    __syncthreads();

    // ---------------- TAPE: SSM state after row t ---------------- h is the row-t post-update state (register-resident across rows, exactly r4d's T loop); one extra float4x16 store per thread per row. state_tape[t] == state_new when t == TMAX-1 (both are materialized; the model path that consumes state_new is untouched).
    {
      float* p =
          state_tape + (((size_t)t * num_v_heads + hv) * head_dim + row) *
              head_dim + k0;
#pragma unroll
      for (int j = 0; j < 64; j += 4)
        *(float4*)(p + j) = *(const float4*)&h[j];
    }

    // ---------------- Phase 5: norm_gated = rms_norm + precise_swiglu ----
    gdnt_rms_norm_row<T, 256, 4>(
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
}
// ---- R1b: the fused sequential slot design. The slot op (per-row snapshots for the COMMIT ROWS 0..TMAX-2; row TMAX-1 is state_new) with (a) the t loop NOT unrolled (one per-row phase for all M rows; the per-row re-execution of the unrolled 4x phase copies is amortized into one pass over the rows; the per-thread weight loads are loop-invariant and hoisted by LICM) and (b) the conv_state_new output (the conv window AFTER the last row, a pure selection, no arithmetic) so the model branch consumes it for the conv-state cache update instead of dispatching concatenate + take_along_axis (launch integration: -2 ops per GDN layer per verify). The per-row phase arithmetic is the verbatim tape kernel body (sequential over positions, reduction order independent of n_tokens; the plan-140 construction); the non-unrolled t loop's per-row numerics vs the unrolled slot kernel carry the documented 1e-5..2e-4 drift class (G2 gates vs the M x T=1 same-codegen reference). n_slots (runtime) = TMAX-1 for the slot rows, same as GdnFusedStepSlot. ----
template <typename T, int TMAX>
__global__ __launch_bounds__(256) void gdn_fused_step_slot2_kernel(
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
    float* __restrict__ state_tape,
    T* __restrict__ conv_state_tape,
    T* __restrict__ conv_state_new,
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
    int n_slots) {
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
  // One reduction buffer PER rms_norm call (sequential calls in one block must not share storage).
  __shared__ float s_ss_q[9];
  __shared__ float s_ss_k[9];
  __shared__ float s_ss_n[9];

  s_one = static_cast<T>(1.0f);

  // Phase-4 geometry (verbatim r4d VS=1): h is loaded ONCE, before the t loop, and updated row-by-row in registers.
  const int row = tid >> 1, half = tid & 1, k0 = half * 64;
  float h[64];
  {
    const float* p =
        state + ((size_t)hv * head_dim + row) * head_dim + k0;
#pragma unroll
    for (int j = 0; j < 64; j += 4)
      *(float4*)&h[j] = *(const float4*)(p + j);
  }

  #pragma unroll 1  // R1b: ONE per-row phase for all M rows (no 4x unrolled copies; the per-row re-execution is amortized into one pass; the per-thread weight loads are loop-invariant and hoisted by LICM)
  for (int t = 0; t < TMAX; ++t) {
    // ---------------- Phase 1: conv1d + silu (+ conv state tape) ------- Verbatim gdn_prep_kernel conv body: 4-value sequential k-order sum in f32, cast to T, silu = Multiply(conv_bf, Sigmoid(conv_bf)) in T. The conv window slides per row (E = [cs0, cs1, cs2, v0..v3]; row t = sum_j E[t+j]*w[j]).
    //
    // TAPE: the conv window AFTER row t is the last-3 of the row-t context, {E[t+1], E[t+2], E[t+3]} — a pure selection (gdnt_cs_tap), no arithmetic. Stored per handled channel (v/k in this branch, q in the else branch).
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
      if (t < n_slots) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = t + 1 + j;
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + cv] =
              gdnt_cs_tap(conv_state, v_raw, value_dim, vch, cv, idx, conv_dim);
        }
      }
      // SLOT2: the conv state AFTER the last row (the post-pass conv state for the model cache update): the window {E[T], E[T+1], E[T+2]} (E = [cs0, cs1, cs2, raw0..rawT-1]) - a pure selection, no arithmetic (bit-exact by construction; the model branch consumes it instead of dispatching concatenate + take_along_axis).
      if (t == TMAX - 1) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = TMAX + j;
          conv_state_new[j * conv_dim + cv] =
              gdnt_cs_tap(conv_state, v_raw, value_dim, vch, cv, idx, conv_dim);
        }
      }

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
      if (t < n_slots) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = t + 1 + j;
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + ck] =
              gdnt_cs_tap(conv_state, k_raw, key_dim, kch, ck, idx, conv_dim);
        }
      }
      if (t == TMAX - 1) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = TMAX + j;
          conv_state_new[j * conv_dim + ck] =
              gdnt_cs_tap(conv_state, k_raw, key_dim, kch, ck, idx, conv_dim);
        }
      }
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
      if (t < n_slots) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = t + 1 + j;
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + cq] =
              gdnt_cs_tap(conv_state, q_raw, key_dim, qch, cq, idx, conv_dim);
        }
      }
      if (t == TMAX - 1) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          int idx = TMAX + j;
          conv_state_new[j * conv_dim + cq] =
              gdnt_cs_tap(conv_state, q_raw, key_dim, qch, cq, idx, conv_dim);
        }
      }
    }
    __syncthreads();

    // ---------------- Phase 2: beta + g ----------------
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

    // ---------------- Phase 3: q/k rms_norm + scale ----------------
    {
      float scale = inv_scale * inv_scale;
      float scale_f =
          static_cast<float>(static_cast<T>(scale));  // pre-round to bf16
      gdnt_rms_norm_row<T, 256, 4>(
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
      gdnt_rms_norm_row<T, 256, 4>(
          s_kconv, &s_one, s_kn_rms, s_ss_k, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 256) {
        float v = static_cast<float>(s_kn_rms[i]) * scale_f;
        s_kn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();

    // ---------------- Phase 4: recurrent state update (one row) + per-row SSM state tape ----------------
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
        qq[j + e] = gdnt_bf2f16(sq[e]);
        kk[j + e] = gdnt_bf2f16(sk[e]);
      }
    }

    // g is rounded to bf16 before the read (the chain's astype at the op boundary), exactly as gdn_fused_step.
    const float eg = __expf(
        fmaxf(__logf(gdnt_bf2f16(gdnt_f2bf_rne(s_g[t]))), -80.0f));
    const float bt = gdnt_bf2f16(
        *reinterpret_cast<unsigned short*>(&s_beta[t]));
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      h[j] *= eg;
      dot += h[j] * kk[j];
    }
    dot += __shfl_xor(dot, 1, 32);

    const float u = (gdnt_bf2f16(
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
      unsigned short out_bf = gdnt_f2bf(on);
      s_out[row] = *reinterpret_cast<T*>(&out_bf);
    }
    __syncthreads();

    // ---------------- TAPE: SSM state after row t ---------------- h is the row-t post-update state (register-resident across rows, exactly r4d's T loop); one extra float4x16 store per thread per row. state_tape[t] == state_new when t == TMAX-1 (both are materialized; the model path that consumes state_new is untouched). n_slots (runtime) gates which rows are snapshotted: the tape op passes TMAX (all rows, unchanged); the slot op passes TMAX-1 (row TMAX-1 is state_new itself - the commit rows are 0..TMAX-2).
    if (t < n_slots) {
      float* p =
          state_tape + (((size_t)t * num_v_heads + hv) * head_dim + row) *
              head_dim + k0;
#pragma unroll
      for (int j = 0; j < 64; j += 4)
        *(float4*)(p + j) = *(const float4*)&h[j];
    }

    // ---------------- Phase 5: norm_gated = rms_norm + precise_swiglu ----
    gdnt_rms_norm_row<T, 256, 4>(
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
}
// ---- rms_norm row body, 2D-block variant for the seq kernel (warp index = threadIdx.y): the tape body indexes warps via threadIdx.x / WARP_SIZE, which is constant 0 in the (32, 4) block (shared_sum races + dropped warp partials). Here the 128 threads read/write via tid = warp_id * WARP_SIZE + lane (4 warps x 32); the 4-warp partial-sum class is the named drift class vs the 8-warp tape variant. The per-element read pattern (threads tid 0..31 read 4 each, the rest 0) matches the 256-thread tape instantiation exactly.
template <typename T, int WARPS, int N_READS = 4>
__device__ void gdnt_rms_norm_row_2d(
    const T* x,
    const T* w,
    T* out,
    float* shared_sum,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;
  float normalizer = 0;
  for (int i = (warp_id * WARP_SIZE + lane) * N_READS; i < (int)axis_size;
       i += WARPS * WARP_SIZE * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < (int)axis_size; ++j) {
      float t = static_cast<float>(x[i + j]);
      normalizer += t * t;
    }
  }
  float warp_sum = gdnt_warp_reduce_sum_rms(normalizer);
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    normalizer = (lane < WARPS) ? shared_sum[lane] : 0;
    normalizer = gdnt_warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();
  if (warp_id == 0 && lane == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);
  for (int i = (warp_id * WARP_SIZE + lane) * N_READS; i < (int)axis_size;
       i += WARPS * WARP_SIZE * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < (int)axis_size; ++j) {
      int idx = i + j;
      T normalized =
          static_cast<T>(static_cast<float>(x[idx]) * normalizer);
      T wi = (w_stride == 0) ? w[0] : w[idx * w_stride];
      out[idx] = wi * normalized;
    }
  }
}

// ---- Phase 2 (W1): the sequential GDN kernel (the llama.cpp GDN design port, reference 29c720ea). The slot op (per-row state slots for the COMMIT ROWS 0..TMAX-2; row TMAX-1 is state_new) on the COLUMN-PARALLEL sequential geometry: one CTA covers cols_per_warp(4) x num_warps(4) = 16 v-columns of the head state; grid (Hv, 1, head_dim/16); block (32, 4) = 128 threads. Per warp: 4 column segments of lanes_per_col(8) lanes; each lane owns one v-column's contiguous 16-k segment in registers (s_shard[16]; the k-contiguous state layout [Hv][128v][128k] supports the float4 vectorization; the store is an index remap of the tape store - no transpose, no adapter). The t loop is a RUNTIME loop over the TMAX rows (sequential over positions; the per-position reduction order is independent of T - the bit-exact-across-T property, verified at G1, not trusted). Per-row phase arithmetic = the tape kernel body (phase 1 conv + silu per channel verbatim; phase 2 beta + g thread-0 body verbatim; phase 3 qk-norm = the 128-CTA variant of gdnt_rms_norm_row - the named drift class: 4 warp partials vs the 8-warp partials; phase 4 recurrence = the reference KDA=false body with OUR per-element order (scale-then-dot: s *= eg, dot = sum (s*k) in-lane ascending; u = (v - dot) * bt; s += u * k; on = sum (s*q)) and our eg (the bf16 round-trip decay, phase 2 verbatim - our class, not the reference's raw expf)). The in-row reduction = in-lane 16 ascending FMA + the 3-step shfl_xor tree over 8 lanes (offsets 4, 2, 1 within the contiguous 8-lane segment) - a NEW reduction class vs the tape kernel's 64+1-shfl split: the named drift class, gated by G1 token-identity (the R1b precedent: 1e-5..2e-4, zero argmax flips over 1065 tokens). The conv_slot + conv_state_new stores are SINGLE-WRITER = colgroup 0 (the colgroup CTAs replicate the prep bit-equal - pure functions of the same inputs - so any colgroup's result is identical; 8x store traffic is the named waste the ruling cuts). The attn output (per-column on, bf16, no in-kernel scale - the scale is folded into phase 3 as today) goes to the op-internal attn intermediate [T, Hv, 128] bf16; the full-row rms + precise swiglu (the tape phase-5 body VERBATIM, 256-thread block per head) runs in the 2nd kernel gdn_fused_step_seq_out_kernel (the cross-CTA row-rms split; 2 launches per GDN layer per step vs 1 today; both kernels = the one GDN-sequential-variant family). Gated by MLX_ROCM_GDN_SEQ=1 (DEFAULT OFF: the op is unreachable without it -> zero-behavior-change). Same shape gate as gdn_fused_step_slot2 (B=1, 2<=T<=4, Dk=Dv=128, Hv%Hk==0, kernel_size=4; T=3 only under MLX_ROCM_GDN_T3=1). The slot2 op, the slot op, the tape op and the gdn_fused_step op are untouched. ----
template <typename T, int TMAX>
__global__ void __launch_bounds__(128) gdn_fused_step_seq_kernel(
    const T* __restrict__ q_raw,
    const T* __restrict__ k_raw,
    const T* __restrict__ v_raw,
    const T* __restrict__ b,
    const T* __restrict__ a,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const T* __restrict__ conv_state,
    const T* __restrict__ conv_weight,
    const float* __restrict__ state,
    T* __restrict__ attn,
    float* __restrict__ state_new,
    float* __restrict__ state_tape,
    T* __restrict__ conv_state_tape,
    T* __restrict__ conv_state_new,
    int key_dim,
    int value_dim,
    int conv_dim,
    int kernel_size,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    float inv_scale,
    float rms_eps,
    int n_slots) {
  constexpr int COLS_PER_WARP = 4;
  constexpr int LANES_PER_COL = WARP_SIZE / COLS_PER_WARP;  // 8
  constexpr int RPL = 16;  // = head_dim / LANES_PER_COL @ head_dim = 128 (the gate Dk=Dv=128; the tape kernel hardcodes its 64/128 constants the same way)
  const int hv = blockIdx.x;
  const int colgroup = blockIdx.z;
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;
  const int col_in_warp = lane / LANES_PER_COL;
  const int lane_in_col = lane - col_in_warp * LANES_PER_COL;
  const int tid = warp_id * WARP_SIZE + lane;  // unique 0..127 thread id for the shared-array channel distribution (threadIdx.x alone only spans 0..31 in the (32, 4) block)
  const int col = (colgroup * blockDim.y + warp_id) * COLS_PER_WARP +
                  col_in_warp;  // v-column 0..head_dim-1 within the head (blockDim.y = 4 warps = 16 cols/CTA)
  const int hk = hv / (num_v_heads / num_k_heads);

  __shared__ T s_qconv[128];
  __shared__ T s_kconv[128];
  __shared__ T s_vconv[128];
  __shared__ T s_qn_rms[128];
  __shared__ T s_kn_rms[128];
  __shared__ __attribute__((aligned(16))) T s_qn[128];
  __shared__ __attribute__((aligned(16))) T s_kn[128];
  __shared__ T s_beta[TMAX];
  __shared__ T s_one;
  __shared__ float s_g[TMAX];
  // One reduction buffer PER rms_norm call (sequential calls in one block must not share storage).
  __shared__ float s_ss_q[9];
  __shared__ float s_ss_k[9];

  s_one = static_cast<T>(1.0f);

  // Register state: the contiguous k-segment [lane_in_col*RPL, +RPL) of my
  // v-column. The state layout state[(hv*head_dim + v)*head_dim + k] is
  // k-contiguous = the reference transposed storage M[col=v][i=k]; the load
  // is 4 x float4, the snapshot store is the same remap (no transpose).
  float s_shard[RPL];
  {
    const float* p =
        state + ((size_t)hv * head_dim + col) * head_dim + lane_in_col * RPL;
#pragma unroll
    for (int j = 0; j < RPL; j += 4)
      *(float4*)&s_shard[j] = *(const float4*)(p + j);
  }

  #pragma unroll 1  // sequential over positions (reduction order independent of T)
  for (int t = 0; t < TMAX; ++t) {
    // ---------------- Phase 1: conv1d + silu (+ conv_slot / conv_state_new, colgroup-0 single-writer) ---- Per-channel 4-tap sequential k-order sum in f32, cast to T, silu in T (the tape body verbatim, order-preserved per channel -> bit-exact under the 128-thread x 3-channel split; the colgroup>0 CTAs replicate the colgroup-0 result bit-equal - pure function of the same inputs).
    {
      const int vch = hv * head_dim + tid;
      const int cv = 2 * key_dim + vch;
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

      const int kch = hk * head_dim + tid;
      const int ck = key_dim + kch;
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

      const int qch = kch;
      const int cq = qch;
      float acc3 = 0.0f;
      acc3 += static_cast<float>(
                  (t <= 2) ? conv_state[t * conv_dim + cq]
                           : q_raw[(t - 3) * key_dim + qch]) *
              static_cast<float>(conv_weight[cq * kernel_size + 0]);
      acc3 += static_cast<float>(
                  (t <= 1) ? conv_state[(t + 1) * conv_dim + cq]
                           : q_raw[(t - 2) * key_dim + qch]) *
              static_cast<float>(conv_weight[cq * kernel_size + 1]);
      acc3 += static_cast<float>(
                  (t == 0) ? conv_state[2 * conv_dim + cq]
                           : q_raw[(t - 1) * key_dim + qch]) *
              static_cast<float>(conv_weight[cq * kernel_size + 2]);
      acc3 += static_cast<float>(q_raw[t * key_dim + qch]) *
              static_cast<float>(conv_weight[cq * kernel_size + 3]);
      T conv_bf3 = static_cast<T>(acc3);
      s_qconv[tid] = Multiply{}(conv_bf3, Sigmoid{}(conv_bf3));
    }
    if (colgroup == 0) {
      if (t < n_slots) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int idx = t + 1 + j;
          const int vch = hv * head_dim + tid;
          const int cv = 2 * key_dim + vch;
          const int kch = hk * head_dim + tid;
          const int ck = key_dim + kch;
          const int cq = kch;
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + cv] =
              gdnt_cs_tap(conv_state, v_raw, value_dim, vch, cv, idx, conv_dim);
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + ck] =
              gdnt_cs_tap(conv_state, k_raw, key_dim, kch, ck, idx, conv_dim);
          conv_state_tape[((size_t)t * 3 + j) * conv_dim + cq] =
              gdnt_cs_tap(conv_state, q_raw, key_dim, kch, cq, idx, conv_dim);
        }
      }
      if (t == TMAX - 1) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int idx = TMAX + j;
          const int vch = hv * head_dim + tid;
          const int cv = 2 * key_dim + vch;
          const int kch = hk * head_dim + tid;
          const int ck = key_dim + kch;
          const int cq = kch;
          conv_state_new[j * conv_dim + cv] =
              gdnt_cs_tap(conv_state, v_raw, value_dim, vch, cv, idx, conv_dim);
          conv_state_new[j * conv_dim + ck] =
              gdnt_cs_tap(conv_state, k_raw, key_dim, kch, ck, idx, conv_dim);
          conv_state_new[j * conv_dim + cq] =
              gdnt_cs_tap(conv_state, q_raw, key_dim, kch, cq, idx, conv_dim);
        }
      }
    }
    __syncthreads();

    // ---------------- Phase 2: beta + g (thread 0 of each CTA; the 8 colgroup CTAs replicate bit-equal - pure scalar compute; the tape body verbatim) ----
    if (lane == 0 && warp_id == 0) {
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

    // ---------------- Phase 3: q/k rms_norm + scale (the 128-CTA variant of the tape body - the named drift class: 4 warp partials vs 8) ----
    {
      float scale = inv_scale * inv_scale;
      float scale_f = static_cast<float>(static_cast<T>(scale));
      gdnt_rms_norm_row_2d<T, 4, 4>(
          s_qconv, &s_one, s_qn_rms, s_ss_q, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 128) {
        float v = static_cast<float>(s_qn_rms[i]) * scale_f;
        s_qn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();
    {
      float scale = inv_scale;
      float scale_f = static_cast<float>(static_cast<T>(scale));
      gdnt_rms_norm_row_2d<T, 4, 4>(
          s_kconv, &s_one, s_kn_rms, s_ss_k, rms_eps, head_dim, 0);
      __syncthreads();
      for (int i = tid; i < head_dim; i += 128) {
        float v = static_cast<float>(s_kn_rms[i]) * scale_f;
        s_kn[i] = static_cast<T>(v);
      }
    }
    __syncthreads();

    // ---------------- Phase 4: recurrent state update (one row) + per-row SSM state slot + attn intermediate ----------------
    // The reference KDA=false body with OUR per-element order (scale-then-dot; the in-row accumulation over the contiguous 16-k segment; the 3-step shfl tree over the 8-lane column segment - the named new reduction class).
    float kk[RPL], qq[RPL];
    {
      const unsigned short* pk =
          reinterpret_cast<const unsigned short*>(s_kn) + lane_in_col * RPL;
      const unsigned short* pqa =
          reinterpret_cast<const unsigned short*>(s_qn) + lane_in_col * RPL;
#pragma unroll
      for (int j = 0; j < RPL; j += 8) {
        const uint4 wk = *(const uint4*)(pk + j);
        const uint4 wq = *(const uint4*)(pqa + j);
        const unsigned short* sk = (const unsigned short*)&wk;
        const unsigned short* sq = (const unsigned short*)&wq;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          kk[j + e] = gdnt_bf2f16(sk[e]);
          qq[j + e] = gdnt_bf2f16(sq[e]);
        }
      }
    }

    // g is rounded to bf16 before the read (the chain's astype at the op boundary), exactly as the tape kernel: eg = exp(max(log(bf16(g)), -80)).
    const float eg = __expf(
        fmaxf(__logf(gdnt_bf2f16(gdnt_f2bf_rne(s_g[t]))), -80.0f));
    const float bt = gdnt_bf2f16(
        *reinterpret_cast<const unsigned short*>(&s_beta[t]));
    const float v_col = gdnt_bf2f16(
        *reinterpret_cast<const unsigned short*>(&s_vconv[col]));

    float dot_shard = 0.0f;
#pragma unroll
    for (int j = 0; j < RPL; ++j) {
      s_shard[j] *= eg;
      dot_shard += s_shard[j] * kk[j];
    }
    float dot_col = dot_shard;
    dot_col += __shfl_xor(dot_col, 4, 32);
    dot_col += __shfl_xor(dot_col, 2, 32);
    dot_col += __shfl_xor(dot_col, 1, 32);

    const float u = (v_col - dot_col) * bt;

    float on_shard = 0.0f;
#pragma unroll
    for (int j = 0; j < RPL; ++j) {
      s_shard[j] += u * kk[j];
      on_shard += s_shard[j] * qq[j];
    }
    float on_col = on_shard;
    on_col += __shfl_xor(on_col, 4, 32);
    on_col += __shfl_xor(on_col, 2, 32);
    on_col += __shfl_xor(on_col, 1, 32);

    if (lane_in_col == 0) {
      unsigned short out_bf = gdnt_f2bf(on_col);
      attn[(size_t)t * num_v_heads * head_dim + hv * head_dim + col] =
          *reinterpret_cast<T*>(&out_bf);
    }

    // ---------------- SLOT: SSM state after row t (commit rows 0..TMAX-2) ---------------- n_slots (runtime) = TMAX-1 (row TMAX-1 is state_new); the store is the tape store's index remap onto the k-segment geometry (4 x float4, k-contiguous - no transpose).
    if (t < n_slots) {
      float* p =
          state_tape + (((size_t)t * num_v_heads + hv) * head_dim + col) *
                           head_dim +
                       lane_in_col * RPL;
#pragma unroll
      for (int j = 0; j < RPL; j += 4)
        *(float4*)(p + j) = *(const float4*)&s_shard[j];
    }
    __syncthreads();
  }

  // Final state store (after the last row's update).
  {
    float* p =
        state_new + ((size_t)hv * head_dim + col) * head_dim +
                   lane_in_col * RPL;
#pragma unroll
    for (int j = 0; j < RPL; j += 4)
      *(float4*)(p + j) = *(const float4*)&s_shard[j];
  }
}

// ---- Phase 2 (W1): the seq out stage (the tape phase-5 body VERBATIM, 256-thread block per head over the attn intermediate [TMAX, Hv, 128] bf16: full-row rms + precise swiglu; the cross-CTA row-rms split of the column-parallel geometry; the numerics = today's phase 5 exactly given the same bf16 input row - the bf16 store of the on value happens in the seq kernel, as today in s_out). ----
template <typename T, int TMAX>
__global__ void __launch_bounds__(256) gdn_fused_step_seq_out_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ z,
    const T* __restrict__ norm_weight,
    T* __restrict__ result,
    int num_v_heads,
    int head_dim,
    float norm_eps) {
  const int hv = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ T s_out[128];
  __shared__ T s_xn[128];
  __shared__ float s_ss_n[9];
  #pragma unroll 1
  for (int t = 0; t < TMAX; ++t) {
    for (int i = tid; i < head_dim; i += 256)
      s_out[i] =
          attn[(size_t)t * num_v_heads * head_dim + hv * head_dim + i];
    __syncthreads();
    gdnt_rms_norm_row<T, 256, 4>(
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
}
}  // namespace rocm

// ---- host launcher ---- Instantiates TMAX in {1, 2, 3, 4}, selected at launch from T (T=3 is admitted by use_fallback only under MLX_ROCM_GDN_T3=1, default off -> T=3 degrades to the clean error as before).
template <int TMAX>
int launch_gdn_fused_step_tape(
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
    void* state_tape,
    void* conv_state_tape,
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
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid(num_v_heads);
  rocm::gdn_fused_step_tape_kernel<T, TMAX><<<grid, block, 0, stream>>>(
      (const T*)q_raw, (const T*)k_raw, (const T*)v_raw, (const T*)b,
      (const T*)a, A_log, dt_bias, (const T*)conv_state,
      (const T*)conv_weight, (const T*)z, state, (const T*)norm_weight,
      (T*)result, (float*)state_new, (float*)state_tape,
      (T*)conv_state_tape, key_dim, value_dim, conv_dim, kernel_size,
      num_v_heads, num_k_heads, head_dim, inv_scale, rms_eps, norm_eps);
  return (int)hipGetLastError();
}


// ---- host launcher (R1b slot2) ---- Instantiates TMAX in {2, 3, 4} (T=3 admitted by use_fallback only under MLX_ROCM_GDN_T3=1, same gate as the slot op), launches the non-unrolled slot2 kernel with the conv_state_new output. The slot op (unrolled) is untouched.
template <int TMAX>
int launch_gdn_fused_step_slot2(
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
    void* state_tape,
    void* conv_state_tape,
    void* conv_state_new,
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
    int n_slots,
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(256);
  dim3 grid(num_v_heads);
  rocm::gdn_fused_step_slot2_kernel<T, TMAX><<<grid, block, 0, stream>>>(
      (const T*)q_raw, (const T*)k_raw, (const T*)v_raw, (const T*)b,
      (const T*)a, A_log, dt_bias, (const T*)conv_state,
      (const T*)conv_weight, (const T*)z, state, (const T*)norm_weight,
      (T*)result, (float*)state_new, (float*)state_tape,
      (T*)conv_state_tape, (T*)conv_state_new, key_dim, value_dim,
      conv_dim, kernel_size, num_v_heads, num_k_heads, head_dim,
      inv_scale, rms_eps, norm_eps, n_slots);
  return (int)hipGetLastError();
}


// ---- host launcher (Phase 2 W1 sequential) ---- Instantiates TMAX in {2, 3, 4} (T=3 admitted by use_fallback only under MLX_ROCM_GDN_T3=1, same gate as the slot2 op), launches the sequential kernel (the column-parallel geometry; grid (Hv, 1, head_dim/16), block (32, 4)) then the out-stage kernel (the tape phase-5 body verbatim, 256 threads per head) over the attn intermediate. The slot2 op and the slot op are untouched.
template <int TMAX>
int launch_gdn_fused_step_seq(
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
    void* attn,
    void* state_new,
    void* state_tape,
    void* conv_state_tape,
    void* conv_state_new,
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
    int n_slots,
    hipStream_t stream) {
  using T = hip_bfloat16;
  dim3 block(32, 4);
  dim3 grid(num_v_heads, 1, head_dim / 16);
  rocm::gdn_fused_step_seq_kernel<T, TMAX><<<grid, block, 0, stream>>>(
      (const T*)q_raw, (const T*)k_raw, (const T*)v_raw, (const T*)b,
      (const T*)a, A_log, dt_bias, (const T*)conv_state,
      (const T*)conv_weight, state, (T*)attn, (float*)state_new,
      (float*)state_tape, (T*)conv_state_tape, (T*)conv_state_new, key_dim,
      value_dim, conv_dim, kernel_size, num_v_heads, num_k_heads, head_dim,
      inv_scale, rms_eps, n_slots);
  int rc1 = (int)hipGetLastError();
  dim3 block2(256);
  dim3 grid2(num_v_heads);
  rocm::gdn_fused_step_seq_out_kernel<T, TMAX><<<grid2, block2, 0, stream>>>(
      (T*)attn, (const T*)z, (const T*)norm_weight, (T*)result, num_v_heads,
      head_dim, norm_eps);
  int rc2 = (int)hipGetLastError();
  return rc1 != 0 ? rc1 : rc2;
}
}  // namespace mlx::core

// ---- eval_gpu implementation ----
namespace mlx::core::fast {

void GdnFusedStepTape::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  // Same row-contiguity forcing as GdnFusedStep::eval_gpu (the F-branch passes row-strided slices; the kernel reads them packed).
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
  auto& state_tape = outputs[2];
  auto& conv_state_tape = outputs[3];

  result.set_data(mlx::core::rocm::malloc_async(result.nbytes(), encoder));
  state_new.set_data(
      mlx::core::rocm::malloc_async(state_new.nbytes(), encoder));
  state_tape.set_data(
      mlx::core::rocm::malloc_async(state_tape.nbytes(), encoder));
  conv_state_tape.set_data(
      mlx::core::rocm::malloc_async(conv_state_tape.nbytes(), encoder));

  // Shape-based dims (B=1 guaranteed by use_fallback; T is 1, 2, 3 or 4): q [1, T, key_dim]  k [1, T, key_dim]  v [1, T, value_dim] z [1, T, Hv, 128]  nw [128]  cs [3, conv_dim]  cw [conv_dim, 4]
  int Tlen = (int)q.shape(1);
  int key_dim = (int)q.shape(2);
  int value_dim = (int)v.shape(2);
  int conv_dim = 2 * key_dim + value_dim;
  int head_dim = (int)nw.size();
  int num_v_heads = (int)zg.size() / head_dim / Tlen;
  int num_k_heads = key_dim / head_dim;

  // TMAX is a template constant so the t loop unrolls and the conv-window ternaries fold (identical to GdnFusedStep::eval_gpu). T=3 (S=3, K=2 spec-decode) maps to the <3> instantiation; the shared s_beta/s_g buffers are sized [4] and the tape buffers are T-row-shaped, so <3> uses strict prefixes only.
  auto launch_one = [&](hipStream_t stream, int tmax) {
    return (tmax == 1)
        ? launch_gdn_fused_step_tape<1>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_tape), gpu_ptr<void>(conv_state_tape),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, stream)
        : (tmax == 2)
        ? launch_gdn_fused_step_tape<2>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_tape), gpu_ptr<void>(conv_state_tape),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, stream)
        : (tmax == 3)
        ? launch_gdn_fused_step_tape<3>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_tape), gpu_ptr<void>(conv_state_tape),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, stream)
        : launch_gdn_fused_step_tape<4>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_tape), gpu_ptr<void>(conv_state_tape),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, stream);
  };
  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_one(stream, Tlen);
    if (rc != 0) {
      throw std::runtime_error("gdn_fused_step_tape: launch failed");
    }
  });
}


// ---- GdnFusedStepSlot2::eval_gpu ---- R1b (the fused sequential slot design): the same fused verify step as GdnFusedStepSlot (slot rows 0..T-2; row T-1 is state_new), but the kernel is the NON-UNROLLED slot2 variant (one per-row phase for all M rows) and it emits the conv_state_new output (the post-pass conv state, a pure selection) which the model branch consumes for the conv-state cache update (the python-side concatenate + take_along_axis is not dispatched on that branch). Identical to GdnFusedStepSlot::eval_gpu except: (a) the launch goes to launch_gdn_fused_step_slot2 with n_slots = Tlen-1, and (b) outputs[4] = conv_state_new [3, conv_dim] bf16. The slot op, the tape op and the gdn_fused_step op are untouched.

void GdnFusedStepSlot2::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  // Same row-contiguity forcing as GdnFusedStepSlot::eval_gpu.
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
  auto& state_slot = outputs[2];
  auto& conv_state_slot = outputs[3];
  auto& conv_state_new = outputs[4];

  result.set_data(mlx::core::rocm::malloc_async(result.nbytes(), encoder));
  state_new.set_data(
      mlx::core::rocm::malloc_async(state_new.nbytes(), encoder));
  state_slot.set_data(
      mlx::core::rocm::malloc_async(state_slot.nbytes(), encoder));
  conv_state_slot.set_data(
      mlx::core::rocm::malloc_async(conv_state_slot.nbytes(), encoder));
  conv_state_new.set_data(
      mlx::core::rocm::malloc_async(conv_state_new.nbytes(), encoder));

  // Shape-based dims (B=1 guaranteed by use_fallback; T is 2, 3 or 4).
  int Tlen = (int)q.shape(1);
  int key_dim = (int)q.shape(2);
  int value_dim = (int)v.shape(2);
  int conv_dim = 2 * key_dim + value_dim;
  int head_dim = (int)nw.size();
  int num_v_heads = (int)zg.size() / head_dim / Tlen;
  int num_k_heads = key_dim / head_dim;

  // Slot rows = the commit rows 0..T-2 (Tlen-1 rows; row T-1 is state_new). The kernel is the NON-UNROLLED slot2 variant (one per-row phase for all M rows); n_slots = Tlen-1 gates the per-row snapshot stores; the conv_state_new store fires on the last row (t == TMAX-1).
  auto launch_one = [&](hipStream_t stream, int tmax) {
    return (tmax == 2)
        ? launch_gdn_fused_step_slot2<2>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_slot), gpu_ptr<void>(conv_state_slot),
              gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream)
        : (tmax == 3)
        ? launch_gdn_fused_step_slot2<3>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_slot), gpu_ptr<void>(conv_state_slot),
              gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream)
        : launch_gdn_fused_step_slot2<4>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(state_new),
              gpu_ptr<void>(state_slot), gpu_ptr<void>(conv_state_slot),
              gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream);
  };
  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_one(stream, Tlen);
    if (rc != 0) {
      throw std::runtime_error("gdn_fused_step_slot2: launch failed");
    }
  });
}


// ---- GdnFusedStepSeq::eval_gpu ---- Phase 2 (W1 sequential, the llama.cpp GDN design port): the slot op contract (commit rows 0..T-2; row T-1 = state_new) + the conv_state_new output (the slot2 launch integration rides along) on the COLUMN-PARALLEL sequential kernel + the out-stage kernel (the tape phase-5 body verbatim). The attn intermediate [T, Hv, 128] bf16 is op-internal (24KB @T=3; 8KB @T=1). Identical to GdnFusedStepSlot2::eval_gpu except: (a) the launch goes to launch_gdn_fused_step_seq (2 kernels), and (b) the attn temporary is allocated here. The slot2 op, the slot op, the tape op and the gdn_fused_step op are untouched.

void GdnFusedStepSeq::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  // Same row-contiguity forcing as GdnFusedStepSlot2::eval_gpu.
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
  auto& state_slot = outputs[2];
  auto& conv_state_slot = outputs[3];
  auto& conv_state_new = outputs[4];

  // Shape-based dims (B=1 guaranteed by use_fallback; T is 2, 3 or 4).
  int Tlen = (int)q.shape(1);
  int key_dim = (int)q.shape(2);
  int value_dim = (int)v.shape(2);
  int conv_dim = 2 * key_dim + value_dim;
  int head_dim = (int)nw.size();
  int num_v_heads = (int)zg.size() / head_dim / Tlen;
  int num_k_heads = key_dim / head_dim;

  result.set_data(mlx::core::rocm::malloc_async(result.nbytes(), encoder));
  state_new.set_data(
      mlx::core::rocm::malloc_async(state_new.nbytes(), encoder));
  state_slot.set_data(
      mlx::core::rocm::malloc_async(state_slot.nbytes(), encoder));
  conv_state_slot.set_data(
      mlx::core::rocm::malloc_async(conv_state_slot.nbytes(), encoder));
  conv_state_new.set_data(
      mlx::core::rocm::malloc_async(conv_state_new.nbytes(), encoder));

  // The attn intermediate (op-internal; the seq kernel's per-column attn values, bf16, consumed by the out-stage kernel; [T, Hv, 128]).
  std::unique_ptr<array> attn_tmp(
      new array(Shape{Tlen, num_v_heads, 128}, bfloat16, nullptr, {}));
  attn_tmp->set_data(
      mlx::core::rocm::malloc_async(attn_tmp->nbytes(), encoder));
  encoder.add_temporary(*attn_tmp);

  // n_slots = Tlen-1 gates the per-row snapshot stores (commit rows 0..T-2; row T-1 is state_new); the conv_state_new store fires on the last row (t == TMAX-1).
  auto launch_one = [&](hipStream_t stream, int tmax) {
    return (tmax == 2)
        ? launch_gdn_fused_step_seq<2>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(*attn_tmp),
              gpu_ptr<void>(state_new), gpu_ptr<void>(state_slot),
              gpu_ptr<void>(conv_state_slot), gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream)
        : (tmax == 3)
        ? launch_gdn_fused_step_seq<3>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(*attn_tmp),
              gpu_ptr<void>(state_new), gpu_ptr<void>(state_slot),
              gpu_ptr<void>(conv_state_slot), gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream)
        : launch_gdn_fused_step_seq<4>(
              gpu_ptr<void>(q), gpu_ptr<void>(k), gpu_ptr<void>(v),
              gpu_ptr<void>(bb), gpu_ptr<void>(aa), gpu_ptr<float>(al),
              gpu_ptr<float>(db), gpu_ptr<void>(cs), gpu_ptr<void>(cw),
              gpu_ptr<void>(zg), gpu_ptr<float>(st), gpu_ptr<void>(nw),
              gpu_ptr<void>(result), gpu_ptr<void>(*attn_tmp),
              gpu_ptr<void>(state_new), gpu_ptr<void>(state_slot),
              gpu_ptr<void>(conv_state_slot), gpu_ptr<void>(conv_state_new),
              key_dim, value_dim, conv_dim, 4, num_v_heads, num_k_heads,
              head_dim, inv_scale_, eps_ * inv_scale_ * inv_scale_,
              norm_eps_, Tlen - 1, stream);
  };
  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = launch_one(stream, Tlen);
    if (rc != 0) {
      throw std::runtime_error("gdn_fused_step_seq: launch failed");
    }
  });
}
}  // namespace mlx::core::fast
