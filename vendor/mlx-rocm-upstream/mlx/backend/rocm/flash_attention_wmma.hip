// WMMA-accelerated flash attention for RDNA 3+ (gfx1100+)
//
// Uses rocwmma 16x16x16 bf16→f32 tiles for Q@K^T and P@V matmuls.
// Implements FlashAttention-2 online softmax.
//
// BLOCK_M=64, BLOCK_N=64, 128 threads (4 waves), each wave owns 16 query rows.
// Shared memory ~50 KB for D=128.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define ROCM_FA_WMMA 1
#include <rocwmma/rocwmma.hpp>
#else
#define ROCM_FA_WMMA 0
#endif

namespace mlx::core {
namespace rocm {

struct FAWmmaParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  // Optional LSE output strides for shape [B, H, qL, 1] (float32). Unused when
  // LSE ptr is null. Contiguous layout: LSE_strides[0]=H*qL, [1]=qL, [2]=1.
  int64_t LSE_strides[3];
};

// Helper: collaborative load from global to shared memory
template <typename T, int STRIDE>
__device__ void load_tile(
    T* __restrict__ dst,
    const T* __restrict__ src_base,
    int64_t row_stride,
    int rows,
    int cols,
    int valid_rows,
    int tid,
    int nthreads) {
  const int total = rows * cols;
  const int per_t = (total + nthreads - 1) / nthreads;
  for (int i = 0; i < per_t; i++) {
    int idx = i * nthreads + tid;
    if (idx < total) {
      int r = idx / cols;
      int c = idx % cols;
      dst[r * STRIDE + c] = (r < valid_rows) ?
          src_base[r * row_stride + c] : static_cast<T>(0.0f);
    }
  }
  // Zero padding columns
  const int pad = STRIDE - cols;
  if (pad > 0) {
    const int total_pad = rows * pad;
    const int per_p = (total_pad + nthreads - 1) / nthreads;
    for (int i = 0; i < per_p; i++) {
      int idx = i * nthreads + tid;
      if (idx < total_pad) {
        int r = idx / pad;
        dst[r * STRIDE + cols + (idx % pad)] = static_cast<T>(0.0f);
      }
    }
  }
}

template <
    typename T,
    bool do_causal,
    int D,
    int BLOCK_M = 64,
    int BLOCK_N = 64>
__global__ void __launch_bounds__(256)
    kernel_sdpa_flash_wmma(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        T* __restrict__ O,
        // Optional logsumexp output (float32), shape [B,H,qL,1]. Null → skip.
        // Written as LSE_i = m_i + log(l_i) for FlashAttention-2 VJP.
        float* __restrict__ LSE,
        const FAWmmaParams params) {
#if ROCM_FA_WMMA
  constexpr int WT = 16; // WMMA tile size
  // Stage Q/K/V one DW-wide head-dim slice at a time so the LDS footprint is
  // capped at the D=128 layout for any head dim (D=256 needs two slices). This
  // keeps the kernel within RDNA's 64 KB LDS budget while still using WMMA.
  constexpr int DW = (D < 128) ? D : 128;
  constexpr int N_DCHUNKS = (D + DW - 1) / DW;
  constexpr int Q_PAD = DW + 4;
  constexpr int KV_PAD = DW + 4;
  constexpr int S_PAD = BLOCK_N + 4;
  constexpr int P_PAD = BLOCK_N + 4;
  constexpr int M_TILES = BLOCK_M / WT;
  constexpr int N_TILES = BLOCK_N / WT;
  constexpr int DW_TILES = DW / WT;
  constexpr int D_TILES = (D + WT - 1) / WT;
  // Wavefront-size-aware threading. A rocWMMA 16x16x16 fragment op is executed
  // collectively by one full hardware wavefront: 64 lanes on CDNA (gfx9xx,
  // MFMA) and 32 lanes on RDNA (gfx10/11/12, incl. gfx1151/1152 WMMA). The
  // block launches one full wavefront per 16-row M-tile (M_TILES waves), so
  // NTHREADS = M_TILES*WAVE (256 on CDNA, 128 on RDNA).
  //
  // CRITICAL: use WARP_SIZE from device/config.h (arch macros), NOT a bare
  // __AMDGCN_WAVEFRONT_SIZE__ fallback to 64. On some hipcc/gfx115x toolchains
  // that macro is missing in device TU and the old `#else 64` path made
  // wave=tid/64 with a 128-thread RDNA launch — only rows 0..31 were produced
  // (L<=32 OK, L>=33 diverge). Matches host launch: (BM/WT)*device.warp_size().
  constexpr int WAVE = WARP_SIZE;
  static_assert(WAVE == 32 || WAVE == 64, "unexpected wavefront size");
  const int NTHREADS = blockDim.x; // = M_TILES * WAVE

  const int bid_b = blockIdx.z;
  const int bid_h = blockIdx.x;
  const int bid_kv_h = bid_h / params.gqa_factor;
  const int q_start = blockIdx.y * BLOCK_M;
  const int tid = threadIdx.x;
  const int wave = tid / WAVE;
  const int lane = tid % WAVE;

  if (q_start >= params.qL) return;

  // ---- Shared memory layout ----
  // Persistent: m[BLOCK_M], l[BLOCK_M]
  // Tile A:     Q_sh [BLOCK_M][Q_PAD]   (bf16, loaded once)
  // Tile B:     KV_sh[BLOCK_N][KV_PAD]  (bf16, K then P_bf16)
  // Tile C:     S_sh [BLOCK_M][S_PAD]   (f32,  scores then V_bf16)
  extern __shared__ char smem[];
  float* m_arr = reinterpret_cast<float*>(smem);
  float* l_arr = m_arr + BLOCK_M;
  T* Q_sh = reinterpret_cast<T*>(l_arr + BLOCK_M);
  T* KV_sh = Q_sh + BLOCK_M * Q_PAD;
  float* S_sh = reinterpret_cast<float*>(KV_sh + BLOCK_N * KV_PAD);

  // Fragment types
  using frag_a     = rocwmma::fragment<rocwmma::matrix_a,   WT, WT, WT, T, rocwmma::row_major>;
  using frag_b_col = rocwmma::fragment<rocwmma::matrix_b,   WT, WT, WT, T, rocwmma::col_major>;
  using frag_b_row = rocwmma::fragment<rocwmma::matrix_b,   WT, WT, WT, T, rocwmma::row_major>;
  using frag_acc   = rocwmma::fragment<rocwmma::accumulator, WT, WT, WT, float>;

  // Output accumulators: each wave owns D_TILES [16x16] f32 tiles
  frag_acc o_acc[D_TILES];
  for (int d = 0; d < D_TILES; d++)
    rocwmma::fill_fragment(o_acc[d], 0.0f);

  // Init online softmax state
  if (tid < BLOCK_M) {
    m_arr[tid] = -1e30f;
    l_arr[tid] = 0.0f;
  }
  __syncthreads();

  const T* Q_base = Q + bid_b * params.Q_strides[0]
                      + bid_h * params.Q_strides[1]
                      + q_start * params.Q_strides[2];
  const int q_valid = min(BLOCK_M, params.qL - q_start);

  // Q is staged per D-chunk inside the S loop below: all chunks share
  // one DW-wide LDS region, so a hoisted multi-chunk stage would leave only
  // the last chunk resident (D=256: chunk 1 overwrites chunk 0 before the
  // S MMA reads it). The finalize section reuses Q_sh as its temp AFTER
  // the loop.

  // acc-fragment layout on gfx1201 (rocWMMA 16x16x16, f32 accumulator),
  // verified by probe (wmma_probe.hip, 2026-11-16): element e of lane L holds
  // row = e + 8*(L>>4), col = L&15. Each row's 16 cols live in 16 consecutive
  // lanes of one 16-lane half; a lane's 8 elements span 8 distinct rows.
  constexpr float LOG2E = 1.4426950408889634f;
#if defined(__gfx1201__)
  const int acc_half = lane >> 4; // 0 or 1
  const int acc_col0 = lane & 15;
#endif

  // ---- K/V block loop ----
  for (int k_start = 0; k_start < params.kL; k_start += BLOCK_N) {

    // Causal skip: if entire K block is after all queries' causal limits
    if constexpr (do_causal) {
      int last_q = q_start + min(BLOCK_M, params.qL - q_start) - 1;
      int max_k_allowed = (params.kL - params.qL) + last_q;
      if (k_start > max_k_allowed) break;
    }

    int k_valid = min(BLOCK_N, params.kL - k_start);

    const T* K_base = K + bid_b * params.K_strides[0]
                        + bid_kv_h * params.K_strides[1]
                        + k_start * params.K_strides[2];

    // ---- S = Q @ K^T via WMMA, accumulated over the head dim in DW slices ----
    // K is staged one DW-wide chunk at a time into KV_sh and consumed
    // immediately, so the LDS footprint stays bounded for D>128. s_acc carries
    // the partial dot products across chunks.
    frag_acc s_acc[N_TILES];
    for (int n = 0; n < N_TILES; n++)
      rocwmma::fill_fragment(s_acc[n], 0.0f);

      for (int dc = 0; dc < N_DCHUNKS; dc++) {
        const int d0 = dc * DW;
        const int dwid = min(DW, D - d0);
        load_tile<T, Q_PAD>(
            Q_sh, Q_base + d0, params.Q_strides[2], BLOCK_M, dwid, q_valid, tid,
            NTHREADS);
        load_tile<T, KV_PAD>(
            KV_sh, K_base + d0, params.K_strides[2], BLOCK_N, dwid, k_valid,
            tid, NTHREADS);
        __syncthreads();

        for (int d = 0; d < DW_TILES; d++) {
          frag_a q_frag;
          rocwmma::load_matrix_sync(
              q_frag, Q_sh + wave * WT * Q_PAD + d * WT, Q_PAD);

          for (int n = 0; n < N_TILES; n++) {
            frag_b_col k_frag;
            // col_major load of K[n*16][d*16] gives K^T
            rocwmma::load_matrix_sync(
                k_frag, KV_sh + n * WT * KV_PAD + d * WT, KV_PAD);
            rocwmma::mma_sync(s_acc[n], q_frag, k_frag, s_acc[n]);
          }
        }
        __syncthreads();
      }

#if defined(__gfx1201__)
      // ---- gfx1201: register-resident online softmax ----
      // No S f32 LDS round-trip: scores stay in s_acc registers, the tile max
      // and per-row sums reduce via shfl, the O rescale multiplies o_acc
      // registers in place, and P is written bf16 straight into KV_sh.
      // The tile max over the whole 16x64 block is a valid per-row upper
      // bound for the running max (FA2 only needs SOME upper bound; the
      // normalization cancels it exactly).
      for (int n = 0; n < N_TILES; n++)
        for (int e = 0; e < 8; e++)
          s_acc[n].x[e] *= params.scale;

      // Tile max over valid keys only (masked scores would widen the bound).
      float tmax = -1e30f;
      #pragma unroll
      for (int n = 0; n < N_TILES; n++) {
        int c0 = n * WT;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r_local = e + 8 * acc_half; // wave-local query row
          int q_idx = q_start + wave * WT + r_local;
          bool use = q_idx < params.qL && (c0 + acc_col0) < k_valid;
          if constexpr (do_causal)
            use = use &&
                (k_start + c0 + acc_col0 <= params.kL - params.qL + q_idx);
          if (use)
            tmax = fmaxf(tmax, s_acc[n].x[e]);
        }
      }
      tmax = fmaxf(tmax, __shfl_xor(tmax, 1));
      tmax = fmaxf(tmax, __shfl_xor(tmax, 2));
      tmax = fmaxf(tmax, __shfl_xor(tmax, 4));
      tmax = fmaxf(tmax, __shfl_xor(tmax, 8));
      tmax = fmaxf(tmax, __shfl_xor(tmax, 16));

      // Per-row running max / normalization state (rows of this half).
      float old_m[8], new_m[8], scale_row[8];
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        int row = wave * WT + e + 8 * acc_half;
        old_m[e] = m_arr[row];
        new_m[e] = fmaxf(old_m[e], tmax);
        scale_row[e] = (old_m[e] > -1e29f) ?
                           exp2f((old_m[e] - new_m[e]) * LOG2E) :
                           0.0f;
      }

      // Fused P store + row-sum: p is computed once per element, written
      // bf16 straight into KV_sh (K is fully consumed; the shared barrier
      // before P@V covers the scalar->WMMA-load visibility), and the f32
      // value feeds the per-row sum. Row sums reduce over the 16 lanes of
      // the half: a lane's element e of every tile n is the same row
      // (e + 8*acc_half), and the 64 cols partition exactly over them.
      float psum[8];
      #pragma unroll
      for (int e = 0; e < 8; e++)
        psum[e] = 0.0f;
      #pragma unroll
      for (int n = 0; n < N_TILES; n++) {
        int c0 = n * WT;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r_local = e + 8 * acc_half;
          int q_idx = q_start + wave * WT + r_local;
          bool use = q_idx < params.qL && (c0 + acc_col0) < k_valid;
          if constexpr (do_causal)
            use = use &&
                (k_start + c0 + acc_col0 <= params.kL - params.qL + q_idx);
          float p = use ?
              exp2f((s_acc[n].x[e] - new_m[e]) * LOG2E) :
              0.0f;
          KV_sh[(wave * WT + r_local) * P_PAD + c0 + acc_col0] =
              static_cast<T>(p);
          psum[e] += p;
        }
      }
      float rowsum[8];
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        float u = psum[e];
        u += __shfl_xor(u, 1);
        u += __shfl_xor(u, 2);
        u += __shfl_xor(u, 4);
        u += __shfl_xor(u, 8);
        rowsum[e] = u;
      }

      // One lane per half publishes the row state (rows are disjoint).
      if (acc_col0 == 0) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int row = wave * WT + e + 8 * acc_half;
          m_arr[row] = new_m[e];
          l_arr[row] = l_arr[row] * scale_row[e] + rowsum[e];
        }
      }

      // Rescale O accumulators IN REGISTERS: element e of every D tile is the
      // same row (e + 8*acc_half) as the acc-fragment layout dictates, so one
      // 8-wide scale vector covers all D_TILES. No LDS, no barriers.
      #pragma unroll
      for (int d = 0; d < D_TILES; d++)
        #pragma unroll
        for (int e = 0; e < 8; e++)
          o_acc[d].x[e] *= scale_row[e];
#else
      // ---- Legacy (non-gfx1201): S f32 via LDS, two-pass row softmax, ----
      // ---- LDS O-rescale, separate P convert. Kept for other targets.  ----

      // Scale and store S to shared memory
      for (int n = 0; n < N_TILES; n++) {
        for (int e = 0; e < s_acc[n].num_elements; e++)
          s_acc[n].x[e] *= params.scale;
        rocwmma::store_matrix_sync(
            S_sh + wave * WT * S_PAD + n * WT, s_acc[n], S_PAD,
            rocwmma::mem_row_major);
      }
    // rocWMMA store_matrix_sync to LDS does NOT guarantee visibility to other
    // waves (or even to scalar loads) with a plain __syncthreads(). Flush the
    // matrix-unit LDS traffic before the online-softmax scalar readers run.
    //
    // Row ownership MUST match the WMMA writer: wave w owns rows [w*16, w*16+16).
    // The old "if (tid < BLOCK_M)" softmax had wave0 lanes 16..31 read rows
    // 16..31 written by wave1 — a cross-wave LDS read of WMMA stores. That is
    // why L<=16 (single wave of real query rows) passed while L>=32 diverged.
    rocwmma::synchronize_workgroup();

    // ---- Online softmax: each wave processes its own 16 query rows ----
    float* wave_scale = reinterpret_cast<float*>(
        reinterpret_cast<char*>(S_sh) + BLOCK_M * S_PAD * sizeof(float));
    constexpr int VALS_PER_LANE = BLOCK_N / 32; // 2 for BLOCK_N=64
    #pragma unroll 1
    for (int r_local = 0; r_local < WT; ++r_local) {
      int row = wave * WT + r_local;
      int q_idx = q_start + row;
      bool valid = q_idx < params.qL;
      float local_max = -1e30f;
      #pragma unroll
      for (int k = 0; k < VALS_PER_LANE; ++k) {
        int j = lane + k * 32;
        bool use = valid && (k_start + j) < params.kL;
        if constexpr (do_causal)
          use = use && ((k_start + j) <= (params.kL - params.qL + q_idx));
        if (use)
          local_max = fmaxf(local_max, S_sh[row * S_PAD + j]);
      }
      local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
      local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
      local_max = fmaxf(local_max, __shfl_xor(local_max, 4));
      local_max = fmaxf(local_max, __shfl_xor(local_max, 8));
      local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
      float new_m = local_max;

      float old_m = m_arr[row];
      float old_l = l_arr[row];
      float scale_old =
          (old_m > -1e29f) ? exp2f((old_m - new_m) * LOG2E) : 0.0f;

      float local_sum = 0.0f;
      #pragma unroll
      for (int k = 0; k < VALS_PER_LANE; ++k) {
        int j = lane + k * 32;
        bool use = valid && (k_start + j) < params.kL;
        if constexpr (do_causal)
          use = use && ((k_start + j) <= (params.kL - params.qL + q_idx));
        if (use) {
          float p = exp2f((S_sh[row * S_PAD + j] - new_m) * LOG2E);
          S_sh[row * S_PAD + j] = p;
          local_sum += p;
        } else {
          S_sh[row * S_PAD + j] = 0.0f;
        }
      }
      local_sum += __shfl_xor(local_sum, 1);
      local_sum += __shfl_xor(local_sum, 2);
      local_sum += __shfl_xor(local_sum, 4);
      local_sum += __shfl_xor(local_sum, 8);
      local_sum += __shfl_xor(local_sum, 16);

      if (lane == 0) {
        m_arr[row] = new_m;
        l_arr[row] = old_l * scale_old + local_sum;
        wave_scale[row] = scale_old;
      }
    }
    __syncthreads();

    // ---- Rescale O accumulators by scale_old via shared memory ----
    {
      float* o_tmp = reinterpret_cast<float*>(KV_sh);
      constexpr int OT_PAD = WT + 4;

      for (int d = 0; d < D_TILES; d++) {
        rocwmma::store_matrix_sync(
            o_tmp + wave * WT * OT_PAD, o_acc[d], OT_PAD,
            rocwmma::mem_row_major);
        rocwmma::synchronize_workgroup();

        if (lane < WT) {
          int row = wave * WT + lane;
          float s = wave_scale[row];
          for (int c = 0; c < WT; c++)
            o_tmp[wave * WT * OT_PAD + lane * OT_PAD + c] *= s;
        }
        __syncthreads();

        rocwmma::load_matrix_sync(
            o_acc[d], o_tmp + wave * WT * OT_PAD, OT_PAD,
            rocwmma::mem_row_major);
        __syncthreads();
      }
    }

    // ---- Convert P (f32 in S_sh) to bf16 in KV_sh for WMMA P@V ----
    {
      int total = BLOCK_M * BLOCK_N;
      int per_t = (total + NTHREADS - 1) / NTHREADS;
      for (int i = 0; i < per_t; i++) {
        int idx = i * NTHREADS + tid;
        if (idx < total) {
          int r = idx / BLOCK_N;
          int c = idx % BLOCK_N;
          KV_sh[r * P_PAD + c] = static_cast<T>(S_sh[r * S_PAD + c]);
        }
      }
      int pad_total = BLOCK_M * 4;
      int pad_per_t = (pad_total + NTHREADS - 1) / NTHREADS;
      for (int i = 0; i < pad_per_t; i++) {
        int idx = i * NTHREADS + tid;
        if (idx < pad_total)
          KV_sh[(idx / 4) * P_PAD + BLOCK_N + (idx % 4)] = static_cast<T>(0.0f);
      }
    }
#endif
    __syncthreads();

    // ---- O += P @ V via WMMA, output head dim tiled in DW slices ----
    // P (bf16) stays resident in KV_sh; V is streamed one DW-wide slice into
    // S_sh (free on gfx1201; the f32 scores on legacy) and consumed
    // immediately, so a full D=256 V tile never has to coexist with P in LDS.
    {
      T* P_sh = KV_sh;
      T* V_sh = reinterpret_cast<T*>(S_sh);
      const T* V_base = V + bid_b * params.V_strides[0]
                          + bid_kv_h * params.V_strides[1]
                          + k_start * params.V_strides[2];

      for (int dc = 0; dc < N_DCHUNKS; dc++) {
        const int d0 = dc * DW;
        const int dwid = min(DW, D - d0);
        load_tile<T, KV_PAD>(
            V_sh, V_base + d0, params.V_strides[2], BLOCK_N, dwid, k_valid, tid,
            NTHREADS);
        __syncthreads();

        // P fragments are D-independent: load once per dc, not per dl.
        frag_a p_frags[N_TILES];
        for (int n = 0; n < N_TILES; n++)
          rocwmma::load_matrix_sync(
              p_frags[n], P_sh + wave * WT * P_PAD + n * WT, P_PAD);

        for (int dl = 0; dl < DW_TILES; dl++) {
          const int dglobal = dc * DW_TILES + dl;
          if (dglobal >= D_TILES)
            break;
          for (int n = 0; n < N_TILES; n++) {
            frag_b_row v_frag;
            rocwmma::load_matrix_sync(
                v_frag, V_sh + n * WT * KV_PAD + dl * WT, KV_PAD);
            rocwmma::mma_sync(o_acc[dglobal], p_frags[n], v_frag,
                              o_acc[dglobal]);
          }
        }
        __syncthreads();
      }
    }
  } // end K/V loop

  // ---- Finalize: normalize O and write to global ----
  {
    float* o_tmp = reinterpret_cast<float*>(Q_sh); // Reuse Q_sh
    constexpr int OT_PAD = WT + 4;

    for (int d = 0; d < D_TILES; d++) {
      rocwmma::store_matrix_sync(
          o_tmp + wave * WT * OT_PAD, o_acc[d], OT_PAD,
          rocwmma::mem_row_major);
      rocwmma::synchronize_workgroup();

      if (lane < WT) {
        int row = wave * WT + lane;
        int q_idx = q_start + row;
        if (q_idx < params.qL) {
          float inv_l = (l_arr[row] > 0.0f) ? (1.0f / l_arr[row]) : 0.0f;
          T* dst = O + bid_b * params.O_strides[0]
                     + bid_h * params.O_strides[1]
                     + q_idx * params.O_strides[2]
                     + d * WT;
          float* src = o_tmp + wave * WT * OT_PAD + lane * OT_PAD;
          for (int c = 0; c < WT && (d * WT + c) < D; c++)
            dst[c] = static_cast<T>(src[c] * inv_l);
        }
      }
      __syncthreads();
    }

    // FlashAttention-2 stats: LSE = m + log(l). Required for fused VJP; written
    // once after O is normalized so m/l match the final online softmax state.
    if (LSE != nullptr && lane < WT) {
      int row = wave * WT + lane;
      int q_idx = q_start + row;
      if (q_idx < params.qL) {
        float m = m_arr[row];
        float l = l_arr[row];
        float lse = (l > 0.0f) ? (m + logf(l)) : -1e30f;
        LSE[bid_b * params.LSE_strides[0] + bid_h * params.LSE_strides[1] +
            q_idx * params.LSE_strides[2]] = lse;
      }
    }
  }
#endif // ROCM_FA_WMMA
}

} // namespace rocm

// ---- Host interface ----

bool supports_sdpa_flash_wmma(
    const array& q, const array& k, const array& v,
    bool has_arr_mask, bool output_logsumexp) {
  // Host-side check: always enabled when compiled for WMMA-capable targets.
  // The kernel itself guards with #if ROCM_FA_WMMA for device code.
  // output_logsumexp is supported (LSE write in finalize); required for fused
  // VJP but harmless for inference when stats are not requested.
  if (has_arr_mask) return false;
  if (q.dtype() != bfloat16 && q.dtype() != float16) return false;
  int D = q.shape(-1);
  if (D != v.shape(-1) || D != k.shape(-1)) return false;
  if (q.shape(1) % k.shape(1) != 0) return false; // GQA factor integer
  (void)output_logsumexp;
  return (D == 64 || D == 128 || D == 256);
}

// Shared-memory (LDS) bytes the WMMA flash kernel needs for a given head dim.
// The kernel stages Q/K/V one DW=min(D,128)-wide head-dim slice at a time, so
// the footprint is capped at the D=128 layout regardless of D. Callers compare
// this against the running device's LDS budget before selecting the kernel.
int sdpa_flash_wmma_smem(int D) {
  constexpr int BM = 64, BN = 64;
  const int DW = (D < 128) ? D : 128;
  // Layout: m[BM] l[BM] | Q[BM,DW+4] | KV[BN,DW+4] | S[BM,BN+4] | wave_scale[BM]
  // wave_scale is dynamic (not static __shared__) so it is accounted here.
  return 2 * BM * (int)sizeof(float)
       + BM * (DW + 4) * (int)sizeof(uint16_t)
       + BN * (DW + 4) * (int)sizeof(uint16_t)
       + BM * (BN + 4) * (int)sizeof(float)
       + BM * (int)sizeof(float);
}

void sdpa_flash_wmma(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    Stream s,
    array* lse /* nullable float32 [B,H,qL,1] */) {
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  int B = q.shape(0), H = q.shape(1), qL = q.shape(2), kL = k.shape(2);
  int D = q.shape(3);

  o.set_data(rocm::malloc_async(o.nbytes(), enc));
  if (lse != nullptr) {
    lse->set_data(rocm::malloc_async(lse->nbytes(), enc));
  }

  rocm::FAWmmaParams p{};
  p.B = B; p.H = H; p.D = D; p.qL = qL; p.kL = kL;
  p.gqa_factor = H / k.shape(1);
  p.scale = scale;
  for (int i = 0; i < 3; i++) {
    p.Q_strides[i] = q.strides(i);
    p.K_strides[i] = k.strides(i);
    p.V_strides[i] = v.strides(i);
    p.O_strides[i] = o.strides(i);
  }
  // Contiguous [B,H,qL,1] layout for LSE.
  p.LSE_strides[0] = static_cast<int64_t>(H) * qL;
  p.LSE_strides[1] = qL;
  p.LSE_strides[2] = 1;

  constexpr int BM = 64, BN = 64;
  constexpr int WT = 16;
  dim3 grid(H, (qL + BM - 1) / BM, B);
  // One full hardware wavefront per 16-row M-tile: 64 lanes on CDNA (MFMA),
  // 32 on RDNA (WMMA). rocWMMA maps the 16x16x16 fragment op onto the whole
  // wavefront, so the launched block width must track the device warp size.
  dim3 block((BM / WT) * d.warp_size());

  // Shared memory: m/l + Q + KV + S, with Q/KV sized to one DW-wide slice.
  int smem = sdpa_flash_wmma_smem(D);

  enc.set_input_array(q);
  enc.set_input_array(k);
  enc.set_input_array(v);
  enc.set_output_array(o);
  if (lse != nullptr) {
    enc.set_output_array(*lse);
  }

  float* lse_ptr = lse ? gpu_ptr<float>(*lse) : nullptr;

  auto launch = [&](auto type_tag, auto causal_tag, auto dim_tag) {
    using DT = decltype(type_tag);
    constexpr bool C = decltype(causal_tag)::value;
    constexpr int DD = decltype(dim_tag)::value;
    enc.add_kernel_node_ex(
        &rocm::kernel_sdpa_flash_wmma<DT, C, DD, BM, BN>,
        grid, block, static_cast<uint32_t>(smem),
        gpu_ptr<const DT>(q), gpu_ptr<const DT>(k),
        gpu_ptr<const DT>(v), gpu_ptr<DT>(o), lse_ptr, p);
  };

  auto dispatch_dim = [&](auto tt, auto ct) {
    if (D == 64)  launch(tt, ct, std::integral_constant<int, 64>());
    else if (D == 128) launch(tt, ct, std::integral_constant<int, 128>());
    else if (D == 256) launch(tt, ct, std::integral_constant<int, 256>());
  };

  if (o.dtype() == bfloat16) {
    if (do_causal) dispatch_dim(hip_bfloat16(), std::true_type());
    else dispatch_dim(hip_bfloat16(), std::false_type());
  } else {
    if (do_causal) dispatch_dim(__half(), std::true_type());
    else dispatch_dim(__half(), std::false_type());
  }
}

} // namespace mlx::core
