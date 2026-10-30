// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip -- the gated-delta-net
// recurrent state update, which is what decode runs where prefill runs the
// chunked scan. Ported from R4D (libr4d e8de4bc / <opt-fork>/vllm-<opt> 1d2e248) for
// the MLX GatedDeltaUpdate ROCm kernel; see reports/k3-gdn-port.md.
//
// The state-resident structure is the R4D one: one thread owns one v row and
// half its k range, so
//
//   * S k and S q are in-lane dot products closed by a single lane<->lane^1
//     exchange, not a tree reduction;
//   * the outer-product update is 64 independent FMAs with no cross-lane
//     traffic at all;
//   * a thread's 64 state elements are contiguous in k, so the load and every
//     store are four 16-byte accesses over one 256-byte run.
//
// MLX contract differences from the R4D kernel (all of it is the vLLM glue):
//   * q / k arrive l2-normalised upstream (the model normalises them);
//   * g is the per-token multiplicative decay e^g in [0, 1] (bf16); the log is
//     taken in-kernel and clamped at -80, the same ceiling the chunk scan uses
//     (see its header on why 80, not 88);
//   * beta is the final per-token delta rate (bf16);
//   * the state is a contiguous fp32 [B, Hv, 128, 128] tensor, not a paged
//     cache, and the output is a contiguous [B*T, Hv, 128] tensor;
//   * no output scale (the MLX primitive does not apply one).
#include <hip/hip_runtime.h>
#include "r4d_gdn_wmma.h"

#define RU_K   128              // head_k
#define RU_V   128              // head_v
#define RU_HK  64              // k elements per thread (RU_K / 2)
#define RU_THR (RU_V * 2)      // two threads per v row
// VS: how many workgroups share a head's 128 v rows. The grid is (batch,
// heads); at one batch with 32 heads that is 32 workgroups on 64 CUs -- 37% of
// the machine, and the kernel is not DRAM bound there (its whole working set
// fits the 64 MB last level). Nothing in a v row's update crosses to another v
// row: the two dot products contract over k, and the outer-product update is
// per (v, k). Only a fused row norm would need the whole row, and the MLX
// model applies its RMS norm separately, so the split is always safe.
#ifndef VSMAX
#define VSMAX 4
#endif

template <int VS>
__global__ __launch_bounds__(RU_V / VS * 2)
void r4d_gdn_recurrent_update_kernel(const unsigned short* __restrict__ q,
                                     const unsigned short* __restrict__ k,
                                     const unsigned short* __restrict__ v,
                                     const unsigned short* __restrict__ g,
                                     const unsigned short* __restrict__ beta,
                                     const float* __restrict__ h0,
                                     unsigned short* __restrict__ o,
                                     float* __restrict__ ht,
                                     int B, int T, int H, int Hg)
{
  constexpr int NR = RU_V / VS;         // v rows this workgroup owns
  constexpr int THR = NR * 2;
  const int b = blockIdx.x, hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int row = (tid >> 1) + (VS > 1 ? blockIdx.z * NR : 0), half = tid & 1, k0 = half * RU_HK;
  const int hk = hv / (H / Hg);

  float h[RU_HK];
  {
    const float* p = h0 + ((size_t)b * H + hv) * RU_V * RU_K + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 4) *(float4*)&h[j] = *(const float4*)(p + j);
  }

  for (int t = 0; t < T; ++t) {
    const int tok = b * T + t;
    const unsigned short* qp = q + ((size_t)tok * Hg + hk) * RU_K + k0;
    const unsigned short* kp = k + ((size_t)tok * Hg + hk) * RU_K + k0;

    float qq[RU_HK], kk[RU_HK];
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) {
      const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
      for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq[e]); kk[j + e] = bf2f(sk[e]); }
    }

    // Per-token multiplicative decay; clamp the log at -80 (the chunk scan's
    // ceiling; see its header).
    const float eg = __expf(fmaxf(__logf(bf2f(g[(size_t)tok * H + hv])), -80.0f));
    const float bt = bf2f(beta[(size_t)tok * H + hv]);

    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    dot += __shfl_xor(dot, 1, 32);

    const float u = (bf2f(v[((size_t)tok * H + hv) * RU_V + row]) - dot) * bt;

    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
    on += __shfl_xor(on, 1, 32);

    if (half == 0) o[((size_t)tok * H + hv) * RU_V + row] = f2bf(on);
  }

  {
    float* p = ht + ((size_t)b * H + hv) * RU_V * RU_K + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 4) *(float4*)(p + j) = *(const float4*)&h[j];
  }
}

extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const float* h0, void* o, float* ht,
    int B, int T, int H, int Hg, int K, int V, void* stream)
{
  if (K != RU_K || V != RU_V) return -1;
  if (H % Hg) return -2;
  // Fill the machine: at one batch the (B, H) grid is 32 workgroups. Split a
  // head's v rows until there are at least 64 workgroups, but never past four
  // (that is 64 threads, two waves).
  int vs = 1;
  {
    const int want = (64 + B * H - 1) / (B * H);
    vs = want >= 4 ? 4 : (want >= 2 ? 2 : 1);
    if (vs > VSMAX) vs = VSMAX;
  }
  dim3 grid(B, H, vs), blk(RU_V / vs * 2);
#define RU_LAUNCH(VSC)                                                                     \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<VSC>), grid, blk, 0,                \
                     (hipStream_t)stream,                                                 \
                     (const unsigned short*)q, (const unsigned short*)k,                  \
                     (const unsigned short*)v, (const unsigned short*)g,                  \
                     (const unsigned short*)beta, (const float*)h0,                       \
                     (unsigned short*)o, (float*)ht, B, T, H, Hg)
  if (vs == 4) RU_LAUNCH(4);
  else if (vs == 2) RU_LAUNCH(2);
  else RU_LAUNCH(1);
#undef RU_LAUNCH
  return (int)hipGetLastError();
}
