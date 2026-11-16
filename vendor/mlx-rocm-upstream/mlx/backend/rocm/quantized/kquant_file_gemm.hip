// File-layout k-quant GEMM / gather / dequant — the dequant-at-GEMM family.
//
// Consumes the raw 256-element super-block stream in the FILE's grouping
// (w shape (K, N/256, block_bytes); super-block = 256 consecutive output
// values at fixed input k) and dequantizes INSIDE the op — no f32/bf16
// weight materialization, no f32 weight cache.
//
// Dequant math: the single-sourced, G1-verified per-class k-quant math,
// UNCHANGED (per-element form of the WS3-verified ggml 6023692 block
// kernels; Q4_K is kquant_q4k_math.hpp's dequant_elem directly). The
// file-layout part is the access/iteration pattern only:
//  - super-block fetch = (k row, n super-block column) of the file, with
//    the 32-unit 4096-element window even/odd map applied to the block
//    grid position (mode/axis/group/keep_ends/pairs — data from the
//    pinned per-tensor tables; the exact same window function the
//    G1-verified host gguf_apply_layout uses);
//  - GEMM: one workgroup = 64 m rows x one 256-element super-block (256
//    n columns, n = c*256 + tid so the dequant index is e = tid); the
//    K-loop dequantizes each (k, n) element on the fly; volatile
//    rounding barriers keep the multiply-chain from contracting, as in
//    the G1-verified math. f32 accumulation in k-order 0..K-1.
//  - gather: one thread per output element (b, k) -> block (k, b/256),
//    element b%256 (the embedding-lookup sibling; a lookup is not a
//    GEMM, so its own op — no weight materialization).
//  - dequant: one thread per output element (u, c*256+e) -> block (u, c),
//    element e (the G1 bit-exact verification path).
//
// Block bytes per 256 elements (format constants): Q8_0 272 (8 x 34 B
// 32-element blocks), Q4_K 144, Q5_K 176, Q6_K 210. ktype codes per
// gguflib.h / python-gguf 0.19: Q8_0 = 8, Q4_K = 12, Q5_K = 13,
// Q6_K = 14.
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "mlx/backend/rocm/quantized/kquant_q4k_math.hpp"
#include "mlx/backend/rocm/quantized/kquants.h"
#include "mlx/kquant_file_gemm.h"
#include "mlx/backend/rocm/quantized/kquant_file_gemm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstring>

namespace mlx::core {

namespace {

// Block bytes per 256 elements (format-level constants, per the pinned
// GGUF table's block_bytes column).
inline int block_bytes_for(int ktype) {
  switch (ktype) {
    case 8:
      return 272;  // Q8_0: 8 x 34 B 32-element blocks
    case 12:
      return 144;  // Q4_K
    case 13:
      return 176;  // Q5_K
    case 14:
      return 210;  // Q6_K
    default:
      return -1;
  }
}

// Canonical inverse of the file's 32-unit 4096-element window map — the
// EXACT same function the G1-verified host gguf_apply_layout uses
// (gguf_loader.cpp apply_window): engine window position q (0..group-1)
// -> file window position. Ends stay; the middle M = group -
// 2*keep_ends is stored in the file as [even engine positions
// 2,4,..,M][odd 1,3,..,M-1]; engine middle position p (1..M) <- file
// middle index f = (p even) ? (p/2 - 1) : (pairs + (p-1)/2).
__device__ __forceinline__ int window_map(
    int pos,
    int group,
    int keep_ends,
    int pairs) {
  int w = pos / group;
  int q = pos % group;
  if (q >= keep_ends && q < group - keep_ends) {
    int p = q - keep_ends + 1;  // engine middle position 1..M
    int f = (p % 2 == 0) ? (p / 2 - 1) : (pairs + (p - 1) / 2);
    return w * group + keep_ends + f;
  }
  return pos;
}

// Exact f16 -> f32 (identical to the WS3-verified f16to32; bit-exact
// IEEE widening, no rounding possible).
__device__ __forceinline__ float f16w(const uint8_t* p) {
  uint32_t h;
  std::memcpy(&h, p, 2);
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0x1Fu) {
    bits = (sign << 31) | 0x7F800000u | (mant << 13);
  } else if (exp == 0u) {
    if (mant == 0) {
      bits = sign << 31;
    } else {
      uint32_t p = 0;
      uint32_t m = mant;
      while (m >> 1) {
        m >>= 1;
        p += 1;
      }
      bits = (sign << 31) | ((p + 103u) << 23) |
             ((mant & ((1u << p) - 1u)) << (23u - p));
    }
  } else {
    bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
  }
  float r;
  std::memcpy(&r, &bits, 4);
  return r;
}

// Pinned-oracle (numpy / IEEE 754-2008-style CPU) NaN propagation for
// a - b (identical to the WS3-verified sub_exact; the GPU fsub applies
// the IEEE 754-2019 XOR sign rule to Inf results — the guard restores
// bit-exactness vs the oracle).
__device__ __forceinline__ float sub_exact(float a, float b) {
  uint32_t av, bv;
  std::memcpy(&av, &a, 4);
  std::memcpy(&bv, &b, 4);
  const uint32_t a_ex = av >> 23 & 0xFFu, b_ex = bv >> 23 & 0xFFu;
  const uint32_t a_s = av >> 31, b_s = bv >> 31;
  const bool a_nan = (a_ex == 0xFFu) && (av & 0x7FFFFFu) != 0u;
  const bool a_inf = (a_ex == 0xFFu) && (av & 0x7FFFFFu) == 0u;
  const bool b_nan = (b_ex == 0xFFu) && (bv & 0x7FFFFFu) != 0u;
  const bool b_inf = (b_ex == 0xFFu) && (bv & 0x7FFFFFu) == 0u;
  if (a_nan) return a;
  if (b_nan) return b;
  if (a_inf || b_inf) {
    if (a_inf && b_inf) {
      if (a_s == b_s) {
        const uint32_t rbits = 0xFFC00000u;
        float r;
        std::memcpy(&r, &rbits, 4);
        return r;
      }
      return a;
    }
    if (b_inf) {
      const uint32_t rbits = b_s ? 0x7F800000u : 0xFF800000u;
      float r;
      std::memcpy(&r, &rbits, 4);
      return r;
    }
    return a;
  }
  return a - b;
}

// Per-element dequant of element e (0..255) of one 256-element super-
// block (block_bytes bytes at `blk`), templated on ktype.
//
// Bit-exact contract (G1): the value a WS3-verified block kernel writes
// at output index e of the same bytes, computed with the same f32 op
// order. Per-element form of the verified block kernels:
//  Q4_K: kquant_q4k_math.hpp dequant_elem directly (the single-sourced
//        G1-verified math — no re-derivation).
//  Q5_K: kq_dequant_q5k loop, element e with g = e/64, l = e%32,
//        h = (e%64)/32 (lo/hi nibble); scale pair index is = 2g + h
//        (the kernel's is counter, +2 per 64-chunk); qh bit g (lo) /
//        g+1 (hi).
//  Q6_K: kq_dequant_q6k loop, e = n + 32s + l with n = e & ~127,
//        l = e%32, s = (e/32)%4; ql byte l + 32*(s>=1) of its 64-byte
//        half, qh byte l of its 32-byte half, scale byte
//        (n/128)*8 + l/16 + {0,2,4,6}[s].
//  Q8_0: 8 consecutive 32-element sub-blocks of 34 B (d f16 @0, 32 x
//        int8 @2) per 256-element block; element e: sub = e/32,
//        i = e%32, v = int8 * d (the WS3 kernel's exact mul order).
// The volatile barriers store the rounding intermediates so the
// (a*b)-c chains can never contract to FMA/HFMA; the rounding sequence
// (mul, mul, sub — each rounded once) is preserved.
template <int ktype>
__device__ __forceinline__ float deq_elem(const uint8_t* __restrict__ blk,
                               int e) {
  if (ktype == 12) {
    return rocm::kquant::dequant_elem(blk, e);
  }
  if (ktype == 13) {
    // G1-verified mapping (bit-exact vs python-gguf 0.19.0): for the
    // 64-chunk g, the low-32 element i uses ql byte i's low nibble +
    // bit (2g) of qh byte i; the high-32 element i+32 uses the high
    // nibble + bit (2g+1). Scales: sc[2g]/sc[2g+1] from the k4 table.
    const float d = f16w(blk + 0);
    const float minv = f16w(blk + 2);
    const uint8_t* sc = blk + 4;   // 12 scale/min bytes (k4 packing)
    const uint8_t* qh = blk + 16;  // 32 B, 1 bit / elem
    const uint8_t* ql = blk + 48;  // 128 B, 4 bit / elem
    const int g = e >> 6;
    const int i = e & 63;
    const int sub = i >> 5;  // 0 = low 32, 1 = high 32
    const int k = i & 31;
    const int f = (qh[k] >> (2 * g + sub)) & 1;
    const int code = sub
        ? ((ql[32 * g + k] >> 4) + (f << 4))
        : ((ql[32 * g + k] & 0xF) + (f << 4));
    const int si = 2 * g + sub;  // 0..7 (k4 table slot)
    int s1, m1;
    if (si < 4) {
      s1 = sc[si] & 63;
      m1 = sc[si + 4] & 63;
    } else {
      s1 = (sc[si + 4] & 0xF) | ((sc[si - 4] >> 6) << 4);
      m1 = (sc[si + 4] >> 4) | ((sc[si] >> 6) << 4);
    }
    volatile float d1 = d * (float)s1;  // round 1
    volatile float m1v = minv * (float)m1;  // round 1
    volatile float p1 = d1 * (float)code;  // round 2
    return sub_exact(p1, m1v);  // round 3 (oracle NaN propagation)
  }
  if (ktype == 14) {
    const float d = f16w(blk + 208);
    const uint8_t* qlb = blk;          // 128 B, 4 bit / elem
    const uint8_t* qhb = blk + 128;  // 64 B, 2 bit / elem
    const int8_t* scb = reinterpret_cast<const int8_t*>(blk + 192);
    const int n = e & ~127;
    const int l = e % 32;
    const int s = (e / 32) % 4;
    const int ql_off = (n / 128) * 64 + ((s == 1 || s == 3) ? l + 32 : l);
    const int qh_off = (n / 128) * 32 + l;
    const int sc_off = (n / 128) * 8 + l / 16;
    const int sc_sel = s == 0 ? 0 : (s == 1 ? 2 : (s == 2 ? 4 : 6));
    int q;
    if (s == 0) {
      q = ((qlb[ql_off] & 0xF) | ((qhb[qh_off] & 3) << 4)) - 32;
    } else if (s == 1) {
      q = ((qlb[ql_off] & 0xF) | (((qhb[qh_off] >> 2) & 3) << 4)) - 32;
    } else if (s == 2) {
      q = ((qlb[ql_off] >> 4) | (((qhb[qh_off] >> 4) & 3) << 4)) - 32;
    } else {
      q = ((qlb[ql_off] >> 4) | (((qhb[qh_off] >> 6) & 3) << 4)) - 32;
    }
    // C: y = d * sc * q (mul, mul — the WS3 kernel's exact order)
    volatile float t1 = d * (float)scb[sc_off + sc_sel];  // round 1
    volatile float p1 = t1 * (float)q;  // round 2
    return p1;
  }
  // ktype 8: Q8_0 — 8 x 34 B 32-element sub-blocks per 256-element block
  const int sub = e / 32;
  const int i = e % 32;
  const float d = f16w(blk + 34 * sub);
  // C: y = qs[i] * d (mul — the WS3 kernel's exact order)
  volatile float p1 = (float)(int8_t)blk[34 * sub + 2 + i] * d;
  return p1;
}

// ---- GEMM: y[m, n] = sum_k x[m, k] * W(k, n) ----
// One workgroup: 64 m rows x one 256-element super-block (256 n columns;
// n = c*256 + tid so the dequant index is e = tid). Per k: one BB-byte
// super-block fetch (L1-shared across the 256 threads) + 64 x loads
// (L1-shared across the 256 threads). f32 accumulation in k-order.
template <typename XT, int ktype>
__global__ void kqfg_gemm_kernel(
    const XT* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    int M,
    int K,
    int NBS,
    int bb,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs) {
  const int m0 = blockIdx.x * 64;
  const int c = blockIdx.y;
  const int tid = threadIdx.x;
  const bool w0 = (mode == 1 && axis == 0);
  const bool w1 = (mode == 1 && axis == 1);
  float acc[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc[i] = 0.0f;
  }
  for (int k = 0; k < K; ++k) {
    int uf = k;
    int cf = c;
    if (w0) {
      uf = window_map(k, group, keep_ends, pairs);
    } else if (w1) {
      // axis-1: the window runs over the whole row-major 256-block
      // sequence (K * NBS units), not per super-block column.
      const int si = window_map(k * NBS + c, group, keep_ends, pairs);
      uf = si / NBS;
      cf = si % NBS;
    }
    const uint8_t* blk = w + ((size_t)uf * (size_t)NBS + (size_t)cf) * (size_t)bb;
    const float wv = deq_elem<ktype>(blk, tid);
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      if (m0 + i < M) {
        acc[i] += static_cast<float>(x[(size_t)(m0 + i) * (size_t)K + k]) * wv;
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    if (m0 + i < M) {
      out[(size_t)(m0 + i) * (size_t)NBS * 256 + (size_t)c * 256 + tid] = acc[i];
    }
  }
}

// ---- GEMM, small-M path r3e (Leg A port, LDS-staged weight blocks,
// 16-element vector dequant from LDS, strict k-order, no volatiles;
// M <= 16) ----
// Family (cloudy bounded extension 2026-11-10): the no-LDS r3d (5.30ms
// down, bit-exact) is flat-load/scratch bound (ISA: ~30 generic flat
// memory ops per k-row per thread - unaligned 16B splits + x + dequant
// temporaries). r3e restores r1's weight fetch (distributed 16B block
// loads into a DOUBLE-BUFFERED 272B x T-row LDS tile, one barrier per
// tile - the r1 parity pattern, verified race-free) but keeps r3's
// dequant (16-element vector runs from LDS: aligned ds_read_b128, the
// G1-verified per-element math straight-unrolled, CPU cross-verified
// 0/256000 per ktype) and r3's accumulation (each thread owns its 16
// elements for ALL K in strict ascending k-order -> bit-exact vs the
// 64-m-row kernel, no cross-thread reduction). No volatiles (r3d
// bit-exact without them; the mul-mul dequant chains cannot contract).
// 256 threads = 4 slice groups (16 n each) x 16 row cells x 4 replicas;
// grid (NBS * 4, 1). Dequant redundancy = 16 (the r3 mapping's ALU
// floor, ~1.4ms/down M=1 - the r3f f32-staging variant cuts it).
template <typename XT, int ktype, int T>
__global__ void kqfg_mr_gemm_kernel(
    const XT* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    int M,
    int K,
    int NBS,
    int bb,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs) {
  constexpr int NVEC = 16;
  constexpr int BB_MAX = 272;
  // Double-buffered on tile parity (required for correctness, the r1
  // pattern): fetch(it+1) writes parity (it+1)&1 while compute(it) reads
  // it&1; barrier(it+1) orders compute(it) before fetch(it+2) to the same
  // parity buffer.
  __shared__ uint8_t sblk[2][T][BB_MAX];
  const int tid = threadIdx.x;
  const int c = blockIdx.x / 4;            // super-block column
  const int slice = blockIdx.x % 4;        // 64-element slice
  const int g16 = (tid / 16) % 4;          // element group within the slice
  const int rc = tid % 16;                 // row cell (r < nrows active)
  const int e0 = slice * 64 + g16 * NVEC;  // element offset in the super-block
  const bool w0 = (mode == 1 && axis == 0);
  const bool w1 = (mode == 1 && axis == 1);
  const int nrows = M;                      // M <= 16 (dispatch gate)
  const bool active = rc < nrows;
  const int full = bb / 16;
  const int rem = bb % 16;
  const int perrow = full + (rem > 0 ? 1 : 0);
  const int niter = (K + T - 1) / T;
  // Thread-invariant dequant geometry (e0 fixed for the life of the
  // thread): computed once, used in every k iteration.
  const int h = e0 / 128;
  const int l0 = e0 & 31;
  const int s32 = (e0 / 32) & 3;
  const int cc = e0 / 64;
  const int sub = (e0 / 32) & 1;
  const int q8sub = e0 / 32;
  const int si = 2 * cc + sub;
  const int j4 = 2 * cc + sub;
  float acc[NVEC];
  #pragma unroll
  for (int j = 0; j < NVEC; ++j) {
    acc[j] = 0.0f;
  }
  for (int it = 0; it < niter; ++it) {
    const int k0 = it * T;
    const int kt = (K - k0) < T ? (K - k0) : T;
    // Tile fetch: perrow*kt byte-chunks over the kt k-rows, each chunk one
    // 16-byte load (or the rem-byte tail); nchunk can exceed 256 threads
    // (T=32 Q6_K: 448), so stride. All threads write disjoint slots; the
    // single barrier after the loop orders every fetch before any deq read.
    const int nchunk = perrow * kt;
    for (int ccu = tid; ccu < nchunk; ccu += 256) {
      const int row = ccu / perrow;
      const int chunk = ccu % perrow;
      int uf = k0 + row;
      int cf = c;
      if (w0) {
        uf = window_map(k0 + row, group, keep_ends, pairs);
      } else if (w1) {
        const int siw =
            window_map((k0 + row) * NBS + c, group, keep_ends, pairs);
        uf = siw / NBS;
        cf = siw % NBS;
      }
      const uint8_t* blk =
          w + ((size_t)uf * (size_t)NBS + (size_t)cf) * (size_t)bb;
      uint8_t* sb = sblk[it & 1][row];
      if (chunk < full) {
        *reinterpret_cast<uint4*>(&sb[chunk * 16]) =
            *reinterpret_cast<const uint4*>(blk + chunk * 16);
      } else if (rem > 0) {
        #pragma unroll
        for (int i = 0; i < rem; ++i) {
          sb[full * 16 + i] = blk[full * 16 + i];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t >= kt) {
        break;
      }
      const int k = k0 + t;
      const uint8_t* sb = sblk[it & 1][t];
      float wv[NVEC];
      if (ktype == 14) {
        // Q6_K: ql 128B, qh 64B @128, sc int8 16B @192, d f16 @208.
        const uint8_t* qlb = sb + h * 64 + ((s32 & 1) ? 32 : 0) + l0;
        const uint8_t* qhb = sb + 128 + h * 32 + l0;
        const int sc_off = 192 + h * 8 + (l0 / 16) +
                           (s32 == 0 ? 0 : (s32 == 1 ? 2 : (s32 == 2 ? 4 : 6)));
        const uint4 ql16 = *reinterpret_cast<const uint4*>(qlb);
        const uint4 qh16 = *reinterpret_cast<const uint4*>(qhb);
        const uint8_t* qb = reinterpret_cast<const uint8_t*>(&ql16);
        const uint8_t* qw = reinterpret_cast<const uint8_t*>(&qh16);
        const float d = f16w(sb + 208);
        const float scv = (float)(int8_t)sb[sc_off];
        const float t1 = d * scv;
        #pragma unroll
        for (int j = 0; j < NVEC; ++j) {
          const int qh2 = (qw[j] >> (2 * s32)) & 3;
          const int qnib = (s32 < 2) ? (qb[j] & 0xF) : (qb[j] >> 4);
          const int q = (qnib | (qh2 << 4)) - 32;
          wv[j] = t1 * (float)q;
        }
      } else if (ktype == 13) {
        // Q5_K: d/dmin f16 @0/2, 12B k4 @4, qh 32B @16, ql 128B @48.
        const uint8_t* qlb = sb + 48 + 32 * cc + l0;
        const uint8_t* qhb = sb + 16 + l0;
        const uint4 ql16 = *reinterpret_cast<const uint4*>(qlb);
        const uint4 qh16 = *reinterpret_cast<const uint4*>(qhb);
        const uint8_t* qb = reinterpret_cast<const uint8_t*>(&ql16);
        const uint8_t* qw = reinterpret_cast<const uint8_t*>(&qh16);
        const float d = f16w(sb + 0);
        const float minv = f16w(sb + 2);
        const int s1 = (si < 4) ? ((sb + 4 + si)[0] & 63)
                                : (((sb + 4 + si + 4)[0] & 0xF) |
                                   ((sb + 4 + si - 4)[0] >> 6) << 4);
        const int m1 = (si < 4) ? ((sb + 4 + si + 4)[0] & 63)
                                : (((sb + 4 + si + 4)[0] >> 4) |
                                   ((sb + 4 + si)[0] >> 6) << 4);
        const float d1 = d * (float)s1;
        const float m1v = minv * (float)m1;
        #pragma unroll
        for (int j = 0; j < NVEC; ++j) {
          const int f = (qw[j] >> si) & 1;
          const int code = sub
              ? ((qb[j] >> 4) + (f << 4))
              : ((qb[j] & 0xF) + (f << 4));
          wv[j] = sub_exact(d1 * (float)code, m1v);
        }
      } else if (ktype == 12) {
        // Q4_K: d/dmin f16 @0/2, 12B scale table @4, 128B qs @16.
        const int scl = (j4 < 4) ? ((sb + 4 + j4)[0] & 63)
                                 : (((sb + 4 + j4 + 4)[0] & 0xF) |
                                    ((sb + 4 + j4 - 4)[0] >> 6) << 4);
        const int mn = (j4 < 4) ? ((sb + 4 + j4 + 4)[0] & 63)
                                : (((sb + 4 + j4 + 4)[0] >> 4) |
                                   ((sb + 4 + j4)[0] >> 6) << 4);
        const uint8_t* qsb = sb + 16 + 32 * cc + l0;
        const uint4 qs16 = *reinterpret_cast<const uint4*>(qsb);
        const uint8_t* qb = reinterpret_cast<const uint8_t*>(&qs16);
        const float d = f16w(sb + 0);
        const float dmin = f16w(sb + 2);
        const float d1 = d * (float)scl;
        const float m1 = dmin * (float)mn;
        #pragma unroll
        for (int j = 0; j < NVEC; ++j) {
          const int code = sub ? (qb[j] >> 4) : (qb[j] & 0xF);
          wv[j] = d1 * (float)code - m1;
        }
      } else {
        // ktype 8: Q8_0 - 8 x 34B sub-blocks (d f16 @0, 32 x int8 @2).
        const uint8_t* qsb = sb + 34 * q8sub + 2 + l0;
        const uint4 qs16 = *reinterpret_cast<const uint4*>(qsb);
        const uint8_t* qb = reinterpret_cast<const uint8_t*>(&qs16);
        const float d = f16w(sb + 34 * q8sub);
        #pragma unroll
        for (int j = 0; j < NVEC; ++j) {
          wv[j] = (float)(int8_t)qb[j] * d;
        }
      }
      if (active) {
        const float xv =
            static_cast<float>(x[(size_t)rc * (size_t)K + k]);
        #pragma unroll
        for (int j = 0; j < NVEC; ++j) {
          acc[j] += xv * wv[j];
        }
      }
    }
  }
  if (active) {
    #pragma unroll
    for (int j = 0; j < NVEC; ++j) {
      out[(size_t)rc * (size_t)NBS * 256 + (size_t)c * 256 +
          (size_t)e0 + j] = acc[j];
    }
  }
}

// ---- gather: y[b, n] = W(n, r) — row r of W. Element (b, k): block
// (k, r/256) window-mapped, element r%256. One thread per (b, k).
template <typename OutT, int ktype>
__global__ void kqfg_gather_kernel(
    const int32_t* __restrict__ idx,
    const uint8_t* __restrict__ w,
    OutT* __restrict__ out,
    int B,
    int K,
    int NBS,
    int bb,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < static_cast<int64_t>(B) * K; i += stride) {
    const int b = (int)(i / K);
    const int k = (int)(i - (int64_t)b * K);
    const int r = idx[b];  // embedding row (the n index)
    // contract: out-of-range idx (negative, or >= N) maps to row 0
    const int rr = (r < 0 || r >= NBS * 256) ? 0 : r;
    const int c = rr / 256;
    const int e = rr % 256;
    const bool w0 = (mode == 1 && axis == 0);
    const bool w1 = (mode == 1 && axis == 1);
    int uf = k;
    int cf = c;
    if (w0) {
      uf = window_map(k, group, keep_ends, pairs);
    } else if (w1) {
      const int si = window_map(k * NBS + c, group, keep_ends, pairs);
      uf = si / NBS;
      cf = si % NBS;
    }
    const uint8_t* blk = w + ((size_t)uf * (size_t)NBS + (size_t)cf) * (size_t)bb;
    out[i] = static_cast<OutT>(deq_elem<ktype>(blk, e));
  }
}

// ---- standalone dequant (the G1 bit-exact verification path): one thread
// per element (u, c*256+e) -> block (u, c) window-mapped, element e.
template <int ktype>
__global__ void kqfg_dequant_kernel(
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    int K,
    int NBS,
    int bb,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < static_cast<int64_t>(K) * NBS * 256; i += stride) {
    const int64_t nb256 = static_cast<int64_t>(NBS) * 256;
    const int64_t r = i % nb256;
    const int u = (int)(i / nb256);
    const int c = (int)(r / 256);
    const int e = (int)(r % 256);
    const bool w0 = (mode == 1 && axis == 0);
    const bool w1 = (mode == 1 && axis == 1);
    int uf = u;
    int cf = c;
    if (w0) {
      uf = window_map(u, group, keep_ends, pairs);
    } else if (w1) {
      const int si = window_map(u * NBS + c, group, keep_ends, pairs);
      uf = si / NBS;
      cf = si % NBS;
    }
    const uint8_t* blk = w + ((size_t)uf * (size_t)NBS + (size_t)cf) * (size_t)bb;
    out[i] = deq_elem<ktype>(blk, e);
  }
}

// ---- element-wise f32 -> OutT cast (RNE for bfloat16; the engine's
// standard conversion, bit-exact vs the RNE check's host numpy RNE).
template <typename OutT>
__global__ void kqfg_cast_kernel(
    const float* __restrict__ in,
    OutT* __restrict__ out,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    out[i] = static_cast<OutT>(in[i]);
  }
}

// ---- launchers ----

template <typename XT>
bool launch_gemm(const XT* x,
                 const uint8_t* wp,
                 float* op,
                 int M,
                 int K,
                 int NBS,
                 int bb,
                 int ktype,
                 int mode,
                 int axis,
                 int group,
                 int keep_ends,
                 int pairs,
                 hipStream_t stream,
                 int mr_gate) {
  // A/B knob (W1 fix): MLX_KQUANT_FILE_MR 0 = force the 64-m-row kernel,
  // 1 (default) = the small-M kernel for M <= 16, 2 = force the small-M
  // kernel. MLX_KQFG_MR_TILE selects the LDS tile T in {16, 32, 64}
  // (default 32). Small-M path = the Leg A port (kqfg_mr_gemm_kernel,
  // r3e: LDS-staged blocks + 16-elem vector dequant + strict k-order).
  static const int mr_tile = [] {
    const char* e = std::getenv("MLX_KQFG_MR_TILE");
    if (!e) {
      return 32;
    }
    if (std::strcmp(e, "16") == 0) {
      return 16;
    }
    if (std::strcmp(e, "64") == 0) {
      return 64;
    }
    return 32;
  }();
  const bool use_mr = (mr_gate == 2) || (mr_gate == 1 && M <= 16);
  if (use_mr) {
    dim3 gridmr(NBS * 4, 1, 1);
    #define KQFG_MR_LAUNCH(kt)                        \
      do {                                            \
        if (mr_tile == 16) {                          \
          kqfg_mr_gemm_kernel<XT, kt, 16><<<gridmr, 256, 0, stream>>>(\
              x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs); \
        } else if (mr_tile == 64) {                   \
          kqfg_mr_gemm_kernel<XT, kt, 64><<<gridmr, 256, 0, stream>>>(\
              x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs); \
        } else {                                      \
          kqfg_mr_gemm_kernel<XT, kt, 32><<<gridmr, 256, 0, stream>>>(\
              x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs); \
        }                                             \
      } while (0)
    if (ktype == 8) {
      KQFG_MR_LAUNCH(8);
    } else if (ktype == 12) {
      KQFG_MR_LAUNCH(12);
    } else if (ktype == 13) {
      KQFG_MR_LAUNCH(13);
    } else if (ktype == 14) {
      KQFG_MR_LAUNCH(14);
    } else {
      return false;
    }
    #undef KQFG_MR_LAUNCH
    return true;
  }
  dim3 grid((M + 63) / 64, (unsigned)NBS, 1);
  if (ktype == 8) {
    kqfg_gemm_kernel<XT, 8><<<grid, 256, 0, stream>>>(
        x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 12) {
    kqfg_gemm_kernel<XT, 12><<<grid, 256, 0, stream>>>(
        x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 13) {
    kqfg_gemm_kernel<XT, 13><<<grid, 256, 0, stream>>>(
        x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 14) {
    kqfg_gemm_kernel<XT, 14><<<grid, 256, 0, stream>>>(
        x, wp, op, M, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else {
    return false;
  }
  return true;
}

template <typename OutT>
void launch_gather(const int32_t* idx,
                   const uint8_t* wp,
                   OutT* op,
                   int B,
                   int K,
                   int NBS,
                   int bb,
                   int ktype,
                   int mode,
                   int axis,
                   int group,
                   int keep_ends,
                   int pairs,
                   hipStream_t stream) {
  const int64_t total = (int64_t)B * K;
  const int64_t blocks = (total + 255) / 256;
  if (ktype == 8) {
    kqfg_gather_kernel<OutT, 8><<<dim3(blocks), 256, 0, stream>>>(
        idx, wp, op, B, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 12) {
    kqfg_gather_kernel<OutT, 12><<<dim3(blocks), 256, 0, stream>>>(
        idx, wp, op, B, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 13) {
    kqfg_gather_kernel<OutT, 13><<<dim3(blocks), 256, 0, stream>>>(
        idx, wp, op, B, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 14) {
    kqfg_gather_kernel<OutT, 14><<<dim3(blocks), 256, 0, stream>>>(
        idx, wp, op, B, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else {
    throw std::invalid_argument("kquant_file_gather: bad ktype");
  }
}

void launch_dequant(const uint8_t* wp,
                    float* op,
                    int K,
                    int NBS,
                    int bb,
                    int ktype,
                    int mode,
                    int axis,
                    int group,
                    int keep_ends,
                    int pairs,
                    hipStream_t stream) {
  const int64_t total = (int64_t)K * NBS * 256;
  const int64_t blocks = (total + 255) / 256;
  if (ktype == 8) {
    kqfg_dequant_kernel<8><<<dim3(blocks), 256, 0, stream>>>(
        wp, op, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 12) {
    kqfg_dequant_kernel<12><<<dim3(blocks), 256, 0, stream>>>(
        wp, op, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 13) {
    kqfg_dequant_kernel<13><<<dim3(blocks), 256, 0, stream>>>(
        wp, op, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else if (ktype == 14) {
    kqfg_dequant_kernel<14><<<dim3(blocks), 256, 0, stream>>>(
        wp, op, K, NBS, bb, mode, axis, group, keep_ends, pairs);
  } else {
    throw std::invalid_argument("kquant_file_dequant: bad ktype");
  }
}

}  // namespace

// ======================= factories ===============================

namespace {

// Shared validation for the (K, NBS, BB) weight contract.
array make_row_contiguous(const array& a, const Stream& s) {
  if (a.flags().row_contiguous) {
    return a;
  }
  array out(a.shape(), a.dtype(), nullptr, {});
  copy_gpu(a, out, CopyType::General, s);
  return out;
}

struct FLayout {
  int bb;
  int K;
  int NBS;
  int N;
};

void check_layout(
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    FLayout& out) {
  out.bb = block_bytes_for(ktype);
  if (out.bb < 0) {
    throw std::invalid_argument(
        "kquant_file_*: ktype must be 8 (Q8_0), 12 (Q4_K), 13 (Q5_K) "
        "or 14 (Q6_K).");
  }
  if (w.dtype() != uint8 || w.shape().size() != 3) {
    throw std::invalid_argument(
        "kquant_file_*: w must be uint8 with shape "
        "(K, N/256, block_bytes).");
  }
  out.K = (int)w.shape()[0];
  out.NBS = (int)w.shape()[1];
  out.N = out.NBS * 256;
  if (w.shape()[2] != (int64_t)out.bb) {
    throw std::invalid_argument(
        "kquant_file_*: w.shape()[2] must equal the block bytes of "
        "the ktype.");
  }
  if (mode != 0 && mode != 1) {
    throw std::invalid_argument("kquant_file_*: mode must be 0 (identity) "
                               "or 1 (mid_transpose).");
  }
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("kquant_file_*: axis must be 0 or 1.");
  }
  if (mode == 1) {
    const int M = group - 2 * keep_ends;
    if (M <= 0 || (M & 1) == 1 || M / 2 != pairs) {
      throw std::invalid_argument(
          "kquant_file_*: bad window shape (group, keep_ends, pairs).");
    }
    // axis-1 windows run over the whole row-major 256-block sequence
    // (K * NBS units) — the old engine's axis-1 convention.
    const int64_t axis_len =
        axis == 0 ? (int64_t)out.K : (int64_t)out.K * out.NBS;
    if (axis_len % group != 0) {
      throw std::invalid_argument("kquant_file_*: window does not divide "
                                  "the windowed axis length.");
    }
  }
}

}  // namespace

// The three free functions below are referenced only from the Python
// extension (dynamic resolution); the HIP objects link in through the
// static kernel archive, so `used` forces the archive member into
// libmlx.so (an unreferenced archive member would be dropped).
__attribute__((used))
array kquant_file_gemm(
    const array& x,
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    StreamOrDevice s) {
  if (x.dtype() != float32 && x.dtype() != bfloat16) {
    throw std::invalid_argument(
        "kquant_file_gemm: x must be float32 or bfloat16.");
  }
  if (x.ndim() < 2) {
    throw std::invalid_argument(
        "kquant_file_gemm: x must be (..., K) with rank >= 2.");
  }
  // N-D x (e.g. (B, S, K) from a model forward): fold the leading dims
  // into M for the 2-D kernel, restore the leading dims on the output.
  std::vector<int64_t> leading;
  int64_t M = 1;
  for (int i = 0; i < x.ndim() - 1; ++i) {
    leading.push_back(x.shape()[i]);
    M *= x.shape()[i];
  }
  const array* px = &x;
  std::unique_ptr<array> x2d;
  if (!leading.empty()) {
    if (!x.flags().row_contiguous) {
      throw std::invalid_argument(
          "kquant_file_gemm: N-D x must be row-contiguous (model "
          "forward inputs always are).");
    }
    x2d = std::make_unique<array>(
        reshape(x, Shape{(int)M, (int)x.shape()[x.ndim() - 1]}));
    px = x2d.get();
  }
  FLayout L;
  check_layout(w, ktype, mode, axis, group, keep_ends, pairs, L);
  if (px->shape()[1] != (int64_t)L.K) {
    throw std::invalid_argument(
        "kquant_file_gemm: x must be (..., K) with K matching w.");
  }
  auto stream = to_stream(s);
  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("kquant_file_gemm only runs on GPU.");
  };
  auto out = array(
      Shape{(int)M, (int)L.N},
      px->dtype(),
      std::make_shared<fast::KQuantFileGemm>(
          stream, fallback, ktype, mode, axis, group, keep_ends, pairs),
      std::vector<array>{*px, w});
  if (leading.empty()) {
    return out;
  }
  Shape full_shape;
  for (auto v : leading) {
    full_shape.push_back((ShapeElem)v);
  }
  full_shape.push_back((ShapeElem)L.N);
  return reshape(out, full_shape);
}

__attribute__((used))
array kquant_file_gather(
    const array& idx,
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    Dtype out_dtype,
    StreamOrDevice s) {
  if (idx.dtype() != int32) {
    throw std::invalid_argument(
        "kquant_file_gather: idx must be int32 (cast before the call).");
  }
  if (idx.ndim() != 1) {
    throw std::invalid_argument(
        "kquant_file_gather: idx must be 1-D (B) — flatten first.");
  }
  FLayout L;
  check_layout(w, ktype, mode, axis, group, keep_ends, pairs, L);
  if (out_dtype != float32 && out_dtype != bfloat16) {
    throw std::invalid_argument(
        "kquant_file_gather: out_dtype must be float32 or bfloat16.");
  }
  auto stream = to_stream(s);
  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("kquant_file_gather only runs on GPU.");
  };
  // y[b, j] = W[j, idx[b]] — the dequantized column (the row of W^T):
  // output (B, K) with K = w.shape()[0].
  return array(
      Shape{(int)idx.shape()[0], (int)L.K},
      out_dtype,
      std::make_shared<fast::KQuantFileGather>(
          stream, fallback, ktype, mode, axis, group, keep_ends, pairs,
          out_dtype),
      std::vector<array>{idx, w});
}

__attribute__((used))
array kquant_file_dequant(
    const array& w,
    int ktype,
    int mode,
    int axis,
    int group,
    int keep_ends,
    int pairs,
    StreamOrDevice s) {
  FLayout L;
  check_layout(w, ktype, mode, axis, group, keep_ends, pairs, L);
  auto stream = to_stream(s);
  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("kquant_file_dequant only runs on GPU.");
  };
  return array(
      Shape{(int)w.shape()[0], (int)L.N},
      float32,
      std::make_shared<fast::KQuantFileDequant>(
          stream, fallback, ktype, mode, axis, group, keep_ends, pairs),
      std::vector<array>{w});
}

// ======================= eval_gpu ===============================

namespace fast {

std::vector<Shape> KQuantFileGemm::output_shapes(
    const std::vector<array>& inputs) {
  const array& x = inputs[0];
  if (x.ndim() < 2) {
    throw std::invalid_argument(
        "KQuantFileGemm: x must have rank >= 2.");
  }
  auto N = (int64_t)inputs[1].shape()[1] * 256;
  Shape shape;
  int64_t M = 1;
  for (int i = 0; i < x.ndim() - 1; ++i) {
    shape.push_back((ShapeElem)x.shape()[i]);
    M *= x.shape()[i];
  }
  shape.push_back((ShapeElem)M);
  shape.push_back((ShapeElem)N);
  return {shape};
}

bool KQuantFileGemm::is_equivalent(const Primitive& other) const {
  const KQuantFileGemm& a_other =
      static_cast<const KQuantFileGemm&>(other);
  return ktype_ == a_other.ktype_ && mode_ == a_other.mode_ &&
         axis_ == a_other.axis_ && group_ == a_other.group_ &&
         keep_ends_ == a_other.keep_ends_ && pairs_ == a_other.pairs_;
}

std::vector<Shape> KQuantFileGather::output_shapes(
    const std::vector<array>& inputs) {
  return {Shape{inputs[0].shape()[0], inputs[1].shape()[0]}};
}

bool KQuantFileGather::is_equivalent(const Primitive& other) const {
  const KQuantFileGather& a_other =
      static_cast<const KQuantFileGather&>(other);
  return ktype_ == a_other.ktype_ && mode_ == a_other.mode_ &&
         axis_ == a_other.axis_ && group_ == a_other.group_ &&
         keep_ends_ == a_other.keep_ends_ && pairs_ == a_other.pairs_ &&
         out_dtype_ == a_other.out_dtype_;
}

std::vector<Shape> KQuantFileDequant::output_shapes(
    const std::vector<array>& inputs) {
  return {Shape{(int)inputs[0].shape()[0],
                (int)((int64_t)inputs[0].shape()[1] * 256)}};
}

bool KQuantFileDequant::is_equivalent(const Primitive& other) const {
  const KQuantFileDequant& a_other =
      static_cast<const KQuantFileDequant&>(other);
  return ktype_ == a_other.ktype_ && mode_ == a_other.mode_ &&
         axis_ == a_other.axis_ && group_ == a_other.group_ &&
         keep_ends_ == a_other.keep_ends_ && pairs_ == a_other.pairs_;
}

void KQuantFileGemm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  std::unique_ptr<array> tmpx, tmpw;
  const array* px = &inputs[0];
  if (!px->flags().row_contiguous) {
    tmpx = std::make_unique<array>(make_row_contiguous(*px, s));
    encoder.add_temporary(*tmpx);
    px = tmpx.get();
  }
  const array* pw = &inputs[1];
  if (!pw->flags().row_contiguous) {
    tmpw = std::make_unique<array>(make_row_contiguous(*pw, s));
    encoder.add_temporary(*tmpw);
    pw = tmpw.get();
  }
  auto& x = *px;
  auto& w = *pw;
  auto& out = outputs[0];

  const int M = x.shape()[0];
  const int K = x.shape()[1];
  const int NBS = (int)w.shape()[1];
  const int bb = block_bytes_for(ktype_);
  const int mode = mode_;
  const int axis = axis_;
  const int group = group_;
  const int keep_ends = keep_ends_;
  const int pairs = pairs_;

  // The kernel accumulates f32. When the output dtype is f32 the
  // temporary IS the output; otherwise an f32 temporary + one element-
  // wise RNE cast.
  array* f32_target = (out.dtype() == float32)
      ? &out
      : nullptr;
  std::unique_ptr<array> tmpf32;
  if (!f32_target) {
    tmpf32 = std::make_unique<array>(
        mlx::core::zeros(Shape{(int)x.shape()[0], (int)((int64_t)NBS * 256)}, float32, s));
    encoder.add_temporary(*tmpf32);
    f32_target = tmpf32.get();
  }
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  if (f32_target != &out) {
    f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder));
  }

  encoder.launch_kernel([=](hipStream_t stream) {
    const void* xp = gpu_ptr<const void>(x);
    const uint8_t* wp = gpu_ptr<const uint8_t>(w);
    float* op = gpu_ptr<float>(*f32_target);
    static const int mr_gate = [] {
      const char* e = std::getenv("MLX_KQUANT_FILE_MR");
      if (!e) {
        return 1;
      }
      if (std::strcmp(e, "0") == 0) {
        return 0;
      }
      if (std::strcmp(e, "2") == 0) {
        return 2;
      }
      return 1;
    }();
    bool ok = (x.dtype() == float32)
        ? launch_gemm<float>(
              (const float*)xp, wp, op, M, K, NBS, bb, ktype_, mode, axis,
              group, keep_ends, pairs, stream, mr_gate)
        : launch_gemm<__hip_bfloat16>(
              (const __hip_bfloat16*)xp, wp, op, M, K, NBS, bb, ktype_, mode,
              axis, group, keep_ends, pairs, stream, mr_gate);
    if (!ok) {
      throw std::runtime_error("kquant_file_gemm: launch failed");
    }
    if (out.dtype() == bfloat16) {
      const int64_t total = (int64_t)M * NBS * 256;
      const int64_t blocks = (total + 255) / 256;
      kqfg_cast_kernel<__hip_bfloat16><<<dim3(blocks), 256, 0, stream>>>(
          op, static_cast<__hip_bfloat16*>(gpu_ptr<void>(out)), total);
    }
  });
}

void KQuantFileGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  std::unique_ptr<array> tmpidx, tmpw;
  const array* pidx = &inputs[0];
  if (!pidx->flags().row_contiguous) {
    tmpidx = std::make_unique<array>(make_row_contiguous(*pidx, s));
    encoder.add_temporary(*tmpidx);
    pidx = tmpidx.get();
  }
  const array* pw = &inputs[1];
  if (!pw->flags().row_contiguous) {
    tmpw = std::make_unique<array>(make_row_contiguous(*pw, s));
    encoder.add_temporary(*tmpw);
    pw = tmpw.get();
  }
  const array& idx = *pidx;
  const array& w = *pw;
  auto& out = outputs[0];

  const int B = idx.shape()[0];
  const int K = (int)w.shape()[0];
  const int NBS = (int)w.shape()[1];
  const int bb = block_bytes_for(ktype_);
  const int mode = mode_;
  const int axis = axis_;
  const int group = group_;
  const int keep_ends = keep_ends_;
  const int pairs = pairs_;

  out.set_data(rocm::malloc_async(out.nbytes(), encoder));

  encoder.launch_kernel([=](hipStream_t stream) {
    const int32_t* ip = gpu_ptr<const int32_t>(idx);
    const uint8_t* wp = gpu_ptr<const uint8_t>(w);
    void* op = (void*)gpu_ptr<void>(out);
    if (out.dtype() == float32) {
      launch_gather<float>(
          ip, wp, (float*)op, B, K, NBS, bb, ktype_, mode, axis, group,
          keep_ends, pairs, stream);
    } else {
      launch_gather<__hip_bfloat16>(
          ip, wp, (__hip_bfloat16*)op, B, K, NBS, bb, ktype_, mode, axis,
          group, keep_ends, pairs, stream);
    }
  });
}

void KQuantFileDequant::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  std::unique_ptr<array> tmpw;
  const array* pw = &inputs[0];
  if (!pw->flags().row_contiguous) {
    tmpw = std::make_unique<array>(make_row_contiguous(*pw, s));
    encoder.add_temporary(*tmpw);
    pw = tmpw.get();
  }
  const array& w = *pw;
  auto& out = outputs[0];

  const int K = (int)w.shape()[0];
  const int NBS = (int)w.shape()[1];
  const int bb = block_bytes_for(ktype_);
  const int mode = mode_;
  const int axis = axis_;
  const int group = group_;
  const int keep_ends = keep_ends_;
  const int pairs = pairs_;

  out.set_data(rocm::malloc_async(out.nbytes(), encoder));

  encoder.launch_kernel([=](hipStream_t stream) {
    const uint8_t* wp = gpu_ptr<const uint8_t>(w);
    float* op = static_cast<float*>(gpu_ptr<void>(out));
    launch_dequant(
        wp, op, K, NBS, bb, ktype_, mode, axis, group, keep_ends, pairs,
        stream);
  });
}

}  // namespace fast

}  // namespace mlx::core
