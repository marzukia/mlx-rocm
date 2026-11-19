// Device code for the ported fattn kernel (attention port, v3 card 35ce5495).
//
// Algorithm, tiling and config follow the ggml fattn mma-f16 path for the
// production instance <256,256,16,4> (ncols1=16 q rows/block, ncols2=4 GQA
// heads, D=256 one pass, nbatch_fa=32 KV rows/softmax tile, 64 threads /
// 2 waves): ref = llama.cpp b12062 @ 29c720eaf, fattn-mma-f16.cuh
// (process_tile + kernel driver) + fattn.cu dispatch (switch_ncols1/2,
// get_best_fattn_kernel). Fragment ops use the rocwmma 16x16x16 bf16->f32
// idiom proven by arm A (flash_attention_wmma.hip) on gfx1201.
//
// v1 scope (instance (16,4), the spec as written):
//   - bf16 in / bf16 out. rocwmma fragments carry 4x the elements of the
//     ref's 16x8 ggml mma B-frags, so two port adjustments (both noted in
//     the PR body; stage 0 is the numeric gate either way):
//       (1) Q is staged in LDS (2 d-halves) and Q fragments load per mma
//           step — the ref's Q_in_reg=true is register-impossible at
//           rocwmma fragment size;
//       (2) O accumulates in a bf16 register array with per-mma rounding
//           (ref-faithful: the ref's T_C_VKQ is a half2 C accumulator),
//           not an f32 accumulator (256 regs, over the 255 budget with the
//           S fragments live).
//   - causal on-the-fly (card default (a)): visibility computed per element
//     (kv_global > q_global -> -inf); no mask tensor, no KV_max scan, no
//     compact-mask kernel.
//   - 1 block per (seq, kv-head, q-tile): grid == tile count, so the ref's
//     stream-K fixup path (dst_meta + fixup kernel) is provably unneeded
//     (kb0_start==0, kb0_stop==iter_k -> needs_fixup==false always).
//     Stream-K load balancing = v2 lever.
//   - no LSE (Q-C4 deferred), no alibi / softcap / sinks / sparse (v1 gates).
//     The (1,4) sparse instance is compiled out for GQA4/D=256 in the ref
//     (may_use_sparse(256,256,1,4)==false); the r3 trace arbitrates
//     empirically.
//
// gfx1201 fragment element layout (rocWMMA 16x16x16, verified by probe,
// arm A 2026-11-16): element e of lane L of an accumulator / A fragment
// holds row = e + 8*(L>>4), col = L&15. Each lane owns a full 16-row
// column per fragment: per-column max / rowsum / rescale / mask / store
// are self-contained per lane (no shuffles).
//
// LDS (bf16), 65,536 B total = EXACTLY the gfx1201 per-block cap (verified
// 2026-11-18: hipGetDeviceProperties smemPerBlock = 65,536 B, RX 9070 XT):
//   Q_lo [NC][128] = 16,384 B   (d-half 0: 128 d, zero pad)
//   Q_hi [NC][128] = 16,384 B   (d-half 1, zero pad)
//   KV   [FA][256] x2 = 32,768 B (K then V, co-resident, zero pad)
// P (bf16, [NC col][FA]) aliases the K region (written after the S mmas
// consume K - barrier-enforced below; read during the O mmas, co-resident
// with V). Zero-pad layout:
// gfx1201 LDS has no NVIDIA-style bank-conflict class and the ref itself
// runs an unaligned 264-elem stride; fallback if exactly-max launch fails =
// Q-in-regs (32,768 B, the ref's Q_in_reg footprint class).

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define FATTN_WMMA 1
#include <rocwmma/rocwmma.hpp>
#else
#define FATTN_WMMA 0
#endif
#include <hip/hip_bfloat16.h>
#include <cfloat>

namespace mlx::core {
namespace rocm {
namespace fattn {

// Tile constants — ref config (256, 256, ncols1=16), instance (16, 4).
constexpr int NC1 = 16;   // q rows per block
constexpr int NC2 = 4;    // gqa heads per block
constexpr int NC = NC1 * NC2;  // 64 S columns
constexpr int FA = 32;    // nbatch_fa: KV rows per softmax tile
constexpr int D = 256;    // head dim, one pass (nbatch_K2 = D/2 pairs)
constexpr int NTHREADS = 64;
constexpr int NWARPS = NTHREADS / 32;
constexpr int CW = NC / NWARPS;  // 32 S columns per warp (2 col chunks of 16)
constexpr int DCHUNKS = D / 16;  // 16 mma steps over the head dim

// Row strides in bf16 element units (pair = 2 bf16 = 4 B = one ref half2
// slot; the ref pads rows by +4 half2 = +8 bf16... v1 uses +4 bf16 pad,
// 4 B aligned — load_matrix_sync needs no 16 B row alignment in LDS).
constexpr int SQH = D / 2;       // 128: Q d-half row (128 d, zero pad)
constexpr int SKV = D;           // 256: K/V row (256 d, zero pad)
constexpr int SP = FA;           // 32:  P row ([64 col][32])

// LDS sizes (bf16 elements); all pointer math below is in bf16 units.
constexpr int QHALF_ELEMS = NC * SQH;          // 8,192 bf16 per d-half
constexpr int KV_ELEMS = 2 * FA * SKV;         // 16,384 bf16 (K + V regions)
constexpr int LDS_ELEMS = 2 * QHALF_ELEMS + KV_ELEMS;  // 32,768
constexpr int LDS_BYTES = LDS_ELEMS * 2;       // 65,536 B (= device cap)

} // namespace fattn

// ---- bf16 conversion helpers (self-contained bit ops) ----
// ROCm 7.2.0 toolchain headers on this box provide no bf16<->f32 intrinsics
// (the tree's proven idiom is the union cast, cf. lm_head_int2.hip
// li2_bf2f16: "ROCm has no __bfloat162float intrinsic").
__device__ __forceinline__ float bf16_to_f32(hip_bfloat16 b) {
  union {
    unsigned short h;
    hip_bfloat16 bf;
  } c;
  c.h = *reinterpret_cast<unsigned short*>(&b);
  unsigned int u = (unsigned int)c.h << 16;  // bf16 bits = upper half of f32
  return *reinterpret_cast<float*>(&u);
}

__device__ __forceinline__ hip_bfloat16 f32_to_bf16(float f) {
  union {
    float fl;
    unsigned int u;
  } c;
  c.fl = f;
  // round-to-nearest-even on bit 16
  unsigned int r = (c.u + 0x7fffu + ((c.u >> 16) & 1u)) >> 16;
  union {
    unsigned short h;
    hip_bfloat16 bf;
  } d;
  d.h = (unsigned short)r;
  return d.bf;
}


template <bool DoCausal>
__global__ void __launch_bounds__(fattn::NTHREADS) kernel_sdpa_flash_fattn(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ o,
    FaFftnParams p) {
#if FATTN_WMMA
  using namespace fattn;
  extern __shared__ char smem_raw[];
  hip_bfloat16* smem = reinterpret_cast<hip_bfloat16*>(smem_raw);

  hip_bfloat16* Q_lo = smem;
  hip_bfloat16* Q_hi = Q_lo + QHALF_ELEMS;
  hip_bfloat16* K_t = Q_hi + QHALF_ELEMS;   // [FA][SKV]
  hip_bfloat16* V_t = K_t + FA * SKV;       // [FA][SKV]
  // P aliases K_t: [NC col][SP]

  // Block mapping: 1 block per (seq, kv-head, q-tile) — the ref's
  // single-block-tile path (kb0_start == 0, kb0_stop == iter_k).
  const int tiles_per_seq = p.KVH * p.iter_j;
  const int seq = blockIdx.x / tiles_per_seq;
  const int zKV = (blockIdx.x / p.iter_j) % p.KVH;
  const int jt = blockIdx.x % p.iter_j;

  const int lane = threadIdx.x & 31;
  const int half = lane >> 4;       // gfx1201 acc row half
  const int col0 = lane & 15;       // acc col within a 16-col chunk
  const int warp = threadIdx.x >> 5;
  // Chunk bases (lane-independent) and the lane's own S column per chunk:
  // S col c -> q row j = c/4, gqa sub-head g = c%4.
  const int chunk0[2] = {warp * CW, warp * CW + 16};
  const int s_col[2] = {chunk0[0] + col0, chunk0[1] + col0};

  // ---- Q tile load (scale applied at load, ref-faithful; for D=256 the
  // scale is 2^-4, exact in bf16, so Q*scale rounds nothing extra) ----
  {
    // 128 row-halves (64 S cols x 2 halves) x 128 bf16 over 64 threads:
    // 2 row-halves per thread. Element-wise (ROCm 7.2.0 hip header has no
    // bf16 pair type / conversion intrinsics in the 7.2.0 headers; the
    // self-contained bit-op helpers below (bf16_to_f32 / f32_to_bf16) are the
    // port, cf. lm_head_int2.hip li2_bf2f16 union-cast idiom.
    const int r0 = threadIdx.x;                     // S col
    for (int hh = 0; hh < 2; hh++) {
      // Each d-half is a separate [NC][128] array: offset within the half
      // row is 0; the global d offset lives in the source pointer.
      const int d0 = hh * (D / 2);                  // bf16 offset in qsrc
      const int j = r0 / NC2;                       // q row within tile
      const int g = r0 % NC2;                       // gqa sub-head
      const int qrow = jt * NC1 + j;
      hip_bfloat16* qdst = (hh ? Q_hi : Q_lo) + r0 * SQH;
      const hip_bfloat16* qsrc =
          q + static_cast<int64_t>(seq) * p.Q_sb
              + static_cast<int64_t>(zKV * NC2 + g) * p.Q_sh
              + static_cast<int64_t>(qrow) * p.Q_sl + d0;
      if (qrow < p.qL) {
#pragma unroll
        for (int i = 0; i < D / 2; i += 8) {
          const float f0 = bf16_to_f32(qsrc[i]);
          const float f1 = bf16_to_f32(qsrc[i + 1]);
          const float f2 = bf16_to_f32(qsrc[i + 2]);
          const float f3 = bf16_to_f32(qsrc[i + 3]);
          const float f4 = bf16_to_f32(qsrc[i + 4]);
          const float f5 = bf16_to_f32(qsrc[i + 5]);
          const float f6 = bf16_to_f32(qsrc[i + 6]);
          const float f7 = bf16_to_f32(qsrc[i + 7]);
          qdst[i] = f32_to_bf16(f0 * p.scale);
          qdst[i + 1] = f32_to_bf16(f1 * p.scale);
          qdst[i + 2] = f32_to_bf16(f2 * p.scale);
          qdst[i + 3] = f32_to_bf16(f3 * p.scale);
          qdst[i + 4] = f32_to_bf16(f4 * p.scale);
          qdst[i + 5] = f32_to_bf16(f5 * p.scale);
          qdst[i + 6] = f32_to_bf16(f6 * p.scale);
          qdst[i + 7] = f32_to_bf16(f7 * p.scale);
        }
      } else {
#pragma unroll
        for (int i = 0; i < D / 2; i++) qdst[i] = f32_to_bf16(0.0f);
      }
    }
  }
  __syncthreads();

  // ---- State ----
  using fA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, hip_bfloat16,
                               rocwmma::row_major>;
  using fB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, hip_bfloat16,
                               rocwmma::col_major>;
  using fBr = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, hip_bfloat16,
                                rocwmma::row_major>;
  using fC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

  // S fragments: [kv16][col chunk] — live across the 16 d-mma steps.
  fC s_acc[2][2];
  float s_max[2] = {-FLT_MAX / 2.0f, -FLT_MAX / 2.0f};  // ref KQ_max init
  float s_rowsum[2] = {0.0f, 0.0f};
  // O accumulator: bf16 (ref half2 C), [col chunk][d16] x 8 bf16.
  hip_bfloat16 o_acc[2][DCHUNKS][8];
#pragma unroll
  for (int cX = 0; cX < 2; cX++)
#pragma unroll
    for (int d16 = 0; d16 < DCHUNKS; d16++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        o_acc[cX][d16][e] = f32_to_bf16(0.0f);

  const int kv_iters = (p.kL + FA - 1) / FA;
  for (int kb = 0; kb < kv_iters; kb++) {
    const int kv0 = kb * FA;

    // Causal early break: last visible kv of the last col of this block.
    if constexpr (DoCausal) {
      const int max_kv = p.kv_offset + (jt * NC1 + NC1 - 1);
      if (kv0 > max_kv) break;
    }

    // ---- K tile load: 32 rows x 256 bf16 over 64 threads (2 rows per
    // thread, 128 contiguous bf16 each) ----
    {
      const int kr = threadIdx.x >> 1;
      const int k2s = (threadIdx.x & 1) * (D / 2);  // bf16 offset
      const int krow = kv0 + kr;
      hip_bfloat16* kdst = K_t + kr * SKV + k2s;
      if (krow < p.kL) {
        const hip_bfloat16* src =
            k + static_cast<int64_t>(seq) * p.K_sb
                + static_cast<int64_t>(zKV) * p.K_sh
                + static_cast<int64_t>(krow) * p.K_sl + k2s;
#pragma unroll
        for (int i = 0; i < D / 2; i++) kdst[i] = src[i];
      } else {
#pragma unroll
        for (int i = 0; i < D / 2; i++) kdst[i] = f32_to_bf16(0.0f);
      }
    }
    __syncthreads();

    // ---- S = K @ Q^T (f32 acc fragments) ----
#pragma unroll
    for (int kv16 = 0; kv16 < 2; kv16++)
#pragma unroll
      for (int cX = 0; cX < 2; cX++)
        rocwmma::fill_fragment(s_acc[kv16][cX], 0.0f);
    for (int d16 = 0; d16 < DCHUNKS; d16++) {
      const int d0 = d16 * 16;
      const hip_bfloat16* qhalf = (d0 < D / 2) ? Q_lo : Q_hi;
#pragma unroll
      for (int kv16 = 0; kv16 < 2; kv16++) {
        fA aK;
        rocwmma::load_matrix_sync(aK, K_t + (kv16 * 16) * SKV + d0, SKV);
#pragma unroll
        for (int cX = 0; cX < 2; cX++) {
          fB aQ;
          // B = Q^T [d16 x col16]: element (d, c) = qtile[c][d] ->
          // col-major load, ldm = SQH (bf16 units); base = chunk base.
          rocwmma::load_matrix_sync(
              aQ, qhalf + chunk0[cX] * SQH + (d0 % (D / 2)), SQH,
              rocwmma::mem_col_major);
          rocwmma::mma_sync(s_acc[kv16][cX], aK, aQ, s_acc[kv16][cX]);
        }
      }
    }
    // Barrier: every thread's S-mma K reads (load_matrix_sync above) must
    // complete before any thread writes P into the aliased K region below.
    // P aliases K_t; a lagging reader of K rows 0..7 would race the write.
    __syncthreads();

    // ---- Causal mask (on-the-fly) + online softmax, per col chunk ----
#pragma unroll
    for (int cX = 0; cX < 2; cX++) {
      const int qrow = jt * NC1 + s_col[cX] / NC2;
      // Finite masked sentinel (not -inf): keeps max/exp NaN-free even when
      // a whole 32-row tile is invisible to this column (causal tail).
      constexpr float MASKED = -FLT_MAX / 4.0f;
      float m = s_max[cX];
#pragma unroll
      for (int kv16 = 0; kv16 < 2; kv16++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int kvrow = kv0 + kv16 * 16 + e + 8 * half;
          float s = s_acc[kv16][cX].x[e];
          if (kvrow >= p.kL || (DoCausal && kvrow > p.kv_offset + qrow)) {
            s = MASKED;
          }
          s_acc[kv16][cX].x[e] = s;
          m = fmaxf(m, s);
        }
      }
      // True per-column max: this lane owns 8 of the 16 kv rows of its S-col
      // (row = e + 8*half, col = L&15, probe-verified); the other half lives
      // in lane ^16.
      m = fmaxf(m, __shfl_xor(m, 16));
      const bool all_masked = (m <= MASKED);
      // Rescale factor = expf(old - new) (ref KQ_max_scale); first iter
      // (old = -FLT_MAX/2) -> 0, so the O accumulator (zero) stays zero.
      float rescale = 1.0f;
      if (!all_masked) {
        rescale = expf(s_max[cX] - m);
        s_max[cX] = m;
      }
      float rs = 0.0f;
#pragma unroll
      for (int kv16 = 0; kv16 < 2; kv16++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float pe = all_masked
                               ? 0.0f
                               : expf(s_acc[kv16][cX].x[e] - m);
          s_acc[kv16][cX].x[e] = pe;
          rs += pe;
        }
      }
      rs += __shfl_xor(rs, 16);  // true per-column rowsum (both halves)
      if (!all_masked) {
        s_rowsum[cX] = s_rowsum[cX] * rescale + rs;
      }

      // P -> LDS (bf16), aliasing the K region: P[s_col][kv_local];
      // element kv_local = e + 8*half (consecutive e = consecutive kv rows
      // of the lane's S column).
      hip_bfloat16* pcol = K_t + s_col[cX] * SP;
#pragma unroll
      for (int kv16 = 0; kv16 < 2; kv16++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          pcol[kv16 * 16 + e + 8 * half] =
              f32_to_bf16(s_acc[kv16][cX].x[e]);
        }
      }
      // O rescale: the O fragment row = S-col (e + 8*half), NOT this lane's
      // S-col (L&15) - gather each element's column rescale from its owning
      // lane (both halves hold the same per-col value after the xor-16 max).
      float rescale_e[8];
#pragma unroll
      for (int e = 0; e < 8; e++)
        rescale_e[e] = __shfl(rescale, e + 8 * half);
#pragma unroll
      for (int d16 = 0; d16 < DCHUNKS; d16++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          o_acc[cX][d16][e] =
              f32_to_bf16(bf16_to_f32(o_acc[cX][d16][e]) * rescale_e[e]);
        }
      }
    }
    __syncthreads();

    // ---- V tile load (co-resident with P; the V region is free) ----
    {
      const int vr = threadIdx.x >> 1;
      const int k2s = (threadIdx.x & 1) * (D / 2);  // bf16 offset
      const int vrow = kv0 + vr;
      hip_bfloat16* vdst = V_t + vr * SKV + k2s;
      if (vrow < p.kL) {
        const hip_bfloat16* src =
            v + static_cast<int64_t>(seq) * p.V_sb
                + static_cast<int64_t>(zKV) * p.V_sh
                + static_cast<int64_t>(vrow) * p.V_sl + k2s;
#pragma unroll
        for (int i = 0; i < D / 2; i++) vdst[i] = src[i];
      } else {
#pragma unroll
        for (int i = 0; i < D / 2; i++) vdst[i] = f32_to_bf16(0.0f);
      }
    }
    __syncthreads();

    // ---- O += P @ V (f32 mma, folded into the bf16 accumulator; ref
    // half2-C precision) ----
    for (int d16 = 0; d16 < DCHUNKS; d16++) {
      const int d0 = d16 * 16;
#pragma unroll
      for (int kv16 = 0; kv16 < 2; kv16++) {
        fBr bV;
        rocwmma::load_matrix_sync(bV, V_t + (kv16 * 16) * SKV + d0, SKV);
#pragma unroll
        for (int cX = 0; cX < 2; cX++) {
          fA aP;
          // A = P [col16 x kv16] row-major, ldm = SP; base = chunk base.
          rocwmma::load_matrix_sync(aP, K_t + chunk0[cX] * SP + kv16 * 16,
                                    SP);
          fC tmp;
          rocwmma::fill_fragment(tmp, 0.0f);
          rocwmma::mma_sync(tmp, aP, bV, tmp);
#pragma unroll
          for (int e = 0; e < 8; e++) {
            o_acc[cX][d16][e] = f32_to_bf16(
                bf16_to_f32(o_acc[cX][d16][e]) + tmp.x[e]);
          }
        }
      }
    }
    __syncthreads();
  }

  // ---- Finalize: O /= rowsum, store bf16 ----
  // O frag element e: S-col local = e + 8*half, d local = lane&15.
#pragma unroll
  for (int cX = 0; cX < 2; cX++) {
    // O fragment row = S-col (e + 8*half): gather that col's rowsum (the
    // per-lane s_rowsum is for col L&15). All-masked col (rs = 0) -> O := 0
    // (the 0*inf NaN guard for the fully-masked class).
    float inv_rs_e[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const float rs_r = __shfl(s_rowsum[cX], e + 8 * half);
      inv_rs_e[e] = (rs_r > 0.0f) ? 1.0f / rs_r : 0.0f;
    }
#pragma unroll
    for (int d16 = 0; d16 < DCHUNKS; d16++) {
      const int d = d16 * 16 + col0;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int sc = chunk0[cX] + e + 8 * half;  // global S col
        const int j = sc / NC2;                    // q row within tile
        const int g = sc % NC2;                    // gqa sub-head
        const int qrow = jt * NC1 + j;
        if (qrow >= p.qL) continue;
        const float val = bf16_to_f32(o_acc[cX][d16][e]);
        o[static_cast<int64_t>(seq) * p.O_sb
             + static_cast<int64_t>(zKV * NC2 + g) * p.O_sh
             + static_cast<int64_t>(qrow) * p.O_sl + d] =
            f32_to_bf16(val * inv_rs_e[e]);
      }
    }
  }
#else  // host pass: stub body so the launcher's &kernel reference links
  (void)q; (void)k; (void)v; (void)o; (void)p;
#endif
}

} // namespace rocm
} // namespace mlx::core
