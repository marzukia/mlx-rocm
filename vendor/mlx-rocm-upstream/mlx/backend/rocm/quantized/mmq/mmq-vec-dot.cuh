#pragma once

#define DBG_T4K 0 // P1B19: probes off (measurement build, cloudy cross-box data point)  // P1B4-AUDIT 07:2xZ: t4k single-fire dump (block0 i0 j0 thread0)  // PATCH60-TMP disabled for the P1 plan-B measurement build (in-kernel printf stalls the model run)
// build96 (2026-11-15): zero-init all mma C accumulator tiles. Uninitialized tile_C C/; the i32 WMMA builtin reads C as input accumulator -> per-k-block garbage (prev-iteration mma result) added to every product. Named by kprobe 2026-11-15: per-k-block const-magnitude random-sign error, prefix rms linear in K, output-relative via GLU amplification, p10-small elements exact.

#include "vecdotq.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_0 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_0], u,
                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_1 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_1], u,
                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    // 93 fix: C.x[l] = (m = 8h+l, n = lane%16) on gfx1201 -> the x
                    // scale row is the C tile-row index (i0 + n*16 + 8h + l), not
                    // the NV-style (i0 + lane%16).
                    const int i = i0 + n*tile_A::I + 8*(threadIdx.x >> 4) + l;
                    const float dA = x_df[i*sram_stride + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA*dB;
                }
            }
        }
    }
#else
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}


template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    // P1B16 (AUDIT18): C tile = I_MAJOR (probed gfx1201 fragment layout, 93 fix) - the
    // J_MAJOR transpose mislabels the raw wmma fragment: += used x-scale row c and
    // y-scale col 8h instead of row 8h+l / col c (AUDIT17 t6a: executed (i,j)=(t%16, 8h)).
    typedef tile<16, 16, int, DATA_LAYOUT_I_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    float2 dmA = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.x*dsB.x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.y*dsB.y;
#if DBG_T4K
                    {
                        const int sl = (j0/tile_C::J + n)*tile_C::ne + l;
                        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && j0 == 0 && k01 == 0 && k00 == 0) {
                            if (false) printf("[t6a] t=%d l=%d i=%d j=%d slot=%d C=%d dma=%08x dsb=%08x T=%.9g\n",
                                   (int)threadIdx.x, (int)l, (int)i, (int)j, (int)sl, (int)C.x[l],
                                   (unsigned)*(const uint32_t*)&dmA, (unsigned)*(const uint32_t*)&dsB,
                                   dmA.x*dsB.x*C.x[l] + dmA.y*dsB.y);
                        }
                    }
#endif
#if DBG_T4K
                    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && l == 0 && j0 == 0 && k0 % 8 == 0 && k0 < 64) {
                        // P1B11 (AUDIT13): per-thread full-K dump - C.x[0..7] + 8 x_dm words (rows 8h..8h+7) + y_dm word (col c)
                        const int h4 = threadIdx.x >> 4;
                        const int c4 = threadIdx.x & 15;
                        if (false) printf("[t4p] t=%d k0=%d C: %d %d %d %d %d %d %d %d dma: %08x %08x %08x %08x %08x %08x %08x %08x dsb: %08x\n",
                               (int)threadIdx.x, (int)k0,
                               (int)C.x[0],(int)C.x[1],(int)C.x[2],(int)C.x[3],(int)C.x[4],(int)C.x[5],(int)C.x[6],(int)C.x[7],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+0)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+1)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+2)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+3)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+4)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+5)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+6)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&x_dm[(8*h4+7)*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&y_dm[c4*MMQ_TILE_Y_K + k01/8]);
                    }
                    if (k0 == 0 && j0 == 0 && l == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                        // P1B17 (AUDIT19): x-tile rows 0..1 words 0..63 for blocks (0,0)/(0,1)/(0,2),
                        // y-tile rows 0..1 words 0..35 for blocks (0,0)/(1,0) - first pass only (k00==0 && k01==0)
                        if ((blockIdx.x == 0 && blockIdx.y < 3) || (blockIdx.x == 1 && blockIdx.y == 0)) {
                            if (k00 == 0 && k01 == 0) {
                                for (int r = 0; r < 2; ++r) {
                                    if (false) printf("[t4qx] k0=0 bx=%d by=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)blockIdx.x, (int)blockIdx.y, r,
                                           (unsigned)x_qs[r*sram_stride+0],(unsigned)x_qs[r*sram_stride+8],(unsigned)x_qs[r*sram_stride+16],(unsigned)x_qs[r*sram_stride+24],(unsigned)x_qs[r*sram_stride+32],(unsigned)x_qs[r*sram_stride+40],(unsigned)x_qs[r*sram_stride+48],(unsigned)x_qs[r*sram_stride+56],
                                           (unsigned)x_qs[r*sram_stride+1],(unsigned)x_qs[r*sram_stride+9],(unsigned)x_qs[r*sram_stride+17],(unsigned)x_qs[r*sram_stride+25],(unsigned)x_qs[r*sram_stride+33],(unsigned)x_qs[r*sram_stride+41],(unsigned)x_qs[r*sram_stride+49],(unsigned)x_qs[r*sram_stride+57],
                                           (unsigned)x_qs[r*sram_stride+2],(unsigned)x_qs[r*sram_stride+10],(unsigned)x_qs[r*sram_stride+18],(unsigned)x_qs[r*sram_stride+26],(unsigned)x_qs[r*sram_stride+34],(unsigned)x_qs[r*sram_stride+42],(unsigned)x_qs[r*sram_stride+50],(unsigned)x_qs[r*sram_stride+58],
                                           (unsigned)x_qs[r*sram_stride+3],(unsigned)x_qs[r*sram_stride+11],(unsigned)x_qs[r*sram_stride+19],(unsigned)x_qs[r*sram_stride+27],(unsigned)x_qs[r*sram_stride+35],(unsigned)x_qs[r*sram_stride+43],(unsigned)x_qs[r*sram_stride+51],(unsigned)x_qs[r*sram_stride+59],
                                           (unsigned)x_qs[r*sram_stride+4],(unsigned)x_qs[r*sram_stride+12],(unsigned)x_qs[r*sram_stride+20],(unsigned)x_qs[r*sram_stride+28],(unsigned)x_qs[r*sram_stride+36],(unsigned)x_qs[r*sram_stride+44],(unsigned)x_qs[r*sram_stride+52],(unsigned)x_qs[r*sram_stride+60],
                                           (unsigned)x_qs[r*sram_stride+5],(unsigned)x_qs[r*sram_stride+13],(unsigned)x_qs[r*sram_stride+21],(unsigned)x_qs[r*sram_stride+29],(unsigned)x_qs[r*sram_stride+37],(unsigned)x_qs[r*sram_stride+45],(unsigned)x_qs[r*sram_stride+53],(unsigned)x_qs[r*sram_stride+61],
                                           (unsigned)x_qs[r*sram_stride+6],(unsigned)x_qs[r*sram_stride+14],(unsigned)x_qs[r*sram_stride+22],(unsigned)x_qs[r*sram_stride+30],(unsigned)x_qs[r*sram_stride+38],(unsigned)x_qs[r*sram_stride+46],(unsigned)x_qs[r*sram_stride+54],(unsigned)x_qs[r*sram_stride+62],
                                           (unsigned)x_qs[r*sram_stride+7],(unsigned)x_qs[r*sram_stride+15],(unsigned)x_qs[r*sram_stride+23],(unsigned)x_qs[r*sram_stride+31],(unsigned)x_qs[r*sram_stride+39],(unsigned)x_qs[r*sram_stride+47],(unsigned)x_qs[r*sram_stride+55],(unsigned)x_qs[r*sram_stride+63]);
                                }
                                if (blockIdx.x == 1 || blockIdx.y == 0) {
                                    for (int r = 0; r < 2; ++r) {
                                        if (false) printf("[t4qy] k0=0 bx=%d by=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)blockIdx.x, (int)blockIdx.y, r,
                                               (unsigned)y_qs[r*MMQ_TILE_Y_K+0],(unsigned)y_qs[r*MMQ_TILE_Y_K+1],(unsigned)y_qs[r*MMQ_TILE_Y_K+2],(unsigned)y_qs[r*MMQ_TILE_Y_K+3],(unsigned)y_qs[r*MMQ_TILE_Y_K+4],(unsigned)y_qs[r*MMQ_TILE_Y_K+5],(unsigned)y_qs[r*MMQ_TILE_Y_K+6],(unsigned)y_qs[r*MMQ_TILE_Y_K+7],
                                               (unsigned)y_qs[r*MMQ_TILE_Y_K+8],(unsigned)y_qs[r*MMQ_TILE_Y_K+9],(unsigned)y_qs[r*MMQ_TILE_Y_K+10],(unsigned)y_qs[r*MMQ_TILE_Y_K+11],(unsigned)y_qs[r*MMQ_TILE_Y_K+12],(unsigned)y_qs[r*MMQ_TILE_Y_K+13],(unsigned)y_qs[r*MMQ_TILE_Y_K+14],(unsigned)y_qs[r*MMQ_TILE_Y_K+15],
                                               (unsigned)y_qs[r*MMQ_TILE_Y_K+16],(unsigned)y_qs[r*MMQ_TILE_Y_K+17],(unsigned)y_qs[r*MMQ_TILE_Y_K+18],(unsigned)y_qs[r*MMQ_TILE_Y_K+19],(unsigned)y_qs[r*MMQ_TILE_Y_K+20],(unsigned)y_qs[r*MMQ_TILE_Y_K+21],(unsigned)y_qs[r*MMQ_TILE_Y_K+22],(unsigned)y_qs[r*MMQ_TILE_Y_K+23],
                                               (unsigned)y_qs[r*MMQ_TILE_Y_K+24],(unsigned)y_qs[r*MMQ_TILE_Y_K+25],(unsigned)y_qs[r*MMQ_TILE_Y_K+26],(unsigned)y_qs[r*MMQ_TILE_Y_K+27],(unsigned)y_qs[r*MMQ_TILE_Y_K+28],(unsigned)y_qs[r*MMQ_TILE_Y_K+29],(unsigned)y_qs[r*MMQ_TILE_Y_K+30],(unsigned)y_qs[r*MMQ_TILE_Y_K+31],
                                               (unsigned)y_qs[r*MMQ_TILE_Y_K+32],(unsigned)y_qs[r*MMQ_TILE_Y_K+33],(unsigned)y_qs[r*MMQ_TILE_Y_K+34],(unsigned)y_qs[r*MMQ_TILE_Y_K+35]);
                                    }
                                }
                            }
                        }
                    } else if (k0 == 0 && j0 == 0 && l == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                        // P1B12 (AUDIT14): full staged code words (x rows 0..15 words 0..63, y rows 0..15 words 0..31), first kb0 only
                        static __device__ unsigned t4q_count;
                        if (t4q_count < 32) {
                            for (int r = 0; r < 16; ++r) {
                                if (false) printf("[t4qx] k0=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)k0, r,
                                   (unsigned)x_qs[r*sram_stride+0],(unsigned)x_qs[r*sram_stride+8],(unsigned)x_qs[r*sram_stride+16],(unsigned)x_qs[r*sram_stride+24],(unsigned)x_qs[r*sram_stride+32],(unsigned)x_qs[r*sram_stride+40],(unsigned)x_qs[r*sram_stride+48],(unsigned)x_qs[r*sram_stride+56],
                                   (unsigned)x_qs[r*sram_stride+1],(unsigned)x_qs[r*sram_stride+9],(unsigned)x_qs[r*sram_stride+17],(unsigned)x_qs[r*sram_stride+25],(unsigned)x_qs[r*sram_stride+33],(unsigned)x_qs[r*sram_stride+41],(unsigned)x_qs[r*sram_stride+49],(unsigned)x_qs[r*sram_stride+57],
                                   (unsigned)x_qs[r*sram_stride+2],(unsigned)x_qs[r*sram_stride+10],(unsigned)x_qs[r*sram_stride+18],(unsigned)x_qs[r*sram_stride+26],(unsigned)x_qs[r*sram_stride+34],(unsigned)x_qs[r*sram_stride+42],(unsigned)x_qs[r*sram_stride+50],(unsigned)x_qs[r*sram_stride+58],
                                   (unsigned)x_qs[r*sram_stride+3],(unsigned)x_qs[r*sram_stride+11],(unsigned)x_qs[r*sram_stride+19],(unsigned)x_qs[r*sram_stride+27],(unsigned)x_qs[r*sram_stride+35],(unsigned)x_qs[r*sram_stride+43],(unsigned)x_qs[r*sram_stride+51],(unsigned)x_qs[r*sram_stride+59],
                                   (unsigned)x_qs[r*sram_stride+4],(unsigned)x_qs[r*sram_stride+12],(unsigned)x_qs[r*sram_stride+20],(unsigned)x_qs[r*sram_stride+28],(unsigned)x_qs[r*sram_stride+36],(unsigned)x_qs[r*sram_stride+44],(unsigned)x_qs[r*sram_stride+52],(unsigned)x_qs[r*sram_stride+60],
                                   (unsigned)x_qs[r*sram_stride+5],(unsigned)x_qs[r*sram_stride+13],(unsigned)x_qs[r*sram_stride+21],(unsigned)x_qs[r*sram_stride+29],(unsigned)x_qs[r*sram_stride+37],(unsigned)x_qs[r*sram_stride+45],(unsigned)x_qs[r*sram_stride+53],(unsigned)x_qs[r*sram_stride+61],
                                   (unsigned)x_qs[r*sram_stride+6],(unsigned)x_qs[r*sram_stride+14],(unsigned)x_qs[r*sram_stride+22],(unsigned)x_qs[r*sram_stride+30],(unsigned)x_qs[r*sram_stride+38],(unsigned)x_qs[r*sram_stride+46],(unsigned)x_qs[r*sram_stride+54],(unsigned)x_qs[r*sram_stride+62],
                                   (unsigned)x_qs[r*sram_stride+7],(unsigned)x_qs[r*sram_stride+15],(unsigned)x_qs[r*sram_stride+23],(unsigned)x_qs[r*sram_stride+31],(unsigned)x_qs[r*sram_stride+39],(unsigned)x_qs[r*sram_stride+47],(unsigned)x_qs[r*sram_stride+55],(unsigned)x_qs[r*sram_stride+63]);
                            }
                            t4q_count += 16;
                        }
                        if (t4q_count >= 32 && t4q_count < 64) {
                            for (int r = 0; r < 16; ++r) {
                                if (false) printf("[t4qy] k0=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)k0, r,
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+0],(unsigned)y_qs[r*MMQ_TILE_Y_K+1],(unsigned)y_qs[r*MMQ_TILE_Y_K+2],(unsigned)y_qs[r*MMQ_TILE_Y_K+3],(unsigned)y_qs[r*MMQ_TILE_Y_K+4],(unsigned)y_qs[r*MMQ_TILE_Y_K+5],(unsigned)y_qs[r*MMQ_TILE_Y_K+6],(unsigned)y_qs[r*MMQ_TILE_Y_K+7],
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+8],(unsigned)y_qs[r*MMQ_TILE_Y_K+9],(unsigned)y_qs[r*MMQ_TILE_Y_K+10],(unsigned)y_qs[r*MMQ_TILE_Y_K+11],(unsigned)y_qs[r*MMQ_TILE_Y_K+12],(unsigned)y_qs[r*MMQ_TILE_Y_K+13],(unsigned)y_qs[r*MMQ_TILE_Y_K+14],(unsigned)y_qs[r*MMQ_TILE_Y_K+15],
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+16],(unsigned)y_qs[r*MMQ_TILE_Y_K+17],(unsigned)y_qs[r*MMQ_TILE_Y_K+18],(unsigned)y_qs[r*MMQ_TILE_Y_K+19],(unsigned)y_qs[r*MMQ_TILE_Y_K+20],(unsigned)y_qs[r*MMQ_TILE_Y_K+21],(unsigned)y_qs[r*MMQ_TILE_Y_K+22],(unsigned)y_qs[r*MMQ_TILE_Y_K+23],
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+24],(unsigned)y_qs[r*MMQ_TILE_Y_K+25],(unsigned)y_qs[r*MMQ_TILE_Y_K+26],(unsigned)y_qs[r*MMQ_TILE_Y_K+27],(unsigned)y_qs[r*MMQ_TILE_Y_K+28],(unsigned)y_qs[r*MMQ_TILE_Y_K+29],(unsigned)y_qs[r*MMQ_TILE_Y_K+30],(unsigned)y_qs[r*MMQ_TILE_Y_K+31]);
                            }
                            t4q_count += 16;
                        }
                    }
                    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && i == 0 && j == 0 && l == 0 && k0 < 64) {
                        // P1B10 (AUDIT12): all 16 mma C lanes + full x_dm/y_dm smem words, every iteration
                        if (false) printf("[t4c] k0=%d C: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d dmA=%08x dsB=%08x\n",
                               (int)k0,
                               (int)C.x[0],(int)C.x[1],(int)C.x[2],(int)C.x[3],(int)C.x[4],(int)C.x[5],(int)C.x[6],(int)C.x[7],
                               (int)C.x[8],(int)C.x[9],(int)C.x[10],(int)C.x[11],(int)C.x[12],(int)C.x[13],(int)C.x[14],(int)C.x[15],
                               (unsigned)*(const uint32_t*)&x_dm[i*sram_stride + k0/8],
                               (unsigned)*(const uint32_t*)&y_dm[j*MMQ_TILE_Y_K + k01/8]);
                    }
                    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && i == 0 && j == 0 && l == 0 && (k0 == 0 || k0 == 32)) {
                        // P1B10: staged code words - all 16 x rows + 16 y rows per call (k0=0 -> x words 0..7, k0=32 -> x words 32..39)
                        if (k0 == 0) if (false) printf("[t4k] sram_stride=%d tile_yk=%d\n", (int)sram_stride, (int)MMQ_TILE_Y_K);
                        for (int r = 0; r < 16; ++r) {
                            if (false) printf("[t4x1] k0=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)k0, r,
                                   (unsigned)x_qs[r*sram_stride+k0+0],(unsigned)x_qs[r*sram_stride+k0+1],(unsigned)x_qs[r*sram_stride+k0+2],(unsigned)x_qs[r*sram_stride+k0+3],
                                   (unsigned)x_qs[r*sram_stride+k0+4],(unsigned)x_qs[r*sram_stride+k0+5],(unsigned)x_qs[r*sram_stride+k0+6],(unsigned)x_qs[r*sram_stride+k0+7]);
                            if (false) printf("[t4y1] k0=%d row%d : %08x %08x %08x %08x %08x %08x %08x %08x\n", (int)k0, r,
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+k01+0],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+1],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+2],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+3],
                                   (unsigned)y_qs[r*MMQ_TILE_Y_K+k01+4],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+5],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+6],(unsigned)y_qs[r*MMQ_TILE_Y_K+k01+7]);
                        }
                    }
                    if (threadIdx.x == 0 && threadIdx.y == 0 && i == 0 && j == 0 && k0 < 40) {
                        if (false) printf("[t4r] l=%d bx=%d by=%d i=%d j=%d k0=%d k01=%d dmA=(%04x,%04x) dsB=(%04x,%04x) C0=%d\n",
                               (int)l, (int)blockIdx.x, (int)blockIdx.y, (int)i, (int)j,
                               (int)k0, (int)k01,
                               (unsigned)*(const uint16_t*)&dmA.x, (unsigned)*(const uint16_t*)&dmA.y,
                               (unsigned)*(const uint16_t*)&dsB.x, (unsigned)*(const uint16_t*)&dsB.y, (int)C.x[l]);
                    }
#endif
                }
            }
        }
#if DBG_T4K
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
            const int tid = threadIdx.x;
            for (int l = 0; l < 8; ++l) {
                if (false) printf("[t6b] k0=%d t=%d s%d: %.9g\n",
                       (int)(k00 + k01), (int)tid, l, sum[l]);
            }
        }
#endif
    }
#else
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// Used for NVFP4, Q3_K, IQ2_S, and IQ2_XS
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, prec_src1);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * x_df[i*sram_stride + k0/4] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, prec_src1);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*sram_stride + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2] = {};
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[J/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K/2; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 2;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }

    // Some compilers fail to unroll the loop over k01 if there is a conditional statement for ns in the inner loop.
    // As a workaround 2 separate loops are used instead.
#pragma unroll
    for (int k01 = MMQ_TILE_NE_K/2; k01 < MMQ_TILE_NE_K; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 1;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y;
            const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);

            tile_C Cm = {};
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
#pragma unroll
                for (int l = 0; l < tile_A::ne; ++l) {
                    A1.x[l] = 0x01010101;
                }
                mma(Cm, A1, B);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd = {};
                mma(Cd, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const float2 dm = __half22float2(x_dm[i*sram_stride + k0/4]);
                    float tmp = Cd.x[l]*dm.x;
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm.x[l]*dm.y;
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*dB;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= dm.y*sB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*sram_stride + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2] = {};
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2] = {};

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);

#if DBG_T4K
                if (k00 == 0 && k01 == 0 && blockIdx.x == 0 && blockIdx.y == 0 && i == 0 && j == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                    if (false) printf("[t4k] bx=%d by=%d i=%d j=%d\n", (int)blockIdx.x, (int)blockIdx.y, (int)i, (int)j);
                    for (int w4 = 0; w4 < 8; w4++) if (false) printf("[t4k] v[%d]=%08x u[%d]=%08x\n", w4, (unsigned)x_qs[w4], w4, (unsigned)y_qs[w4]);
                    if (false) printf("[t4k] sc=%02x%02x%02x%02x m=%02x%02x%02x%02x dm=%04x ds8_0=%04x ds8_1=%04x\n",
                        sc[0], sc[1], sc[2], sc[3], sc[8], sc[9], sc[10], sc[11],
                        *(const unsigned short*)&x_dm[0], *(const unsigned short*)&y_ds[0], *(const unsigned short*)&y_ds[1]);
                }
#endif
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0),         sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*sram_stride];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2] = {};
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------

// Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.
// Both quantizations encode values as e2m1 (FP4) and produce one uint32 scale per
// m16n8k64 MMA call; only the PTX kind (scale_vec::2X ue8m0 vs scale_vec::4X ue4m3)
// and the per-type stride constant differ.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_fp4_fp4_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q4);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q4);
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            // accumulate in place into the output sum array
            tile_C & C = *reinterpret_cast<tile_C *>(sum + (j0 / tile_C::J + n) * tile_C::ne);
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                mma_block_scaled_fp4<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
            }
        }
    }
}

