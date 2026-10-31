/*
 * mlx-rocm MVP — minimal hipBLASLt GEMM (C ABI).
 *
 * Provenance: written for the mlx-rocm MVP skeleton (ticket 405), modeled on
 * the persistent-handle + heuristic-cache design of
 * vendor/mlx-rocm-upstream/mlx/backend/rocm/gemms/hipblaslt_gemm.cpp
 * (the vendor ref @ rocm-current; forward-merge of the ROCm base).
 *
 * Conventions:
 *   - All GEMMs are row-major:  D[m,n] = A[m,k] * B[k,n]
 *   - *_hip_*hostio* functions take host pointers; the library does H2D/D2H.
 *   - bf16 payloads are uint16_t bit patterns (IEEE bfloat16).
 *   - fp8 payloads are uint8_t e4m3 code values (0x7F/0xFF are NaN, avoid).
 *   - Return: 0 = success, non-zero = failure (diagnostic on stderr).
 */
#ifndef MLX_ROCM_GEMM_H
#define MLX_ROCM_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Host (CPU) reference GEMM: f32 accumulation, bf16 in/out. Deterministic. */
int mlx_rocm_gemm_host_bf16(const uint16_t *A, const uint16_t *B, uint16_t *D,
                            int m, int n, int k);

/* GPU GEMM (bf16 in/out), host I/O convenience wrapper. Requires a build
 * with MLX_ROCM_WITH_HIP; returns 2 if built without ROCm support. */
int mlx_rocm_gemm_hip_bf16_hostio(const uint16_t *A, const uint16_t *B,
                                  uint16_t *D, int m, int n, int k);

/* GPU GEMM (fp8 e4m3 in, bf16 out), host I/O convenience wrapper.
 * The known gfx1201 fp8 GEMM driver bug (garbled long-context prefill,
 * upstream hipblaslt_gemm.cpp) may surface here as a numeric mismatch;
 * set MLX_ROCM_GEMM_BF16=1 to force bf16 in the serving stack. */
int mlx_rocm_gemm_hip_fp8_hostio(const uint8_t *A, const uint8_t *B,
                                 uint16_t *D, int m, int n, int k);

/* Self-test: runs a small fp8 GEMM on device and cross-checks it against the
 * host reference. 0 = fp8 GEMM looks healthy, 1 = mismatch/unsupported. */
int mlx_rocm_gemm_fp8_probe(void);

/* Number of HIP devices (0 = none / no ROCm build). */
int mlx_rocm_hip_device_count(void);

/* Recommended input precision for this build/device:
 * "fp8" unless MLX_ROCM_GEMM_BF16=1, in which case "bf16". */
const char *mlx_rocm_preferred_precision(void);

/* Backend one-liner, e.g. "hip=1 n_dev=2 gfx1201 workspace=32MB". */
int mlx_rocm_backend_info(char *buf, int buflen);

#ifdef __cplusplus
}
#endif

#endif /* MLX_ROCM_GEMM_H */
