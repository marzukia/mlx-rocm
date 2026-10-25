/*
 * mlx-rocm MVP — header-only host reference kernels (no dependencies).
 *
 * Used by the smoke driver for (a) the CPU GEMM reference and (b) the fp8
 * self-test. bf16 packing is round-to-nearest-even; e4m3 decode follows the
 * OCP FP8 e4m3 spec (bias 7, no Inf, NaN = 0x7F).
 */
#ifndef MLX_ROCM_GEMM_REF_H
#define MLX_ROCM_GEMM_REF_H

#include <cstdint>
#include <cmath>

namespace mlx_rocm::gemm_ref {

inline float bf16_to_f32(std::uint16_t x) {
  union {
    std::uint32_t u;
    float f;
  } t;
  t.u = (std::uint32_t)x << 16;
  return t.f;
}

inline std::uint16_t f32_to_bf16(float f) {
  union {
    float f;
    std::uint32_t u;
  } t;
  t.f = f;
  const std::uint32_t lsb = (t.u >> 16) & 1u;
  return (std::uint16_t)((t.u + 0x7FFFu + lsb) >> 16); // RNE
}

inline float fp8e4m3_to_f32(std::uint8_t x) {
  const float s = (x & 0x80u) ? -1.0f : 1.0f;
  const int e = (x >> 3) & 0xF; // 4-bit exponent (e4m3: 1s + 4e + 3m)
  const int m = x & 0x7;        // 3-bit mantissa
  if (e == 0xF && m == 0x7)
    return std::nanf(""); // 0x7F/0xFF = NaN (no Inf in e4m3fn)
  if (e == 0) {
    // subnormal: 0.mmm * 2^-6  ==  m * 2^-9
    return s * (float)m * 0x1p-9f;
  }
  return s * (1.0f + (float)m * 0x1p-3f) * std::ldexpf(1.0f, e - 7);
}

/* D[m,n] = A[m,k] * B[k,n], bf16 in/out, f32 accumulation. */
inline void gemm_host_bf16(const std::uint16_t *A, const std::uint16_t *B,
                           std::uint16_t *D, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      const std::uint16_t *ar = A + (std::int64_t)i * k;
      for (int l = 0; l < k; ++l) {
        acc += bf16_to_f32(ar[l]) * bf16_to_f32(B[(std::int64_t)l * n + j]);
      }
      D[(std::int64_t)i * n + j] = f32_to_bf16(acc);
    }
  }
}

/* Deterministic LCG (SplitMix64-style) in [-1, 1) as float. */
inline float lcg_f32(std::uint64_t &state) {
  state ^= state << 31;
  state ^= state >> 31;
  state ^= state << 19;
  std::uint32_t r = (std::uint32_t)(state >> 40);
  return (float)((int)(r % 20001) - 10000) / 10000.0f;
}

} // namespace mlx_rocm::gemm_ref

#endif // MLX_ROCM_GEMM_REF_H
