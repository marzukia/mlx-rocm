// kquant_q4k_math.hpp — shared Q4_K (ggml k-quant) dequantization math.
//
// Single source of truth, used by BOTH:
//   - the ROCm GPU kernel (kquant_q4k.hip), and
//   - the hermetic CPU G1 test (tests/kquant_q4k/kq4k_cpu_dequant.cpp).
//
// Bit-exact contract (G1): given the 144-byte Q4_K block encoding,
// kq_dequant_block must produce, element by element, exactly the values the
// python-gguf 0.19.0 oracle (gguf.dequantize) produces as float32. The math
// below is a 1:1 port of the pinned ggml reference
// (ggml-quants.c @ 584365bc, dequantize_row_q4_K + get_scale_min_k4), which
// was refgate-verified bit-exact in this project's D2 work.
//
// 144-byte block = f16 d, f16 dmin, 12 packed scale bytes, 128 qs bytes;
// 8 sub-blocks of 32 elements. qs is 4 chunks of 32 bytes; chunk t carries
// sub-blocks (2t, 2t+1): low nibble of byte l = element l of sub-block 2t,
// high nibble = element l of sub-block 2t+1. Codes are raw 4-bit values
// 0..15 (offset-carrying form).
//
// Per sub-block j:
//   j < 4 : sc = scales[j] & 63;         m = scales[j+4] & 63
//   j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//          m  = (scales[j+4] >> 4) | ((scales[j-0] >> 6) << 4)
//   d1 = f32(d) * sc;  m1 = f32(dmin) * m;
//   v  = d1 * code - m1        (all float32; no FMA contraction)
//
// The volatile barriers below store each intermediate product to volatile
// memory so the (a*b) - c pattern can never be contracted into an FMA/HFMA;
// the rounding sequence (mul, mul, sub — each rounded once) is preserved
// bit-exactly on GPU and host alike.
#pragma once

#include <cstdint>
#include <cstring>

// Host/device annotation: this header is the single math source for both the
// ROCm GPU kernel (compiled by hipcc) and the hermetic CPU test (g++).
#if defined(__HIP__) || defined(__CUDACC__)
#define KQ_HOSTDEV __host__ __device__
#else
#define KQ_HOSTDEV
#endif

namespace mlx::core::rocm::kquant {

constexpr int kQ4KBlockBytes = 144;   // one 256-element Q4_K block
constexpr int kQ4KBlockElems = 256;   // 8 sub-blocks of 32
constexpr int kQ4KSubElems = 32;      // elements per sub-block
// Offset of the 128 qs bytes inside a 144-byte block.
constexpr int kQ4KQSOffset = 16;      // 2 (d) + 2 (dmin) + 12 (scales)

// Exact (rounding-free) f16 -> f32. Host AND device.
// Normal/inf/nan: pure bit re-biasing. Subnormal: exact integer construction
// (mant * 2^-24 is exactly representable in f32; no rounding ever occurs).
KQ_HOSTDEV inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (uint32_t)(h & 0x7C00u) >> 10;
  uint32_t mant = (uint32_t)(h & 0x03FFu);
  float r;
  if (exp == 0) {
    if (mant == 0) {
      // signed zero
      memcpy(&r, &sign, 4);
      return r;
    }
    // value = mant * 2^-24, mant in [1, 1023] (b = position of top bit:
    // 2^b <= mant < 2^(b+1)).
    int b = 0;
    while (b < 10 && (1u << (b + 1)) <= mant) {
      ++b;
    }
    uint32_t M = mant << (23 - b);          // in [2^23, 2^24)
    uint32_t e = (uint32_t)(b + 103);       // field: E=b-24, field=E+127
    uint32_t bits = sign | (e << 23) | (M - 0x00800000u);
    memcpy(&r, &bits, 4);
    return r;
  } else if (exp == 0x1F) {
    if (mant == 0) {
      uint32_t inf = sign | 0x7F800000u;
      memcpy(&r, &inf, 4);
    } else {
      uint32_t bits = sign | 0x7FC00000u | (mant << 13);
      memcpy(&r, &bits, 4);
    }
    return r;
  }
  uint32_t bits = sign | (((exp - 0x0Fu) + 0x7Fu) << 23) | (mant << 13);
  memcpy(&r, &bits, 4);
  return r;
}

// Decode sub-block j (0..7) scale/min from the 12 packed scale bytes
// (bit-packed get_scale_min_k4, pinned ggml reference).
KQ_HOSTDEV inline void get_scale_min(const uint8_t* __restrict__ scales,
                          int j,
                          int* __restrict__ sc,
                          int* __restrict__ m) {
  if (j < 4) {
    *sc = scales[j] & 63;
    *m = scales[j + 4] & 63;
  } else {
    *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// Dequantize one element e (0..255) of one 144-byte Q4_K block.
// `block` must point at 144 consecutive bytes (any alignment).
// Element order matches the row-major flattened tensor:
//   chunk t = e/64 (0..3), l = e%32;
//   sub-block 2t carries elements [64t, 64t+32) via the LOW nibble of
//   qs byte l; sub-block 2t+1 carries [64t+32, 64t+64) via the HIGH nibble.
// This is the single math both the 256-element batch function below and
// the one-thread-per-element GPU kernel use (one source of truth).
KQ_HOSTDEV inline float dequant_elem(const uint8_t* __restrict__ block, int e) {
  const uint16_t d_raw =
      (uint16_t)(block[0]) | (uint16_t(block[1]) << 8);
  const uint16_t dmin_raw =
      (uint16_t)(block[2]) | (uint16_t(block[3]) << 8);
  const float d = f16_to_f32(d_raw);
  const float dmin = f16_to_f32(dmin_raw);

  const int t = e / 64;
  const int j = 2 * t + ((e & 32) != 0 ? 1 : 0);
  const int l = e & 31;
  const uint8_t byte = block[kQ4KQSOffset + 32 * t + l];
  const int code = (e & 32) ? (byte >> 4) : (byte & 0xF);

  int sc, m;
  get_scale_min(block + 4, j, &sc, &m);
  volatile float d1 = d * sc; // round 1
  volatile float m1 = dmin * m; // round 1
  volatile float p1 = d1 * (float)(code); // round 2
  return p1 - m1; // round 3
}

// Dequantize one 144-byte Q4_K block into 256 float32 values.
// `out` must point at 256 consecutive floats.
KQ_HOSTDEV inline void dequant_block(const uint8_t* __restrict__ block,
                          float* __restrict__ out) {
  for (int e = 0; e < 256; ++e) {
    out[e] = dequant_elem(block, e);
  }
}

} // namespace mlx::core::rocm::kquant
