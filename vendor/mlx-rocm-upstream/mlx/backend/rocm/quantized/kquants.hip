// Copyright © 2025 Apple Inc.
//
// WS3: K-quant dequantization — Q2_K / Q5_K / Q6_K (+ Q8_0) — from the
// compressed GGUF byte stream to f32, on the GPU, at GEMM time.
//
// Bit-exact port of dequantize_row_q2_K / dequantize_row_q5_K /
// dequantize_row_q6_K / dequantize_row_q8_0 from ggml-quants.c (llama.cpp
// 6023692, the pinned ref). No re-quantization: raw compressed bytes in,
// f32 out. Block geometry is format-level constants (not model-specific):
//   Q8_0:  34 B per 32 elems   (d f16 @0, 32 x 1B @2)
//   Q2_K:  84 B per 256 elems  (scales 16 x u8 @0, qs 64 x 2-bit @16,
//                               d f16 @80, dmin f16 @82)
//   Q5_K: 176 B per 256 elems  (d @0, dmin @2, scales 12B k4-packed @4,
//                               qh 32 x 1-bit @16, qs 128 x 4-bit @48)
//   Q6_K: 210 B per 256 elems  (ql 128 x 4-bit @0, qh 64 x 2-bit @128,
//                               scales 16 x SIGNED i8 @192, d f16 @208)
//
// ktype codes follow gguflib.h (== the pinned GGUF file's own header, ==
// python-gguf 0.19): Q8_0 = 8, Q2_K = 10, Q5_K = 13, Q6_K = 14.
//
// Every output element is computed with the reference's exact f32 op order
// (all intermediate values — f16 scales, 0..63/0..15/0..3 scale bytes,
// -32..31 weights — convert to f32 exactly, so the only rounding is the
// same multiplications/additions the C reference performs, in the same
// order). One thread per compressed block.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/backend/rocm/quantized/kquants.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <cstring>

namespace mlx::core {

namespace rocm {

// Exact f16 -> f32 widening (no rounding possible; f16 has a subset of the
// f32 patterns). Matches GGML_FP16_TO_FP32. Device-side (manual bit
// rearrangement, no conversion library needed).
__device__ inline float f16to32(const uint8_t* p) {
  uint32_t h;
  std::memcpy(&h, p, 2);
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0x1Fu) {
    // Inf (mant == 0) / NaN: spec widening — payload is mant << 13,
    // which preserves the quiet/signaling bit exactly (bit-exact with
    // the IEEE f16->f32 conversion).
    bits = (sign << 31) | 0x7F800000u | (mant << 13);
  } else if (exp == 0u) {
    if (mant == 0) {
      bits = sign << 31;  // signed zero
    } else {
      // Subnormal: mant * 2^-24. Normalize mant (leading bit position p),
      // then f32 exponent p + 103 (103 = 127 - 24), fractional mantissa.
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

__global__ void kq_dequant_q8_0(const uint8_t* __restrict__ in,
                               float* __restrict__ out,
                               int nblocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const uint8_t* blk = in + (size_t)b * 34;
  float d = f16to32(blk);
  float* o = out + (size_t)b * 32;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    // C: y[i*32+j] = x[i].qs[j] * d;  qs is int8 (two's complement)
    o[i] = (int8_t)blk[2 + i] * d;
  }
}

__global__ void kq_dequant_q2k(const uint8_t* __restrict__ in,
                              float* __restrict__ out,
                              int nblocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const uint8_t* blk = in + (size_t)b * 84;
  float d = f16to32(blk + 80);
  float minv = f16to32(blk + 82);
  const uint8_t* sc = blk;       // 16 scale/min bytes @0 (pinned ggml-common.h:
                                 // scales first, then qs, then d/dmin)
  const uint8_t* q = blk + 16;   // qs: 64 B (2 bit / elem)
  float* o = out + (size_t)b * 256;
  int is = 0;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      int s = sc[is++];
      float dl = d * (s & 0xF);
      float ml = minv * (s >> 4);
      #pragma unroll
      for (int l = 0; l < 16; ++l) {
        // C: *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
        o[n + l] = dl * ((q[l] >> shift) & 3) - ml;
      }
      s = sc[is++];
      dl = d * (s & 0xF);
      ml = minv * (s >> 4);
      #pragma unroll
      for (int l = 0; l < 16; ++l) {
        o[n + 16 + l] = dl * ((q[16 + l] >> shift) & 3) - ml;
      }
      shift += 2;
    }
    q += 32;
  }
}

__global__ void kq_dequant_q5k(const uint8_t* __restrict__ in,
                              float* __restrict__ out,
                              int nblocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const uint8_t* blk = in + (size_t)b * 176;
  float d = f16to32(blk + 0);
  float minv = f16to32(blk + 2);
  const uint8_t* sc = blk + 4;   // 12 scale/min bytes (k4 packing)
  const uint8_t* qh = blk + 16;  // 32 bytes, 1 bit / elem
  const uint8_t* ql = blk + 48;  // 128 bytes, 4 bit / elem
  float* o = out + (size_t)b * 256;
  // C: get_scale_min_k4(j, q) — 4-bit-packing of 12 x 6-bit (scale, min)
  // pairs over a 12-byte field.
  auto k4 = [&](int j) -> std::pair<int, int> {
    if (j < 4) {
      return {(int)(sc[j] & 63), (int)(sc[j + 4] & 63)};
    }
    return {(int)((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)),
            (int)((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))};
  };
  int is = 0;
  unsigned u1 = 1, u2 = 2;
  for (int j = 0; j < 256; j += 64) {
    auto s1 = k4(is + 0);
    float d1 = d * s1.first;
    float m1 = minv * s1.second;
    auto s2 = k4(is + 1);
    float d2 = d * s2.first;
    float m2 = minv * s2.second;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
      // C: d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1
      o[j + l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    }
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
      o[j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    }
    is += 2;
    ql += 32;
    u1 <<= 2;
    u2 <<= 2;
  }
}

__global__ void kq_dequant_q6k(const uint8_t* __restrict__ in,
                              float* __restrict__ out,
                              int nblocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const uint8_t* blk = in + (size_t)b * 210;
  float d = f16to32(blk + 208);
  const uint8_t* ql = blk;         // 128 B, 4 bit / elem
  const uint8_t* qh = blk + 128;   // 64 B, 2 bit / elem
  const int8_t* sc =
      reinterpret_cast<const int8_t*>(blk + 192);  // 16 SIGNED scale bytes
  float* o = out + (size_t)b * 256;
  for (int n = 0; n < 256; n += 128) {
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
      int is = l / 16;
      // C: (int8_t)(...) - 32, then y = d * sc[is + k] * q
      int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      o[n + l] = d * (float)sc[is + 0] * (float)q1;
      o[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
      o[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
      o[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
    }
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// One thread per compressed block. Returns 0 on success.
inline int launch_kq_dequant(
    const uint8_t* in,
    float* out,
    size_t nblocks,
    int ktype,
    hipStream_t stream) {
  int nblocks_i = static_cast<int>(nblocks);
  int nwb = (nblocks_i + 255) / 256;
  if (ktype == 8) {
    kq_dequant_q8_0<<<nwb, 256, 0, stream>>>(in, out, nblocks_i);
  } else if (ktype == 10) {
    kq_dequant_q2k<<<nwb, 256, 0, stream>>>(in, out, nblocks_i);
  } else if (ktype == 13) {
    kq_dequant_q5k<<<nwb, 256, 0, stream>>>(in, out, nblocks_i);
  } else if (ktype == 14) {
    kq_dequant_q6k<<<nwb, 256, 0, stream>>>(in, out, nblocks_i);
  } else {
    return 1;
  }
  return 0;
}

}  // namespace rocm

namespace fast {

std::vector<Shape> KQuantDequantize::output_shapes(
    const std::vector<array>& inputs) {
  auto g = rocm::kq_block_geometry(ktype_);
  if (g.first < 0) {
    throw std::invalid_argument(
        "[KQuantDequantize] ktype must be 8 (Q8_0), 10 (Q2_K), 13 (Q5_K) "
        "or 14 (Q6_K).");
  }
  if (inputs[0].size() % (size_t)g.first != 0) {
    throw std::invalid_argument(
        "[KQuantDequantize] byte count is not a multiple of the block size.");
  }
  size_t nblocks = inputs[0].size() / (size_t)g.first;
  return {Shape{static_cast<int>(nblocks * (size_t)g.second)}};
}

bool KQuantDequantize::is_equivalent(const Primitive& other) const {
  const KQuantDequantize& a_other =
      static_cast<const KQuantDequantize&>(other);
  return ktype_ == a_other.ktype_;
}

void KQuantDequantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  // One thread per compressed block; the input must be a contiguous
  // {nblocks, block_bytes} (or flat) uint8 stream.
  std::unique_ptr<array> tmp;
  const array* px = &inputs[0];
  if (!px->flags().row_contiguous) {
    tmp = std::unique_ptr<array>(
        new array(mlx::core::contiguous_copy_gpu(*px, s)));
    encoder.add_temporary(*tmp);
    px = tmp.get();
  }
  auto& x = *px;
  auto& out = outputs[0];
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));

  auto g = rocm::kq_block_geometry(ktype_);
  if (g.first < 0) {
    throw std::invalid_argument(
        "[kquant_dequantize] ktype must be 8 (Q8_0), 10 (Q2_K), 13 (Q5_K) "
        "or 14 (Q6_K).");
  }
  size_t nblocks = x.size() / (size_t)g.first;
  const uint8_t* xdata = gpu_ptr<uint8_t>(x);
  float* outdata = gpu_ptr<float>(out);
  encoder.launch_kernel([&](hipStream_t stream) {
    if (rocm::launch_kq_dequant(xdata, outdata, nblocks, ktype_, stream) != 0) {
      throw std::runtime_error("kquant_dequantize: launch failed");
    }
  });
}

}  // namespace fast

}  // namespace mlx::core
