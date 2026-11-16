// Copyright © 2025 Apple Inc.
//
// WS3: K-quant block geometry (format-level constants, shared by the op in
// fast.cpp and the launcher in kquants.hip).

#pragma once

#include <utility>

namespace mlx::core {

namespace rocm {

// ktype follows the GGUF type codes (gguflib.h == the pinned GGUF file's
// own header == python-gguf 0.19): Q8_0 = 8, Q2_K = 10, Q5_K = 13,
// Q6_K = 14. Returns (bytes per compressed block, elems per block);
// (-1, -1) for an unknown ktype.
//   Q8_0:  34 B per 32 elems   (d f16, 32 x 1B weights)
//   Q2_K:  84 B per 256 elems  (64 x 2-bit, 16 x u8 scales, d, dmin)
//   Q5_K: 176 B per 256 elems  (d, dmin, 12B k4 scales, 32 x 1-bit,
//                               128 x 4-bit)
//   Q6_K: 210 B per 256 elems  (128 x 4-bit, 64 x 2-bit, 16 x SIGNED
//                               i8 scales, d f16)
inline std::pair<int, int> kq_block_geometry(int ktype) {
  switch (ktype) {
    case 8:
      return {34, 32};
    case 10:
      return {84, 256};
    case 13:
      return {176, 256};
    case 14:
      return {210, 256};
    default:
      return {-1, -1};
  }
}

}  // namespace rocm

}  // namespace mlx::core
