// ent4_kernel.hip — in-CU Huffman decode for the ent4 compressed weight stream
// (docs/entropy4.md). Device-only; compiled in the .hip translation unit that
// includes it (qmm.hip). See ent4.h for the host side.
//
// Decode is a single LUT lookup: the host builds a 15-bit -> symbol table for
// the tensor's canonical codebook (prefix-free, so the entry at any code
// boundary is unambiguous). The device does NO canonical computation and NO
// symbol loop — a fixed 15-bit window index, one read, one length add. This
// sidesteps a ROCm/hipcc codegen bug that mis-compiled the naive
// length-sorted symbol loop (it matched the wrong symbol on the first code).
//
// Output is the DENSE 4bit layout the wide qmv kernel reads: k/2 bytes per
// row, byte j = (code[2j]<<4) | code[2j+1] (high nibble first — the same
// order the compressor's unpack_u32 used, so repacking is bit-exact).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace mlx {
namespace core {
namespace rocm {
namespace ent4 {

// Decode one symbol from the MSB-first 15-bit window at bit position `pos`.
// `lut` is the host-built 15-bit -> symbol table (0..15, 255 = no match).
// Advances `pos` by the matched symbol's code length.
__device__ __forceinline__ uint32_t ent4_decode_one(
    const uint8_t* __restrict__ stream, uint64_t& pos,
    const uint8_t* __restrict__ lut, const uint16_t* __restrict__ code_len) {
  uint32_t byte = static_cast<uint32_t>(pos >> 3),
           off = static_cast<uint32_t>(pos & 7);
  uint32_t W = (static_cast<uint32_t>(stream[byte]) << 24) |
              (static_cast<uint32_t>(stream[byte + 1]) << 16) |
              (static_cast<uint32_t>(stream[byte + 2]) << 8) |
              static_cast<uint32_t>(stream[byte + 3]);
  uint32_t w15 = (W >> (17 - off)) & 0x7FFF;
  uint32_t sym = lut[w15];
  if (sym < 16u) pos += code_len[sym];
  return sym;
}

// Decode one row (k codes) from the compressed stream into the packed dense
// layout (k/2 bytes). `out` points at row*(k/2).
__device__ __forceinline__ void ent4_decode_row(
    const uint8_t* __restrict__ stream,
    const uint32_t* __restrict__ row_start,
    const uint8_t* __restrict__ lut,
    const uint16_t* __restrict__ code_len,
    int row, uint32_t k, uint8_t* __restrict__ out) {
  uint64_t pos = row_start[row];
  uint32_t j = 0;
  for (uint32_t i = 0; i < k; i += 2) {
    uint32_t c0 = ent4_decode_one(stream, pos, lut, code_len);
    uint32_t c1 = ent4_decode_one(stream, pos, lut, code_len);
    out[j++] = static_cast<uint8_t>((c0 << 4) | (c1 & 0xF));
  }
}

// Full-layer decode: one thread per row -> dense [n_rows, k/2] scratch.
__global__ void ent4_decode_kernel(const uint8_t* __restrict__ stream,
                                  const uint32_t* __restrict__ row_start,
                                  const uint8_t* __restrict__ lut,
                                  const uint16_t* __restrict__ code_len,
                                  uint8_t* __restrict__ scratch, uint32_t n_rows,
                                  uint32_t k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows)
    ent4_decode_row(stream, row_start, lut, code_len, row, k,
                   scratch + static_cast<size_t>(row) * (k >> 1));
}

}  // namespace ent4
}  // namespace rocm
}  // namespace core
}  // namespace mlx
