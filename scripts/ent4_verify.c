/* ent4_verify: full-stream Huffman decoder for the ent4 format (docs/entropy4.md).
 * Per-tensor: rowstart[] gives the absolute stream bit offset of each row;
 * decode k codes per row from a 16-symbol canonical codebook (lengths16).
 * Verifies row end alignment (pos == rowstart[r+1]) to catch stream/offset skew.
 * Exposed to Python via ctypes (scripts/ent4_verify.py).
 */
#include <stdint.h>
#include <string.h>

#define WID 15

/* 1 = decode error (bad code), 2 = row-end misalignment, 0 = ok.
 * out is indexed by absolute code position (row-major over the tensor). */
int ent4_decode(const uint8_t *stream, uint32_t n_rows, uint32_t k,
               const uint32_t *rowstart, const uint8_t *lengths, uint32_t total_bits,
               uint8_t *out, uint64_t *err_pos, uint32_t *err_row) {
  uint32_t codes[16];
  int order[16], no = 0;
  memset(codes, 0, sizeof codes);
  for (int l = 1; l <= 15; ++l)
    for (int s = 0; s < 16; ++s)
      if (lengths[s] == (uint8_t)l)
        order[no++] = s;
  uint32_t c = 0;
  int prev = 0;
  for (int i = 0; i < no; ++i) {
    int s = order[i];
    int l = lengths[s];
    c <<= l - prev;
    codes[s] = c;
    c++;
    prev = l;
  }

  for (uint32_t r = 0; r < n_rows; ++r) {
    uint64_t pos = rowstart[r];
    for (uint32_t i = 0; i < k; ++i) {
      uint32_t byte = (uint32_t)(pos >> 3), off = (uint32_t)(pos & 7);
      /* MSB-first window: byte 0 = bits 24..31 */
      uint32_t W = ((uint32_t)stream[byte] << 24) | (stream[byte + 1] << 16) |
                  (stream[byte + 2] << 8) | stream[byte + 3];
      uint32_t w15 = (W >> (17 - off)) & 0x7FFF;
      /* match the TOP lengths[s] bits only (lower bits = next code) */
      int sym = -1;
      for (int s = 0; s < 16; ++s) {
        int l = lengths[s];
        if (l && (w15 >> (WID - l)) == codes[s]) { sym = s; break; }
      }
      if (sym < 0) {
        if (err_pos) *err_pos = pos;
        if (err_row) *err_row = r;
        return 1;
      }
      out[r * (uint64_t)k + i] = (uint8_t)sym;
      pos += lengths[sym];
    }
    uint32_t want = (r + 1 < n_rows) ? rowstart[r + 1] : total_bits;
    if (r + 1 < n_rows) {
      /* chunk padding (<=7 bits) may separate this row from the next */
      if (pos > want || (uint64_t)want - pos > 7) {
        if (err_pos) *err_pos = pos;
        if (err_row) *err_row = r;
        return 2;
      }
    } else {
      /* end-of-stream: content end + at most 7 padding bits per chunk */
      uint64_t maxend = (uint64_t)total_bits + 7 * (((uint64_t)n_rows + 4095) / 4096);
      if (pos < total_bits || pos > maxend) {
        if (err_pos) *err_pos = pos;
        if (err_row) *err_row = r;
        return 2;
      }
    }
  }
  return 0;
}
