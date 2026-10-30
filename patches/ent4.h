// ent4.h — lossless compressed 4-bit weight stream, in-CU decode (Phase 3,
// lever 1). Per-tensor shared canonical Huffman over 16 4-bit code values.
//
// The 4-bit weight buffer holds the *compressed* stream (leading comp_bytes);
// the dense tail is never read. On the qmv path, a decode kernel expands the
// stream into a dense [N, K/2] scratch (bit-identical code values — the same
// decoder the offline C verifier used, verified 250/250 on qwen3-9b), and the
// existing wide qmv kernel reads the scratch. Lossless by construction.
//
// Gated by MLX_ROCM_ENTROPY=1 (+ MLX_ROCM_ENTROPY_FILE=/path/to/model.ent4).
// Default path and CI are untouched.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "mlx/allocator.h"
#include "mlx/backend/rocm/allocator.h"

namespace mlx {
namespace core {
namespace rocm {
namespace ent4 {

// ---------------------------------------------------------------------------
// Host payload: compressed stream + side tables for one 4-bit tensor.
// ---------------------------------------------------------------------------
struct Payload {
  std::vector<uint8_t> stream;       // byte-aligned per 4096-row chunk (+3 pad, see reader)
  std::vector<uint32_t> row_start;   // absolute stream-bit offset per row
  uint8_t code_len[16] = {0};        // 0 = absent
  uint32_t n_rows = 0;
  uint32_t k = 0;
  uint32_t comp_bytes = 0;
  uint32_t total_bits = 0;
  bool plain = false;                // 1 = canonical 4-bit (above entropy gate)
};

// Device registration (side tables uploaded to VRAM, kept alive here).
struct Reg {
  const uint8_t* stream = nullptr;   // == dense weight buffer (leading comp_bytes)
  uint32_t* row_start_dev = nullptr;
  const uint8_t* lut_dev = nullptr;  // 32768 u8 (15-bit -> symbol)
  uint16_t* code_len_dev = nullptr;  // 16 u16
  uint32_t n_rows = 0, k = 0, comp_bytes = 0, total_bits = 0;
  std::unique_ptr<allocator::Buffer> row_start_keep;
  std::unique_ptr<allocator::Buffer> lut_keep;
  std::unique_ptr<allocator::Buffer> code_len_keep;
};

class Registry {
 public:
  static Registry& instance() {
    static Registry r;
    return r;
  }
  void register_(uintptr_t w, Reg&& r) {
    std::lock_guard<std::mutex> g(mu_);
    m_[w] = std::move(r);
  }
  const Reg* lookup(uintptr_t w) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = m_.find(w);
    return it == m_.end() ? nullptr : &it->second;
  }
  size_t size() const {
    std::lock_guard<std::mutex> g(mu_);
    return m_.size();
  }

 private:
  Registry() = default;
  mutable std::mutex mu_;
  std::unordered_map<uintptr_t, Reg> m_;
};

// (The device decode kernels live in ent4_kernel.hip, compiled in the .hip
// translation unit (qmm.hip). load.cpp is a plain C++ translation unit and
// includes only this host-only header.)
// ---------------------------------------------------------------------------
// Persistent scratch (one buffer, reused across ops on the serial stream).
// ---------------------------------------------------------------------------
inline uint8_t* scratch(size_t needed) {
  static std::unique_ptr<allocator::Buffer> buf;
  static size_t cap = 0;
  if (needed > cap) {
    buf = std::make_unique<allocator::Buffer>(allocator::malloc(needed));
    cap = needed;
  }
  return static_cast<uint8_t*>(static_cast<RocmBuffer*>(buf->ptr())->data);
}

// The .ent4 file reader + lazy env-gated table (ent4_table) live in
// ent4_json.h (they need nlohmann::json, whose include path is only available
// to the main 'mlx' target, not the mlx_rocm_kernels_lib target that compiles
// qmm.hip).

}  // namespace ent4
}  // namespace rocm
}  // namespace core
}  // namespace mlx
