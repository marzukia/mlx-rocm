// ent4_json.h — .ent4 file reader + lazy env-gated name->Payload table.
// Needs nlohmann::json (include path only available to the main 'mlx' target),
// so it is included ONLY by plain-C++ translation units (load.cpp,
// io/safetensors.cpp) — never by qmm.hip (mlx_rocm_kernels_lib target).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <json.hpp>

#include "mlx/backend/rocm/quantized/ent4.h"

namespace mlx {
namespace core {
namespace rocm {
namespace ent4 {

// Build the 15-bit -> symbol decode LUT for a canonical codebook from 16 code
// lengths. Host-side (canonical computation is correct on the host; the device
// codegen of the naive symbol loop is buggy, so we ship the finished table).
// lut[w] = the symbol whose code is a prefix of the 15-bit window w
// (unambiguous at code boundaries because the codebook is prefix-free), or 255.
inline void build_lut(const uint8_t* code_len, uint8_t lut[32768]) {
  uint32_t codes[16] = {0};
  int order[16], no = 0;
  for (int l = 1; l <= 15; ++l)
    for (int s = 0; s < 16; ++s)
      if (code_len[s] == l) order[no++] = s;
  uint32_t c = 0;
  int prev = 0;
  for (int i = 0; i < no; ++i) {
    int s = order[i];
    int l = code_len[s];
    c <<= (l - prev);
    codes[s] = c;
    c++;
    prev = l;
  }
  for (int i = 0; i < 32768; ++i) lut[i] = 255;
  for (int s = 0; s < 16; ++s) {
    int l = code_len[s];
    if (l == 0) continue;
    uint32_t base = codes[s] << (15 - l);
    for (uint32_t o = 0; o < (1u << (15 - l)); ++o) lut[base + o] = (uint8_t)s;
  }
}

// .ent4 file reader (host). Format: docs/entropy4.md. Returns the tensors in
// file order; the caller zips them with the .index.json names.
inline bool load_ent4_file(
    const std::string& path,
    std::vector<std::shared_ptr<Payload>>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f)
    return false;
  uint8_t magic[16];
  if (fread(magic, 1, 16, f) != 16 || memcmp(magic, "MLXENT4V0", 9) != 0) {
    fclose(f);
    return false;
  }
  uint32_t version = 0, n_tensors = 0;
  (void)fread(&version, 4, 1, f);
  (void)fread(&n_tensors, 4, 1, f);
  bool ok = true;
  out.clear();
  for (uint32_t t = 0; t < n_tensors; ++t) {
    uint32_t n_rows, k, comp_bytes, total_bits;
    uint8_t plain;
    if (fread(&n_rows, 4, 1, f) != 1 || fread(&k, 4, 1, f) != 1 ||
        fread(&comp_bytes, 4, 1, f) != 1 || fread(&total_bits, 4, 1, f) != 1) {
      ok = false;
      break;
    }
    if (fread(&plain, 1, 1, f) != 1) {
      ok = false;
      break;
    }
    auto p = std::make_shared<Payload>();
    p->n_rows = n_rows;
    p->k = k;
    p->comp_bytes = comp_bytes;
    p->total_bits = total_bits;
    p->plain = plain;
    if (fread(p->code_len, 1, 16, f) != 16) {
      ok = false;
      break;
    }
    p->row_start.resize(n_rows);
    if (fread(p->row_start.data(), 4, n_rows, f) != n_rows) {
      ok = false;
      break;
    }
    // +3 zero bytes: the 15-bit window decoder reads a 4-byte window and the
    // last code's window can extend up to 3 bytes past the content end
    // (t1044 8ec0580).
    p->stream.resize(comp_bytes + 3);
    if (comp_bytes &&
        fread(p->stream.data(), 1, comp_bytes, f) != comp_bytes) {
      ok = false;
      break;
    }
    out.push_back(std::move(p));
  }
  fclose(f);
  return ok && out.size() == n_tensors;
}

// Lazy, env-gated: MLX_ROCM_ENTROPY=1 + MLX_ROCM_ENTROPY_FILE. Builds the
// name -> Payload map once (zips file-order tensors with .index.json names).
inline const std::unordered_map<std::string, std::shared_ptr<Payload>>*
ent4_table() {
  static std::unordered_map<std::string, std::shared_ptr<Payload>> table;
  static bool done = false;
  if (!done) {
    done = true;
    const char* en = std::getenv("MLX_ROCM_ENTROPY");
    const char* fp = std::getenv("MLX_ROCM_ENTROPY_FILE");
    if (!en || en[0] != '1' || !fp)
      return &table;
    std::vector<std::shared_ptr<Payload>> payloads;
    if (!load_ent4_file(fp, payloads)) {
      std::fprintf(stderr, "[ent4] failed to read %s\n", fp);
      return &table;
    }
    std::string idx = std::string(fp) + ".index.json";
    std::ifstream in(idx);
    if (!in) {
      std::fprintf(stderr, "[ent4] failed to open %s\n", idx.c_str());
      return &table;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, false);
    if (j.is_discarded() || !j.contains("tensors")) {
      std::fprintf(stderr, "[ent4] bad index %s\n", idx.c_str());
      return &table;
    }
    auto& names = j["tensors"];
    if (names.size() != payloads.size()) {
      std::fprintf(stderr,
                  "[ent4] index/file tensor count mismatch (%zu vs %zu)\n",
                  names.size(), payloads.size());
      return &table;
    }
    for (size_t i = 0; i < payloads.size(); ++i)
      table.emplace(names[i].get<std::string>(), payloads[i]);
  }
  return &table;
}

}  // namespace ent4
}  // namespace rocm
}  // namespace core
}  // namespace mlx
