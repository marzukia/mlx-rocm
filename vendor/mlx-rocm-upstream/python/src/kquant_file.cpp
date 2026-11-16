// Copyright © 2026. File-layout k-quant op bindings (dequant-at-GEMM family).
//
// Exposes the W1-build file-layout k-quant ops as `mlx.core.kquant_file`:
// GEMM / gather / standalone dequant, each consuming the raw 256-element
// super-block stream in the FILE's grouping (w shape (K, N/256, BB),
// uint8) with the per-tensor layout parameters (mode/axis/group/
// keep_ends/pairs) from the pinned tables — dequantized inside the op,
// no weight materialization.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/kquant_file_gemm.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nanobind::literals;  // arg-name literal operator""_a

void init_kquant_file(nb::module_& m) {
  auto sub = m.def_submodule(
      "kquant_file",
      "File-layout k-quant ops (dequant-at-GEMM): GEMM / gather / "
      "dequant of the raw 256-element super-block stream in the file's "
      "grouping, with the per-tensor window layout parameters.");

  sub.def(
      "kquant_file_gemm",
      [](const mx::array& x,
         const mx::array& w,
         int ktype,
         int mode,
         int axis,
         int group,
         int keep_ends,
         int pairs) {
        return mx::kquant_file_gemm(
            x, w, ktype, mode, axis, group, keep_ends, pairs);
      },
      "x"_a, "w"_a, "ktype"_a, "mode"_a, "axis"_a, "group"_a, "keep_ends"_a,
      "pairs"_a,
      R"pbdoc(
y = x @ W with W dequantized IN-KERNEL from the file's raw 256-element
super-block stream: x (..., M, K) f32/bf16 (N-D leading dims fold into
M), w (K, N/256, block_bytes) uint8, y (..., M, N) in x's dtype. ktype: 8 = Q8_0 (BB 272), 12 = Q4_K (BB 144),
13 = Q5_K (BB 176), 14 = Q6_K (BB 210). mode: 0 = identity, 1 =
mid_transpose (the 32-unit 4096-element window even/odd map). f32
accumulation in k-order; per-element dequant bit-exact vs the pinned GGUF
oracle (G1).
)pbdoc");

  sub.def(
      "kquant_file_gather",
      [](const mx::array& idx,
         const mx::array& w,
         int ktype,
         int mode,
         int axis,
         int group,
         int keep_ends,
         int pairs,
         std::string out_dtype) {
        mx::Dtype dt = out_dtype == "f32" ? mx::float32 : mx::bfloat16;
        return mx::kquant_file_gather(
            idx, w, ktype, mode, axis, group, keep_ends, pairs, dt);
      },
      "idx"_a, "w"_a, "ktype"_a, "mode"_a, "axis"_a, "group"_a,
      "keep_ends"_a, "pairs"_a, "out_dtype"_a = "f32",
      R"pbdoc(
Gather of W's columns (the embedding-lookup sibling of the GEMM op):
idx (B) int32, w (K, N/256, block_bytes) uint8, y (B, K) in out_dtype
("f32" or "bf16"): y[b, j] = W[j, idx[b]] (the row of W^T). A lookup
is not a GEMM — its own op; no weight materialization.
)pbdoc");

  sub.def(
      "kquant_file_dequant",
      [](const mx::array& w,
         int ktype,
         int mode,
         int axis,
         int group,
         int keep_ends,
         int pairs) {
        return mx::kquant_file_dequant(
            w, ktype, mode, axis, group, keep_ends, pairs);
      },
      "w"_a, "ktype"_a, "mode"_a, "axis"_a, "group"_a, "keep_ends"_a,
      "pairs"_a,
      R"pbdoc(
Standalone dequantization in the file layout (the kernel's dequant path,
exposed for G1 bit-exact verification): w (K, N/256, block_bytes) uint8
-> (K, N) f32, in the [K rows][N/256 super-blocks] element order, with
the window map applied.
)pbdoc");
}
