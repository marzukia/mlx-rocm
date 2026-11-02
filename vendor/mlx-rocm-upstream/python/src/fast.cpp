// (reconstructed graft-base region — see README.md "Graft-base reconstruction": python/ is not vendored (see the "Not vendored" note in README.md); the 75ca881 single-commit snapshot is not in the public fork refs. This file carries the init_fast region the gdn_fused_step / gdn_fused_step_tape graft chain references: the gdn_norm_gated binding (base) + the metal_kernel binding (base), in the order both graft patches' python hunks require.)

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/fast.h"
#include "utils.h"

namespace py = pybind11;
using namespace py::literals;
namespace nb = pybind11;

namespace mlx::core::fast {

void init_fast(nb::module_& parent_module) {
  auto m = parent_module.def_submodule("fast");

  m.def(
      "gdn_norm_gated",
      &mlx::core::fast::gdn_norm_gated,
      "out"_a,
      "z"_a,
      "weight"_a,
      "eps"_a,
      "stream"_a = nb::none(),
      R"(
                The normalized + gated output.
        )");

  m.def(
      "metal_kernel",
      [](const std::string& name,
         const std::vector<array>& inputs,
         const std::vector<std::string>& input_names,
         const std::vector<std::string>& output_names,
         const std::vector<Shape>& output_shapes,
         const std::vector<Dtype>& output_dtypes,
         const std::vector<ScalarArg>& scalars,
         const std::string& source,
         const std::string& header,
         bool ensure_row_contiguous,
         int shared_memory,
         const std::vector<std::pair<int, int>>& output_input_aliases,
         const StreamOrDevice& s) {
        return metal_kernel(
            name,
            inputs,
            input_names,
            output_names,
            output_shapes,
            output_dtypes,
            scalars,
            source,
            header,
            ensure_row_contiguous,
            shared_memory,
            output_input_aliases,
            s);
      },
      "name"_a,
      "inputs"_a,
      "input_names"_a,
      "output_names"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "scalars"_a,
      "source"_a,
      "header"_a,
      "ensure_row_contiguous"_a,
      "shared_memory"_a,
      "output_input_aliases"_a,
      "s"_a);
}

}  // namespace mlx::core::fast
