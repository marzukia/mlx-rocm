// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

} // namespace

namespace mlx::core {

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = rocm::get_command_encoder(stream());
  auto size = out.size();
  auto nbytes = size * out.itemsize();
  out.set_data(mlx::core::rocm::malloc_async(nbytes, encoder));
  // Stage through PINNED host memory. An async H2D copy from pageable memory is
  // unreliable on a discrete GPU over a non-coherent link (TB5 eGPU): the
  // driver must internally stage it, which can stall the stream (queue stuck,
  // GPU shows busy, the eval's sync never returns). Pinned memory DMAs directly
  // and lets the copy actually run asynchronously.
  void* out_ptr = nullptr;
  if (hipHostMalloc(&out_ptr, nbytes, hipHostMallocDefault) != hipSuccess ||
      out_ptr == nullptr) {
    // Fallback: pageable + synchronous copy (still correct, just slower).
    out_ptr = malloc(nbytes);
    reader_->read(static_cast<char*>(out_ptr), nbytes, offset_);
    if (swap_endianness_) {
      switch (out.itemsize()) {
        case 2:
          swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 4:
          swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 8:
          swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
      }
    }
    (void)hipMemcpy(gpu_ptr<void>(out), out_ptr, nbytes, hipMemcpyHostToDevice);
    free(out_ptr);
    return;
  }
  reader_->read(static_cast<char*>(out_ptr), nbytes, offset_);
  if (swap_endianness_) {
    switch (out.itemsize()) {
      case 2:
        swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
      case 4:
        swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
      case 8:
        swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
    }
  }
  (void)hipMemcpyAsync(
      gpu_ptr<void>(out),
      out_ptr,
      nbytes,
      hipMemcpyHostToDevice,
      encoder.stream());
  // Do NOT free the staging buffer via hipLaunchHostFunc: calling a HIP
  // runtime API (hipHostFree) from inside a stream host function deadlocks the
  // stream executor on ROCm/gfx12xx (the free waits on progress that can only
  // happen after the host function itself returns). The H2D copy, the event
  // record after it, and the eval's completion wait then never retire. Free on
  // the HIP-initialized worker thread instead: it runs once the stream has
  // drained past this copy, so the pinned source stays alive long enough and
  // the free is outside the executor.
  encoder.add_completed_handler([out_ptr]() { (void)hipHostFree(out_ptr); });
}

} // namespace mlx::core
