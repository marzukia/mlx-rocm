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
  // GD-idx (t1099 R3): small per-step host loads (the decode token idx, <=16 B)
  // in a capture/replay step stage through the persistent fixed-VA idx slot
  // instead of a per-step fresh pinned buffer + per-step out re-allocation. The
  // re-allocation moved the out VA (arena-active at replay, buffer_cache at a
  // fresh capture) under the pointer the capture baked into the embedding
  // gather, so the replayed graph read the capture step's value (the R2
  // collapse); and the recorded in-graph H2D node's source was the freed
  // capture-step pinned buffer, so the pre-launch re-delivery replayed that
  // stale value. The slot's device VA is stable across steps and
  // incarnations; the 16 B H2D re-executes before every graph launch (the
  // direct copy here is not launch_kernel-wrapped, so it runs at replay too,
  // stream-ordered before the graph), and the pre-launch re-delivery sources
  // the persistent pinned buffer, which the current step's reader just
  // rewrote in place. A fresh capture's token load happens BEFORE the step's
  // first op (the mode flag is not set yet) — the embedding gather stages the
  // then-produced idx array into the slot with a D2D (recorded at capture,
  // suppressed at replay, where this path delivers).
  if (rocm::graph_decode_mode() && nbytes <= 16) {
    auto sl = rocm::allocator().decode_idx_slot();
    if (sl.host && sl.data) {
      // Zero the full 16 B first: a 4 B (int32) load into an 8 B (int64)
      // reader would otherwise leave stale high bytes in the slot.
      memset(sl.host, 0, 16);
      reader_->read(static_cast<char*>(sl.host), nbytes, offset_);
      if (swap_endianness_) {
        switch (out.itemsize()) {
          case 2:
            swap_endianness<2>(reinterpret_cast<uint8_t*>(sl.host), size);
            break;
          case 4:
            swap_endianness<4>(reinterpret_cast<uint8_t*>(sl.host), size);
            break;
          case 8:
            swap_endianness<8>(reinterpret_cast<uint8_t*>(sl.host), size);
            break;
        }
      }
      out.set_data(rocm::allocator().make_buffer(sl.data, 16));
      (void)hipMemcpyAsync(
          sl.data,
          sl.host,
          16,
          hipMemcpyHostToDevice,
          encoder.stream());
      rocm::allocator().decode_idx_slot_arm();
      return;
    }
  }
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
