#pragma once
// G1: targeted HIP call logger for the capture-invalidation hunt.
// Logs the synchronous (non-capturable) HIP calls with thread ID so we can
// identify the exact call that sets the 900 "operation not permitted when
// stream is capturing" during a decode-step capture.
#include <cstdio>
#include <hip/hip_runtime.h>
#include <thread>

namespace mlx::core::rocm::g1hiplog {

inline bool on() {
  static const bool v = std::getenv("MLX_G1_HIPLOG") != nullptr;
  return v;
}

inline void log_call(const char* name, hipError_t err, const char* phase) {
  if (!on()) return;
  const bool errish = (err != hipSuccess);
  if (!errish && !(phase && phase[0])) return; // log all when phase given, else only errors
  fprintf(stderr,
          "[g1hl] tid=%lu %s -> %d (%s) [%s]\n",
          (unsigned long)std::hash<std::thread::id>{}(std::this_thread::get_id()),
          name, (int)err, hipGetErrorString(err), phase ? phase : "");
}

} // namespace mlx::core::rocm::g1hiplog

#define G1_HL_SYNC(name, expr, phase) \
  ([&] { \
    hipError_t _e = (expr); \
    mlx::core::rocm::g1hiplog::log_call(name, _e, phase); \
    return _e; \
  }())
