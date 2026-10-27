// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mlx::core::rocm {

// Forward declarations
class HipEvent;

// Run tasks in worker thread, synchronized with HIP stream.
class Worker {
 public:
  explicit Worker(int device);
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a pending |task| that will run when consumed or committed.
  void add_task(std::function<void()> task);

  // Inform worker thread to run current batches after kernels in |stream|
  // finish running.
  void commit(hipStream_t stream);

  // Signal the worker to run any batches deferred during a capture (see
  // commit). Called from decode_capture_end_record after the capture ends.
  void signal_now();

  // Wait until the worker has finished running all committed batches (is
  // idle). Called from decode_capture_begin before starting a capture, so
  // that in-flight buffer frees (hipFreeAsync on the worker thread) do not
  // race the thread-local capture and invalidate it.
  void wait_idle();

 private:
  static void signal(void*);

  void thread_fn();
  std::mutex mtx_;
  std::condition_variable cond_;

  uint64_t committed_batch_{0};
  uint64_t signaled_batch_{0};
  // The highest committed batch that was signaled (captured at signal time).
  // The worker only runs batches <= signal_batch_, so batches committed after
  // a signal are not run by that signal (avoids a race where a batch committed
  // during a capture is run by a pre-capture signal).
  uint64_t signal_batch_{0};
  bool worker_idle_{true};

  bool stop_{false};

  // The HIP device this worker's stream-completion callbacks run against. The
  // worker thread must hipSetDevice(device_) before running any task: HIP's
  // current device is per-thread and a freshly spawned thread defaults to
  // device 0. Running device-1 stream callbacks/frees from a device-0-bound
  // thread is a cross-device coupling that wedges the queue on a discrete GPU.
  int device_{0};

  // Tasks are put in |pending_tasks_| first, and then moved to
  // |worker_tasks_| when end_batch() is called.
  using Tasks = std::vector<std::function<void()>>;
  Tasks pending_tasks_;
  std::map<uint64_t, Tasks> worker_tasks_;
  std::thread worker_;
};

} // namespace mlx::core::rocm
