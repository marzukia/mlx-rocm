// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/worker.h"
#include "mlx/backend/rocm/utils.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mlx::core::rocm {

// Defined in device.cpp. True during a full decode-step stream capture.
extern std::atomic<bool> g_decode_capturing;
// G1 first-error tracking (defined in device.cpp).
extern void g1_check_hip(const char* call, hipError_t err, const char* phase = nullptr);

Worker::Worker(int device)
    : device_(device), worker_(&Worker::thread_fn, this) {}

Worker::~Worker() {
  {
    std::lock_guard lock(mtx_);
    stop_ = true;
  }
  cond_.notify_one();
  worker_.join();
}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::signal(void* data) {
  auto w = static_cast<Worker*>(data);
  // G1f: log every host-func firing with the capture flag — if this fires
  // while g_decode_capturing is true, the worker (or this callback) is a
  // cross-thread actor during the capture region.
  if (getenv("MLX_WORKER_DEBUG")) {
    fprintf(stderr,
            "[worker] signal: HOSTFUNC FIRED (capturing=%d committed=%llu)\n",
            (int)g_decode_capturing.load(std::memory_order_relaxed),
            (unsigned long long)w->committed_batch_);
  }
  {
    std::lock_guard lock(w->mtx_);
    w->signaled_batch_++;
    w->signal_batch_ = w->committed_batch_;
  }
  w->cond_.notify_one();
}

void Worker::signal_now() {
  if (getenv("MLX_WORKER_DEBUG")) {
    fprintf(stderr, "[worker] signal_now: signaled_batch_=%llu committed_batch_=%llu\n",
            (unsigned long long)signaled_batch_, (unsigned long long)committed_batch_);
  }
  signal(this);
}

void Worker::wait_idle() {
  if (getenv("MLX_WORKER_DEBUG")) {
    fprintf(stderr, "[worker] wait_idle: ENTER (committed=%llu signaled=%llu idle=%d)\n",
            (unsigned long long)committed_batch_, (unsigned long long)signaled_batch_,
            (int)worker_idle_);
  }
  // Signal the worker so it runs any committed-but-unrun batches, then wait
  // for it to finish (idle). This ensures in-flight buffer frees (hipFreeAsync
  // on the worker thread) complete before the caller begins a capture.
  signal(this);
  std::unique_lock<std::mutex> lk(mtx_);
  cond_.wait(lk, [this] {
    return this->worker_idle_;
  });
  if (getenv("MLX_WORKER_DEBUG")) {
    fprintf(stderr, "[worker] wait_idle: EXIT (committed=%llu signaled=%llu idle=%d)\n",
            (unsigned long long)committed_batch_, (unsigned long long)signaled_batch_,
            (int)worker_idle_);
  }
}

void Worker::commit(hipStream_t stream) {
  // Move pending tasks into tasks
  if (pending_tasks_.empty()) {
    return;
  }
  {
    std::lock_guard lock(mtx_);
    // Move pending tasks into ready tasks
    worker_tasks_[++committed_batch_] = std::move(pending_tasks_);
  }
  // During a full decode-step capture, do NOT signal the worker inline.
  // The completion handlers free temporaries via hipFreeAsync on the worker
  // thread; a cross-thread hipFreeAsync on a thread-local capture stream
  // invalidates the capture (the next kernel launch then fails with
  // "operation failed due to a previous error during capture"). Defer the
  // tasks: they stay queued in worker_tasks_ and run after the capture ends
  // (signaled from decode_capture_end_record). The deterministic decode arena
  // makes deferring the frees safe: arena buffers are not actually freed and
  // persistent buffers keep a refcount > 1.
  if (g_decode_capturing.load(std::memory_order_relaxed)) {
    if (getenv("MLX_WORKER_DEBUG")) {
      fprintf(stderr, "[worker] commit: DEFERRED (capturing) committed_batch_=%llu\n",
              (unsigned long long)committed_batch_);
    }
    return;
  }
  // Use hipLaunchHostFunc to signal when stream operations complete
  hipError_t hf = hipLaunchHostFunc(stream, signal, this);
  g1_check_hip("hipLaunchHostFunc (worker commit)", hf);
  (void)hf;
}

void Worker::thread_fn() {
  // Bind this thread to the encoder's device before running any task.
  // Completion handlers free temporaries / return buffers to the pool and may
  // issue HIP calls; they must hit the same device the stream lives on, not the
  // default device 0. Without this the discrete-GPU queue wedges on a multi-GPU
  // host.
  hipError_t sd = hipSetDevice(device_);
  g1_check_hip("hipSetDevice (worker thread start)", sd, "worker");
  uint64_t current_batch = 0;
  while (!stop_) {
    Tasks tasks;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this, current_batch] {
        if (this->stop_)
          return true;
        // Do not run deferred tasks (buffer frees) while a decode-step
        // capture is in flight: a previous eager commit's
        // hipLaunchHostFunc fires on the GPU during the capture and would
        // wake us mid-capture; a cross-thread hipFreeAsync on the
        // capturing stream (ThreadLocal mode) invalidates the capture.
        // The frees run after EndCapture, when signal_now() signals on a
        // non-capturing stream. stop_ still wins so the destructor can
        // always join us.
        return this->signaled_batch_ > current_batch &&
            !g_decode_capturing.load(std::memory_order_relaxed);
      });
      current_batch = signaled_batch_;
      auto end = worker_tasks_.upper_bound(signal_batch_);
      for (auto it = worker_tasks_.begin(); it != end; ++it) {
        if (tasks.empty()) {
          tasks = std::move(it->second);
        } else {
          std::move(
              it->second.begin(), it->second.end(), std::back_inserter(tasks));
        }
      }
      worker_tasks_.erase(worker_tasks_.begin(), end);
    }
    // Make sure tasks are cleared before the next wait
    if (!tasks.empty()) {
      std::lock_guard lock(mtx_);
      worker_idle_ = false;
    }
    if (getenv("MLX_WORKER_DEBUG")) {
      fprintf(stderr, "[worker] thread_fn: running %zu tasks (current=%llu signal_batch=%llu committed=%llu)\n",
              tasks.size(), (unsigned long long)current_batch,
              (unsigned long long)signal_batch_, (unsigned long long)committed_batch_);
    }
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto task = std::move(tasks[i]);
      task();
    }
    if (!tasks.empty()) {
      std::lock_guard lock(mtx_);
      worker_idle_ = true;
      cond_.notify_all();
    }
  }
}

} // namespace mlx::core::rocm
