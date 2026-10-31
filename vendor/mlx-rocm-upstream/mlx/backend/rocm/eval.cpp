// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// G1 root-cause instrumentation (defined in device.cpp, gated by MLX_G1_TRACE).
namespace mlx::core::rocm {
extern long g1_trace_n;
extern std::atomic<bool> g_decode_capturing;
// G1 first-error tracking (defined in device.cpp).
extern void g1_check_hip(const char* call, hipError_t err, const char* phase = nullptr);
}

namespace mlx::core::gpu {

// --- Graph decode autopilot (serve/lm path wiring) -------------------------
// The build-once decode-graph backend (decode_capture_*, decode_arena_*) was
// complete but DORMANT: only the C API gpu_set_graph_decode_mode set the
// mode and nothing in-tree or in mlx-lm ever called it, so every decode step
// ran eager (~1,070 launches + ~20 commits/token). This autopilot wires the
// serve/lm path: on the default stream it watches the eval-step pattern
// (one dispatch burst per mx.eval/mx.async_eval, ending with the no-op
// "Synchronize" primitive) and, once it sees the decode-shaped pattern — a
// large step (prefill chunk) followed by two small, stable steps (decode
// tokens) — it captures the next small step into ONE HIP graph (the
// deterministic decode arena keeps every token's buffer addresses identical)
// and, for each subsequent matching step, refreshes the captured exec via
// hipGraphExecUpdate and launches it once per token.
//
// OFF BY DEFAULT; MLX_ROCM_GRAPH_DECODE=1 enables the whole autopilot.
// Any capture failure or shape mismatch falls back to eager (and cools
// down), so the worst case is exactly today's behavior.
//
// Tunables: MLX_ROCM_GRAPH_DECODE_BIG_OPS (default 4000 — a step with at
// least this many evals is "large" and resets the arming streak),
// MLX_ROCM_GRAPH_DECODE_ARENA_MB (default 2048 — decode arena capacity,
// doubled on overflow up to 8 GB).

namespace {
bool g1_trace_on() {
  static const bool on = std::getenv("MLX_G1_TRACE") != nullptr;
  return on;
}
long g1_trace_ops_limit() {
  static const long lim = [] {
    const char* e = std::getenv("MLX_G1_TRACE_OPS");
    return e ? std::atol(e) : 0;
  }();
  return lim;
}

struct GraphDecodePilot {
  enum class Arming { Cold, PrefillSeen, Cooldown };
  std::mutex mu;
  Arming arming{Arming::Cold};
  bool active{false};        // decode arena is on
  bool next_capture{false};  // the next step is a capture step
  long step_ops{0};         // evals in the current (default-stream) step
  long small_streak{0};
  long prev_small_ops{0};
  long capture_ref_ops{0};  // shape the captured graph must match
  long step_seq{0};        // slot parity for the captured execs
  size_t arena_cap{0};
  bool arena_realloc{false};  // deferred: arena base must be (re)allocated
  long arena_incarnation{0};   // bumped on every (re)allocation of the base
  long floor_incarnation{-1};  // incarnation in which the floor was frozen
  long log_n{0};

  void log(const char* fmt, ...) {
    // Rate-limited: first 8 events, then every 256th, so a long generation
    // does not flood stderr but a failure is always visible.
    if (log_n < 8 || (log_n % 256) == 0) {
      va_list ap;
      va_start(ap, fmt);
      fprintf(stderr, "[g1] ");
      vfprintf(stderr, fmt, ap);
      fprintf(stderr, "\n");
      va_end(ap);
    }
    log_n++;
  }
};

GraphDecodePilot& pilot() {
  static GraphDecodePilot p;
  return p;
}

bool gd_env_enabled() {
  static const bool e = [] {
    const char* v = std::getenv("MLX_ROCM_GRAPH_DECODE");
    return v && std::string(v) == "1";
  }();
  return e;
}

long gd_big_ops() {
  static const long t = [] {
    const char* e = std::getenv("MLX_ROCM_GRAPH_DECODE_BIG_OPS");
    return e ? std::max<long>(100, std::atol(e)) : 4000;
  }();
  return t;
}

size_t gd_arena_cap() {
  static const size_t c = [] {
    const char* e = std::getenv("MLX_ROCM_GRAPH_DECODE_ARENA_MB");
    return static_cast<size_t>(e ? std::max<long>(64, std::atol(e)) : 2048) *
        1024 * 1024;
  }();
  return c;
}

bool gd_is_default_stream(Stream s) {
  auto d = default_device();
  if (d.type != Device::gpu) {
    return false;
  }
  auto ds = default_stream(d);
  return s.device == ds.device && s.index == ds.index;
}

bool gd_shape_match(long a, long b) {
  return std::abs(a - b) <= std::max<long>(64, b / 4);
}

void gd_tear_down(GraphDecodePilot& p) {
  // Back to eager: mode off. The arena is ended and the captured execs
  // destroyed DEFERRED (at the next gd_begin_capture): the just-launched
  // graph may still be in flight, and destroying its exec / freeing its
  // arena base while it runs would read freed memory.
  rocm::set_graph_decode_mode(false);
  p.arena_realloc = true;  // also ends an idle arena from a prior capture
}

// First op of a step that is to be captured. The arena makes every token's
// allocations land at identical addresses, so the captured exec's baked
// pointers stay valid for every later token.
void gd_begin_capture(GraphDecodePilot& p, Stream s) {
  auto& enc = rocm::get_command_encoder(s);
  auto stream = static_cast<hipStream_t>(enc.stream());
  if (p.arena_realloc) {
    // The previous capture's launch (if any) must be done before the old
    // arena base is freed. In the normal generate/serve flow the CPU has
    // already waited on the step event; hipStreamQuery makes it explicit.
    if (hipStreamQuery(stream) != hipSuccess) {
      (void)hipStreamSynchronize(stream);
      (void)hipGetLastError();
    }
    if (p.active) {
      rocm::allocator().decode_arena_end();
      p.active = false;
    }
    p.arena_realloc = false;
  }
  if (!p.active) {
    if (p.arena_cap == 0) {
      p.arena_cap = gd_arena_cap();
    }
    if (!rocm::allocator().decode_arena_begin(
            p.arena_cap, s.device.index, static_cast<void*>(stream))) {
      p.log(
          "graph decode: arena begin failed (cap=%zu MB); staying eager",
          p.arena_cap >> 20);
      p.arming = GraphDecodePilot::Arming::Cooldown;
      p.next_capture = false;
      return;
    }
    p.active = true;
    p.arena_incarnation++;
    // The floor is not valid for the fresh base.
  } else {
    // Idle arena from a deferred teardown: fresh capture starts at 0.
    rocm::allocator().decode_arena_reset();
  }
  if (p.floor_incarnation == p.arena_incarnation) {
    // Replay step: same addresses as the captured step (skip the one-time
    // prefix frozen after the first capture).
    rocm::allocator().decode_arena_reset_to_floor();
  }
  rocm::set_graph_decode_mode(true);
  if (!enc.decode_capture_begin(p.step_seq & 1)) {
    p.log("graph decode: capture begin failed; staying eager");
    gd_tear_down(p);
    p.next_capture = false;
  }
}

// Last op of a step (the "Synchronize" primitive of every eval burst).
// Ends the capture (if this step was captured), validates the shape, and
// advances the arming state machine for the next step.
void gd_step_end(GraphDecodePilot& p, Stream s, long ops) {
  auto& enc = rocm::get_command_encoder(s);
  const bool big = ops >= gd_big_ops();
  p.step_seq++;
  if (p.active && p.next_capture) {
    // This step was captured: end the capture, refresh (ExecUpdate) or
    // re-instantiate the slot's exec, and launch it (the only real
    // execution of this step's kernels).
    bool ok = enc.decode_capture_end_record(p.step_seq & 1);
    if (!ok) {
      p.log(
          "graph decode: capture end_record failed (ops=%ld); staying eager",
          ops);
      gd_tear_down(p);
      p.next_capture = false;
      p.arming = GraphDecodePilot::Arming::Cooldown;
      p.small_streak = 0;
      return;
    }
    p.log(
        "graph decode: step captured+launched (ops=%ld, slot=%ld, active)",
        ops,
        p.step_seq & 1);
    if (big) {
      // A large step was captured (a prefill slipped in): the launch is
      // correct for this step, but the exec is the wrong shape for decode.
      p.log("graph decode: large step captured (ops=%ld); re-arming", ops);
      gd_tear_down(p);
      p.next_capture = false;
      p.arming = GraphDecodePilot::Arming::PrefillSeen;
      p.small_streak = 0;
      return;
    }
    if (!gd_shape_match(ops, p.capture_ref_ops)) {
      p.log(
          "graph decode: shape mismatch (ops=%ld ref=%ld); cooling down",
          ops,
          p.capture_ref_ops);
      gd_tear_down(p);
      p.next_capture = false;
      p.arming = GraphDecodePilot::Arming::Cooldown;
      p.small_streak = 0;
      return;
    }
    if (rocm::allocator().decode_arena_overflowed()) {
      // Some allocations escaped the arena: this token's launch is still
      // correct, but re-capture the next step with a bigger arena. The
      // reallocation is DEFERRED: the just-launched graph still references
      // the old base.
      if (p.arena_cap < 8ULL * 1024 * 1024 * 1024) {
        p.log(
            "graph decode: arena overflowed (cap=%zu MB); doubling and "
            "re-capturing",
            p.arena_cap >> 20);
        p.arena_cap <<= 1;
        p.arena_realloc = true;
      } else {
        p.log("graph decode: arena overflowed at cap; cooling down");
        gd_tear_down(p);
        p.next_capture = false;
        p.arming = GraphDecodePilot::Arming::Cooldown;
        p.small_streak = 0;
      }
      return;
    }
    // Good capture: freeze the one-time prefix (first capture step only) so
    // replay steps re-allocate from the same addresses.
    if (p.floor_incarnation != p.arena_incarnation) {
      rocm::allocator().decode_arena_freeze_floor();
      p.floor_incarnation = p.arena_incarnation;
    }
    p.next_capture = true;  // keep capturing/replaying
    return;
  }
  // Eager step: advance the arming state machine.
  //
  // NOTE: the original design required a "big" (prefill) step before arming,
  // but a step's OP COUNT does not scale with sequence length — prefill and
  // decode steps of the same model have (near-)identical op counts (~350 for
  // 0.6B, ~1,070 for 9B, both far below the 4000 "big" threshold), so the
  // big-step requirement can never be satisfied and the pilot would stay Cold
  // forever. Instead: arm after two consecutive small steps with matching op
  // counts, with or without a big step seen first. A big step (if one ever
  // appears) still resets the streak.
  if (big) {
    p.arming = GraphDecodePilot::Arming::PrefillSeen;
    p.small_streak = 0;
    return;
  }
  p.small_streak++;
  if (p.small_streak == 1) {
    p.prev_small_ops = ops;
  } else if (p.small_streak == 2 && gd_shape_match(ops, p.prev_small_ops)) {
    p.capture_ref_ops = p.prev_small_ops;
    p.next_capture = true;  // next small step is the capture step
    p.log(
        "graph decode: armed (small steps %ld/%ld ops); capturing next",
        p.prev_small_ops,
        ops);
  } else if (p.small_streak == 2) {
    // No match: slide the window.
    p.prev_small_ops = ops;
    p.small_streak = 1;
  }
}

// Each op of a step (non-Synchronize evals on the default stream). The
// first op of a capture step starts the capture (and, for replay steps,
// resets the arena to the frozen floor so every allocation lands at the
// same address as in the captured step). next_capture stays true for the
// whole step (it is only cleared at step end / on a begin failure), so
// guard on step_ops: re-beginning per op would discard the partial capture
// on every op (decode_capture_begin's defensive end+discard) and the
// recorded step would end up 0 nodes.
void gd_on_op(GraphDecodePilot& p, Stream s) {
  if (p.next_capture && p.step_ops == 0) {
    gd_begin_capture(p, s);
  }
  p.step_ops++;
}

void graph_decode_pilot(array& arr) {
  if (!gd_env_enabled()) {
    return;
  }
  auto s = arr.primitive().stream();
  if (!gd_is_default_stream(s)) {
    return;
  }
  auto& p = pilot();
  std::lock_guard<std::mutex> lock(p.mu);
  const bool sync = std::strcmp(arr.primitive().name(), "Synchronize") == 0;
  if (sync) {
    gd_step_end(p, s, p.step_ops);
    p.step_ops = 0;
  } else {
    gd_on_op(p, s);
  }
}

} // namespace

void init() {
  // Initialize the SELECTED default GPU's primary context — not device 0. On a
  // multi-GPU host, creating a context/queue on the other GPU (the APU) too is
  // what differs from HIP_VISIBLE_DEVICES, and that cross-device queue
  // coexistence is what wedges the discrete GPU's command queue over a TB5
  // link. Touch only the chosen device so the runtime behaves as if it were the
  // only one.
  auto d = mlx::core::default_device();
  if (d.type == mlx::core::Device::gpu) {
    (void)hipSetDevice(d.index);
  }
  (void)hipFree(nullptr);}

void new_stream(Stream s) {
  // Bind the stream's device FIRST (creates/selects its Device + context), then
  // warm the event pool on that device — creating the HipEvent before binding
  // would put it (and its queue interaction) on whatever device is current.
  rocm::get_command_encoder(s);
  rocm::HipEvent(hipEventDefault);
}

void new_thread_unsafe_stream(Stream s) {
  rocm::get_command_encoder(s);
  rocm::HipEvent(hipEventDefault);
}

// Ops whose kernels corrupt when batched into a multi-node HIP graph with
// neighbors (a ROCm CLR kernarg-pool interaction; found by per-op force-execute
// bisection). Isolate them: flush the graph before AND after so they run alone.
static bool is_graph_split_op(const char* name) {
  static const bool no_split = std::getenv("MLX_NO_CONCAT_SPLIT") != nullptr;
  // Decode-mode keeps the whole forward in one graph, so Concatenate must be a
  // graph node (validated bit-identical via ExecUpdate), never a split point.
  if (no_split || rocm::graph_decode_mode())
    return false;
  return std::strcmp(name, "Concatenate") == 0;
}

void eval(array& arr) {
  graph_decode_pilot(arr);
  auto outputs = arr.outputs();
  auto& encoder = rocm::get_command_encoder(arr.primitive().stream());
  const bool split =
      rocm::use_hip_graphs() && is_graph_split_op(arr.primitive().name());
  if (split) {
    encoder.commit(); // flush ops accumulated before this one
  }
  // Bind the stream's device before eval_gpu so output buffers allocate on the
  // same device the kernels run on. Otherwise (multi-GPU) outputs land on
  // whatever device is current (often device 0) while kernels run on the
  // stream's device, stranding the model on the wrong GPU.
  encoder.device().make_current();
  if (rocm::use_hip_graphs()) {
    rocm::set_current_prim(arr.primitive().name());
  }
  // G1 trace: print the op name BEFORE eval_gpu so the throwing op is named.
  if (g1_trace_on() && rocm::g_decode_capturing.load(std::memory_order_relaxed)) {
    fprintf(stderr,
            "[g1t] pre op %ld %s stream=%p\n",
            rocm::g1_trace_n + 1,
            arr.primitive().name(),
            static_cast<void*>(encoder.stream()));
  }
  {
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  // G1 first-error: after each op, probe the stream's last error. An
  // unchecked failed hip call inside eval_gpu (or a cross-thread free on
  // the capture stream) sets the last error without being checked; this
  // names the op that contains the first failure. (Per-thread error state:
  // a worker-thread error is named by the direct check at the free site.)
  if (rocm::g_decode_capturing.load(std::memory_order_relaxed)) {
    hipError_t le = hipGetLastError();
    rocm::g1_check_hip("in-op (after eval_gpu)", le);
    if (true) {
      for (auto& o : outputs) {
        fprintf(stderr, "[g1t]   out %p sz=%zu %s\n",
                o.buffer().ptr(), o.data_size(),
                rocm::allocator().in_decode_arena(o.buffer().ptr())
                    ? "ARENA"
                    : "");
      }
      for (auto& i : arr.inputs()) {
        fprintf(stderr, "[g1t]   in  %p sz=%zu %s\n",
                i.buffer().ptr(), i.data_size(),
                rocm::allocator().in_decode_arena(i.buffer().ptr())
                    ? "ARENA"
                    : "");
      }
    }
  }

  // G1 trace: while capturing, watch the capture status per op. Optionally
  // end the capture early at MLX_G1_TRACE_OPS to count recorded nodes.
  if (g1_trace_on() && rocm::g_decode_capturing.load(std::memory_order_relaxed)) {
    rocm::g1_trace_n++;
    if (rocm::g1_trace_n <= 500 || rocm::g1_trace_n % 100 == 0) {
      hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
      hipError_t ic = hipStreamIsCapturing(
          static_cast<hipStream_t>(encoder.stream()), &st);
      rocm::g1_check_hip("hipStreamIsCapturing (per-op trace)", ic);
      fprintf(stderr,
              "[g1t] op %ld %s stream=%p status=%d\n",
              rocm::g1_trace_n,
              arr.primitive().name(),
              static_cast<void*>(encoder.stream()),
              static_cast<int>(st));
    }
    const long lim = g1_trace_ops_limit();
    if (lim > 0 && rocm::g1_trace_n >= lim) {
      hipGraph_t g = nullptr;
      hipError_t ee = hipStreamEndCapture(
          static_cast<hipStream_t>(encoder.stream()), &g);
      rocm::g1_check_hip("hipStreamEndCapture (early-end trace)", ee);
      size_t nn = 0;
      if (ee == hipSuccess && g) {
        hipError_t gn = hipGraphGetNodes(g, nullptr, &nn);
        rocm::g1_check_hip("hipGraphGetNodes (early-end trace)", gn);
      }
      fprintf(stderr,
              "[g1t] early-end at op %ld: end=%s nodes=%zu\n",
              rocm::g1_trace_n,
              hipGetErrorString(ee),
              nn);
      if (g) {
        (void)hipGraphDestroy(g);
      }
      rocm::g_decode_capturing.store(false, std::memory_order_relaxed);
      (void)hipGetLastError();
    }
  }

  for (auto& in : arr.inputs()) {
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (rocm::use_hip_graphs()) {
    auto& stream = arr.primitive().stream();
    if (split || encoder.needs_commit()) {
      scheduler::notify_new_task(stream);
      encoder.add_completed_handler(
          [stream]() { scheduler::notify_task_completion(stream); });
      encoder.commit();
    }
  } else {
    encoder.maybe_commit();
  }
}

void finalize(Stream s) {
  rocm::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  rocm::get_command_encoder(s).synchronize();
}

void clear_streams() {
  rocm::clear_all_encoders();
}

} // namespace mlx::core::gpu

// --- GPU memcpy for direct KV cache writes ---
extern "C" void mlx_gpu_memcpy_async(void* dst, const void* src, size_t bytes) {
  // Use the SELECTED default device's stream, not Device::gpu (which is the
  // device TYPE = gpu index 0). On a multi-GPU box, --device 1 would otherwise
  // memcpy KV data on device 0's stream while the data lives on device 1.
  auto& enc = mlx::core::rocm::get_command_encoder(
      mlx::core::default_stream(mlx::core::default_device()));
  enc.launch_kernel([=](hipStream_t stream) {
    hipError_t mc =
        hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
    mlx::core::rocm::g1_check_hip("hipMemcpyAsync D2D (mlx_gpu_memcpy_async)", mc);
    (void)mc;
  });
}

// --- Arena + Graph wrappers (called from engine code without HIP headers) ---
namespace mlx::core {

void gpu_set_graph_decode_mode(bool v) {
  rocm::set_graph_decode_mode(v);
}

} // namespace mlx::core
