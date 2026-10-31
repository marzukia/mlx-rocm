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
#include <chrono>
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
  std::string candidate_first_op;  // first op of the last real eager step
  long step_seq{0};        // slot parity for the captured execs
  size_t arena_cap{0};
  bool arena_realloc{false};  // deferred: arena base must be (re)allocated
  long arena_incarnation{0};   // bumped on every (re)allocation of the base
  long floor_incarnation{-1};  // incarnation in which the floor was frozen
  long log_n{0};

  // --- Exec-cache replay: capture once per shape, then replay the
  // cached graph (hipGraphLaunch) instead of re-capturing every step.
  bool replay_armed{false}; // next small step: replay, not capture
  long replay_slot{-1};    // slot holding the last good captured exec
  std::string step_first_name;  // first op name of this step
  std::vector<std::string> ref_names;     // captured step's op-name sequence
  std::vector<std::string> capture_names;  // names recorded during a capture
  enum class Kind { Eager, Capture, Replay };
  Kind step_kind{Kind::Eager};
  bool step_diverged{false}; // op sequence diverged from the capture
  // Per-step CPU cost of the replay path (first op -> graph launch).
  double cpu_total_ms{0.0};
  long cpu_n{0};
  double cpu_max_ms{0.0};
  std::chrono::steady_clock::time_point cpu_t0;

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

// Shared arena setup for capture AND replay steps: deferred arena
// teardown, arena begin/reset, floor (re)freeze, decode mode on.
// Returns false when the step must run eager (arena setup failed).
bool gd_step_prepare(GraphDecodePilot& p, Stream s) {
  auto& enc = rocm::get_command_encoder(s);
  auto stream = static_cast<hipStream_t>(enc.stream());
  if (p.active) {
    // The previous captured step's launch may still be in flight on this
    // stream, and its input/idx buffers live in the decode arena. The
    // arena reset-to-floor and the step's re-allocations below must not
    // run until the in-flight kernels have finished reading those slots
    // (the input H2D happens outside the capture; replay depends on the
    // slot content surviving). Synchronize before touching the arena.
    if (hipStreamQuery(stream) != hipSuccess) {
      (void)hipStreamSynchronize(stream);
      (void)hipGetLastError();
    }
  }
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
      return false;
    }
    p.active = true;
    p.arena_incarnation++;
    // The floor is not valid for the fresh base.
  } else {
    // Idle arena from a deferred teardown: fresh capture starts at 0.
    rocm::allocator().decode_arena_reset();
  }
  if (p.floor_incarnation == p.arena_incarnation) {
    // Replay step. The between-capture allocations (the token
    // input/idx buffer) live above the prefix, at the previous
    // capture's end offset. The reset above reclaims the prefix but
    // not the input slots. The original reset-to-floor moved the
    // offset up to the floor (== the input slots' start), and the
    // step's first re-allocation then landed in the input slot,
    // aliasing a gather's output with the idx buffer (observed
    // 2nd-capture fault: the embedding gather writes its row into
    // the idx slot and the next gather reads the row's float bits
    // as an index). Re-freezing the floor at the (reset) offset
    // keeps the step's re-allocations inside the prefix, so the
    // input slots are never reclaimed; they are stably reused each
    // step, and the sync at the top of this function covers that
    // reuse.
    rocm::allocator().decode_arena_freeze_floor();
  }
  // Reserve the slot region (5 wrappers: value slots 0-3 + zero slot)
  // so capture and replay steps advance the offset identically: a
  // capture step allocates the slots (bumping the offset), a replay
  // step reuses them (no bump) — reserving in both keeps the step's
  // re-allocations at identical relative addresses.
  rocm::allocator().decode_arena_reserve_wrappers(5);
  // Pre-allocate the value-slot device buffers (arena bump, no hip
  // calls) and their pinned host buffers BEFORE the capture window:
  // hipHostMalloc during a capture invalidates it on ROCm 10.1
  // (observed: first capture step died at the first slot use, the
  // next launch failing with hipErrorStreamCaptureInvalidated).
  for (int i = 0; i < 4; ++i) {
    (void)rocm::allocator().decode_value_slot(i);
  }
  (void)rocm::allocator().decode_zero_slot();
  rocm::set_graph_decode_mode(true);
  return true;
}

// First op of a step that is to be captured. The arena makes every token's
// allocations land at identical addresses, so the captured exec's baked
// pointers stay valid for every later token.
void gd_begin_capture(GraphDecodePilot& p, Stream s) {
  p.capture_names.clear();
  if (!gd_step_prepare(p, s)) {
    return;
  }
  auto& enc = rocm::get_command_encoder(s);
  if (!enc.decode_capture_begin(p.step_seq & 1)) {
    p.log("graph decode: capture begin failed; staying eager");
    gd_tear_down(p);
    p.next_capture = false;
  }
}

// First op of a replay step: same arena setup as a capture step, but
// no stream capture — the cached exec is launched at step end and the
// step's per-op launches are suppressed (the graph nodes recorded at
// the initial capture re-run them; per-step values ride the persistent
// value slots, rewritten from the host before the graph launch).
void gd_begin_replay(GraphDecodePilot& p, Stream s) {
  if (!gd_step_prepare(p, s)) {
    p.step_kind = GraphDecodePilot::Kind::Eager;
    return;
  }
  rocm::get_command_encoder(s).set_decode_replay_suppress(true);
  p.cpu_t0 = std::chrono::steady_clock::now();
}

// T>1 detection on a step's first op: a prefill chunk (or speculative
// decode) step's token indices are [1, T] with T > 1; a decode step's
// are [1, 1]. Weight inputs are [V, D] (shape(0) != 1). Such steps
// run fully eager (they would also blow the small-op-count budget at
// step end).
bool gd_step_is_big(const array& arr) {
  if (std::getenv("MLX_G1Y_DEBUG") != nullptr) {
    static int n = 0;
    if (n < 6) {
      n++;
      fprintf(stderr, "[g1] first-op %s inputs:", arr.primitive().name());
      for (auto& in : arr.inputs()) {
        fprintf(stderr, " [ndim=%ld", in.ndim());
        for (size_t i = 0; i < in.ndim() && i < 4; ++i) {
          fprintf(stderr, ":%ld", in.shape(i));
        }
        fprintf(stderr, "]");
      }
      fprintf(stderr, "\n");
    }
  }
  for (auto& in : arr.inputs()) {
    if (in.ndim() == 2 && in.shape(0) == 1 && in.shape(1) > 1) {
      return true;
    }
  }
  return false;
}

// Last op of a step (the "Synchronize" primitive of every eval burst).
// Ends the capture (if this step was captured), validates the shape, and
// advances the arming state machine for the next step.
void gd_step_end(GraphDecodePilot& p, Stream s, long ops) {
  auto& enc = rocm::get_command_encoder(s);
  const bool big = ops >= gd_big_ops();
  p.step_seq++;
  if (p.active && p.step_kind == GraphDecodePilot::Kind::Replay) {
    // This step replayed the cached exec (per-op launches were
    // suppressed). Time the CPU side (first op -> graph launch).
    double dt = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - p.cpu_t0)
                    .count();
    p.cpu_total_ms += dt;
    p.cpu_n++;
    p.cpu_max_ms = std::max(p.cpu_max_ms, dt);
    bool ok = !p.step_diverged && !big;
    if (ok && rocm::allocator().decode_arena_overflowed()) {
      // Some allocations escaped the arena: same treatment as the
      // capture-time overflow — this step's launch is best-effort, the
      // next small step re-captures with a bigger arena.
      if (p.arena_cap < 8ULL * 1024 * 1024 * 1024) {
        p.log(
            "graph decode: replay arena overflowed (cap=%zu MB); "
            "doubling and re-capturing",
            p.arena_cap >> 20);
        p.arena_cap <<= 1;
        p.arena_realloc = true;
      } else {
        p.log("graph decode: replay arena overflowed at cap; cooling down");
        gd_tear_down(p);
        p.arming = GraphDecodePilot::Arming::Cooldown;
        p.small_streak = 0;
        p.prev_small_ops = -1;
      }
      p.next_capture = true;
      p.replay_armed = false;
      (void)enc.decode_capture_replay((int)p.replay_slot);
    } else if (ok) {
      if (std::getenv("MLX_G1Y_DEBUG") != nullptr && p.cpu_n < 4) {
        auto rstream = static_cast<hipStream_t>(enc.stream());
        for (int i = 0; i < 4; ++i) {
          auto sl = rocm::allocator().decode_value_slot(i);
          int64_t dv = -12345;
          if (sl.data) {
            (void)hipMemcpy(&dv, sl.data, 8, hipMemcpyDeviceToHost);
            (void)hipStreamSynchronize(rstream);
          }
          fprintf(stderr,
                  "[g1] slot %d host=%lld dev=%lld\n",
                  i,
                  sl.host ? (long long)*sl.host : -1LL,
                  (long long)dv);
        }
      }
      if (enc.decode_capture_replay((int)p.replay_slot)) {
        // Stay armed: the next small step replays too.
        p.replay_armed = true;
        if (p.cpu_n % 256 == 0) {
          p.log(
              "[g1] replay cpu/step mean=%.2f max=%.2f ms (n=%ld)",
              p.cpu_total_ms / (double)p.cpu_n,
              p.cpu_max_ms,
              p.cpu_n);
        }
      } else {
        p.log(
            "graph decode: replay launch failed (ops=%ld); re-arming",
            ops);
        gd_tear_down(p);
        p.arming = GraphDecodePilot::Arming::Cooldown;
        p.small_streak = 0;
        p.prev_small_ops = -1;
        p.replay_armed = false;
      }
      p.next_capture = false;
    } else {
      // Diverged (or oversized mid-step): the prefix ran via the
      // graph, the suffix ran eagerly and produced this step's real
      // outputs. Launch the graph (its prefix is correct; its suffix
      // nodes write stale slots no one reads) and re-arm: the next
      // small step is a fresh capture (new reference).
      p.log(
          "graph decode: replay diverged (ops=%ld, big=%d, div=%d); "
          "prefix launched, re-capturing",
          ops,
          (int)big,
          (int)p.step_diverged);
      (void)enc.decode_capture_replay((int)p.replay_slot);
      p.arming = GraphDecodePilot::Arming::PrefillSeen;
      p.small_streak = 0;
      p.prev_small_ops = -1;
      p.next_capture = false;
      p.replay_armed = false;
    }
    p.step_kind = GraphDecodePilot::Kind::Eager;
    p.step_diverged = false;
    enc.set_decode_replay_suppress(false);
    return;
  }
  if (p.active && p.step_kind == GraphDecodePilot::Kind::Capture) {
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
      if (ops < 100) {
        // A micro step started as a capture candidate: launch its
        // recorded (1-op) graph — that is exactly this step — and keep
        // the capture armed: the next real step re-captures.
        p.log("graph decode: micro step (ops=%ld); launched, re-capturing next", ops);
        (void)enc.decode_capture_replay((int)(p.step_seq & 1));
        return;
      }
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
    // Arm the exec cache: from the next small step on, replay this
    // exec (hipGraphLaunch) instead of re-capturing. The exec lives
    // in the slot end_record just wrote (post-increment parity).
    p.replay_slot = p.step_seq & 1;
    p.ref_names = std::move(p.capture_names);
    p.replay_armed = true;
    p.next_capture = false;
    p.log(
        "graph decode: exec cache armed (ops=%ld, slot=%ld, %zu ref names)",
        ops,
        p.replay_slot,
        p.ref_names.size());
    return;
  }
  // Eager step: advance the arming state machine.
  if (ops >= 100) {
    p.candidate_first_op = p.step_first_name;
  }
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
  // Micro step (a between-step scalar eval, a cache resize, ...): do not
  // disturb the arming streak (it interleaves with the 1740-op decode
  // steps); it is not a capture candidate either.
  if (ops < 100) {
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
// first op of a step starts the capture (or the replay, where the step's
// per-op launches are suppressed and the cached exec is launched at step
// end); every op of a replay step is validated against the captured
// op-name sequence (a KVCache resize inserts extra ops). next_capture
// stays true for the whole step (it is only cleared at step end / on a
// begin failure), so guard on step_ops: re-beginning per op would discard
// the partial capture on every op (decode_capture_begin's defensive
// end+discard) and the recorded step would end up 0 nodes.
void gd_on_op(GraphDecodePilot& p, const array& arr, Stream s) {
  if (p.step_ops == 0) {
    p.step_first_name = arr.primitive().name();
    if (gd_step_is_big(arr)) {
      // T>1 step: run it fully eager; re-arm on the next small step.
      p.step_kind = GraphDecodePilot::Kind::Eager;
      p.next_capture = false;
      p.replay_armed = false;
    } else if (p.active && p.replay_armed && p.replay_slot >= 0 &&
               !p.ref_names.empty() &&
               p.ref_names[0] == arr.primitive().name()) {
      p.step_kind = GraphDecodePilot::Kind::Replay;
      p.replay_armed = false;
      gd_begin_replay(p, s);
    } else if (p.next_capture &&
               p.candidate_first_op == arr.primitive().name()) {
      // p.active may be false here: the capture's prepare is what
      // activates the arena for the first time. A step whose first op
      // does not match the candidate's (a between-step scalar eval,
      // a cache resize, ...) runs eager; the flag stays set.
      p.step_kind = GraphDecodePilot::Kind::Capture;
      gd_begin_capture(p, s);
    } else {
      p.step_kind = GraphDecodePilot::Kind::Eager;
    }
  } else if (p.step_kind == GraphDecodePilot::Kind::Replay &&
             !p.step_diverged) {
    // Validate this op against the captured reference (strict name
    // sequence). On the first divergence, run the rest of the step
    // eagerly (the graph's prefix still launches at step end).
    if (p.step_ops >= (long)p.ref_names.size() ||
        p.ref_names[p.step_ops] != arr.primitive().name()) {
      p.step_diverged = true;
      p.log(
          "graph decode: replay diverged at op %ld (%s)",
          p.step_ops,
          arr.primitive().name());
      rocm::get_command_encoder(s).set_decode_replay_suppress(false);
    }
  }
  if (p.step_kind == GraphDecodePilot::Kind::Capture) {
    p.capture_names.push_back(arr.primitive().name());
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
    gd_on_op(p, arr, s);
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
