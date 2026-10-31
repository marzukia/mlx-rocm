// The pure (HIP-free) state machine of the graph decode pilot
// (eval.cpp): the arming streak, the per-step kind gating (T>1 eager,
// replay, capture), the replay divergence check, and the step-end
// resolution. eval.cpp's GraphDecodePilot extends this struct with the
// HIP-facing bookkeeping (arena, encoder, slot data) and mirrors these
// decisions around the HIP calls. Unit-tested hermetically:
// tests/test_gd_pilot_logic.py compiles
// tests/cpp/gd_pilot_logic_test.cpp against this header and runs it (no
// GPU needed).
#pragma once
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace mlx::core::rocm {

struct GdPilotLogic {
  enum class Arming { Cold, PrefillSeen, Cooldown };
  enum class Kind { Eager, Capture, Replay };
  // Step-end resolution for a replay step. Launch: the normal replay —
  // launch the cached exec. GrowArena: the arena overflowed — double
  // the capacity and re-capture; this step's launch is best-effort.
  // Discard: diverged or big — do NOT launch the exec. Its node
  // sequence is the capture step's, not this step's: on a divergence,
  // launching it would re-run the stale sequence (its suffix nodes
  // re-write the persistent state buffers after the eager suffix,
  // which itself ran on the suppressed prefix's stale arena
  // intermediates, and it lacks the inserted ops — a mixed,
  // inconsistent post-step state); on a big step the step's real
  // outputs would never be computed by the smaller cached graph. The
  // token is lost; re-arm so the next small step is a fresh capture
  // that overwrites the exec.
  enum class ReplayEnd { Launch, GrowArena, Discard };
  // Step-end shape resolution for a capture step. Launch: the shape
  // matched the arming reference. MicroLaunch: a micro step started as
  // a capture candidate — launch its recorded (1-op) graph, which is
  // exactly this step, and keep the capture armed. Cooldown: shape
  // mismatch.
  enum class CaptureEnd { Launch, MicroLaunch, Cooldown };

  long big_ops{4000};  // a step with >= this many evals is "big"
  long micro_max{100};  // a step with < this many evals is "micro"

  Arming arming{Arming::Cold};
  bool active{false};  // decode arena is on
  bool next_capture{false};  // the next step is a capture step
  long small_streak{0};
  long prev_small_ops{0};
  long capture_ref_ops{0};  // shape the captured graph must match
  std::string candidate_first_op;  // first op of the last real eager step

  bool replay_armed{false};  // next small step: replay, not capture
  long replay_slot{-1};  // slot holding the last good captured exec
  std::vector<std::string> ref_names;  // captured step's op-name sequence

  Kind step_kind{Kind::Eager};
  bool step_diverged{false};  // op sequence diverged from the capture
  long step_ops{0};  // evals in the current (default-stream) step
  std::string step_first_name;

  static bool shape_match(long a, long b) {
    return std::abs(a - b) <= std::max<long>(64, b / 4);
  }

  // First op of a step: choose the step kind and start the per-step
  // state. big_shape: a T>1 shape on the first op (a 2-D input [1, T]
  // with T > 1; such steps run fully eager). embedding_lookup: the
  // first input is 2-D (a decode embedding gather, not a micro-step
  // lm-head gather — matching on the first op NAME alone mis-arms a
  // replay on the micro-step).
  Kind first_op(const std::string& name, bool big_shape,
               bool embedding_lookup) {
    step_first_name = name;
    if (big_shape) {
      // T>1 step: run it fully eager; re-arm on the next small step.
      step_kind = Kind::Eager;
      next_capture = false;
      replay_armed = false;
    } else if (active && replay_armed && replay_slot >= 0 &&
               !ref_names.empty() && ref_names[0] == name &&
               embedding_lookup) {
      step_kind = Kind::Replay;
      replay_armed = false;
    } else if (next_capture && candidate_first_op == name) {
      // active may be false here: the capture's prepare is what
      // activates the arena for the first time. A step whose first op
      // does not match the candidate's (a between-step scalar eval, a
      // cache resize, ...) runs eager; the flag stays set.
      step_kind = Kind::Capture;
    } else {
      step_kind = Kind::Eager;
    }
    step_ops = 1;
    return step_kind;
  }

  // A non-first op of a step: the replay divergence check (strict name
  // sequence; a KVCache resize inserts extra ops). Returns true on the
  // FIRST divergence — the caller un-suppresses so the rest of the
  // step runs eagerly.
  bool op(const std::string& name) {
    bool diverged_now = false;
    if (step_kind == Kind::Replay && !step_diverged) {
      if (step_ops >= (long)ref_names.size() ||
          ref_names[step_ops] != name) {
        step_diverged = true;
        diverged_now = true;
      }
    }
    step_ops++;
    return diverged_now;
  }

  // Step-end resolution for a replay step.
  ReplayEnd replay_end(long ops, bool overflowed) {
    const bool big = ops >= big_ops;
    const bool ok = !step_diverged && !big;
    if (ok && overflowed)
      return ReplayEnd::GrowArena;
    if (ok)
      return ReplayEnd::Launch;
    return ReplayEnd::Discard;
  }

  // Step-end shape resolution for a capture step.
  CaptureEnd capture_end(long ops) {
    if (shape_match(ops, capture_ref_ops))
      return CaptureEnd::Launch;
    if (ops < micro_max)
      return CaptureEnd::MicroLaunch;
    return CaptureEnd::Cooldown;
  }

  // Arm the exec cache after a successful capture: from the next small
  // step on, replay this exec (the slot the end_record just wrote,
  // post-increment parity).
  void arm_replay(long slot) {
    replay_slot = slot;
    replay_armed = true;
    next_capture = false;
  }

  // Re-arm after a replay step's Discard (and after a capture's shape
  // mismatch re-capture): the next small step is a fresh capture (a new
  // reference) that overwrites the cached exec, which is dropped with
  // it.
  void discard_rearm() {
    arming = Arming::PrefillSeen;
    small_streak = 0;
    prev_small_ops = -1;
    next_capture = false;
    replay_armed = false;
  }

  // An eager step end: advance the arming state machine. Returns true
  // when a capture was armed (the next small step is the capture step).
  // Arm after two consecutive small steps with matching op counts, with
  // or without a big step seen first (a step's OP COUNT does not scale
  // with sequence length — prefill and decode steps of the same model
  // have near-identical op counts, both far below the big threshold —
  // so the original big-step-before-arming requirement can never be
  // satisfied). A big step still resets the streak.
  bool eager_end(long ops) {
    const bool big = ops >= big_ops;
    if (ops >= micro_max)
      candidate_first_op = step_first_name;
    if (big) {
      arming = Arming::PrefillSeen;
      small_streak = 0;
      return false;
    }
    // Micro step (a between-step scalar eval, a cache resize, ...): do
    // not disturb the arming streak (it interleaves with the 1,740-op
    // decode steps); it is not a capture candidate either.
    if (ops < micro_max)
      return false;
    small_streak++;
    if (small_streak == 1) {
      prev_small_ops = ops;
    } else if (small_streak == 2 && shape_match(ops, prev_small_ops)) {
      capture_ref_ops = prev_small_ops;
      next_capture = true;  // next small step is the capture step
      return true;
    } else if (small_streak == 2) {
      // No match: slide the window.
      prev_small_ops = ops;
      small_streak = 1;
    }
    return false;
  }

  // Reset the per-step state after a step end (the step_kind/
  // step_diverged resets the replay path already did are a no-op for
  // the other paths: a step's kind is re-chosen at its first op).
  void step_done() {
    step_kind = Kind::Eager;
    step_diverged = false;
    step_ops = 0;
  }
};

}  // namespace mlx::core::rocm
