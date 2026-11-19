// The pure (HIP-free) state machine of the graph decode pilot (eval.cpp): the arming streak, the per-step kind gating (per-class), the replay divergence check, and the step-end resolution. eval.cpp's GraphDecodePilot extends this struct with the HIP-facing bookkeeping (arena, encoder, slot data) and mirrors these decisions around the HIP calls. Unit-tested hermetically: tests/test_gd_pilot_logic.py compiles tests/cpp/gd_pilot_logic_test.cpp against this header and runs it (no GPU needed).
//
// Step classes (MTP 140): the T=1 AR decode step and the S>1 verify
// (batched) steps are SEPARATE arming classes. S (the token count) is
// the only shape dimension that varies across steps of a class, so
// each S arms independently: S=3 and S=4 verify steps interleave with
// T=1 draft steps without disturbing each other's arming streaks
// (a shared single class can never arm in MTP mode: the verify steps'
// op counts differ from the draft steps' and reset the streak every
// cycle). verify = false (the default) reproduces the pre-verify
// behavior bit for bit: a T>1 step runs eager, clears the T=1 arming,
// and counts as a small step in the T=1 arming window.
#pragma once
#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mlx::core::rocm {

struct GdPilotLogic {
  enum class Arming { Cold, PrefillSeen, Cooldown };
  enum class Kind { Eager, Capture, Replay };
  // Step-end resolution for a replay step. Launch: the normal replay — launch the cached exec. GrowArena: the arena overflowed — double the capacity and re-capture; this step's launch is best-effort.
  // Discard: diverged or big — do NOT launch the exec. Its node sequence is the capture step's, not this step's: on a divergence, launching it would re-run the stale sequence (its suffix nodes re-write the persistent state buffers after the eager suffix, which itself ran on the suppressed prefix's stale arena intermediates, and it lacks the inserted ops — a mixed, inconsistent post-step state); on a big step the step's real outputs would never be computed by the smaller cached graph. The token is lost; re-arm so the next small step is a fresh capture that overwrites the exec.
  enum class ReplayEnd { Launch, GrowArena, Discard };
  // Step-end shape resolution for a capture step. Launch: the shape matched the arming reference. MicroLaunch: a micro step started as a capture candidate — launch its recorded (1-op) graph, which is exactly this step, and keep the capture armed. Cooldown: shape mismatch.
  enum class CaptureEnd { Launch, MicroLaunch, Cooldown };

  long big_ops{4000};  // a step with >= this many evals is "big"
  long micro_max{100};  // a step with < this many evals is "micro"

  // --- T=1 AR decode class ------------------------------------------------
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

  // --- S>1 verify (batched) class ------------------------------------------
  // Per-S arming: the same two-consecutive-stable-shapes rule as the T=1
  // class, keyed by the batch size S. S is stable across a run (a 140-style
  // MTP serve alternates a fixed S verify with fixed T=1 drafts), so two
  // consecutive matching S steps arm the next S step as a capture; the
  // following S steps replay the S-keyed exec (a separate exec slot per S,
  // so an S=3 and an S=4 exec never clobber each other).
  bool verify{false};  // S>1 capture/replay enabled (MLX_ROCM_GRAPH_DECODE_VERIFY=1)
  struct VerifyKey {
    long small_streak{0};
    long prev_small_ops{0};
    long capture_ref_ops{0};
    std::string candidate_first_op;
    bool next_capture{false};
    bool replay_armed{false};
    long replay_slot{-1};
    std::vector<std::string> ref_names;
  };
  std::map<long, VerifyKey> verify_keys;

  // Exec-slot map: T=1 AR -> 0, S=3 -> 1, S=4 -> 2, any other S>1 -> 3.
  // Four exec slots: the two S=3/S=4 verify execs, the T=1 AR exec, and a
  // shared slot for other S (one model serves one MTP S at a time; a mixed
  // other-S run shares slot 3 and re-captures on S change, like T=1 does).
  static constexpr long kExecSlots = 4;
  static long slot_for_batch(long batch) {
    if (batch <= 1) return 0;
    if (batch == 3) return 1;
    if (batch == 4) return 2;
    return 3;
  }
  void enable_verify(bool on) {
    verify = on;
  }

  Kind step_kind{Kind::Eager};
  bool step_diverged{false};  // op sequence diverged from the capture
  long step_ops{0};  // evals in the current (default-stream) step
  std::string step_first_name;
  long step_batch{0};  // the current step's class: 0 = T=1 AR, S>1 = verify

  static bool shape_match(long a, long b) {
    return std::abs(a - b) <= std::max<long>(64, b / 4);
  }

  // First op of a step: choose the step kind and start the per-step state. batch: 1 (or 0) for a T=1 shape, the token count S > 1 for a verify shape (a 2-D input [1, T] with T > 1). embedding_lookup: the first input is 2-D (a decode embedding gather, not a micro-step lm-head gather — matching on the first op NAME alone mis-arms a replay on the micro-step).
  Kind first_op(const std::string& name, long batch, bool embedding_lookup) {
    step_first_name = name;
    step_batch = batch <= 1 ? 0 : batch;
    if (step_batch > 0 && verify) {
      VerifyKey& k = verify_keys[step_batch];
      if (active && k.replay_armed && k.replay_slot >= 0 &&
          !k.ref_names.empty() && k.ref_names[0] == name &&
          embedding_lookup) {
        step_kind = Kind::Replay;
        k.replay_armed = false;
      } else if (k.next_capture && k.candidate_first_op == name &&
                 embedding_lookup) {
        // active may be false here: the capture's prepare is what activates the arena for the first time. A step whose first op does not match the candidate's (a between-step scalar eval, a cache resize, ...) runs eager; the flag stays set.
        step_kind = Kind::Capture;
      } else {
        step_kind = Kind::Eager;
      }
    } else if (step_batch > 0) {
      // Pre-verify T>1: run it fully eager; clear the T=1 arming (the next small step re-arms).
      step_kind = Kind::Eager;
      next_capture = false;
      replay_armed = false;
    } else if (active && replay_armed && replay_slot >= 0 &&
               !ref_names.empty() && ref_names[0] == name &&
               embedding_lookup) {
      step_kind = Kind::Replay;
      replay_armed = false;
    } else if (next_capture && candidate_first_op == name) {
      // active may be false here: the capture's prepare is what activates the arena for the first time.
      step_kind = Kind::Capture;
    } else {
      step_kind = Kind::Eager;
    }
    step_ops = 1;
    return step_kind;
  }
  // Legacy 2-arg entry (bit-identical pre-verify behavior; big_shape maps to the smallest verify shape so a verify-disabled session takes the pre-verify T>1 branch).
  Kind first_op(const std::string& name, bool big_shape, bool embedding_lookup) {
    return first_op(name, static_cast<long>(big_shape ? 2 : 1), embedding_lookup);
  }

  // A non-first op of a step: the replay divergence check (strict name sequence; a KVCache resize inserts extra ops). Returns true on the FIRST divergence — the caller un-suppresses so the rest of the step runs eagerly.
  bool op(const std::string& name) {
    bool diverged_now = false;
    if (step_kind == Kind::Replay && !step_diverged) {
      const std::vector<std::string>& ref =
          (step_batch > 0 && verify) ? verify_keys[step_batch].ref_names
                                    : ref_names;
      if (step_ops >= (long)ref.size() || ref[step_ops] != name) {
        step_diverged = true;
        diverged_now = true;
      }
    }
    step_ops++;
    return diverged_now;
  }

  // Step-end resolution for a replay step (per class).
  ReplayEnd replay_end(long ops, bool overflowed) {
    const bool big = ops >= big_ops;
    const long slot =
        (step_batch > 0 && verify) ? verify_keys[step_batch].replay_slot
                                  : replay_slot;
    const bool ok = !step_diverged && !big && slot >= 0;
    if (ok && overflowed)
      return ReplayEnd::GrowArena;
    if (ok)
      return ReplayEnd::Launch;
    return ReplayEnd::Discard;
  }

  // Step-end shape resolution for a capture step (per class).
  CaptureEnd capture_end(long ops) {
    const long ref =
        (step_batch > 0 && verify) ? verify_keys[step_batch].capture_ref_ops
                                  : capture_ref_ops;
    if (shape_match(ops, ref))
      return CaptureEnd::Launch;
    if (ops < micro_max)
      return CaptureEnd::MicroLaunch;
    return CaptureEnd::Cooldown;
  }

  // Arm the exec cache after a successful capture: from the next step of this class on, replay this exec (the slot the end_record just wrote).
  void arm_replay(long slot) {
    if (step_batch > 0 && verify) {
      VerifyKey& k = verify_keys[step_batch];
      k.replay_slot = slot;
      k.replay_armed = true;
      k.next_capture = false;
      return;
    }
    replay_slot = slot;
    replay_armed = true;
    next_capture = false;
  }

  // The active class's armed replay slot (-1 when unarmed).
  long cur_replay_slot() const {
    if (step_batch > 0) {
      auto it = verify_keys.find(step_batch);
      if (it != verify_keys.end())
        return it->second.replay_slot;
      return -1;
    }
    return replay_slot;
  }

  // The active class' captured op-name sequence (the replay divergence
  // reference; an unarmed class has an empty sequence).
  const std::vector<std::string>& cur_ref_names() const {
    if (step_batch > 0) {
      static const std::vector<std::string> empty;
      auto it = verify_keys.find(step_batch);
      if (it != verify_keys.end())
        return it->second.ref_names;
      return empty;
    }
    return ref_names;
  }

  // Re-arm after a replay step's GrowArena: the next step of this class is a fresh capture (a new reference) that overwrites the cached exec, which is dropped with it.
  void rearm_capture() {
    if (step_batch > 0 && verify) {
      VerifyKey& k = verify_keys[step_batch];
      k.next_capture = true;
      k.replay_armed = false;
      return;
    }
    next_capture = true;
    replay_armed = false;
  }

  // Re-arm after a replay step's Discard (and after a capture's shape mismatch re-capture): the next step of this class is a fresh capture (a new reference) that overwrites the cached exec, which is dropped with it.
  void discard_rearm() {
    if (step_batch > 0 && verify) {
      VerifyKey& k = verify_keys[step_batch];
      k.small_streak = 0;
      k.prev_small_ops = -1;
      k.next_capture = false;
      k.replay_armed = false;
      return;
    }
    arming = Arming::PrefillSeen;
    small_streak = 0;
    prev_small_ops = -1;
    next_capture = false;
    replay_armed = false;
  }

  // Cool down the active class (an arena setup/launch failure or an
  // overflow at cap): no further captures or replays until the arming
  // streak rebuilds.
  void cool_down() {
    if (step_batch > 0 && verify) {
      VerifyKey& k = verify_keys[step_batch];
      k.small_streak = 0;
      k.prev_small_ops = -1;
      k.next_capture = false;
      k.replay_armed = false;
      return;
    }
    arming = Arming::Cooldown;
    small_streak = 0;
    prev_small_ops = -1;
    next_capture = false;
    replay_armed = false;
  }

  // An eager step end: advance the arming state machine of the step's class. Returns true when a capture was armed (the next step of the class is the capture step).
  // Arm after two consecutive small steps with matching op counts, with or without a big step seen first (a step's OP COUNT does not scale with sequence length — prefill and decode steps of the same model have near-identical op counts, both far below the big threshold — so the original big-step-before-arming requirement can never be satisfied). A big step still resets the streak. With verify on, a T>1 step advances only its own S-keyed streak — the T=1 class is undisturbed (the MTP interleaving: a verify step between two draft steps must not break the draft arming).
  bool eager_end(long ops) {
    if (step_batch > 0)
      return verify ? verify_eager_end(ops) : t1_eager_end(ops);
    return t1_eager_end(ops);
  }

  // Reset the per-step state after a step end (the step_kind/ step_diverged resets the replay path already did are a no-op for the other paths: a step's kind is re-chosen at its first op).
  void step_done() {
    step_kind = Kind::Eager;
    step_diverged = false;
    step_ops = 0;
    step_batch = 0;
  }

 private:
  bool t1_eager_end(long ops) {
    const bool big = ops >= big_ops;
    if (ops >= micro_max)
      candidate_first_op = step_first_name;
    if (big) {
      arming = Arming::PrefillSeen;
      small_streak = 0;
      return false;
    }
    // Micro step (a between-step scalar eval, a cache resize, ...): do not disturb the arming streak (it interleaves with the 1,740-op decode steps); it is not a capture candidate either.
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
  bool verify_eager_end(long ops) {
    VerifyKey& k = verify_keys[step_batch];
    const bool big = ops >= big_ops;
    if (ops >= micro_max)
      k.candidate_first_op = step_first_name;
    if (big) {
      k.small_streak = 0;
      return false;
    }
    // Micro step: do not disturb the streak; not a capture candidate either.
    if (ops < micro_max)
      return false;
    k.small_streak++;
    if (k.small_streak == 1) {
      k.prev_small_ops = ops;
    } else if (k.small_streak == 2 && shape_match(ops, k.prev_small_ops)) {
      k.capture_ref_ops = k.prev_small_ops;
      k.next_capture = true;  // next step of this S is the capture step
      return true;
    } else if (k.small_streak == 2) {
      // No match: slide the window.
      k.prev_small_ops = ops;
      k.small_streak = 1;
    }
    return false;
  }

};

}  // namespace mlx::core::rocm
