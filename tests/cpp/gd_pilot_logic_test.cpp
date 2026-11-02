// Hermetic unit tests for the graph decode pilot state machine
// (vendor/mlx-rocm-upstream/mlx/backend/rocm/graph_decode_pilot_logic.h).
// No GPU, no HIP: plain C++ main with checks. Driven by
// tests/test_gd_pilot_logic.py (compile with the system C++ compiler,
// run, expect exit 0).
#include "graph_decode_pilot_logic.h"
#include <cstdio>
#include <string>
#include <vector>

using mlx::core::rocm::GdPilotLogic;

static int g_failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      g_failures++;                                                   \
    }                                                                 \
  } while (0)

static GdPilotLogic armed_capture() {
  GdPilotLogic p;
  p.active = true;
  p.step_first_name = "Gather";
  CHECK(!p.eager_end(1740));  // streak 1
  CHECK(p.eager_end(1740));   // streak 2, matched -> capture armed
  CHECK(p.next_capture);
  return p;
}

// Arming state machine.
static void test_arming() {
  GdPilotLogic p;
  p.step_first_name = "Gather";
  // Two consecutive small steps with matching op counts arm a capture.
  CHECK(!p.eager_end(1740));
  CHECK(p.small_streak == 1);
  CHECK(!p.next_capture);
  CHECK(p.eager_end(1740));
  CHECK(p.next_capture);
  CHECK(p.capture_ref_ops == 1740);
  // A big step resets the streak (and marks a prefill seen).
  GdPilotLogic b;
  b.step_first_name = "Gather";
  b.eager_end(1740);
  b.eager_end(4000);
  CHECK(b.arming == GdPilotLogic::Arming::PrefillSeen);
  CHECK(b.small_streak == 0);
  CHECK(!b.next_capture);
  // A micro step does not disturb the streak and is not a candidate.
  GdPilotLogic m;
  m.step_first_name = "Gather";
  m.eager_end(1740);
  const long streak_before = m.small_streak;
  const std::string cand_before = m.candidate_first_op;
  m.step_first_name = "Squeeze";  // a micro step's first op
  m.eager_end(2);
  CHECK(m.small_streak == streak_before);
  CHECK(m.candidate_first_op == cand_before);  // not updated by a micro
  // A micro step on an empty pilot arms nothing.
  GdPilotLogic m2;
  m2.step_first_name = "Squeeze";
  CHECK(!m2.eager_end(2));
  CHECK(m2.small_streak == 0);
  // Two mismatched small steps slide the window (no arm).
  GdPilotLogic s;
  s.step_first_name = "Gather";
  s.eager_end(1740);
  CHECK(!s.eager_end(999));  // 999 vs 1740: no shape match
  CHECK(!s.next_capture);
  CHECK(s.small_streak == 1);  // window slid: 999 is the new first
  CHECK(s.prev_small_ops == 999);
  CHECK(s.eager_end(999));  // now they match
  CHECK(s.next_capture);
  // Shape match tolerance: +/- (64, b/4) inclusive.
  GdPilotLogic t;
  t.step_first_name = "Gather";
  t.eager_end(1740);
  CHECK(t.eager_end(1740 + 64));  // 64 is within max(64, b/4)=435
  CHECK(t.next_capture);
}

// Kind gating on the first op.
static void test_kind_gating() {
  // T>1 (big shape) runs fully eager and clears capture/replay arming.
  GdPilotLogic p = armed_capture();
  p.replay_armed = true;  // also test that T>1 clears the replay arm
  p.replay_slot = 0;
  p.ref_names = {"Gather", "Squeeze"};
  CHECK(p.first_op("Gather", /*big_shape=*/true, true) ==
        GdPilotLogic::Kind::Eager);
  CHECK(!p.next_capture);
  CHECK(!p.replay_armed);
  // A replay-armed session replays a matching embedding-lookup step.
  GdPilotLogic r = armed_capture();
  r.arm_replay(1);
  r.ref_names = {"Gather", "Squeeze"};
  CHECK(r.first_op("Gather", false, true) == GdPilotLogic::Kind::Replay);
  CHECK(!r.replay_armed);  // consumed by this step
  // The micro-step trap: a matching op NAME without the 2-D first input
  // (the lm-head weights gather) must NOT replay.
  GdPilotLogic r2 = armed_capture();
  r2.arm_replay(1);
  r2.ref_names = {"Gather", "Squeeze"};
  CHECK(r2.first_op("Gather", false, /*embedding_lookup=*/false) ==
        GdPilotLogic::Kind::Eager);
  CHECK(r2.replay_armed);  // the arm is NOT consumed
  // A non-matching first op name runs eager and keeps the flags
  // (the candidate step still captures next).
  GdPilotLogic r3 = armed_capture();
  CHECK(r3.first_op("Add", false, true) == GdPilotLogic::Kind::Eager);
  CHECK(r3.next_capture);
  CHECK(!r3.replay_armed);  // was never replay-armed
  // A capture-armed session captures the matching candidate step.
  GdPilotLogic c = armed_capture();
  CHECK(c.first_op("Gather", false, false) == GdPilotLogic::Kind::Capture);
  // A replay arm without an active arena stays eager (the capture path
  // is the one that activates the arena).
  GdPilotLogic i;
  i.active = false;
  i.replay_armed = true;
  i.replay_slot = 0;
  i.ref_names = {"Gather"};
  CHECK(i.first_op("Gather", false, true) == GdPilotLogic::Kind::Eager);
  // An empty ref_names vector never replays.
  GdPilotLogic e;
  e.active = true;
  e.replay_armed = true;
  e.replay_slot = 0;
  CHECK(e.first_op("Gather", false, true) == GdPilotLogic::Kind::Eager);
  // step_done resets the per-step state.
  i.step_done();
  CHECK(i.step_ops == 0);
  CHECK(i.step_kind == GdPilotLogic::Kind::Eager);
}

// The replay divergence check.
static void test_divergence() {
  GdPilotLogic p = armed_capture();
  p.arm_replay(0);
  p.ref_names = {"Gather", "MatMul", "Add"};
  p.first_op("Gather", false, true);
  CHECK(p.step_kind == GdPilotLogic::Kind::Replay);
  CHECK(!p.op("MatMul"));  // op index 1 matches
  CHECK(!p.step_diverged);
  CHECK(p.op("Concatenate"));  // op index 2 diverges from "Add"
  CHECK(p.step_diverged);
  CHECK(!p.op("Add"));  // a late op no longer re-triggers
  // A step longer than the reference diverges past the end.
  GdPilotLogic q = armed_capture();
  q.arm_replay(0);
  q.ref_names = {"Gather", "MatMul"};
  q.first_op("Gather", false, true);
  CHECK(!q.op("MatMul"));
  CHECK(q.op("Extra"));  // index 2 >= ref size
  CHECK(q.step_diverged);
  // An exact reference-length step never diverges.
  GdPilotLogic ok = armed_capture();
  ok.arm_replay(0);
  ok.ref_names = {"Gather", "MatMul"};
  ok.first_op("Gather", false, true);
  CHECK(!ok.op("MatMul"));
  CHECK(!ok.step_diverged);
}

// The step-end resolution (the F5/F6 contract).
static void test_step_end() {
  // A clean small replay step launches.
  GdPilotLogic p = armed_capture();
  p.arm_replay(0);
  p.ref_names = {"Gather", "MatMul"};
  p.first_op("Gather", false, true);
  CHECK(p.replay_end(1740, false) == GdPilotLogic::ReplayEnd::Launch);
  // An overflowed clean step grows the arena (best-effort launch).
  GdPilotLogic g = armed_capture();
  g.arm_replay(0);
  g.ref_names = {"Gather"};
  g.first_op("Gather", false, true);
  CHECK(g.replay_end(1740, true) == GdPilotLogic::ReplayEnd::GrowArena);
  // F5: a DIVERGED step discards (no launch) — the graph's node
  // sequence is stale, and launching it would re-write the persistent
  // state buffers after the eager suffix ran on the suppressed
  // prefix's stale intermediates.
  GdPilotLogic d = armed_capture();
  d.arm_replay(0);
  d.ref_names = {"Gather", "MatMul"};
  d.first_op("Gather", false, true);
  d.op("Concatenate");  // diverge
  CHECK(d.step_diverged);
  CHECK(d.replay_end(1740, false) == GdPilotLogic::ReplayEnd::Discard);
  // F6: a BIG step discards (no launch) — its per-op launches were
  // suppressed for the whole step, and the smaller cached graph would
  // never compute this step's real outputs.
  GdPilotLogic b = armed_capture();
  b.arm_replay(0);
  b.ref_names = {"Gather"};
  b.first_op("Gather", false, true);
  CHECK(b.replay_end(4000, false) == GdPilotLogic::ReplayEnd::Discard);
  // Diverged AND big still discards.
  GdPilotLogic db = armed_capture();
  db.arm_replay(0);
  db.ref_names = {"Gather", "MatMul"};
  db.first_op("Gather", false, true);
  db.op("Concatenate");
  CHECK(db.replay_end(4000, false) == GdPilotLogic::ReplayEnd::Discard);
  // The discard re-arm resets to a fresh capture.
  d.discard_rearm();
  CHECK(d.arming == GdPilotLogic::Arming::PrefillSeen);
  CHECK(d.small_streak == 0);
  CHECK(d.prev_small_ops == -1);
  CHECK(!d.next_capture);
  CHECK(!d.replay_armed);
  // The capture-step shape resolution.
  GdPilotLogic c = armed_capture();
  CHECK(c.capture_end(1740) == GdPilotLogic::CaptureEnd::Launch);
  CHECK(c.capture_end(1740 + 64) == GdPilotLogic::CaptureEnd::Launch);
  GdPilotLogic cm = armed_capture();
  cm.capture_ref_ops = 1740;
  CHECK(cm.capture_end(2) == GdPilotLogic::CaptureEnd::MicroLaunch);
  CHECK(cm.capture_end(999) == GdPilotLogic::CaptureEnd::Cooldown);
}

int main() {
  test_arming();
  test_kind_gating();
  test_divergence();
  test_step_end();
  if (g_failures) {
    std::printf("gd_pilot_logic: %d CHECK failure(s)\n", g_failures);
    return 1;
  }
  std::printf("gd_pilot_logic: all checks passed\n");
  return 0;
}
