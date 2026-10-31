"""Hermetic unit test for the graph decode pilot state machine.

Compiles tests/cpp/gd_pilot_logic_test.cpp (which includes the pure,
HIP-free header
vendor/mlx-rocm-upstream/mlx/backend/rocm/graph_decode_pilot_logic.h)
with the system C++ compiler and runs it. No GPU needed: the header
holds only the pilot's transition logic (arming streak, kind gating,
replay divergence, step-end resolution incl. the F5/F6 discard
contract); eval.cpp mirrors those decisions around the HIP calls.

If the machine has no C++ compiler (g++/c++/clang++), the compile-and-
run test skips; the source-presence test always runs.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tests" / "cpp" / "gd_pilot_logic_test.cpp"
HDR = (
    ROOT
    / "vendor"
    / "mlx-rocm-upstream"
    / "mlx"
    / "backend"
    / "rocm"
    / "graph_decode_pilot_logic.h"
)


def _cc() -> str | None:
    for name in ("g++", "c++", "clang++"):
        p = shutil.which(name)
        if p:
            return p
    return None


def test_pilot_logic_header_and_test_exist():
    assert SRC.is_file(), "tests/cpp/gd_pilot_logic_test.cpp missing"
    assert HDR.is_file(), "graph_decode_pilot_logic.h missing"
    # The test includes the header by name; the header must be
    # self-contained (no HIP/MLX includes).
    src = SRC.read_text()
    assert '#include "graph_decode_pilot_logic.h"' in src
    hdr = HDR.read_text()
    incs = [l for l in hdr.splitlines() if l.startswith("#include")]
    assert all("hip" not in i.lower() for i in incs), "the logic header must stay HIP-free (its #includes)"


def test_eval_cpp_mirrors_the_pilot_logic():
    # The transition logic must live in the header (unit-tested), and
    # eval.cpp must consume it (not carry a second copy of the arming
    # state machine).
    eval_src = (
        ROOT
        / "vendor"
        / "mlx-rocm-upstream"
        / "mlx"
        / "backend"
        / "rocm"
        / "eval.cpp"
    ).read_text()
    assert "graph_decode_pilot_logic.h\"" in eval_src
    assert "GdPilotLogic" in eval_src
    for call in ("first_op(", "op(", "replay_end(", "capture_end(", "eager_end(", "arm_replay(", "discard_rearm("):
        assert call in eval_src, f"eval.cpp must call the logic's {call}"


@pytest.mark.skipif(_cc() is None, reason="no C++ compiler on this machine")
def test_pilot_logic_transitions():
    out = ROOT / "build" / "gd_pilot_logic_test"
    out.parent.mkdir(exist_ok=True)
    r = subprocess.run(
        [_cc(), "-std=c++17", "-Wall", "-Werror", "-I", str(HDR.parent), str(SRC), "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"compile failed:\n{r.stderr}"
    r2 = subprocess.run([str(out)], capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0, f"test main failed (exit {r2.returncode}):\n{r2.stdout}\n{r2.stderr}"
    assert "all checks passed" in r2.stdout
    # The check count guards against a vacuous test: the main must
    # report the single success line and nothing else.
    lines = [l for l in r2.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, lines
