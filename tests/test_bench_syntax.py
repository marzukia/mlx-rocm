"""Hermetic syntax gate: every bench/*.py must ast.parse and every bench/*.sh must pass bash -n; added by the round-6 fix (639e141's reflow joined code lines as prose and left six bench evidence files unparseable, invisible because they are extend-excluded from ruff and unimported) so this class of breakage cannot re-enter silently (REVIEW-PR6-R5, PR #6)."""

import ast
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench"


def test_bench_python_files_parse():
    py_files = sorted(BENCH.glob("*.py"))
    assert py_files, "no bench/*.py found in the worktree"
    bad = []
    for p in py_files:
        try:
            ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as e:
            bad.append(f"{p.name}: {e}")
    assert not bad, "unparseable bench python files:\n" + "\n".join(bad)


def test_bench_shell_files_pass_bash_n():
    sh_files = sorted(BENCH.glob("*.sh"))
    assert sh_files, "no bench/*.sh found in the worktree"
    bad = []
    for p in sh_files:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{p.name}: {r.stderr.strip()}")
    assert not bad, "bench shell files failing bash -n:\n" + "\n".join(bad)
