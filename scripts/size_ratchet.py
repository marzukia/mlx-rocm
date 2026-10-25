#!/usr/bin/env python3
"""Deterministic size gate. No inference. Fails on new bloat or growth.

Usage: uv run python scripts/size_ratchet.py [--update] [--base <git-ref>]
  --base: enable the AST export gate, comparing top-level exports vs that ref
  --base HEAD: commit time (pre-commit) — the changed set is what is staged
  --update: rewrite size-ratchet.json from current state (run after a split)

Thresholds: 800 = flag (must be ratcheted), 2000 = hard stop (no new
top-level exports). Test files (test_*.py / *_test.py / tests.py) get a x3
flag threshold; a ratcheted file's budget is its ratcheted value (only shrinks).
Stdlib only: json, os, ast, subprocess.
"""

import ast
import json
import os
import re
import subprocess
import sys

FLAG = 800
HARD = 2000
TEST_MULT = 3
RATCHET = "size-ratchet.json"
# Test files: test_*.py, *_test.py, and Django's per-app tests.py convention.
TEST_FILE = re.compile(r"^(test_.*|.*_test|tests)\.py$")
# Generated / non-live-code dirs (adjust per repo).
EXCLUDE = [
    re.compile(r"\.venv"),
    re.compile(r"__pycache__"),
    re.compile(r"(^|/)migrations/"),
]

# Anchor to the repo root (the parent of scripts/) so the gate is cwd-independent.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def walk(d: str) -> list[str]:
    out: list[str] = []
    for e in sorted(os.listdir(d)):
        if e == ".git":
            continue
        p = os.path.normpath(os.path.join(d, e))
        if os.path.isdir(p):
            out.extend(walk(p))
        elif e.endswith(".py") and not any(r.search(p) for r in EXCLUDE):
            out.append(p)
    return out


def count(p: str) -> int:
    with open(p, encoding="utf-8") as fh:
        return sum(1 for line in fh.read().split("\n") if line.strip())


def top_exports(src: str) -> int:
    tree = ast.parse(src)
    return sum(1 for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))


def sh(c: list[str]) -> str:
    return subprocess.run(c, capture_output=True, text=True, check=True).stdout


def gate_err(e: Exception) -> str:
    if isinstance(e, subprocess.CalledProcessError):
        first = (e.stderr or "").strip().splitlines()
        return first[0] if first else str(e)
    return str(e)


def main() -> int:
    args = sys.argv[1:]
    update = "--update" in args
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1] if i + 1 < len(args) else ""  # bare --base: no value
    else:
        base = ""
    if base in ("", "origin/"):  # e.g. CI push to main, where base_ref is empty
        base = ""
    changed: list[str] = []
    renames: dict[str, str] = {}  # new name -> old name (R entries, rename detection)
    if base:
        # base == "HEAD" is commit time (pre-commit): a HEAD...HEAD ref diff is
        # always empty, so the changed set is what is staged for the commit.
        flags = ["--cached"] if base == "HEAD" else [f"{base}...HEAD"]
        try:
            changed = sh(["git", "diff", *flags, "--name-only"]).strip().split("\n")
            status = sh(["git", "diff", *flags, "--name-status", "-M"])
        except subprocess.CalledProcessError as e:
            print(f"GATE-ERROR base {base}: {gate_err(e)}")
            return 1
        # Rename detection collapses R entries to the new name in --name-only;
        # remember the sources so the prev read below can resolve them.
        for line in status.splitlines():
            parts = line.split("\t")
            if parts[0].startswith("R") and len(parts) >= 3:
                renames[parts[2]] = parts[1]
    try:
        with open(RATCHET, encoding="utf-8") as fh:
            ratchet: dict[str, int] = json.load(fh)
    # Parentheses matter: the 3.14-only unparenthesized form is a SyntaxError on
    # python <=3.13; ruff format (target py314) wants to strip them, so skip it.
    except (FileNotFoundError, json.JSONDecodeError):  # fmt: skip
        ratchet = {}

    fail = False
    over: set[str] = set()
    counts: dict[str, int] = {}
    for f in walk("."):
        try:
            n = count(f)
        except (OSError, UnicodeDecodeError) as e:
            print(f"GATE-ERROR count {f}: {e}")
            return 1
        mult = TEST_MULT if TEST_FILE.match(os.path.basename(f)) else 1
        if n > FLAG * mult:
            over.add(f)
            counts[f] = n
            if f not in ratchet:
                if update:
                    ratchet[f] = n
                else:
                    fail = True
                    print(f"NEW BLOAT {f}: {n} lines > {FLAG * mult} - split, do not ratchet")
            elif n > ratchet[f]:
                if update:
                    ratchet[f] = n
                else:
                    fail = True
                    print(f"GROWTH    {f}: {n} > budget {ratchet[f]}")
        if base and n > HARD and f in changed:
            try:
                prev_src = sh(["git", "show", f"{base}:{f}"])
            except subprocess.CalledProcessError:
                if f not in renames:
                    prev_src = ""  # new file (no rename maps to f)
                else:
                    # rename target: --name-only collapsed it to the new name;
                    # read prev from the rename source instead
                    try:
                        prev_src = sh(["git", "show", f"{base}:{renames[f]}"])
                    except subprocess.CalledProcessError as e:
                        print(f"GATE-ERROR ast {f} (base {base}): {gate_err(e)}")
                        return 1
            try:
                prev = top_exports(prev_src) if prev_src.strip() else 0
            except (SyntaxError, ValueError) as e:
                print(f"GATE-ERROR ast {f} (base {base}): {e}")
                return 1
            if base == "HEAD":
                # commit time: gate the staged content, not the worktree
                try:
                    cur_src = sh(["git", "show", f":{f}"])
                except subprocess.CalledProcessError as e:
                    if "not in the index" not in (e.stderr or ""):
                        print(f"GATE-ERROR ast {f}: {gate_err(e)}")
                        return 1
                    cur = 0  # staged untrack (git rm --cached): exports leave
                else:
                    try:
                        cur = top_exports(cur_src)
                    except (SyntaxError, ValueError) as e:
                        print(f"GATE-ERROR ast {f}: {e}")
                        return 1
            else:
                try:
                    with open(f, encoding="utf-8") as fh:
                        cur_src = fh.read()
                    cur = top_exports(cur_src)
                except (OSError, SyntaxError, ValueError) as e:
                    print(f"GATE-ERROR ast {f}: {gate_err(e)}")
                    return 1
            if cur > prev:
                fail = True
                print(f"EXPORT GATE {f}: {cur} top-level exports (was {prev}) - extract to a sibling module")
    if update:
        # Rewrite from current state: drop keys for files no longer over the
        # flag (deleted or shrunk back under 800); budgets for surviving files
        # only shrink per the ratchet rule (min lets a shrunken file lower its
        # budget; new/grown entries were already raised to n in the loop).
        ratchet = {f: min(ratchet[f], counts[f]) for f in sorted(over)}
        with open(RATCHET, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(ratchet, indent=2) + "\n")
    print("SIZE GATE: FAIL" if fail else f"SIZE GATE: OK ({len(ratchet)} ratcheted)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
