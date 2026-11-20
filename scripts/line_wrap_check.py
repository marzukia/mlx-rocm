#!/usr/bin/env python3
"""Deterministic hard-wrap detector (the no-manual-line-wrap rule).

Counts "hard-wrapped pairs": two adjacent, non-blank, column-0 prose
lines in an in-scope file where BOTH lines are shorter than the wrap
budget (i.e. the author wrapped the line by hand at the budget width;
flowing prose never emits two budget-short lines back to back).

In scope: .md / .markdown / .txt / .html files, plus a PR body file
(--body). Out of scope: fenced code blocks, indented code, list items
(-, *, + at column 0), table rows (| at column 0), headings (#),
blockquotes (>), and tag-only HTML lines. Code files and commit
messages are never scanned.

A hard-wrapped pair is flagged when len(a) < BUDGET and len(b) < BUDGET.

Usage:
  python3 scripts/line_wrap_check.py FILE [FILE ...] [--body BODYFILE]
Exit 1 if any in-scope hits; prints one line per hit:
  file:line <len_a>+1+len_b (<= BUDGET)
Exit 0 with the total hit count on the last line.
"""

import sys

BUDGET = 80  # a pair is flagged when both lines are < BUDGET chars

SCOPED_SUFFIXES = (".md", ".markdown", ".txt", ".html")

MARKERS = ("#", "|", ">", "-", "*", "+")


def is_prose(line: str, in_fence: bool) -> bool:
    if in_fence or not line.strip():
        return False
    if line[:1] in (" ", "\t"):  # indented code / continuation
        return False
    s = line.lstrip()
    if s[:1] in MARKERS:
        # a bare "-"/"*"/"+" only counts as prose if it is not a list item
        if s[:1] in ("-", "*", "+") and (len(s) == 1 or s[1] == " "):
            return False
        return False
    return not (s.startswith("<") and s.endswith(">") and " " not in s and "\t" not in s)


def scan(lines, name: str):
    hits = []
    in_fence = False
    prev = None
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")
        if stripped.lstrip()[:3] == "```" or stripped.lstrip()[:4] == "~~~":
            in_fence = not in_fence
            prev = None
            continue
        prose = is_prose(stripped, in_fence)
        if prose:
            if prev is not None and len(prev) < BUDGET and len(stripped) < BUDGET:
                hits.append(f"{name}:{i} {len(prev)},{len(stripped)} (both < {BUDGET})")
            prev = stripped
        else:
            prev = None
    return hits


def main(argv):
    files = []
    body = None
    i = 0
    while i < len(argv):
        if argv[i] == "--body":
            body = argv[i + 1]
            i += 2
            continue
        files.append(argv[i])
        i += 1

    total = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            hits = scan(fh.read().split("\n"), f)
        total += len(hits)
        for h in hits:
            print(h)
    if body:
        with open(body, encoding="utf-8") as fh:
            hits = scan(fh.read().split("\n"), "PR-body")
        total += len(hits)
        for h in hits:
            print(h)

    print(f"total hard-wrap pairs: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
