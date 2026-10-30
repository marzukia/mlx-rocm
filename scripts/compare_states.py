#!/usr/bin/env python3
"""Cross-process GDN state comparison (fused vs unfused).

Usage: compare_states.py <state_A.jsonl> <state_B.jsonl>
Exits 0 only if every (step, layer) entry matches byte-for-byte (sha256) and
the captured count is non-trivial (n == n_expected > 0, all states nonzero).
"""

import json
import sys


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d[(r["step"], r["layer"])] = (r["sha"], r["nbytes"], r["nonzero"])
    return d


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    n_expected = len(a)
    assert n_expected > 0, "vacuous: no state entries in A"
    assert len(b) == n_expected, f"count mismatch: A={len(a)} B={len(b)}"
    assert set(a) == set(b), "step/layer key sets differ"
    mism = [k for k in a if a[k][0] != b[k][0]]
    assert not mism, f"state sha mismatch at {len(mism)} entries, e.g. {mism[:3]}"
    nz_a = sum(1 for v in a.values() if v[2])
    nz_b = sum(1 for v in b.values() if v[2])
    assert nz_a == n_expected, f"vacuous: {n_expected - nz_a} zero states in A"
    assert nz_b == n_expected, f"vacuous: {n_expected - nz_b} zero states in B"
    print(f"STATE-MATCH PASS n={n_expected} steps={sorted({k[0] for k in a})} layers={len({k[1] for k in a})}")


if __name__ == "__main__":
    main()
