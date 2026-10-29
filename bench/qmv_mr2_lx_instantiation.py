#!/usr/bin/env python3
"""t889 (t903 MINOR-2 fix): the template-instantiation assert for the
qmv_wide_mr_kernel A/B — non-vacuous on mangled symbol names.

The MR kernel is qmv_wide_mr_kernel<T, ScaleT, GROUP_SIZE, BITS,
LXHOIST> (scripts/apply-qmv-mr-lxhoist.py P1): the bool LXHOIST is the
LAST template parameter, so a symbol name carries it as
  - mangled (rocprofv3 --kernel-trace on this toolchain): Lb0
    (false = t825 path) / Lb1 (true = hoist), immediately before the E
    that terminates the template-argument list — the t889 trace names
    end ...Li4ELb0E (LX=0) / ...Li4ELb1E (LX=1);
  - demangled: ", false>" / ", true>" (the bool is the last template
    argument, so the marker is unambiguous).
A wrong instantiation (a stale .so, or the other template branch
dispatched) carries the opposite marker and fails. The per-row
qmv_wide_kernel (wide fallback / drift control) has no LX template
parameter and is not checked.

The t889 runner's original check was conditional on the name containing
"<" (demangled) and was a no-op for the mangled names rocprofv3
actually reports; this module is the sound replacement (proven
non-vacuous offline by tests/test_qmv_mr2_lx_instantiation.py).

Usage (gate (c) of qmv_mr2_lx_run.sh):
  python3 qmv_mr2_lx_instantiation.py <combos0.json> <combos1.json>
Exit 0 = PASS, 1 = FAIL (the first failing combo is printed).
"""

import json
import sys

# LXHOIST is the last template parameter: the mangled bool argument is
# followed by the E that closes the template-argument list.
_MANGLED = {False: "Lb0E", True: "Lb1E"}
_DEMANGLED = {False: ", false>", True: ", true>"}


def lx_instantiation_ok(full_name: str, want_lx: bool) -> bool:
    """True iff full_name is a qmv_wide_mr_kernel symbol instantiated
    with LXHOIST == want_lx (False = t825 path, True = hoist).

    False for any other kernel (e.g. the per-row qmv_wide_kernel, which
    has no LX template parameter) and for a name that carries the
    opposite marker (a stale .so / the other template branch).
    """
    if "qmv_wide_mr_kernel" not in full_name:
        return False
    if "<" in full_name:  # demangled
        want, got = _DEMANGLED[want_lx], _DEMANGLED[not want_lx]
    else:  # mangled (rocprofv3 --kernel-trace)
        want, got = _MANGLED[want_lx], _MANGLED[not want_lx]
    return want in full_name and got not in full_name


def check_combos(combos0: list, combos1: list) -> None:
    """Gate (c): the kernel family per shape + the template
    instantiation (non-vacuous on mangled names). Raises AssertionError
    on the first failure."""
    for tag, expect in (("0", combos0), ("1", combos1)):
        want_lx = tag == "1"
        for c in expect:
            names = c["kernel_names"]
            full = c["kernel_names_full"]
            # down M=1 fits the MR gate (LDS 24576 <= 32768) — MR kernel;
            # down M>=2 (LDS > 32 KB) is the per-row wide fallback for
            # BOTH env values (the drift control).
            wide_fallback = c["tag"] == "down" and c["M"] >= 2
            want = "qmv_wide_kernel" if wide_fallback else "qmv_wide_mr_kernel"
            assert want in names, (tag, c, names)
            if not wide_fallback:
                for f in full:
                    assert lx_instantiation_ok(f, want_lx), (tag, c["tag"], c["M"], f)


def main() -> None:
    with open(sys.argv[1]) as f:
        a = json.load(f)
    with open(sys.argv[2]) as f:
        b = json.load(f)
    check_combos(a, b)
    print("CONFIG-ASSERT: PASS (kernel family per shape + template instantiation)", flush=True)


if __name__ == "__main__":
    main()
