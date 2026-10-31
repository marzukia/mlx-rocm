"""Non-vacuity test for the run-889 template-instantiation assert (run-903
MINOR-2 fix): the check must pass on a known-correct symbol name and
fail on a known-wrong one (the other template branch / a stale .so).
No GPU needed.

The names are the mangled qmv_wide_mr_kernel symbol names recorded in
the run-889 A/B artifacts (combos0/1.json; quoted verbatim in the run-903
review, section 2): the template-argument list ends in the bool LXHOIST
— the LAST template parameter (P1: <T, ScaleT, GROUP_SIZE, BITS,
LXHOIST>) — followed by the E that closes the list:
  LX=0: ...Li4ELb0E   (LXHOIST = false, run-825 path)
  LX=1: ...Li4ELb1E   (LXHOIST = true, hoist)
The appended tail is the mangled function signature (a representative
one — pointers and ints only, no further template arguments, so no
further Lb sequences).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from qmv_mr2_lx_instantiation import check_combos, lx_instantiation_ok  # noqa: E402

# run-889 artifact names (quoted verbatim in the run-903 review, section 2);
# the appended tail is the mangled function signature (representative).
SIG = "vPK12hip_bfloat16PKhS8_S9_PS0_iiib"
NAME_LX0 = "qmv_wide_mr_kernelI12hip_bfloat16S3_Li64ELi4ELb0E" + SIG
NAME_LX1 = "qmv_wide_mr_kernelI12hip_bfloat16S3_Li64ELi4ELb1E" + SIG
# the per-row qmv_wide_kernel (wide fallback / drift control) has no LX
# template parameter — a representative mangled name.
NAME_WIDE = "qmv_wide_kernelI12hip_bfloat16S3_Li64ELi4E" + SIG
# demangled form (the run-889 runner's original check path, kept for
# toolchains that demangle).
NAME_DEMANGLED_LX0 = (
    "mlx::amd::qmv_wide_mr_kernel<hip_bfloat16, int, 64, 4, false>("
    "const hip_bfloat16*, const unsigned char*, const int*, const int*,"
    " hip_bfloat16*, int, int, int, bool)"
)
NAME_DEMANGLED_LX1 = NAME_DEMANGLED_LX0.replace(", false>", ", true>")


def test_mangled_correct_instantiation_passes():
    # LX=0 env: the run-825 path (LXHOIST = false)
    assert lx_instantiation_ok(NAME_LX0, False)
    # LX=1 env: the hoist (LXHOIST = true)
    assert lx_instantiation_ok(NAME_LX1, True)


def test_mangled_wrong_instantiation_fails():
    # a stale .so, or the other template branch dispatched: the name
    # carries the opposite marker and must fail
    assert not lx_instantiation_ok(NAME_LX1, False)
    assert not lx_instantiation_ok(NAME_LX0, True)


def test_wide_fallback_kernel_is_not_the_mr_kernel():
    # the per-row qmv_wide_kernel has no LX template parameter
    assert not lx_instantiation_ok(NAME_WIDE, False)
    assert not lx_instantiation_ok(NAME_WIDE, True)


def test_demangled_names_still_checked():
    assert lx_instantiation_ok(NAME_DEMANGLED_LX0, False)
    assert not lx_instantiation_ok(NAME_DEMANGLED_LX1, False)
    assert lx_instantiation_ok(NAME_DEMANGLED_LX1, True)


def _combo(tag: str, m: int, names: list, full: list) -> dict:
    return {
        "tag": tag,
        "K": 4096,
        "N": 12288,
        "M": m,
        "kernel_names": names,
        "kernel_names_full": full,
    }


def test_check_combos_passes_on_the_correct_instantiation():
    a = [_combo("gate_up", 1, ["qmv_wide_mr_kernel"], [NAME_LX0])]
    b = [_combo("gate_up", 1, ["qmv_wide_mr_kernel"], [NAME_LX1])]
    check_combos(a, b)  # must not raise


def test_check_combos_fails_on_a_wrong_instantiation():
    # LX=0 env but the LX=1 kernel (a stale .so / a dispatch regression)
    a = [_combo("gate_up", 1, ["qmv_wide_mr_kernel"], [NAME_LX1])]
    b = [_combo("gate_up", 1, ["qmv_wide_mr_kernel"], [NAME_LX1])]
    with pytest.raises(AssertionError):
        check_combos(a, b)
