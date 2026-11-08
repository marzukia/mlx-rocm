"""t2261 PH battery (cloudy ruling 11-06 12:5xZ): per-verify-step GPU span
(span.so) + roctx markers for the rocprofv3 kernel CSV join (7.2.4
userspace stack, env-driven via LD_PRELOAD) + t_q host marker after the
S=3 lm() return (enqueue/evalwait split). All no-ops when the libs are
absent. No added syncs: span_end runs on the drained queue (each verify
step drains via the int(argmax) syncs).

Split out of mtp-bench6-stepdec-ph.py for the size gate (830 > 800
non-blank lines - "split, do not ratchet"): behavior and module-load side
effects are byte-identical (same env gates, same lib paths, same print
markers); the box as-run copy of the bench (md5 d12fae08) keeps the
inline form and stays untouched (tree-copy-only split).
"""

import ctypes as _ct
import os

_SP = None
if os.environ.get("SPAN", "1") == "1":
    _SP = _ct.CDLL(os.environ.get("SPAN_SO", "/opt/fleet/span.so"))
    _SP.span_init.argtypes = []
    _SP.span_init.restype = _ct.c_int
    _SP.span_begin.argtypes = []
    _SP.span_end.argtypes = []
    _SP.span_end.restype = _ct.c_double


def _span_begin():
    if _SP is not None:
        _SP.span_begin()


def _span_end():
    return _SP.span_end() if _SP is not None else -1.0


_ROCTX = None
if os.environ.get("ROCTX", "1") == "1":
    try:
        _ROCTX = _ct.CDLL(os.environ.get("ROCTX_SO", "/opt/fleet/librocprofiler-sdk-roctx.so"))
        _ROCTX.roctxRangeStartA.argtypes = [_ct.c_char_p]
        _ROCTX.roctxRangeStartA.restype = None
        _ROCTX.roctxRangeStop.argtypes = []
        _ROCTX.roctxRangeStop.restype = None
        print("ROCTX_INIT_OK", flush=True)
    except OSError as _e:
        _ROCTX = None
        print("ROCTX_DISABLED %s" % _e, flush=True)  # noqa: UP031 (verbatim from the as-run bench block)


def _mstart(i):
    if _ROCTX is not None:
        _ROCTX.roctxRangeStartA(("ph_v%04d" % i).encode())  # noqa: UP031 (verbatim from the as-run bench block)


def _mend():
    if _ROCTX is not None:
        _ROCTX.roctxRangeStop()
