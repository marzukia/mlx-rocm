#!/usr/bin/env python3
"""Transform mtp-bench6-canonical.py (box copy) into mtp-bench6-stepspan.py:
wrap each bench loop iteration with wall + span (HIP-event) measurement.

- Inserts the span import block after the EOS constant.
- Wraps the `for _ in range(MAX_TOKENS):` body in try/finally with
  SPANSTEP printing. `break` (EOS) still runs the finally block, so the
  partial final iteration is measured and tagged; nothing else changes.
"""

import sys

with open(sys.argv[1]) as f:
    src = f.read().splitlines(keepends=False)
out = []
i = 0
N = len(src)

# 1) import block after EOS line
while i < N:
    out.append(src[i])
    if src[i].startswith("EOS = 248044"):
        out += [
            "",
            "# t2246 seam: per-step GPU span via HIP events on the legacy default",
            "# stream (span.so). SPAN=0 disables. span_end() syncs the GPU; the",
            "# bench already syncs each step via int(mx.argmax(...)), so the",
            "# extra cost is ~0.",
            "import ctypes as _ct",
            "_SP = None",
            'if os.environ.get("SPAN", "1") == "1":',
            '    _SP = _ct.CDLL(os.environ.get("SPAN_SO", os.path.expanduser("~/t2246/span.so")))',
            "    _SP.span_init.argtypes = []",
            "    _SP.span_init.restype = _ct.c_int",
            "    _SP.span_begin.argtypes = []",
            "    _SP.span_end.argtypes = []",
            "    _SP.span_end.restype = _ct.c_double",
            '    print("SPAN_INIT_OK", flush=True)',
            "else:",
            '    print("SPAN_DISABLED", flush=True)',
            "",
            "",
            "def _span_begin():",
            "    if _SP is not None:",
            "        _SP.span_begin()",
            "",
            "",
            "def _span_end():",
            "    return _SP.span_end() if _SP is not None else -1.0",
        ]
    i += 1
src = out

# 2) wrap the bench loop
out = []
i = 0
N = len(src)
body_start = None
body_end = None
while i < N:
    if src[i] == "    t1 = time.time()":
        out.append('    if _SP is not None:\n        assert _SP.span_init() == 0, "span_init failed"\n')
        out.append(src[i])
        i += 1
        continue
    if src[i] == "    for _ in range(MAX_TOKENS):":
        out.append("    for it in range(MAX_TOKENS):")
        out.append("        _ws = time.perf_counter()")
        out.append('        _kind = "ar" if (draft1 is None) else "verify"')
        out.append("        _span_begin()")
        out.append("        try:")
        body_start = i + 1
        i += 1
        continue
    if src[i] == "    dt = time.time() - t1" and body_start is not None:
        # re-indent captured body lines under try
        for j in range(body_start, i):
            line = src[j]
            if line.strip():
                out.append("    " + line)
            else:
                out.append("")
        out.append("        finally:")
        out.append("            _we = time.perf_counter()")
        out.append("            _span_ms = _SP.span_end() if _SP is not None else -1.0")
        out.append("            _wall_ms = (_we - _ws) * 1000.0")
        out.append("            _seam_ms = (_wall_ms - _span_ms) if _span_ms >= 0 else -1.0")
        out.append(
            '            print(f"SPANSTEP {it} {_kind} wall={_wall_ms:.3f} '
            'span={_span_ms:.3f} seam={_seam_ms:.3f}", flush=True)'
        )
        out.append(src[i])
        body_start = None
        i += 1
        continue
    if body_start is not None:
        i += 1
        continue
    out.append(src[i])
    i += 1

assert body_start is None, "loop body not closed"
with open(sys.argv[2], "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote", sys.argv[2])
