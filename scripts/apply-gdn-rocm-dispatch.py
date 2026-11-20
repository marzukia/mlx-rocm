#!/usr/bin/env python3
"""Apply the GDN-ROCM-DISPATCH (3b) patch to an installed mlx_lm
gated_delta.py in a venv.

The stock mlx-lm routing in gated_delta_update() sends every GDN call to
the per-token Python loop (gated_delta_ops) when mx.metal.is_available()
is False -- which is always the case on the ROCm wheel. This patch adds
a dispatch branch, checked BEFORE the stock routing, that routes to the
fused fast::GatedDeltaUpdate (bound at mx.fast.gated_delta_update) when
the C++ use_fallback geometry guard matches:

  GPU device, no mask, Dk == 128, Dv == 128, Hv % Hk == 0, Hv / Hk <= 8

Non-matching geometries keep the existing Python fallback (the branch is
NOT extended). Qwen3.8-27B GDN geometry (Dk = Dv = 128, Hk = 16, Hv = 48,
ratio = 3, scalar g, bf16, no mask) is in coverage.

Usage:
  python apply-gdn-rocm-dispatch.py --venv /path/to/venv          # apply
  python apply-gdn-rocm-dispatch.py --venv /path/to/venv --check  # verify
"""

import argparse
import ast
import glob
import os
import sys

MARKER = "GDN-ROCM-DISPATCH"

BACKUP_SUFFIX = ".bak-gdn3b"

# The text that the stock file has at the insertion point.
ANCHOR = """    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)"""

REPLACEMENT = """    # GDN-ROCM-DISPATCH (3b): on the ROCm wheel mx.metal.is_available() is
    # False, so the stock routing falls through to the per-token Python
    # loop (gated_delta_ops) for every GDN call. The fused
    # fast::GatedDeltaUpdate (chunked scan; recurrent kernel for T <= 8)
    # is bound at mx.fast.gated_delta_update. Route there when the C++
    # use_fallback geometry guard matches: GPU device, no mask,
    # Dk == 128, Dv == 128, Hv % Hk == 0, Hv / Hk <= 8. Non-matching
    # geometries keep the Python fallback below (not extended).
    # Qwen3.8-27B: Dk = Dv = 128, Hk = 16, Hv = 48, ratio = 3, scalar g,
    # bf16, no mask -> in coverage.
    if (
        use_kernel
        and mx.default_device() == mx.gpu
        and not mx.metal.is_available()
        and mask is None
        and q.shape[-1] == 128
        and v.shape[-1] == 128
        and v.shape[-2] % q.shape[-2] == 0
        and v.shape[-2] // q.shape[-2] <= 8
    ):
        return mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)"""


def find_target(venv: str):
    pats = [
        os.path.join(venv, "lib", "python3.*", "site-packages", "mlx_lm", "models", "gated_delta.py"),
        os.path.join(venv, "lib64", "python3.*", "site-packages", "mlx_lm", "models", "gated_delta.py"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    target = find_target(args.venv)
    if target is None:
        print(f"[gdn3b] FATAL: gated_delta.py not found under {args.venv}", file=sys.stderr)
        sys.exit(1)
    print(f"[gdn3b] target: {target}")

    with open(target) as f:
        text = f.read()

    patched = MARKER in text
    stock = (not patched) and (ANCHOR in text)

    if args.check:
        if patched and "mx.fast.gated_delta_update" in text:
            print(f"[gdn3b] OK: already patched ({MARKER} present)")
            sys.exit(0)
        print(f"[gdn3b] NOT patched (marker={'yes' if patched else 'no'}, stock-anchor={'yes' if stock else 'no'})")
        sys.exit(2)

    if patched:
        print(f"[gdn3b] SKIP: already patched ({MARKER} present)")
        sys.exit(0)
    if not stock:
        print(
            f"[gdn3b] FATAL: stock anchor not found in {target}; refusing to patch a file we do not recognize",
            file=sys.stderr,
        )
        sys.exit(1)

    backup = target + BACKUP_SUFFIX
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            f.write(text)
        print(f"[gdn3b] backup: {backup}")

    new_text = text.replace(ANCHOR, REPLACEMENT, 1)
    if new_text == text:
        print("[gdn3b] FATAL: replacement had no effect", file=sys.stderr)
        sys.exit(1)
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        print(f"[gdn3b] FATAL: patched file fails to parse: {e}", file=sys.stderr)
        sys.exit(1)

    with open(target, "w") as f:
        f.write(new_text)
    print(
        f"[gdn3b] APPLIED: {target} now dispatches to "
        f"mx.fast.gated_delta_update on ROCm (guard mirrors C++ "
        f"use_fallback)"
    )


if __name__ == "__main__":
    main()
