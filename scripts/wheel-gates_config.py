#!/usr/bin/env python3
"""rule-6 config print+assert: env, wheel sha; kernel names asserted by the runner from a short rocprof pass.

Prints the effective config of the gate run and asserts it matches the expected
from-ref build. Exits non-zero on any assertion failure (a gate run on the wrong
wheel or wrong device is a vacuous gate).
"""

import importlib.metadata
import os
import sys

EXPECTED_WHEEL_SHA = os.environ.get("RUN_789_EXPECTED_WHEEL_SHA", "75ca881")


def main():
    venv = sys.prefix
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    fusion = os.environ.get("MLX_GDN_TAIL_FUSION", "<unset>")  # read from env, not positional (run-608 lesson)
    print(f"CONFIG: env venv={venv} python={pyver} LD_LIBRARY_PATH={ldp} MLX_GDN_TAIL_FUSION={fusion}", flush=True)

    ver = importlib.metadata.version("mlx")
    sha = ver.rsplit("+", 1)[-1] if "+" in ver else "<none>"
    print(f"CONFIG: wheel={ver} (sha={sha})", flush=True)
    assert sha == EXPECTED_WHEEL_SHA, f"wheel sha {sha} != expected {EXPECTED_WHEEL_SHA}"

    import subprocess

    import mlx.core as mx

    dev = mx.default_device()
    assert str(dev).startswith("Device(gpu"), f"default device {dev} != gpu"
    r = subprocess.run(
        ["/opt/rocm/core-10.1/bin/rocminfo"],
        capture_output=True,
        text=True,
        env={**os.environ, "LD_LIBRARY_PATH": "/opt/rocm/core-10.1/lib"},
    )
    assert "gfx1201" in r.stdout, "rocminfo does not report gfx1201"
    print(f"CONFIG: device={dev} arch=gfx1201 (ROCm 10.1 userland, /opt/rocm/core-10.1)", flush=True)
    print("CONFIG-OK", flush=True)


if __name__ == "__main__":
    main()
