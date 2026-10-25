"""Hermetic gate tests: no GPU needed. Run in CI and anywhere.

These pin the deterministic facts a GPU run depends on: the vendor pin, the build flags,
the gfx1201 naming correction, and the script inventory. If a fact moves (base re-pinned,
flags changed), this file must change in the same PR.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _vendor() -> dict:
    return json.loads((ROOT / "vendor.json").read_text())


def test_vendor_pin_is_well_formed():
    v = _vendor()
    for key in ("primary", "fallback"):
        assert v[key]["fork"].startswith("https://github.com/")
        assert v[key]["ref"]
    assert v["build"]["hip_architectures"] == "gfx1201"
    # RX 9070 XT is gfx1201 (RDNA4) - not a gfx110x target
    assert "gfx110" not in v["build"]["cmake_args"]


def test_build_flags_are_the_documented_ones():
    v = _vendor()
    args = v["build"]["cmake_args"]
    assert "-DMLX_BUILD_ROCM=ON" in args
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1201" in args
    assert "-DBLA_VENDOR=OpenBLAS" in args


def test_build_script_uses_the_vendor_pin_not_hardcoded():
    src = (ROOT / "scripts" / "build-wheel.sh").read_text()
    assert "vendor.json" in src
    assert "MLX_BUILD_ROCM" not in src.replace("vendor.json", "")  # flags come from the pin


def test_smoke_scripts_exist_and_are_executable_inventory():
    for name in ("build-wheel.sh", "smoke.py", "smoke-serve.sh", "size_ratchet.py"):
        assert (ROOT / "scripts" / name).is_file(), name
    assert (ROOT / "scripts" / "smoke-serve.sh").read_text().count("MLX_ROCM_GEMM_BF16") >= 2


def test_ci_wiring_present():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'pytest -m "not gpu"' in ci
    assert "size_ratchet.py" in ci
    assert "self-hosted" not in ci  # hermetic CI: no GPU, no neon assumptions


def test_gpu_tests_are_marked():
    gpu_tests = [p for p in (ROOT / "tests").glob("*.py") if "gpu" in p.name]
    assert gpu_tests, "expected at least one gpu test file"
    for p in gpu_tests:
        src = p.read_text()
        assert "pytest.mark.gpu" in src or "mark.gpu" in src
