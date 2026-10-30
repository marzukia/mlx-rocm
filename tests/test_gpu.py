"""GPU tests: run ON the build VM only (uv run pytest -m gpu). Skipped off-GPU.

Phase 0 exit criteria live here; Phase 1 (QMM/SDPA/fp8/allocator) extends this file.
"""

import os
import sys

import pytest

# mlx is only importable where the (ROCm) wheel is installed - skip the whole module elsewhere
mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not (sys.platform == "linux" and os.environ.get("MLX_ROCM_GPU") == "1"):
        pytest.skip("gpu tests run on the build VM with MLX_ROCM_GPU=1")
    if not hasattr(mx, "rocm") or not mx.rocm.is_available():
        pytest.skip("no ROCm backend in this wheel")
    mx.set_default_device(mx.gpu)


def test_is_available():
    assert mx.rocm.is_available()


def test_matmul_vs_cpu():
    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    c_gpu = a @ b
    mx.eval(c_gpu)
    c_cpu = (a.astype(mx.float32) @ b.astype(mx.float32)).astype(a.dtype)
    mx.eval(c_cpu)
    assert float(mx.max(mx.abs(c_gpu - c_cpu)).item()) < 0.5


def test_softmax_rows():
    a = mx.random.normal((128, 256))
    s = mx.softmax(a, axis=1)
    mx.eval(s)
    assert float(mx.max(mx.abs(mx.sum(s, axis=1) - 1.0)).item()) < 1e-2
