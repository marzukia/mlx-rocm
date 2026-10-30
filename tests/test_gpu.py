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


def test_gdn_fused_step_bit_exact_vs_chain():
    """Phase 2: the single-op fused decode step must be byte-identical to the
    default 4-op chain (same delta-rule arithmetic; fusion removes only
    memory round-trips). In-process arm; the cross-process gate is
    scripts/phase2-gdn-fuse-runner.sh."""
    if not hasattr(mx.fast, "gdn_fused_step"):
        pytest.skip("wheel without gdn_fused_step (pre-phase2 build)")
    import numpy as np

    rng = np.random.default_rng(1234)

    def b16(x):
        return mx.array(x).astype(mx.bfloat16)

    q = b16(rng.standard_normal(2048))
    k = b16(rng.standard_normal(2048))
    v = b16(rng.standard_normal(4096))
    b = b16(rng.standard_normal(32))
    a = b16(rng.standard_normal(32))
    A_log = mx.array(rng.standard_normal(32) + 1.0)  # f32 [32]
    dt_bias = mx.array(rng.standard_normal(32))  # f32 [32]
    cs = b16(rng.standard_normal((3, 8192)))
    cw = b16(rng.standard_normal((8192, 4)))
    z = b16(rng.standard_normal((1, 1, 32, 128)))
    state = mx.zeros((1, 32, 128, 128), dtype=mx.float32)
    w = b16(1.0 + 0.05 * rng.standard_normal(128))
    inv_scale = 128**-0.5

    def inputs():
        # Fresh copies so the two arms read identical bytes.
        return (
            mx.array(q),
            mx.array(k),
            mx.array(v),
            mx.array(b),
            mx.array(a),
            mx.array(A_log),
            mx.array(dt_bias),
            mx.array(cs),
            mx.array(cw),
            mx.array(z),
            mx.array(state),
            mx.array(w),
        )

    def chain():
        q_, k_, v_, b_, a_, al_, db_, cs_, cw_, z_, st_, w_ = inputs()
        qn, kn, vc, beta, g = mx.fast.gdn_prep(q_, k_, v_, b_, a_, al_, db_, cs_, cw_, inv_scale=inv_scale, eps=1e-6)
        out2, st2 = mx.fast.gated_delta_update(
            qn.reshape(1, 1, 16, 128),
            kn.reshape(1, 1, 16, 128),
            vc.reshape(1, 1, 32, 128),
            g.reshape(1, 1, 32),
            beta.reshape(1, 1, 32),
            st_,
            None,
        )
        out3 = mx.fast.gdn_norm_gated(out2, z_, w_, eps=1e-6)[0]
        return out3, st2

    def fused():
        q_, k_, v_, b_, a_, al_, db_, cs_, cw_, z_, st_, w_ = inputs()
        return mx.fast.gdn_fused_step(
            q_, k_, v_, b_, a_, al_, db_, cs_, cw_, z_, st_, w_, inv_scale=inv_scale, prep_eps=1e-6, norm_eps=1e-6
        )

    os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    try:
        out_f, st_f = fused()
        out_c, st_c = chain()
        mx.eval(out_f, st_f, out_c, st_c)
        import numpy as np

        assert out_f.size == 4096 and st_f.size == 32 * 128 * 128
        assert np.ascontiguousarray(np.asarray(out_f)).tobytes() == np.ascontiguousarray(np.asarray(out_c)).tobytes(), (
            "out not byte-identical"
        )
        assert np.ascontiguousarray(np.asarray(st_f)).tobytes() == np.ascontiguousarray(np.asarray(st_c)).tobytes(), (
            "state not byte-identical"
        )
    finally:
        del os.environ["MLX_ROCM_GDN_FUSE"]
