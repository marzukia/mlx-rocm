"""Thin ctypes binding for the mlx-rocm GEMM library (libmlx_rocm.so).

The C++ library is built with CMake (see README). This module:
  * locates libmlx_rocm.so (MLX_ROCM_LIB env > next to this file > find_library),
  * exposes host_gemm_bf16() (works in any build) and, in ROCm builds,
    hip_gemm_bf16() / hip_gemm_fp8_e4m3() / fp8_probe(),
  * falls back to a pure-numpy bf16 reference when no .so is present.

bf16 arrays are numpy uint16 views (IEEE bfloat16 bit patterns);
fp8 arrays are numpy uint8 (e4m3 codes, avoid 0x7F/0xFF = NaN).
"""

import ctypes
import ctypes.util
import os
from pathlib import Path

_C = ctypes.c_int

try:
    import numpy as _np
except ImportError:  # numpy is required for the fallback path only
    _np = None


class MlxRocmLibError(RuntimeError):
    pass


def _find_lib():
    env = os.environ.get("MLX_ROCM_LIB")
    if env and os.path.exists(env):
        return env
    here = Path(__file__).resolve().parent
    for name in ("libmlx_rocm.so",):
        p = here / name
        if p.exists():
            return str(p)
    # Common build-tree location (repo checkout)
    repo = here.parent.parent.parent.parent
    for cand in (
        repo / "build" / "libmlx_rocm.so",
        repo / "build" / "mlx_rocm" / "libmlx_rocm.so",
    ):
        if cand.exists():
            return str(cand)
    found = ctypes.util.find_library("mlx_rocm")
    return found


_lib = None
_lib_path = None


def load():
    """Load libmlx_rocm.so; raises MlxRocmLibError if it cannot be found."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    p = _find_lib()
    if not p:
        raise MlxRocmLibError(
            "libmlx_rocm.so not found. Build it: cmake -S . -B build && "
            "cmake --build build (add -DMLX_ROCM_WITH_HIP=ON on a ROCm box), "
            "or set MLX_ROCM_LIB=/path/to/libmlx_rocm.so"
        )
    _lib = ctypes.CDLL(p)
    _lib_path = p
    _lib.mlx_rocm_gemm_host_bf16.restype = _C
    _lib.mlx_rocm_gemm_host_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        _C,
        _C,
        _C,
    ]
    for name in (
        "mlx_rocm_gemm_hip_bf16_hostio",
        "mlx_rocm_gemm_hip_fp8_hostio",
    ):
        f = getattr(_lib, name)
        f.restype = _C
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, _C, _C, _C]
    _lib.mlx_rocm_gemm_fp8_probe.restype = _C
    _lib.mlx_rocm_gemm_fp8_probe.argtypes = []
    _lib.mlx_rocm_hip_device_count.restype = _C
    _lib.mlx_rocm_hip_device_count.argtypes = []
    _lib.mlx_rocm_preferred_precision.restype = ctypes.c_char_p
    _lib.mlx_rocm_preferred_precision.argtypes = []
    return _lib


def library_path():
    return _lib_path


def _check(rc):
    if rc:
        raise MlxRocmLibError(f"GEMM call failed (rc={rc}); see stderr")


def _as_u16(arr):
    return _np.ascontiguousarray(arr, dtype=_np.uint16)


def _as_u8(arr):
    return _np.ascontiguousarray(arr, dtype=_np.uint8)


def _out_16(shape):
    return _np.empty(shape, dtype=_np.uint16)


def host_gemm_bf16(a, b):
    """D[m,n] = A[m,k] x B[k,n], bf16 (uint16 views). CPU reference."""
    lib = None
    try:
        lib = load()
    except MlxRocmLibError:
        if _np is None:
            raise
        # numpy fallback: bf16 -> f32 via bit shift, compute, round to bf16.
        # (shift in uint32: a 16-bit shift in a uint16 container masks to 0)
        af = (_as_u16(a).astype(_np.uint32) << 16).view(_np.float32)
        bf = (_as_u16(b).astype(_np.uint32) << 16).view(_np.float32)
        c32 = af @ bf
        u = c32.view(_np.uint32)
        lsb = (u >> 16) & 1
        bits = (u + _np.uint32(0x7FFF) + lsb) >> 16
        return bits.astype(_np.uint16)
    a = _as_u16(a)
    b = _as_u16(b)
    d = _out_16((a.shape[0], b.shape[1]))
    _check(
        lib.mlx_rocm_gemm_host_bf16(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            _C(a.shape[0]),
            _C(b.shape[1]),
            _C(a.shape[1]),
        )
    )
    return d


def hip_gemm_bf16(a, b):
    """GPU bf16 GEMM (hipBLASLt), host I/O. Requires a ROCm build + device."""
    lib = load()
    a = _as_u16(a)
    b = _as_u16(b)
    d = _out_16((a.shape[0], b.shape[1]))
    _check(
        lib.mlx_rocm_gemm_hip_bf16_hostio(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            _C(a.shape[0]),
            _C(b.shape[1]),
            _C(a.shape[1]),
        )
    )
    return d


def hip_gemm_fp8_e4m3(a, b):
    """GPU fp8 e4m3 GEMM (uint8 in, bf16 out). gfx1201 may need bf16 fallback."""
    lib = load()
    a = _as_u8(a)
    b = _as_u8(b)
    d = _out_16((a.shape[0], b.shape[1]))
    _check(
        lib.mlx_rocm_gemm_hip_fp8_hostio(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            _C(a.shape[0]),
            _C(b.shape[1]),
            _C(a.shape[1]),
        )
    )
    return d


def fp8_probe():
    """0 if the fp8 GEMM looks healthy on this driver, 1 otherwise."""
    return load().mlx_rocm_gemm_fp8_probe()


def hip_device_count():
    try:
        return load().mlx_rocm_hip_device_count()
    except MlxRocmLibError:
        return 0


def preferred_precision():
    try:
        return load().mlx_rocm_preferred_precision().decode()
    except MlxRocmLibError:
        return "fp8"  # matches the C++ default without env override
