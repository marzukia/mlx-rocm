"""mlx-rocm — ROCm backend for MLX (pre-alpha).

Faithful mlx <-> ROCm adapter/kernel layer for AMD RDNA4 (gfx1201):
a complete `mlx/backend/rocm/` tree (vendored, MIT) + a minimal
standalone hipBLASLt GEMM layer (bf16 + fp8 e4m3) with a CPU reference
for validation.
"""

__version__ = "0.1.2"
__all__ = ["gemm", "__version__"]
