"""mlx-rocm — ROCm backend for MLX (MVP).

MVP skeleton: vendored mlx ROCm backend tree (the vendor ref,
forward-merge of the ROCm base) + a minimal standalone hipBLASLt
GEMM layer (bf16 + fp8 e4m3) with a CPU reference for validation.
"""

__version__ = "0.0.1"
__all__ = ["gemm", "__version__"]
