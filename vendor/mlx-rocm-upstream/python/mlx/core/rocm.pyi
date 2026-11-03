"""mlx.rocm"""

import mlx.core


def is_available() -> bool:
    """Check if the ROCm back-end is available."""

def train_arena_begin(capacity_bytes: int) -> bool:
    """
    Allocate/arm a fixed HBM bump region for stable device addresses.
    Returns False on failure. Call train_arena_reset() before each step
    when capturing train graphs (not required for eager train).
    """

def train_arena_reset() -> None: ...

def train_arena_end() -> None: ...

def train_arena_active() -> bool: ...

def train_arena_high_water() -> int: ...

def train_arena_overflowed() -> bool: ...

def moe_swiglu_sorted(x: mlx.core.array, w_gate: mlx.core.array, w_up: mlx.core.array, w_down: mlx.core.array, expert_ids: mlx.core.array, *, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Fused sorted-MoE SwiGLU (compile-safe Primitive; one host sync).

    x: [T, D] bf16
    w_gate, w_up: [E, D, I] bf16 (after swapaxes of Linear [E,I,D])
    w_down: [E, I, D] bf16 (after swapaxes of Linear [E,D,I])
    expert_ids: [T] uint32, sorted by expert
    Returns y: [T, D] bf16
    """

def moe_swiglu_sorted_vjp(x: mlx.core.array, w_gate: mlx.core.array, w_up: mlx.core.array, w_down: mlx.core.array, expert_ids: mlx.core.array, dy: mlx.core.array, *, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> list[mlx.core.array]:
    """
    Fused sorted-MoE SwiGLU VJP (one host sync for recompute+grads).

    Same weight layouts as moe_swiglu_sorted. Returns
    (dx, dw_gate, dw_up, dw_down) all bf16.
    """
