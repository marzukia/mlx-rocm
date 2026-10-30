#!/usr/bin/env python3
"""m6: apply the rope_q row-invariant patch to the venv's qwen3_next.py.

Usage:
  python3 apply-m6-patch.py /path/to/venv          # apply (idempotent)
  python3 apply-m6-patch.py /path/to/venv --check  # verify state, no write

Conventions (mirrors scripts/patch-models-k2.py):
  - pattern-based, idempotent (re-running is a no-op);
  - the pre-patch file is backed up as qwen3_next.py.bak-an earlier run (the
    qwen3_next.py.bak-an earlier runc convention); the backup is taken once, from
    the current (pre-patch) file, and is never overwritten.

Edits (recorded in patches/m6-ropeq-rowinvariant.diff):
  1. import math (module top)
  2. RowInvariantRoPE class (before Qwen3NextAttention)
  3. Qwen3NextAttention.__init__: the default (no rope_scaling) path uses
     RowInvariantRoPE instead of the fused mx.fast.rope; non-default
     scaling configs keep the original initialize_rope path.
"""

import sys
from pathlib import Path

IMPORT_OLD = "from __future__ import annotations\nimport os\n"
IMPORT_NEW = "from __future__ import annotations\nimport math\nimport os\n"

CLASS_NEW = '''class RowInvariantRoPE(nn.Module):
    """m6: explicit row-invariant RoPE (replaces the fused mx.fast.rope).

    The fused ROCm rope kernel dispatches T-dependent variants (a 1D
    single-token kernel for T=1, a batched kernel for T>1) that can differ
    by 1 ULP (m5: rope_q 1/4096, the first diverging component of the
    layer-3 full-attention block). This module computes the same rotation
    with plain elementwise ops so that row r depends only on (row r,
    offset + r) and T=1 / T=2 are bit-identical per row:

      - inv_freqs[i] = base ** (-i / (dims // 2))  (fp32 constant; the
        kernel computes exp2f(-(i / half) * log2(base)))
      - theta = (offset + row) * inv_freqs  (fp32, per row)
      - cos/sin precomputed in fp32
      - split/rotate/multiply/add in the model dtype (bf16)

    Preserves the nn.RoPE contract exactly: dims (partial_rotary_factor -
    the first ``dims`` dims rotated, the rest passthrough), traditional=False
    (pair (x[i], x[i + dims // 2]) rotated by theta_i), per-row offset
    semantics (pos = offset + row_idx), no mask change, no padding.
    """

    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0, scale: float = 1.0):
        super().__init__()
        if traditional:
            raise ValueError("RowInvariantRoPE supports traditional=False only")
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        half = dims // 2
        # Host double -> float32 cast mirrors the kernel's
        # (float)std::log2(base) + exp2f path as closely as the API allows.
        lf = math.log2(base)
        self._inv_freqs = mx.array(
            [math.exp2(-(i / half) * lf) for i in range(half)],
            dtype=mx.float32,
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (B, H, L, D)
        L = x.shape[-2]
        half = self.dims // 2
        # Kernel: L = scale * (float)(row + offset); theta = L * inv_freq.
        pos = (mx.arange(L, dtype=mx.int32) + offset).astype(mx.float32)
        Lp = pos * self.scale
        theta = Lp[:, None] * self._inv_freqs[None, :]
        cos_t = mx.cos(theta)[None, None, :, :]
        sin_t = mx.sin(theta)[None, None, :, :]
        x1 = x[..., :half].astype(mx.float32)
        x2 = x[..., half : self.dims].astype(mx.float32)
        r1 = x1 * cos_t - x2 * sin_t
        r2 = x1 * sin_t + x2 * cos_t
        rot = mx.concatenate([r1, r2], axis=-1).astype(x.dtype)
        if self.dims < x.shape[-1]:
            return mx.concatenate([rot, x[..., self.dims :]], axis=-1)
        return rot


'''

CLASS_ANCHOR = "class Qwen3NextAttention(nn.Module):\n"

ROPE_OLD = """        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
"""
ROPE_NEW = """        # m6: the default (no rope_scaling) path uses an explicit
        # row-invariant rotation instead of the fused mx.fast.rope (see
        # RowInvariantRoPE). Non-default scaling configs keep the original
        # initialize_rope path.
        if args.rope_scaling is None:
            self.rope = RowInvariantRoPE(
                int(self.head_dim * args.partial_rotary_factor),
                traditional=False,
                base=args.rope_theta,
            )
        else:
            self.rope = initialize_rope(
                int(self.head_dim * args.partial_rotary_factor),
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
"""

BAK = "qwen3_next.py.bak-an earlier run"


def apply(p: Path) -> None:
    s = p.read_text()
    changed = False
    if IMPORT_NEW not in s:
        if IMPORT_OLD not in s:
            raise SystemExit(f"PATCH FAIL (import): pattern not found in {p}")
        s = s.replace(IMPORT_OLD, IMPORT_NEW, 1)
        changed = True
    if "class RowInvariantRoPE" not in s:
        if CLASS_ANCHOR not in s:
            raise SystemExit(f"PATCH FAIL (class): anchor not found in {p}")
        s = s.replace(CLASS_ANCHOR, CLASS_NEW + CLASS_ANCHOR, 1)
        changed = True
    if ROPE_NEW not in s:
        if ROPE_OLD not in s:
            raise SystemExit(f"PATCH FAIL (rope init): pattern not found in {p}")
        s = s.replace(ROPE_OLD, ROPE_NEW, 1)
        changed = True
    if changed:
        bak = p.with_name(BAK)
        if not bak.exists():
            bak.write_text(p.read_text())
            print(f"backup: {bak}")
        p.write_text(s)
        print(f"patched: {p}")
    else:
        print(f"already patched: {p}")


def check(p: Path) -> None:
    s = p.read_text()
    ok = (
        IMPORT_NEW in s
        and "class RowInvariantRoPE" in s
        and "if args.rope_scaling is None:" in s
        and "self.rope = RowInvariantRoPE(" in s
    )
    print(f"{'PATCHED' if ok else 'NOT-PATCHED'}: {p}")
    sys.exit(0 if ok else 1)


def main() -> None:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--check"):
        raise SystemExit(__doc__)
    venv = Path(sys.argv[1])
    p = venv / "lib" / "python3.12" / "site-packages" / "mlx_lm" / "models" / "qwen3_next.py"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    if len(sys.argv) == 3:
        check(p)
    apply(p)


if __name__ == "__main__":
    main()
