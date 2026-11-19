#!/usr/bin/env python3
"""inproj-pad: apply the GDN in_proj N=128 pad patch to the venv's
qwen3_5.py (mlx_lm model file).

Usage:
  python3 patch-inproj-pad.py /path/to/venv          # apply (idempotent)
  python3 patch-inproj-pad.py /path/to/venv --check # verify state, no write

Conventions (mirrors scripts/apply-m6-patch.py):
  - pattern-based, idempotent (re-running is a no-op);
  - the pre-patch file is backed up as qwen3_5.py.bak-inprojpad; the
    backup is taken once, from the current (pre-patch) file, and is
    never overwritten.

Edit (recorded in patches/inproj-pad.diff):
  GatedDeltaNet.__call__: one new branch, selected by
  MLX_ROCM_INPROJ_PAD=1 (DEFAULT OFF = zero behavior change), inserted
  before the MLX_ROCM_W1_INBA_FUSE branch. When on, in_proj_b/in_proj_a
  run through ONE fused GEMM padded to N = next multiple of 128
  (num_v_heads=16: 32 -> 128). The padded rows are a zero tensor
  re-quantized in place (dequant exactly 0), the output is sliced back
  to N = 2 * num_v_heads. The N % 128 == 0 shape dispatches to the
  stock q4_k mmq / wmma-dense prefill path - existing kernels only, no
  new kernels. Weights are built lazily on the first call (after load()
  has installed the real weight/scales/biases); the one-time fill lands
  in the warm pass.
"""

import sys
from pathlib import Path

ANCHOR = '        if os.environ.get("MLX_ROCM_W1_INBA_FUSE", "0") == "1":\n'
MARKER = "MLX_ROCM_INPROJ_PAD"

NEW = """        if os.environ.get("MLX_ROCM_INPROJ_PAD", "0") == "1":
            # INPROJ-PAD (2026-11-19): in_proj_b/a -> ONE fused GEMM padded
            # to N = next multiple of 128 (num_v_heads=16: 32 -> 128). The
            # padded rows are a zero tensor re-quantized in place, so their
            # dequant is exactly 0 (weight == zero-point); the output is
            # sliced back to N = 2 * num_v_heads. Existing kernels only:
            # the N % 128 == 0 shape dispatches to the stock q4_k mmq /
            # wmma-dense prefill path. Default OFF = zero behavior change.
            # Built lazily on the first call (after load() has installed
            # the real weight/scales/biases); the one-time fill lands in
            # the warm pass.
            pad = getattr(self, "_inba_pad", None)
            if pad is None:
                ib, ia = self.in_proj_b, self.in_proj_a
                n = 2 * self.num_v_heads
                m = 128
                N = n if n % m == 0 else (n // m + 1) * m
                if getattr(ib, "scales", None) is not None:  # QuantizedLinear
                    fused = nn.QuantizedLinear(
                        self.hidden_size,
                        N,
                        bits=ib.bits,
                        group_size=ib.group_size,
                        bias=False,
                    )
                    zq = mx.quantize(
                        mx.zeros((N - n, self.hidden_size)),
                        bits=ib.bits,
                        group_size=ib.group_size,
                    )
                    # PAD-FIX (t3208 ARM B, 2026-11-19): mx.quantize in this
                    # wheel returns a [weight, scales, biases] list (not a
                    # QuantizedLinear); handle both return shapes.
                    if isinstance(zq, list):
                        zqw, zqs, zqb = zq[0], zq[1], zq[2]
                    else:
                        zqw, zqs, zqb = zq.weight, zq.scales, zq.biases
                    fused.weight = mx.concatenate(
                        [ib.weight, ia.weight, zqw], axis=0
                    )
                    fused.scales = mx.concatenate(
                        [ib.scales, ia.scales, zqs], axis=0
                    )
                    fused.biases = mx.concatenate(
                        [ib.biases, ia.biases, zqb], axis=0
                    )
                else:  # plain Linear
                    fused = nn.Linear(self.hidden_size, N, bias=False)
                    fused.weight = mx.concatenate(
                        [
                            ib.weight,
                            ia.weight,
                            mx.zeros(
                                (N - n, self.hidden_size),
                                dtype=ib.weight.dtype,
                            ),
                        ],
                        axis=0,
                    )
                pad = (fused, n)
                self._inba_pad = pad
            fused, n = pad
            ba = fused(inputs)
            b = mx.contiguous(ba[..., : self.num_v_heads])
            a = mx.contiguous(ba[..., self.num_v_heads : 2 * self.num_v_heads])
"""

BAK = "qwen3_5.py.bak-inprojpad"


def find_target(venv: Path) -> Path:
    cands = sorted((venv / "lib").glob("python3.*/site-packages/mlx_lm/models/qwen3_5.py"))
    if not cands:
        raise SystemExit(f"missing mlx_lm/models/qwen3_5.py under {venv}")
    return cands[0]


def apply(p: Path) -> None:
    s = p.read_text()
    if MARKER in s:
        print(f"already patched: {p}")
        return
    if ANCHOR not in s:
        raise SystemExit(f"PATCH FAIL (anchor): pattern not found in {p}")
    if s.count(ANCHOR) != 1:
        raise SystemExit(f"PATCH FAIL (anchor): not unique in {p}")
    bak = p.with_name(BAK)
    if not bak.exists():
        bak.write_text(s)
        print(f"backup: {bak}")
    p.write_text(s.replace(ANCHOR, NEW + ANCHOR, 1))
    print(f"patched: {p}")


def check(p: Path) -> None:
    s = p.read_text()
    ok = MARKER in s and ANCHOR in s
    print(f"{'PATCHED' if ok else 'NOT-PATCHED'}: {p}")
    sys.exit(0 if ok else 1)


def main() -> None:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--check"):
        raise SystemExit(__doc__)
    p = find_target(Path(sys.argv[1]))
    if len(sys.argv) == 3:
        check(p)
    apply(p)


if __name__ == "__main__":
    main()
