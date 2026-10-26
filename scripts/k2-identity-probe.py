"""K2 MAJOR-1 env-gate identity: plain rms_norm, bit-exact output hash.

With the K2 env unset (unpatched model), the pre-existing rms_norm op must be
bit-identical to the base: the fix removed the one-warp-per-row decode fast
path from RMSNorm::eval_gpu, so plain rms_norm always uses the block kernel.

Run the SAME script in the fixed wheel (venv-k2) and the baseline wheel
(/mnt/models/venv) and diff the sha256 lines:
  - non-decode shapes (n_rows>64 or axis>8192): both wheels use the block
    kernel -> must be bit-identical (fix changed nothing there).
  - decode shapes (n_rows<=64, axis<=8192): the pre-fix wheel used the warp
    fast path; the fixed wheel uses the block kernel -> may differ by ~1 ulp
    (that delta is exactly the MAJOR-1 fix). The fixed wheel must still match
    the f32 reference (correctness).
Deterministic: fixed seeds, no GPU nondeterminism (single-kernel ops).
"""

import hashlib

import mlx.core as mx

# (shape, note): decode = old fast path eligible; block = always block kernel
CASES = [
    ((1, 4096), "decode"),
    ((32, 128), "decode"),
    ((1, 128), "decode"),
    ((8, 4096), "decode"),
    ((65, 128), "block"),
    ((1, 16384), "block"),
    ((128, 512), "block"),
    ((4, 8193), "block"),
]

for dt in (mx.bfloat16, mx.float16, mx.float32):
    for shape, note in CASES:
        mx.random.seed(5)
        x = mx.random.normal(shape).astype(dt)
        w = mx.random.normal((shape[-1],)).astype(dt)
        eps = 1e-5
        got = mx.fast.rms_norm(x, w, eps)
        mx.eval(got)
        h = hashlib.sha256(got.tobytes()).hexdigest()[:16]
        # f32-accumulation reference (correctness, not the identity check)
        xf = x.astype(mx.float32)
        ref = (
            xf * mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + eps)
            * w.astype(mx.float32)
        ).astype(dt)
        ok = bool(mx.allclose(got, ref, rtol=1e-2, atol=1e-3))
        print(f"{dt} {shape} [{note}] sha256={h} refclose={ok}")
print("IDENTITY-PROBE-DONE")
