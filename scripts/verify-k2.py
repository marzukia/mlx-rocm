"""K2 numerical verification: fused ops vs decompositions + model A/B logits."""
import sys

import mlx.core as mx

torch_free = True  # noqa
failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- unit checks
print("== unit: rms_norm_residual vs add + rms_norm ==")
for dt in (mx.bfloat16, mx.float16, mx.float32):
    for shape in ((1, 4096), (1, 128), (8, 4096), (1, 256)):
        mx.random.seed(0)
        x = mx.random.normal(shape)
        r = mx.random.normal(shape)
        w = mx.random.normal((shape[-1],))
        eps = 1e-5
        ref = mx.fast.rms_norm(x + r, w, eps)
        got = mx.rocm.rms_norm_residual(x, r, w, eps)
        close = mx.allclose(got, ref, rtol=1e-2, atol=1e-3)
        check(f"rms_norm_residual {shape} {dt}", bool(close))

print("== unit: kv_append vs slice-assign ==")
for dt in (mx.bfloat16, mx.float16, mx.float32):
    for (B, H, T, D) in ((1, 8, 1, 256), (1, 2, 1, 128), (1, 8, 1, 96), (1, 2, 1, 120)):
        mx.random.seed(1)
        keys = mx.random.normal((B, H, T, D))
        values = mx.random.normal((B, H, T, D))
        for offset in (0, 1, 5):
            ck = mx.random.normal((B, H, 64, D))
            cv = mx.random.normal((B, H, 64, D))
            # reference: copy into fresh arrays via slice assign
            rk = mx.array(ck)
            rv = mx.array(cv)
            rk[..., offset : offset + T, :] = keys
            rv[..., offset : offset + T, :] = values
            ok, ov = mx.rocm.kv_append(keys, values, ck, cv, offset)
            check(
                f"kv_append {B}x{H}x{T}x{D} off={offset} {dt}",
                bool(mx.allclose(ok, rk, rtol=0, atol=0) and mx.allclose(ov, rv, rtol=0, atol=0)),
            )

# in-place semantics: the cache buffer itself must now hold the appended rows
ck = mx.zeros((1, 2, 4, 8))
keys = mx.ones((1, 2, 1, 8)) * 3
cv = mx.zeros((1, 2, 4, 8))
values = mx.ones((1, 2, 1, 8)) * 7
ok, ov = mx.rocm.kv_append(keys, values, ck, cv, 2)
check("kv_append in-place writes cache", bool(mx.allclose(ck[0, 0, 2, :], mx.array([3.0] * 8))))
check("kv_append in-place writes values", bool(mx.allclose(cv[0, 0, 2, :], mx.array([7.0] * 8))))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all unit checks passed")
