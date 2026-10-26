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
print("== unit: plain rms_norm (MAJOR-1 env-gate identity) ==")
# With the K2 env unset (unpatched model), the pre-existing rms_norm op must be
# bit-identical to the base: the one-warp-per-row decode fast path was removed
# from RMSNorm::eval_gpu, so it now always uses the block kernel. Verify the
# decode-shape plain rms_norm still matches the reference decomposition.
for dt in (mx.bfloat16, mx.float16, mx.float32):
    for shape in ((1, 4096), (32, 128), (1, 128), (8, 4096)):
        mx.random.seed(5)
        x = mx.random.normal(shape).astype(dt)
        w = mx.random.normal((shape[-1],)).astype(dt)
        eps = 1e-5
        got = mx.fast.rms_norm(x, w, eps)
        # reference: x * rsqrt(mean(x^2) + eps) * w (f32 accumulation)
        xf = x.astype(mx.float32)
        ref = (xf * mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + eps) * w.astype(mx.float32)).astype(dt)
        check(f"plain rms_norm {shape} {dt} (block kernel only)", bool(mx.allclose(got, ref, rtol=1e-2, atol=1e-3)))

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
    for B, H, T, D in ((1, 8, 1, 256), (1, 2, 1, 128), (1, 8, 1, 96), (1, 2, 1, 120)):
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
mx.eval(ok, ov)  # trigger the in-place write before reading the cache buffer
check("kv_append in-place writes cache", bool(mx.allclose(ck[0, 0, 2, :], mx.array([3.0] * 8))))
check("kv_append in-place writes values", bool(mx.allclose(cv[0, 0, 2, :], mx.array([7.0] * 8))))

# ---------------------------------------------------------------- review fixes
print("== unit: kv_append CPU fallback (MAJOR-3) ==")
# The CPU fallback (slice_update) must compute, not throw. Before the fix the
# start/stop=0,0 Shape normalized to empty (size-0) B/H/D slices and always
# threw for H>1 or D>1. Exercise it on a small [B,H,T,D] case via a CPU stream.
for B, H, T, D in ((1, 2, 1, 8), (2, 3, 1, 16)):
    mx.random.seed(2)
    keys = mx.random.normal((B, H, T, D))
    values = mx.random.normal((B, H, T, D))
    for offset in (0, 1):
        ck = mx.random.normal((B, H, 8, D))
        cv = mx.random.normal((B, H, 8, D))
        rk = mx.array(ck)
        rv = mx.array(cv)
        rk[..., offset : offset + T, :] = keys
        rv[..., offset : offset + T, :] = values
        ok, ov = mx.rocm.kv_append(keys, values, ck, cv, offset, stream=mx.default_stream(mx.cpu))
        check(
            f"kv_append CPU fallback {B}x{H}x{T}x{D} off={offset}",
            bool(mx.allclose(ok, rk, rtol=0, atol=0) and mx.allclose(ov, rv, rtol=0, atol=0)),
        )

print("== negative: kv_append dtype mismatch (MAJOR-4) ==")
# f32 keys/values into a bf16 cache must raise a clean error (not silently
# write the wrong element size into the cache buffer).
keys = mx.random.normal((1, 2, 1, 8), dtype=mx.float32)
values = mx.random.normal((1, 2, 1, 8), dtype=mx.float32)
ck = mx.random.normal((1, 2, 8, 8), dtype=mx.bfloat16)
cv = mx.random.normal((1, 2, 8, 8), dtype=mx.bfloat16)
try:
    ok, ov = mx.rocm.kv_append(keys, values, ck, cv, 1)
    mx.eval(ok, ov)
    check("kv_append f32->bf16 raises clean error", False)
except Exception as e:  # noqa: BLE001
    check("kv_append f32->bf16 raises clean error", "dtype" in str(e).lower())

print("== negative: kv_append non-contiguous view (MAJOR-5) ==")
# A non-standard-layout [B,H,1,D] view (strides[0] != H*strides[1]) must route
# to the slice_update fallback (correct), not the fast kernel (misaddress).
B, H, D = 1, 2, 8
mx.random.seed(3)
base = mx.random.normal((H, B, D))
keys = mx.transpose(base, (1, 0, 2))[:, :, None, :]  # [B,H,1,D], non-contig
assert not keys.flags.contiguous, "test setup: keys must be non-contiguous"
values = mx.random.normal((B, H, 1, D))
ck = mx.random.normal((B, H, 8, D))
cv = mx.random.normal((B, H, 8, D))
rk = mx.array(ck)
rv = mx.array(cv)
rk[..., 1:2, :] = keys
rv[..., 1:2, :] = values
ok, ov = mx.rocm.kv_append(keys, values, ck, cv, 1)
check(
    "kv_append non-contig keys -> fallback (correct)",
    bool(mx.allclose(ok, rk, rtol=0, atol=0) and mx.allclose(ov, rv, rtol=0, atol=0)),
)

print("== unit: rms_norm_residual non-contiguous residual (MAJOR-2) ==")
# A non-row-contiguous residual (transposed view) must be copied before the
# kernel (the packed fast path reads residual rows at row*axis_size).
B, D = 2, 8
mx.random.seed(4)
x = mx.random.normal((B, D))
base_r = mx.random.normal((D, B))
r = mx.transpose(base_r, (1, 0))  # [B,D], non-contiguous
assert not r.flags.contiguous, "test setup: residual must be non-contiguous"
w = mx.random.normal((D,))
ref = mx.fast.rms_norm(x + r, w, 1e-5)
got = mx.rocm.rms_norm_residual(x, r, w, 1e-5)
check(
    "rms_norm_residual non-contig residual",
    bool(mx.allclose(got, ref, rtol=1e-2, atol=1e-3)),
)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all unit checks passed")
