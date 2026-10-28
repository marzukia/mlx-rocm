"""K2 uint4-verify: D=8 copy probe (ok/ck/ov arrays) + rms_norm_residual.

Reproduces the previous run's probe: the old kernel wrote only 4 of 8
elements for f32 D=8 (VEC=4). The uint4 fix must write all 8.
"""

import sys

import mlx.core as mx

fails = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def d8_probe(dt, tag):
    # ok/ck/ov: distinct values so any partial write is visible
    B, H, T, D = 1, 2, 1, 8
    keys = mx.arange(B * H * D, dtype=dt).reshape(B, H, T, D) + 1
    values = mx.arange(B * H * D, dtype=dt).reshape(B, H, T, D) * 3 + 2
    ck = mx.zeros((B, H, 4, D), dtype=dt)
    cv = mx.zeros((B, H, 4, D), dtype=dt)
    ok, ov = mx.rocm.kv_append(keys, values, ck, cv, 2)
    mx.eval(ok, ov)
    got_k = [ok[0, 0, 2, :].tolist()]
    got_v = [ov[0, 0, 2, :].tolist()]
    want_k = keys[0, 0, 0, :].tolist()
    want_v = values[0, 0, 0, :].tolist()
    check(f"kv_append D=8 {tag} keys all 8 written", got_k[0] == want_k)
    check(f"kv_append D=8 {tag} values all 8 written", got_v[0] == want_v)
    # in-place: the cache buffer itself must hold the appended row
    check(f"kv_append D=8 {tag} ck in-place", ck[0, 0, 2, :].tolist() == want_k)
    check(f"kv_append D=8 {tag} cv in-place", cv[0, 0, 2, :].tolist() == want_v)
    # untouched rows stay zero
    check(f"kv_append D=8 {tag} row0 untouched", ck[0, 0, 0, :].tolist() == [0] * 8)


for dt, tag in ((mx.float32, "f32"), (mx.bfloat16, "bf16"), (mx.float16, "f16")):
    d8_probe(dt, tag)

# offset variants for f32
for off in (0, 1, 3):
    B, H, T, D = 1, 2, 1, 8
    keys = mx.full((B, H, T, D), float(off + 1), dtype=mx.float32)
    values = mx.full((B, H, T, D), float(off + 2), dtype=mx.float32)
    ck = mx.zeros((B, H, 4, D), dtype=mx.float32)
    cv = mx.zeros((B, H, 4, D), dtype=mx.float32)
    ok, ov = mx.rocm.kv_append(keys, values, ck, cv, off)
    mx.eval(ok, ov)
    check(f"kv_append D=8 f32 off={off} keys", ok[0, 0, off, :].tolist() == [off + 1] * 8)
    check(f"kv_append D=8 f32 off={off} values", ov[0, 0, off, :].tolist() == [off + 2] * 8)

# rms_norm_residual at D=8
for dt, tag in ((mx.float32, "f32"), (mx.bfloat16, "bf16")):
    mx.random.seed(0)
    x = mx.random.normal((1, 8)).astype(dt)
    r = mx.random.normal((1, 8)).astype(dt)
    w = mx.random.normal((8,)).astype(dt)
    ref = mx.fast.rms_norm(x + r, w, 1e-5)
    got = mx.rocm.rms_norm_residual(x, r, w, 1e-5)
    mx.eval(got, ref)
    check(f"rms_norm_residual D=8 {tag}", bool(mx.allclose(got, ref, rtol=1e-2, atol=1e-3)))

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("D8-PROBE-PASS")
