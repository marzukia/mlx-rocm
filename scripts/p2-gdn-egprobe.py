#!/usr/bin/env python3
"""Phase 2 GDN-fuse: direct per-head eg measurement.

With q=k=v=0, conv_state=0, conv_weight=0, b=1.0 (beta=1), a/dt_bias=0
(same g for all heads? no -- A_log varies per head), norm z=0, and
h0 = 1.0 everywhere, ONE decode step gives:
  chain: state[hv, row, k] = 1.0 * eg_chain[hv]  (u = 0*1 = 0, dot = 0)
  fused: state[hv, row, k] = 1.0 * eg_fused[hv]
so the per-head state value IS the applied decay factor. The host-side
replica (low-u16-of-f32 -> bf16 -> exp(max(log, -80)) of the g value)
is printed alongside to identify which read the kernels actually use.

Run: venv-fused python, under the GPU lock.
"""

import os

os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
os.environ["MLX_ROCM_GDN_FUSE"] = "1"

import mlx.core as mx
import numpy as np

Hk, Hv, Dk, Dv = 16, 32, 128, 128
key_dim = Hk * Dk
value_dim = Hv * Dv
conv_dim = 2 * key_dim + value_dim
KS = 4
INV_SCALE = Dk**-0.5

rng = np.random.default_rng(7)

q = mx.zeros((1, 1, key_dim), dtype=mx.bfloat16)
k = mx.zeros((1, 1, key_dim), dtype=mx.bfloat16)
v = mx.zeros((1, 1, value_dim), dtype=mx.bfloat16)
b = mx.array(np.full(Hv, 1.0)).astype(mx.bfloat16)  # sigmoid(1) != 1!
# beta = sigmoid(b) -> use b = 100 so sigmoid ~ 1.0 in bf16 (0x3F80)
b = mx.array(np.full(Hv, 100.0)).astype(mx.bfloat16)
a = mx.zeros((1, 1, Hv), dtype=mx.bfloat16)
al = mx.array(rng.standard_normal(Hv) - 5.0)  # f32 [Hv] realistic decay
db = mx.zeros((Hv,), dtype=mx.float32)
cs = mx.zeros((KS - 1, conv_dim), dtype=mx.bfloat16)
cw = mx.zeros((conv_dim, KS), dtype=mx.bfloat16)
z = mx.zeros((1, 1, Hv, Dv), dtype=mx.bfloat16)
w = mx.zeros((Dv,), dtype=mx.bfloat16)
st = mx.ones((1, Hv, Dv, Dk), dtype=mx.float32)


def to_np(a, np_dtype):
    return np.frombuffer(bytes(bytearray(a)), np.uint8).view(np_dtype).reshape(-1)


def host_eg(g):
    g = to_np(g, np.float32).copy()
    b = g.view(np.uint32)
    u16 = (b & 0xFFFF).astype(np.uint32) << 16
    v = u16.view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.abs(v))
    return np.exp(np.maximum(logv, -80.0)), g


# ---- chain arm ----
qn, kn, vc, beta, g = mx.fast.gdn_prep(q, k, v, b, a, al, db, cs, cw, inv_scale=INV_SCALE, eps=1e-6)
out2, st2 = mx.fast.gated_delta_update(
    qn.reshape(1, 1, Hk, Dk),
    kn.reshape(1, 1, Hk, Dk),
    vc.reshape(1, 1, Hv, Dv),
    g.reshape(1, 1, Hv),
    beta.reshape(1, 1, Hv),
    st,
    None,
)
mx.eval(out2, st2, g, beta)
print("CHAIN per-head state(=eg) + host_eg:", flush=True)
eg_c = to_np(st2, np.float32).reshape(Hv, Dv, Dk)
eg_h, g_c = host_eg(g)
for hv in range(Hv):
    m = np.abs(eg_c[hv]).max()
    print(
        f"  h{hv:2d} chain_eg={m:.6e} g={g_c[hv]:.8e} host_eg={eg_h[hv]:.6e} beta={to_np(beta, np.uint16)[hv]:#06x}",
        flush=True,
    )

# ---- fused arm ----
of, sf = mx.fast.gdn_fused_step(
    q,
    k,
    v,
    b,
    a,
    al,
    db,
    cs,
    cw,
    z,
    st,
    w,
    inv_scale=INV_SCALE,
    prep_eps=1e-6,
    norm_eps=1e-6,
)
mx.eval(of, sf)
eg_f = to_np(sf, np.float32).reshape(Hv, Dv, Dk)
print("FUSED per-head state(=eg):", flush=True)
for hv in range(Hv):
    print(
        f"  h{hv:2d} fused_eg={np.abs(eg_f[hv]).max():.6e}",
        flush=True,
    )
print("EG-PROBE-DONE", flush=True)
