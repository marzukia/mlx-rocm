#!/usr/bin/env python3
"""t789 short rocprof pass: dispatch the three fused GDN kernels (T=1 shapes) so a
rocprofv3 run can report the kernel names actually dispatched (rule 6)."""

import os

os.environ["MLX_GDN_TAIL_FUSION"] = "1"

import mlx.core as mx

mx.random.seed(42)
key_dim, value_dim, conv_dim, K = 2048, 4096, 8192, 4

q = (mx.random.normal((1, 1, key_dim)) * 0.1).astype(mx.bfloat16)
k = (mx.random.normal((1, 1, key_dim)) * 0.1).astype(mx.bfloat16)
v = (mx.random.normal((1, 1, value_dim)) * 0.1).astype(mx.bfloat16)
b = (mx.random.normal((1, 1, 32)) * 0.1).astype(mx.bfloat16)
a = (mx.random.normal((1, 1, 32)) * 0.1).astype(mx.bfloat16)
A_log = mx.random.normal((32,)) * 0.5 - 2.0
dt_bias = (mx.random.normal((32,)) * 0.1).astype(mx.bfloat16)
conv_state = (mx.random.normal((3, conv_dim)) * 0.1).astype(mx.bfloat16)
conv_weight = (mx.random.normal((conv_dim, K, 1)) * 0.1).astype(mx.bfloat16)

# gdn_prep (fuses conv1d+silu, q/k rms-norm, beta sigmoid, compute_g; includes gdn_qk_norm)
qn, kn, vc, beta_f, g_f = mx.fast.gdn_prep(
    q, k, v, b, a, A_log, dt_bias, conv_state, conv_weight.reshape(conv_dim, K), inv_scale=128**-0.5, eps=1e-6
)

# gdn_norm_gated (fuses rms-norm x precise_swiglu)
out = (mx.random.normal((1, 1, 32, 128)) * 0.1).astype(mx.bfloat16)
z = (mx.random.normal((1, 1, 32, 128)) * 0.1).astype(mx.bfloat16)
weight = (mx.random.normal((128,)) * 0.1 + 1.0).astype(mx.bfloat16)
fused = mx.fast.gdn_norm_gated(out, z, weight, eps=1e-6)

mx.eval(qn, kn, vc, beta_f, g_f, fused)
print("KERNELS-DISPATCHED", flush=True)
