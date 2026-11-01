#!/usr/bin/env python3
"""T=1 vs T=2 op discrimination (real layer-0 values). From the same state, run: - fused op T=1 (row 0) vs Mx.fast.gated_delta_update T=1 (row 0) - fused op T=2 vs mx.fast.gated_delta_update T=2 If T=1 matches and T=2 diverges -> row-carry/indexing bug in the fused op."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import compute_g, normalize_qk
from mlx_lm.utils import load

model_full, tokenizer = load(os.environ["MODELS"] + "/qwen3-9b-mlx")
mx.eval(model_full.parameters())
lm = model_full.language_model
lay = lm.layers[0].linear_attn

prompt_ids = tokenizer.encode(
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive.",
    add_special_tokens=False,
)
cache = lm.make_cache()
logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
t1 = int(np.argmax(np.asarray(logits[0, -1].astype(mx.float32))))
t2 = 8160

cap = {}
layer0 = lm.layers[0]
orig_call = type(layer0).__call__


def hook_call(self, hidden_states, mask=None, cache=None, **kw):
    cap.setdefault("h", hidden_states)
    return orig_call(self, hidden_states, mask, cache, **kw)


type(layer0).__call__ = hook_call
lm(mx.array([[t1, t2]]), cache=cache, return_hidden=True, return_prenorm=True)
type(layer0).__call__ = orig_call
h = cap["h"]  # [1, 2, 4096]


def f2bf(x):
    u = x.view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


conv_state_np = np.asarray(cache[0][0].astype(mx.float32)).copy()
ssm0 = np.asarray(cache[0][1].astype(mx.float32)).copy()
print("t1:", t1, "ssm0:", ssm0.shape, flush=True)


def prep(ht):
    qkv = lay.in_proj_qkv(ht)
    z = lay.in_proj_z(ht).reshape(1, ht.shape[1], lay.num_v_heads, lay.head_v_dim)
    b = lay.in_proj_b(ht)
    a = lay.in_proj_a(ht)
    return qkv, z, b, a


def run_default_t(T):
    ht = h[:, :T]
    qkv, z, b, a = prep(ht)
    cs = mx.array(conv_state_np).astype(mx.bfloat16)
    if cs.ndim == 2:
        cs = cs[None]
    conv_in = mx.concatenate([cs, qkv], axis=1)
    conv_out = nn.silu(lay.conv1d(conv_in))
    q, k, v = [
        t.reshape(1, T, hh, dd)
        for t, hh, dd in zip(
            mx.split(conv_out, [lay.key_dim, 2 * lay.key_dim], -1),
            [lay.num_k_heads, lay.num_k_heads, lay.num_v_heads],
            [lay.head_k_dim, lay.head_k_dim, lay.head_v_dim],
        )
    ]
    state = mx.array(ssm0)
    qn, kn = normalize_qk(q, k, inv_scale=lay.head_k_dim**-0.5, eps=1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(lay.A_log, a, lay.dt_bias)
    out, state = mx.fast.gated_delta_update(qn, kn, v, g, beta, state, None)
    return out, state


def run_fused_t(T):
    ht = h[:, :T]
    qkv, z, b, a = prep(ht)
    cs = mx.array(conv_state_np).astype(mx.bfloat16)
    if cs.ndim == 2:
        cs = cs[None]
    q3 = qkv[:, :, : lay.key_dim]
    k3 = qkv[:, :, lay.key_dim : 2 * lay.key_dim]
    v3 = qkv[:, :, 2 * lay.key_dim :]
    state = mx.array(ssm0)
    os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    out, state = mx.fast.gdn_fused_step(
        q3,
        k3,
        v3,
        b,
        a,
        lay.A_log,
        lay.dt_bias,
        cs.reshape(lay.conv_kernel_size - 1, lay.conv_dim),
        lay.conv1d.weight.reshape(lay.conv_dim, lay.conv_kernel_size),
        z,
        state,
        lay.norm.weight,
        inv_scale=lay.head_k_dim**-0.5,
        prep_eps=1e-6,
        norm_eps=lay.layer_norm_epsilon,
    )
    os.environ["MLX_ROCM_GDN_FUSE"] = "0"
    return out, state


for T in (1, 2):
    od, sd = run_default_t(T)
    of, sf = run_fused_t(T)
    od = np.asarray(od.astype(mx.float32))
    of = np.asarray(of.astype(mx.float32))
    sd = np.asarray(sd.astype(mx.float32))
    sf = np.asarray(sf.astype(mx.float32))
    print(
        f"T={T}: out maxdiff {float(np.abs(od - of).max())} "
        f"state maxdiff {float(np.abs(sd - sf).max())} "
        f"state ndiff {int((sd != sf).sum())}/{sd.size}",
        flush=True,
    )
print("DONE", flush=True)
