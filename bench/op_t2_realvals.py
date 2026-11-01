#!/usr/bin/env python3
"""Op-level T=2 comparison with REAL model values (layer 0). Compares mx.fast.gdn_fused_step (the F-branch op) against the Model-default chain (nn.Conv1d + silu + compute_g + normalize_qk + mx.fast.gated_delta_update) on the exact layer-0 Inputs of the bench's V2 step (tokens t1, 8160 from the prefill state). Determines whether the divergence is (a) an op Kernel bug or (b) an F-branch glue bug."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load

model_dir = os.environ["MODELS"] + "/qwen3-9b-mlx"
model_full, tokenizer = load(model_dir)
mx.eval(model_full.parameters())
lm = model_full.language_model
model = lm

prompt = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
cache = lm.make_cache()
logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
t1 = int(np.argmax(np.asarray(logits[0, -1].astype(mx.float32))))
t2 = 8160
print("prefill done; t1 =", t1, flush=True)

lay = model.layers[0].linear_attn
c0 = cache[0]
conv0 = np.asarray(c0[0].astype(mx.float32)).copy()
ssm0 = np.asarray(c0[1].astype(mx.float32)).copy()
print("conv0 shape:", c0[0].shape, "ssm0 shape:", c0[1].shape, flush=True)

# ---- capture the layer-0 input residual ----
cap = {}
layer0 = model.layers[0]
orig_call = type(layer0).__call__


def hook_call(self, hidden_states, mask=None, cache=None, **kw):
    cap.setdefault("h", hidden_states)
    return orig_call(self, hidden_states, mask, cache, **kw)


type(layer0).__call__ = hook_call
lm(mx.array([[t1, t2]]), cache=cache, return_hidden=True, return_prenorm=True)
type(layer0).__call__ = orig_call
h = cap["h"]
print("h shape:", tuple(h.shape), flush=True)


def run_default():
    qkv = lay.in_proj_qkv(h)
    z = lay.in_proj_z(h).reshape(1, 2, lay.num_v_heads, lay.head_v_dim)
    b = lay.in_proj_b(h)
    a = lay.in_proj_a(h)
    conv_state = mx.array(conv0).astype(mx.bfloat16)
    if conv_state.ndim == 2:
        conv_state = conv_state[None, :, :]
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(lay.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, 2, hh, dd)
        for t, hh, dd in zip(
            mx.split(conv_out, [lay.key_dim, 2 * lay.key_dim], -1),
            [lay.num_k_heads, lay.num_k_heads, lay.num_v_heads],
            [lay.head_k_dim, lay.head_k_dim, lay.head_v_dim],
        )
    ]
    state = mx.array(ssm0)
    from mlx_lm.models.gated_delta import compute_g, normalize_qk

    q_n, k_n = normalize_qk(q, k, inv_scale=lay.head_k_dim**-0.5, eps=1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(lay.A_log, a, lay.dt_bias)
    out, state = mx.fast.gated_delta_update(q_n, k_n, v, g, beta, state, None)
    out = lay.norm(out, z)
    return out, state


def run_fused():
    qkv = lay.in_proj_qkv(h)
    z = lay.in_proj_z(h).reshape(1, 2, lay.num_v_heads, lay.head_v_dim)
    b = lay.in_proj_b(h)
    a = lay.in_proj_a(h)
    conv_state = mx.array(conv0).astype(mx.bfloat16)
    if conv_state.ndim == 2:
        conv_state = conv_state[None, :, :]
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
        conv_state.reshape(lay.conv_kernel_size - 1, lay.conv_dim),
        lay.conv1d.weight.reshape(lay.conv_dim, lay.conv_kernel_size),
        z,
        state,
        lay.norm.weight,
        inv_scale=lay.head_k_dim**-0.5,
        prep_eps=1e-6,
        norm_eps=lay.layer_norm_epsilon,
    )
    return out, state


od, sd = run_default()
of_, sf = run_fused()
od = np.asarray(od.astype(mx.float32))
of_ = np.asarray(of_.astype(mx.float32))
sd = np.asarray(sd.astype(mx.float32))
sf = np.asarray(sf.astype(mx.float32))
print("out shape:", od.shape, sd.shape, flush=True)
print("out maxdiff:", float(np.abs(od - of_).max()), flush=True)
print("state maxdiff:", float(np.abs(sd - sf).max()), flush=True)
print("state ndiff:", int((sd != sf).sum()), "/", sd.size, flush=True)
# per-row detail
sd2 = sd[0]
sf2 = sf[0]
for r in range(4):
    print(
        f"  row {r}: maxdiff {float(np.abs(sd2[r] - sf2[r]).max())} "
        f"ndiff {int((sd2[r] != sf2[r]).sum())} "
        f"rowmax {float(np.abs(sd2[r]).max())}",
        flush=True,
    )
