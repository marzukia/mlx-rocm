#!/usr/bin/env python3
"""Bisect which t=1 window input causes the divergence: zero each candidate (cs row 1, cs row 2, qkv row 0, qkv row 1) in BOTH the fused op and the default chain, compare states."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import compute_g, normalize_qk
from mlx_lm.utils import load

MODELS = os.environ.get("MODELS", "/models")
model_full, tokenizer = load(f"{MODELS}/qwen3-9b-mlx")
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
h = cap["h"]

cs0 = np.asarray(cache[0][0].astype(mx.float32)).copy()
if cs0.ndim == 2:
    cs0 = cs0[None]
sm0 = np.asarray(cache[0][1].astype(mx.float32)).copy()
qkv0 = np.asarray(lay.in_proj_qkv(h).astype(mx.float32))
b = np.asarray(lay.in_proj_b(h).astype(mx.float32))
a = np.asarray(lay.in_proj_a(h).astype(mx.float32))
print("cs0:", cs0.shape, "qkv0:", qkv0.shape, flush=True)


def run(cs, qkv, name):
    # default chain
    conv_in = mx.concatenate([mx.array(cs).astype(mx.bfloat16), mx.array(qkv).astype(mx.bfloat16)], axis=1)
    conv_out = nn.silu(lay.conv1d(conv_in))
    q = mx.split(conv_out, [lay.key_dim, 2 * lay.key_dim], -1)[0].reshape(1, 2, 16, 128)
    k = mx.split(conv_out, [lay.key_dim, 2 * lay.key_dim], -1)[1].reshape(1, 2, 16, 128)
    v = mx.split(conv_out, [lay.key_dim, 2 * lay.key_dim], -1)[2].reshape(1, 2, 32, 128)
    qn, kn = normalize_qk(q, k, inv_scale=lay.head_k_dim**-0.5, eps=1e-6)
    beta = mx.sigmoid(mx.array(b).astype(mx.bfloat16))
    g = compute_g(mx.array(lay.A_log), mx.array(a).astype(mx.bfloat16), mx.array(lay.dt_bias).astype(mx.bfloat16))
    od, sd = mx.fast.gated_delta_update(qn, kn, v, g, beta, mx.array(sm0), None)
    sd = np.asarray(sd.astype(mx.float32)).copy()
    # fused op
    os.environ["MLX_ROCM_FUSE" if False else "MLX_ROCM_GDN_FUSE"] = "1"
    qkv_mx = mx.array(qkv).astype(mx.bfloat16)
    cs_mx = mx.array(cs).astype(mx.bfloat16)
    zz = lay.in_proj_z(h).reshape(1, 2, 32, 128)
    of, sf = mx.fast.gdn_fused_step(
        qkv_mx[:, :, : lay.key_dim],
        qkv_mx[:, :, lay.key_dim : 2 * lay.key_dim],
        qkv_mx[:, :, 2 * lay.key_dim :],
        mx.array(b).astype(mx.bfloat16),
        mx.array(a).astype(mx.bfloat16),
        lay.A_log,
        lay.dt_bias,
        cs_mx.reshape(lay.conv_kernel_size - 1, lay.conv_dim),
        lay.conv1d.weight.reshape(lay.conv_dim, lay.conv_kernel_size),
        zz,
        mx.array(sm0),
        lay.norm.weight,
        inv_scale=lay.head_k_dim**-0.5,
        prep_eps=1e-6,
        norm_eps=lay.layer_norm_epsilon,
    )
    sf = np.asarray(sf.astype(mx.float32)).copy()
    os.environ["MLX_ROCM_GDN_FUSE"] = "0"
    nd = int((sd != sf).sum())
    print(f"{name}: state ndiff {nd}/{sd.size} maxdiff {float(np.abs(sd - sf).max())}", flush=True)
    return nd


cs = cs0.copy()
qkv = qkv0.copy()
run(cs, qkv, "baseline (all inputs)")
cs = cs0.copy()
qkv = qkv0.copy()
cs[0, 1] = 0.0
run(cs, qkv, "zero cs row 1")
cs = cs0.copy()
qkv = qkv0.copy()
cs[0, 2] = 0.0
run(cs, qkv, "zero cs row 2")
cs = cs0.copy()
qkv = qkv0.copy()
qkv[0, 0] = 0.0
run(cs, qkv, "zero qkv row 0")
cs = cs0.copy()
qkv = qkv0.copy()
qkv[0, 1] = 0.0
run(cs, qkv, "zero qkv row 1")
cs = cs0.copy()
qkv = qkv0.copy()
cs[0, 1] = 0.0
qkv[0, 0] = 0.0
run(cs, qkv, "zero cs1 + qkv0")
print("DONE", flush=True)

# shifted-window probe: cs0 = big, all other window inputs zero
cs = cs0.copy()
qkv = qkv0.copy()
for r in range(3):
    cs[0, r] = 0.0
qkv[0, 0] = 0.0
qkv[0, 1] = 0.0
cs[0, 0] = 3.5
run(cs, qkv, "cs0=3.5, rest zero (shift probe)")
cs = cs0.copy()
qkv = qkv0.copy()
for r in range(3):
    cs[0, r] = 0.0
qkv[0, 0] = 0.0
qkv[0, 1] = 0.0
cs[0, 1] = 3.5
run(cs, qkv, "cs1=3.5, rest zero (shift probe)")
