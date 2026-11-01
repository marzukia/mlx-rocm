#!/usr/bin/env python3
"""Locate the g-formula difference: model compute_g (f32) vs the kernel's
gdn_prep-style f32 softplus, on the real layer-0 a/A_log/dt_bias values."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import compute_g
from mlx_lm.utils import load

model_full, tokenizer = load("/mnt/models/qwen3-9b-mlx")
mx.eval(model_full.parameters())
lm = model_full.language_model
lay = lm.layers[0].linear_attn

# build the real layer-0 a / b for the V2 tokens
toks = None
# just use the prefilled state: prefill then capture a,b for t=[t1, 8160]
cap = {}
orig = type(lay).__call__


def hook(self, x, mask=None, cache=None):
    o = orig(self, x, mask, cache)
    return o


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

# capture the layer-0 input residual, then compute a/b directly
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
a_t = lay.in_proj_a(h)
b_t = lay.in_proj_b(h)

a = np.asarray(a_t.astype(mx.float32))  # [1,2,32] f32 of bf16
a = np.asarray(a_t.astype(mx.float32))  # [1,2,32] f32 of bf16
b = np.asarray(b_t.astype(mx.float32))
A_log = np.asarray(lay.A_log.astype(mx.float32))  # [32]
dt_bias = np.asarray(lay.dt_bias.astype(mx.float32))  # [32] bf16->f32
print("a shape:", a.shape, "A_log:", A_log.shape, flush=True)

# model path (f32): g = exp(-exp(A_log) * softplus_bf16(a + dt_bias))
g_model = np.asarray(compute_g(lay.A_log, a_t, lay.dt_bias).astype(mx.float32))


# kernel path (f32): sum = bf16(a_f32 + bias_f32); sp_f32 = mxv + log1p(exp(mnv-mxv));
#                   sp = bf16(sp_f32); arg = -expf(A_log)*sp; g = expf(arg)
def bf16(x):
    return x.astype(np.float32).astype(np.float32)  # placeholder


def f2bf(x):
    # f32 -> bf16 -> f32 (RNE via bit ops)
    u = x.view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return (r).view(np.float32).astype(np.float32)


af = a  # already f32 of bf16 values
sumf = f2bf(af + dt_bias[None, None, :])  # bf16 rounding
fx = sumf
mxv = np.maximum(fx, 0.0)
mnv = np.minimum(fx, 0.0)
sp_f = mxv + np.log1p(np.exp(mnv - mxv))  # kernel's stable softplus
sp_k = f2bf(sp_f)
arg_k = -np.exp(A_log[None, None, :]) * sp_k
g_kernel = np.exp(arg_k)
# alternative: straight softplus (nn.softplus style)
sp_s = np.log1p(np.exp(fx))
sp_s2 = f2bf(sp_s)
arg_s = -np.exp(A_log[None, None, :]) * sp_s2
g_straight = np.exp(arg_s)

print(
    "g vs kernel-stable ndiff:",
    int((g_model != g_kernel).sum()),
    "/",
    g_model.size,
    "maxdiff",
    float(np.abs(g_model - g_kernel).max()),
    flush=True,
)
print(
    "g vs straight ndiff:",
    int((g_model != g_straight).sum()),
    "/",
    g_model.size,
    "maxdiff",
    float(np.abs(g_model - g_straight).max()),
    flush=True,
)
# where they differ: print first few
idx = np.argwhere(g_model != g_kernel)
if len(idx):
    i0, j0, k0 = idx[0]
    print(
        "first diff at",
        idx[0],
        "model",
        g_model[i0, j0, k0],
        "kernel",
        g_kernel[i0, j0, k0],
        "fx",
        fx[i0, j0, k0],
        flush=True,
    )
idx2 = np.argwhere(g_model != g_straight)
print("straight ndiff:", len(idx2), flush=True)
if len(idx2):
    i0, j0, k0 = idx2[0]
    print(
        "first straight diff at",
        idx2[0],
        "model",
        g_model[i0, j0, k0],
        "straight",
        g_straight[i0, j0, k0],
        "fx",
        fx[i0, j0, k0],
        flush=True,
    )
# also check bf16-rounded g (what the state update sees)
gb_m = f2bf(g_model)
gb_k = f2bf(g_kernel)
gb_s = f2bf(g_straight)
print("bf16(g) model vs kernel ndiff:", int((gb_m != gb_k).sum()), flush=True)
print("bf16(g) model vs straight ndiff:", int((gb_m != gb_s).sum()), flush=True)
print("DONE", flush=True)
