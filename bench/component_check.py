#!/usr/bin/env python3
"""Component check: which of conv+silu / beta sigmoid / qk rms_norm differs between the model-default chain and the
Kernel's verbatim formulas (real V2 layer-0 values)."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import normalize_qk
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
h = cap["h"]


def f2bf(x):
    u = x.view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


qkv = np.asarray(lay.in_proj_qkv(h).astype(mx.float32))  # [1,2,14336]
b_raw = np.asarray(lay.in_proj_b(h).astype(mx.float32))
z = np.asarray(lay.in_proj_z(h).astype(mx.float32))
conv_state = np.asarray(cache[0][0].astype(mx.float32))  # [1,3,14336]
cw = np.asarray(lay.conv1d.weight.reshape(lay.conv_dim, lay.conv_kernel_size).astype(mx.float32))  # [D,4]
print("qkv", qkv.shape, "conv_state", conv_state.shape, "cw", cw.shape, flush=True)
D = qkv.shape[2]

# ---- 1) conv + silu ---- model path
conv_input = mx.concatenate([mx.array(conv_state).astype(mx.bfloat16), mx.array(qkv).astype(mx.bfloat16)], axis=1)
conv_out_m = np.asarray(nn.silu(lay.conv1d(conv_input)).astype(mx.float32))  # [1,5,D]


# kernel source order (lines 225-234 / 245-254 / 266-275), per row t: w[0] * (t==0 ? cs[0] : cs[1]) + w[1] * (t==0 ?
# Cs[1] : cs[2]) + w[2] * (t==0 ? cs[2] : x[t-1...]) ... need exact; do it from the file: row t: [c_{2-t+?}] -- use the
# Documented window: row0 = [cs0, cs1, cs2, x0]; row1 = [cs1, cs2, x0, x1] additive order in kernel: w0*e0 + w1*e1 +
# W2*e2 + w3*e3 (sequential FMAs, but each e_i is a f32 multiply; the kernel does: (cond) * w0 + (cond) * w1 + ... as a
# Nested sum in source order)
def kernel_conv_row(t):
    e = [
        conv_state[0, 0 + t - 1 if t else 0],
        conv_state[0, 1 + t - 1 if t else 1],
        conv_state[0, 2] if t == 0 else qkv[0, 0],
        qkv[0, t],
    ]
    e = [
        e[0] if t == 0 else e[0],  # e0 = cs0 (t0) / cs1 (t1)
        e[1],
        e[2],
        e[3],
    ]
    e0 = conv_state[0, 0] if t == 0 else conv_state[0, 1]
    e1 = conv_state[0, 1] if t == 0 else conv_state[0, 2]
    e2 = conv_state[0, 2] if t == 0 else qkv[0, 0]
    e3 = qkv[0, t]
    acc = cw[:, 0] * e0
    acc = acc + cw[:, 1] * e1
    acc = acc + cw[:, 2] * e2
    acc = acc + cw[:, 3] * e3
    return acc


for t in range(2):
    cm = conv_out_m[0, t]
    cf = kernel_conv_row(t)
    # silu: model nn.silu (f32 compute? on bf16 input) vs kernel: Multiply(conv_bf, Sigmoid(conv_bf)): conv is cast to
    # Bf16 first? in the kernel the conv sum (f32) is rounded to bf16, THEN sigmoid in f32, then product cast to bf16?
    # Read from source: s_qconv = Multiply(conv_bf, Sigmoid(conv_bf)) with conv_bf = T(sum) -- so: bf16 sum -> f32
    # Sigmoid of bf16 value -> product f32 -> bf16.
    sum_bf = f2bf(cf)
    sig = 1.0 / (1.0 + np.exp(-sum_bf))
    silu_k = f2bf(sig * sum_bf)
    nd = int((cm != silu_k).sum())
    print(f"conv+silu row {t}: ndiff {nd}/{D} maxdiff {float(np.abs(cm - silu_k).max())}", flush=True)

# ---- 2) beta sigmoid ----
beta_m = np.asarray(mx.sigmoid(mx.array(b_raw).astype(mx.bfloat16)).astype(mx.float32))
sig_k = f2bf(1.0 / (1.0 + np.exp(-b_raw)))
print(
    f"beta: ndiff {int((beta_m != sig_k).sum())}/{beta_m.size} maxdiff {float(np.abs(beta_m - sig_k).max())}",
    flush=True,
)

# ---- 3) qk rms_norm ---- model: normalize_qk uses mx.fast.rms_norm (256-thread ROCm kernel) get conv_out (model,
# Bf16) split into q,k
q3 = qkv[:, :, : lay.key_dim]
k3 = qkv[:, :, lay.key_dim : 2 * lay.key_dim]
# need silu'd q,k from the MODEL conv output (last 2 rows)
co = conv_out_m  # f32 of bf16
print("conv_out_m shape:", co.shape, flush=True)
co2 = co[0, -2:]
q_m = co2[:, : lay.key_dim].reshape(2, -1)
k_m = co2[:, lay.key_dim : 2 * lay.key_dim].reshape(2, -1)
HD = q_m.shape[1] // lay.num_k_heads
print("HD (head dim):", HD, "key_dim:", lay.key_dim, flush=True)
qm = mx.array(q_m).astype(mx.bfloat16).reshape(1, 2, lay.num_k_heads, HD)
km = mx.array(k_m).astype(mx.bfloat16).reshape(1, 2, lay.num_k_heads, HD)
qn_m, kn_m = normalize_qk(qm, km, inv_scale=lay.head_k_dim**-0.5, eps=1e-6)


# kernel: rms = 256-thread tree over 128 elems (threads 128..255 = 0), then x/rms * scale -- the reduction order: warp
# Tree (shfl_xor 16,8,4,2,1) across 256 threads. Replicate: each thread has 1 element (128) or 0; sum = sequential
# Tree. With 128 active of 256: level1 pairs (t,t+1)... do the tree in numpy per the standard pattern (pairs of
# Adjacent, halving).
def tree256(xv):
    n = 256
    x = np.concatenate([xv, np.zeros(n - len(xv))])
    while n > 1:
        x = x[:n:2] + x[1:n:2]
        n //= 2
    return x[0]


inv_scale = lay.head_k_dim**-0.5
rms_eps = 1e-6 * inv_scale**2
for name, arr in (("q", q_m), ("k", k_m)):
    x = arr  # [2, nheads*HD] f32 of bf16
    x = x.reshape(2, lay.num_k_heads, HD)
    ss = np.array([tree256(np.square(r)) for r in x.reshape(-1, HD)])
    ss = ss.reshape(2, lay.num_k_heads, 1)
    rms = np.sqrt(ss / HD + rms_eps)
    scale_q = f2bf(np.full_like(x[0][0], inv_scale**2))
    scale_k = f2bf(np.full_like(x[0][0], inv_scale))
    scale = scale_q if name == "q" else scale_k
    normed = f2bf((x / rms) * scale)
    ref = np.asarray((qn_m if name == "q" else kn_m).astype(mx.float32))[0]
    print(f"rmsnorm {name}: shapes {normed.shape} {ref.shape}", flush=True)
    nd = int((ref != normed).sum())
    print(f"rmsnorm {name}: ndiff {nd}/{ref.size} maxdiff {float(np.abs(ref - normed).max())}", flush=True)
print("DONE", flush=True)
