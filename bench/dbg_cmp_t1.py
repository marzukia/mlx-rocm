#!/usr/bin/env python3
"""T=1 dbg: kernel's row-0 intermediates vs model row-0 (same post-fwd state)."""

import os

os.environ["MLX_ROCM_GDN_FUSE"] = "0"
os.environ["P2_GDN_DEBUG"] = "1"
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
h = cap["h"]

# T=1 direct op call
os.environ["MLX_ROCM_GDN_FUSE"] = "1"
h1 = h[:, :1]
qkv1 = np.asarray(lay.in_proj_qkv(h1).astype(mx.float32))
qkv_mx = mx.array(qkv1).astype(mx.bfloat16)
cs_np = np.asarray(cache[0][0].astype(mx.float32)).copy()
if cs_np.ndim == 2:
    cs_np = cs_np[None]
cs_mx = mx.array(cs_np).astype(mx.bfloat16)
sstate = mx.array(np.asarray(cache[0][1].astype(mx.float32)).copy())
zz = lay.in_proj_z(h1).reshape(1, 1, 32, 128)
out_f, st_f = mx.fast.gdn_fused_step(
    qkv_mx[:, :, : lay.key_dim],
    qkv_mx[:, :, lay.key_dim : 2 * lay.key_dim],
    qkv_mx[:, :, 2 * lay.key_dim :],
    lay.in_proj_b(h1),
    lay.in_proj_a(h1),
    lay.A_log,
    lay.dt_bias,
    cs_mx.reshape(lay.conv_kernel_size - 1, lay.conv_dim),
    lay.conv1d.weight.reshape(lay.conv_dim, lay.conv_kernel_size),
    zz,
    sstate,
    lay.norm.weight,
    inv_scale=lay.head_k_dim**-0.5,
    prep_eps=1e-6,
    norm_eps=lay.layer_norm_epsilon,
)
mx.eval(out_f, st_f)
os.environ["MLX_ROCM_GDN_FUSE"] = "0"


def f2bf(x):
    u = x.view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


# model chain row 0, same post-fwd state
conv_in = mx.concatenate([cs_mx, mx.array(qkv1).astype(mx.bfloat16)], axis=1)
conv_out = np.asarray(nn.silu(lay.conv1d(conv_in)).astype(mx.float32))
qn, kn = normalize_qk(
    mx.array(conv_out).astype(mx.bfloat16)[0, 0, : lay.key_dim].reshape(1, 1, 16, 128),
    mx.array(conv_out).astype(mx.bfloat16)[0, 0, lay.key_dim : 2 * lay.key_dim].reshape(1, 1, 16, 128),
    inv_scale=lay.head_k_dim**-0.5,
    eps=1e-6,
)
b = np.asarray(lay.in_proj_b(h1).astype(mx.float32))
a = np.asarray(lay.in_proj_a(h1).astype(mx.float32))
beta = np.asarray(mx.sigmoid(mx.array(b).astype(mx.bfloat16)).astype(mx.float32))
g = np.asarray(
    compute_g(mx.array(lay.A_log), mx.array(a).astype(mx.bfloat16), mx.array(lay.dt_bias).astype(mx.bfloat16))
    .astype(mx.bfloat16)
    .astype(mx.float32)
)

dbg = np.fromfile("/tmp/p2_gdn_dbg.bin", dtype=np.uint8)
print("dbg bytes:", dbg.size, flush=True)
H = 32
mism = 0
for hv in range(H):
    p = dbg[hv * 1042 : (hv + 1) * 1042]
    qn_k = (p[0:256].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    kn_k = (p[256:512].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    vc_k = (p[512:768].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    qn_m = np.asarray(qn[0, 0, hv // 2].astype(mx.float32))
    kn_m = np.asarray(kn[0, 0, hv // 2].astype(mx.float32))
    vc_m = conv_out[0, 0, 2 * lay.key_dim + hv * 128 : 2 * lay.key_dim + (hv + 1) * 128]
    for name, k, m in (("qn", qn_k, qn_m), ("kn", kn_k, kn_m), ("v", vc_k, vc_m)):
        nd = int((k != m).sum())
        if nd:
            mism += 1
            print(f"hv={hv} {name}: ndiff {nd}/128 maxdiff {float(np.abs(k - m).max())}", flush=True)
    bf_k = int.from_bytes(p[768:770], "little")
    bf_m = int.from_bytes(np.float32(f2bf(np.array([beta[0, 0, hv]]))[0]).tobytes()[2:], "little")
    if bf_k != bf_m:
        mism += 1
        print(f"hv={hv} beta: kernel {bf_k:04x} model {bf_m:04x}", flush=True)
    gk = np.frombuffer(p[770:774].tobytes(), dtype=np.float32)
    if f2bf(gk)[0] != g[0, 0, hv]:
        mism += 1
        print(f"hv={hv} g: kernel {f2bf(gk)[0]} model {g[0, 0, hv]}", flush=True)
print("MISMATCHES:", mism, "of", H * 5, flush=True)
print("DONE", flush=True)
