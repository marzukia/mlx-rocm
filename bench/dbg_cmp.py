#!/usr/bin/env python3
"""Compare the fused op's captured row-1 intermediates (dbg bin) against the model chain's row-1 values (qn, kn, v, beta, G). P2_GDN_DEBUG=1 required. FUSE=0 during the model forward (capture only), direct op call after."""

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

# direct fused op call (writes /tmp/p2_gdn_dbg.bin for layer-0 values)
os.environ["MLX_ROCM_GDN_FUSE"] = "1"
qkv_mx = mx.array(qkv0 := np.asarray(lay.in_proj_qkv(h).astype(mx.float32))).astype(mx.bfloat16)
qx = qkv_mx[:, :, : lay.key_dim]
kx = qkv_mx[:, :, lay.key_dim : 2 * lay.key_dim]
vv = qkv_mx[:, :, 2 * lay.key_dim :]
cs_mx = mx.array(np.asarray(cache[0][0].astype(mx.float32)).copy()).astype(mx.bfloat16)
if cs_mx.ndim == 2:
    cs_mx = cs_mx[None]
sstate = mx.array(np.asarray(cache[0][1].astype(mx.float32)).copy())
zz = lay.in_proj_z(h).reshape(1, 2, 32, 128)
out_f, st_f = mx.fast.gdn_fused_step(
    qx,
    kx,
    vv,
    lay.in_proj_b(h),
    lay.in_proj_a(h),
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


# model chain row-1 values
qkv = np.asarray(lay.in_proj_qkv(h).astype(mx.float32))
b = np.asarray(lay.in_proj_b(h).astype(mx.float32))
a = np.asarray(lay.in_proj_a(h).astype(mx.float32))
cs = np.asarray(cache[0][0].astype(mx.float32)).copy()
if cs.ndim == 2:
    cs = cs[None]
conv_in = mx.concatenate([mx.array(cs).astype(mx.bfloat16), mx.array(qkv).astype(mx.bfloat16)], axis=1)
conv_out = np.asarray(nn.silu(lay.conv1d(conv_in)).astype(mx.float32))  # [1,2,D]
# model qn/kn: full chain (rms + scale) — recompute via mx ops
q3 = mx.array(qkv).astype(mx.bfloat16)[:, :, : lay.key_dim]
k3 = mx.array(qkv).astype(mx.bfloat16)[:, :, lay.key_dim : 2 * lay.key_dim]
v3 = mx.array(qkv).astype(mx.bfloat16)[:, :, 2 * lay.key_dim :]
qn, kn = normalize_qk(
    mx.array(conv_out).astype(mx.bfloat16)[:, :, : lay.key_dim].reshape(1, 2, 16, 128),
    mx.array(conv_out).astype(mx.bfloat16)[:, :, lay.key_dim : 2 * lay.key_dim].reshape(1, 2, 16, 128),
    inv_scale=lay.head_k_dim**-0.5,
    eps=1e-6,
)
beta = np.asarray(mx.sigmoid(mx.array(b).astype(mx.bfloat16)).astype(mx.float32))
g = np.asarray(
    compute_g(
        mx.array(lay.A_log).astype(mx.float32) if lay.A_log.dtype == mx.bfloat16 else mx.array(lay.A_log),
        mx.array(a).astype(mx.bfloat16),
        mx.array(lay.dt_bias).astype(mx.bfloat16),
    )
    .astype(mx.bfloat16)
    .astype(mx.float32)
)

dbg = np.fromfile("/tmp/p2_gdn_dbg.bin", dtype=np.uint8)
print("dbg bytes:", dbg.size, flush=True)
H = 32
mism = 0
for hv in range(H):
    p = dbg[hv * 1042 : (hv + 1) * 1042]
    qn_k = p[0:256].view(np.uint16)
    kn_k = p[256:512].view(np.uint16)
    vc_k = p[512:768].view(np.uint16)
    out_k = p[774:1030].view(np.uint16)
    b_k = np.array([p[768:770].tobytes()], dtype="S2")
    g_k = np.frombuffer(p[770:774].tobytes(), dtype=np.float32)
    # model row 1
    qn_m = np.asarray(qn[0, 1, hv // 2].astype(mx.float32))
    kn_m = np.asarray(kn[0, 1, hv // 2].astype(mx.float32))
    vc_m = conv_out[0, 1, 2 * lay.key_dim + hv * 128 : 2 * lay.key_dim + (hv + 1) * 128]
    b_m = beta[0, 1, hv]
    g_m = g[0, 1, hv]
    if hv == 0:
        kk = (p[512:768].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        for i in np.where(kk != vc_m)[0][:8]:
            print(f"hv0 v[{i}]: kernel {kk[i]} model {vc_m[i]}", flush=True)
        kk2 = (p[0:256].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        for i in np.where(kk2 != qn_m)[0][:8]:
            print(f"hv0 qn[{i}]: kernel {kk2[i]} model {qn_m[i]}", flush=True)

    def cmp(name, k_u16, m_f32):
        global mism
        kf = k_u16.astype(np.uint32) << 16
        kf = kf.view(np.float32)
        d = float(np.abs(kf - m_f32).max())
        nd = int((kf != m_f32).sum())
        if nd:
            mism += 1
            print(f"hv={hv} {name}: ndiff {nd}/128 maxdiff {d}", flush=True)

    cmp("qn", qn_k, qn_m)
    cmp("kn", kn_k, kn_m)
    cmp("v", vc_k, vc_m)
    # beta: bf16 u16 vs f32 (model value already bf16-exact)
    bf_k = int.from_bytes(p[768:770], "little")
    bf_m = int.from_bytes(np.float32(f2bf(np.array([b_m]))[0]).tobytes()[2:], "little")
    if bf_k != bf_m:
        mism += 1
        print(f"hv={hv} beta: kernel {bf_k:04x} model {bf_m:04x}", flush=True)
    # g: kernel writes the f32 s_g; the chain sees bf16 of it
    if f2bf(g_k)[0] != g_m:
        mism += 1
        print(f"hv={hv} g: kernel f32 {g_k[0]} (bf {f2bf(g_k)[0]}) model {g_m}", flush=True)
print("MISMATCHES:", mism, "of", H * 5, flush=True)
print("DONE", flush=True)
