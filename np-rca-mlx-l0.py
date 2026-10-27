import os
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import numpy as np
import mlx.core as mx
import mlx.nn as mnn
from mlx_lm import load
import mlx_lm.models.gated_delta as gd
D = "/mnt/models/qwen3-9b-mlx"
model, tok = load(D)
tm = model.language_model.model
mx.eval(model.parameters())
PROMPT = "What color are the fruits watermelon, strawberry, orange and banana?"
prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
x = tm.embed_tokens(mx.array([prompt_ids]))
layer = tm.layers[0]
h1 = layer.input_layernorm(x)
la = layer.linear_attn
def n(t):
    return float(np.linalg.norm(np.array(t.astype(mx.float32))))
def mean(t):
    return float(np.array(t.astype(mx.float32)).mean())

def run_gdn(use_kernel):
    # replicate GatedDeltaNet.__call__ but control the recurrence path
    B, S, _ = h1.shape
    qkv = la.in_proj_qkv(h1)
    z = la.in_proj_z(h1).reshape(B, S, la.num_v_heads, la.head_v_dim)
    b = la.in_proj_b(h1)
    a = la.in_proj_a(h1)
    conv_state = mx.zeros((B, la.conv_kernel_size - 1, la.conv_dim), dtype=h1.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = mnn.silu(la.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
        [la.num_k_heads, la.num_k_heads, la.num_v_heads],
        [la.head_k_dim, la.head_k_dim, la.head_v_dim])]
    state = mx.zeros((B, la.num_v_heads, la.head_v_dim, la.head_k_dim), dtype=mx.float32)
    q, k = gd.normalize_qk(q, k, inv_scale=la.head_k_dim ** -0.5, eps=1e-6)
    beta = mx.sigmoid(b)
    g = gd.compute_g(la.A_log, a, la.dt_bias)
    if use_kernel:
        out, state = mx.fast.gated_delta_update(q, k, v, g, beta, state, None)
    else:
        out, state = gd.gated_delta_ops(q, k, v, g, beta, state, None)
    out = la.norm(out, z)
    out = la.out_proj(out.reshape(B, S, -1))
    return out, z, q, k, v, g, beta

# K3 kernel (default ROCm path)
os.environ.pop("MLX_ROCM_GDN_FALLBACK", None)
os.environ.pop("MLX_GDN_MLXFAST", None)
r_kernel, z_k, q_k, k_k, v_k, g_k, b_k = run_gdn(True)
# dump before the op loop (which may crash on JIT)
print(f"h1 norm={n(h1):.3f}", flush=True)
print(f"z:  kernel norm={n(z_k):.3f} mean={mean(z_k):.4f}", flush=True)
print(f"q:  kernel norm={n(q_k):.3f}", flush=True)
print(f"k:  kernel norm={n(k_k):.3f}", flush=True)
print(f"g:  kernel norm={n(g_k):.3f} mean={mean(g_k):.4f}", flush=True)
print(f"beta: kernel mean={mean(b_k):.4f}", flush=True)
print(f"r:  K3 kernel norm={n(r_kernel):.3f}", flush=True)
np.save("/tmp/np-rca-mlx-h1.npy", np.array(h1.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-z.npy", np.array(z_k.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-ipz.npy", np.array(la.in_proj_z.weight.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-rk.npy", np.array(r_kernel.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-iln.npy", np.array(layer.input_layernorm.weight.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-emb.npy", np.array(x.astype(mx.float32)))
# numpy reference for comparison
np_r = np.load("/tmp/np-rca-np-r.npy")
np_z = np.load("/tmp/np-rca-np-z.npy")
np_h1 = np.load("/tmp/np-rca-np-h1.npy")
np_ipz = np.load("/tmp/np-rca-np-ipz.npy")
print(f"r:  numpy(op-ref) norm={np.linalg.norm(np_r):.3f}", flush=True)
d_k = np.abs(np.array(r_kernel.astype(mx.float32)) - np_r)
print(f"r:  maxdiff kernel-vs-numpy={d_k.max():.4f}", flush=True)
dz = np.abs(np.array(z_k.astype(mx.float32)) - np_z)
print(f"z:  maxdiff kernel-vs-numpy={dz.max():.4f} np_norm={np.linalg.norm(np_z):.3f}", flush=True)
dh = np.abs(np.array(h1.astype(mx.float32)) - np_h1)
print(f"h1: maxdiff kernel-vs-numpy={dh.max():.4f} np_norm={np.linalg.norm(np_h1):.3f}", flush=True)
print(f"ipz: mlx_shape={np.array(la.in_proj_z.weight).shape} np_shape={np_ipz.shape} np_norm={np.linalg.norm(np_ipz):.3f}", flush=True)
dil = np.abs(np.array(layer.input_layernorm.weight.astype(mx.float32)) - np.load("/tmp/np-rca-np-iln.npy"))
print(f"iln: mlx_norm={np.linalg.norm(np.array(layer.input_layernorm.weight.astype(mx.float32))):.3f} np_norm={np.linalg.norm(np.load('/tmp/np-rca-np-iln.npy')):.3f} maxdiff={dil.max():.4f}", flush=True)
demb = np.abs(np.array(x.astype(mx.float32)) - np.load("/tmp/np-rca-np-emb.npy"))
print(f"emb: mlx_norm={np.linalg.norm(np.array(x.astype(mx.float32))):.3f} np_norm={np.linalg.norm(np.load('/tmp/np-rca-np-emb.npy')):.3f} maxdiff={demb.max():.4f}", flush=True)
print("mlx-l0-compare-done", flush=True)
# op loop (may crash on JIT; run last)
os.environ["MLX_ROCM_GDN_FALLBACK"] = "1"
r_ops, z_o, q_o, k_o, v_o, g_o, b_o = run_gdn(False)
os.environ.pop("MLX_ROCM_GDN_FALLBACK", None)
d_o = np.abs(np.array(r_ops.astype(mx.float32)) - np_r)
print(f"r:  oploop norm={n(r_ops):.3f} maxdiff oploop-vs-numpy={d_o.max():.4f}", flush=True)
