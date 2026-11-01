#!/usr/bin/env python3
"""conv+silu check with FMA semantics (HIP compiles acc += a*b to fmaf).""" import os os.environ["MLX_ROCM_GDN_FUSE"] =
"0" import math import mlx.core as mx import mlx.nn as nn import numpy as np from mlx_lm.utils import load model_full,
Tokenizer = load(os.environ["MODELS"] + "/qwen3-9b-mlx") mx.eval(model_full.parameters()) lm =
Model_full.language_model lay = lm.layers[0].linear_attn prompt_ids = tokenizer.encode( "Explain in great detail how a
Modern smartphone works, from the " "hardware components to the operating system, covering every " "subsystem. Be
Exhaustive.", add_special_tokens=False, ) cache = lm.make_cache() logits, hidden = lm(mx.array([prompt_ids]),
Cache=cache, return_hidden=True, return_prenorm=True) t1 = int(np.argmax(np.asarray(logits[0, -1].astype(mx.float32))))
T2 = 8160 cap = {} layer0 = lm.layers[0] orig_call = type(layer0).__call__ def hook_call(self, hidden_states,
Mask=None, cache=None, **kw): cap.setdefault("h", hidden_states) return orig_call(self, hidden_states, mask, cache,
**kw) type(layer0).__call__ = hook_call lm(mx.array([[t1, t2]]), cache=cache, return_hidden=True, return_prenorm=True)
Type(layer0).__call__ = orig_call h = cap["h"] def f2bf(x): u = x.view(np.uint32) r = (u + 0x7FFF + ((u >> 16) & 1)) &
Np.uint32(0xFFFF0000) return r.view(np.float32) qkv = np.asarray(lay.in_proj_qkv(h).astype(mx.float32)) conv_state =
Np.asarray(cache[0][0].astype(mx.float32)) # [1,3,8192] cw = np.asarray(lay.conv1d.weight.reshape(lay.conv_dim,
Lay.conv_kernel_size).astype(mx.float32)) D = qkv.shape[2] print("D:", D, "cw:", cw.shape, flush=True) # MX model conv
Output conv_in = mx.concatenate([mx.array(conv_state).astype(mx.bfloat16), mx.array(qkv).astype(mx.bfloat16)], axis=1)
Conv_out_m = np.asarray(nn.silu(lay.conv1d(conv_in)).astype(mx.float32)) print("conv_out_m:", conv_out_m.shape,
Flush=True) cm0 = conv_out_m[0, 0] if conv_out_m.ndim == 3 else conv_out_m[0] cm1 = conv_out_m[0, 1] if conv_out_m.ndim
== 3 else conv_out_m[1] def krow(t): e = np.stack( [ conv_state[0, 0] if t == 0 else conv_state[0, 1], conv_state[0, 1]
If t == 0 else conv_state[0, 2], conv_state[0, 2] if t == 0 else qkv[0, 0], qkv[0, t], ], axis=0, ) # [4, D] f32 of
Bf16 values w = cw # [D, 4] out = np.empty((4, D), dtype=np.float32) for d in range(D): acc = np.float32(0.0) for k in
Range(4): acc = np.float32(math.fma(e[k, d], w[d, k], acc)) # f32 FMA out[:, d] = acc return out for t, cm in ((0,
Cm0), (1, cm1)): acc = krow(t) sum_bf = f2bf(acc[3]) # placeholder not used; do per-element # per element: silu of
Bf16(sum) sb = f2bf(acc[3]) # sigmoid via MX's unary op formula: 1/(1+exp(-x)) sig = 1.0 / (1.0 + np.exp(-sb)) silu_k =
F2bf(sig * sb) nd = int((cm != silu_k).sum()) print(f"FMA conv+silu row {t}: ndiff {nd}/{D} maxdiff {float(np.abs(cm -
Silu_k).max())}", flush=True) # also compare just the conv sum (before silu) with MX conv (no silu) conv_m =
Np.asarray(lay.conv1d(conv_in).astype(mx.float32)) if conv_m.ndim == 3: csum_m = conv_m[0, t] else: csum_m = conv_m[0]
If t == 0 else conv_m[1] csum_k = f2bf(acc[3]) nd2 = int((csum_m != csum_k).sum()) print(f"FMA conv-sum row {t}: ndiff
{nd2}/{D} maxdiff {float(np.abs(csum_m - csum_k).max())}", flush=True) print("DONE", flush=True)"""