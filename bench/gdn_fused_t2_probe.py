#!/usr/bin/env python3
"""Op-level bit-exactness probe for the T<=4 fused GDN step (gate 0).

Reference = the MODEL DEFAULT PATH (not the op's fallback chain, which is
1-D-only in gdn_prep and thus invalid for T=2/4):
  conv:  mx.nn.Conv1d sliding window over [conv_state(3), qkv(T)]
  beta:  mx.sigmoid(b)
  g:     model compute_g  (exp(-exp(A_log) * softplus(a + dt_bias)), f32)
  qk:    normalize_qk (inv_scale**2 * rms_norm, eps=1e-6*inv_scale**2)
  gdu:   mx.fast.gated_delta_update (r4d recurrent for T<=8)
  norm:  mx.fast.rms_norm + precise_swiglu (RMSNormGated call)

Checks:
  1. fused T=1 == model-default chain T=1 (regression)
  2. fused T=2 == model-default chain T=2 (regression)
  3. invariance: model chain T=2 == two sequential model-chain T=1 steps
     (conv-state updated between), fused T=2 == two sequential fused T=1
  4. fused T=4 == model-default chain T=4 (new path, S=4 MTP verify)
  5. invariance: model chain T=4 == four sequential model-chain T=1 steps,
     fused T=4 == four sequential fused T=1 steps (conv-state updated
     between rows)
"""

import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import precise_swiglu
from mlx_lm.models.gated_delta import gated_delta_update as model_gdu
from mlx_lm.models.gated_delta import normalize_qk


def mk(T, seed):
    g = np.random.default_rng(seed)

    def bf(*shape):
        return mx.array(g.normal(0.0, 1.0, shape).astype(np.float32)).astype(mx.bfloat16)

    def f32(*shape, lo=0.0, hi=1.0):
        return mx.array(g.uniform(lo, hi, shape).astype(np.float32))

    conv_dim, key_dim, value_dim = 6144, 1024, 4096
    q = bf(1, T, key_dim)
    k = bf(1, T, key_dim)
    v = bf(1, T, value_dim)
    b = bf(1, T, 32)
    a = bf(1, T, 32)
    A_log = f32(32, lo=0.5, hi=2.0)
    dt_bias = mx.array(g.normal(0.0, 0.5, (32,)).astype(np.float32))
    cs = bf(3, conv_dim)
    cw = mx.array(g.normal(0.0, 0.1, (conv_dim, 4)).astype(np.float32)).astype(mx.bfloat16)
    z = bf(1, T, 32, 128)
    st = mx.array(g.normal(0.0, 0.3, (1, 32, 128, 128)).astype(np.float32))
    nw = mx.ones((128,), mx.bfloat16)
    return dict(
        q=q,
        k=k,
        v=v,
        b=b,
        a=a,
        A_log=A_log,
        dt_bias=dt_bias,
        conv_state=cs,
        conv_weight=cw,
        z_raw=z,
        state_raw=st,
        norm_weight=nw,
        conv_dim=conv_dim,
        key_dim=key_dim,
        value_dim=value_dim,
    )


def fused(inp, Tlen):
    out, st = mx.fast.gdn_fused_step(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["b"],
        inp["a"],
        inp["A_log"],
        inp["dt_bias"],
        inp["conv_state"],
        inp["conv_weight"],
        inp["z_raw"],
        inp["state_raw"],
        inp["norm_weight"],
        inv_scale=128**-0.5,
        prep_eps=1e-6,
        norm_eps=1e-6,
    )
    mx.eval(out, st)
    return out, st


def chain_default(inp, Tlen):
    """Model default path for Tlen rows, using inp's qkv rows as-is."""
    conv = nn.Conv1d(inp["conv_dim"], inp["conv_dim"], 4, groups=inp["conv_dim"], bias=False)
    conv.weight = inp["conv_weight"].reshape(inp["conv_dim"], 4, 1)
    qkv = mx.concatenate([inp["q"], inp["k"], inp["v"]], axis=-1)  # (1, T, 6144)
    conv_in = mx.concatenate([inp["conv_state"][None], qkv], axis=1)
    conv_out = nn.silu(conv(conv_in))  # (1, T, 6144)
    kd = inp["key_dim"]
    q3 = conv_out[..., :kd].reshape(1, Tlen, 8, 128)
    k3 = conv_out[..., kd : 2 * kd].reshape(1, Tlen, 8, 128)
    v3 = conv_out[..., 2 * kd :].reshape(1, Tlen, 32, 128)
    q3, k3 = normalize_qk(q3, k3, inv_scale=128**-0.5, eps=1e-6)
    out2, st2 = model_gdu(
        q3, k3, v3, inp["a"], inp["b"], inp["A_log"], inp["dt_bias"], inp["state_raw"], None, use_kernel=True
    )
    normed = mx.fast.rms_norm(out2, inp["norm_weight"], 1e-6)
    out3 = precise_swiglu(out2, inp["z_raw"], normed)
    mx.eval(out3, st2)
    return out3, st2


def check(name, x, y):
    x = np.asarray(mx.eval(x))
    y = np.asarray(mx.eval(y))
    ok = bool((x == y).all())
    print(("PASS" if ok else "FAIL"), name, "bit-exact", ok)
    if not ok:
        d = x != y
        print("   mismatches:", int(d.sum()), "/", x.size)
        if x.dtype != np.object_:
            print("   maxdiff:", float(np.abs(x - y).max()))
    return ok


def single_row(i, t):
    d = dict(i)
    for key in ("q", "k", "v", "b", "a", "z_raw"):
        d[key] = i[key][:, t : t + 1, :]
    d["conv_state"] = i["conv_state"]
    return d


def advance_cs(d):
    """Roll the conv state one row forward (in place on d): after a row-t
    forward the 3-row window becomes [cs1, cs2, row_t]."""
    d["conv_state"] = mx.stack(
        [
            d["conv_state"][1],
            d["conv_state"][2],
            mx.concatenate([d["q"][0, 0], d["k"][0, 0], d["v"][0, 0]], axis=-1),
        ],
        axis=0,
    )
    return d


def main():
    ok = True
    i1 = mk(1, seed=42)
    f1, fs1 = fused(i1, 1)
    c1, cs1 = chain_default(i1, 1)
    ok &= check("T=1 out  (fused vs model default)", f1, c1)
    ok &= check("T=1 state(fused vs model default)", fs1, cs1)

    i2 = mk(2, seed=42)
    f2, fs2 = fused(i2, 2)
    c2, cs2 = chain_default(i2, 2)
    ok &= check("T=2 out  (fused vs model default)", f2, c2)
    ok &= check("T=2 state(fused vs model default)", fs2, cs2)

    # Invariance, model-default path: T=2 == two T=1 with conv-state update
    a_ = mk(2, seed=42)
    r0, s0 = chain_default(single_row(a_, 0), 1)
    advance_cs(a_)
    a1 = single_row(a_, 1)
    r1, s1 = chain_default(a1, 1)
    ok &= check("invar chain out  (T=2 vs two T=1)", c2[0, 1], r1[0, 0])
    ok &= check("invar chain state(T=2 vs two T=1)", cs2, s1)

    # Invariance, fused path: fused T=2 == two fused T=1
    fr0, fso = fused(single_row(i2, 0), 1)
    advance_cs(i2)
    i2b = single_row(i2, 1)
    fr1, fs1b = fused(i2b, 1)
    ok &= check("invar fused out  (T=2 vs two T=1)", f2[0, 1], fr1[0, 0])
    ok &= check("invar fused state(T=2 vs two T=1)", fs2, fs1b)

    # T=4 (S=4 MTP verify): fused vs model default
    i4 = mk(4, seed=42)
    f4, fs4 = fused(i4, 4)
    c4, cs4 = chain_default(i4, 4)
    ok &= check("T=4 out  (fused vs model default)", f4, c4)
    ok &= check("T=4 state(fused vs model default)", fs4, cs4)

    # Invariance, model-default path: T=4 == four T=1 with conv-state update
    a4 = mk(4, seed=42)
    for t in range(4):
        rt, st = chain_default(single_row(a4, t), 1)
        if t < 3:
            advance_cs(a4)
    ok &= check("invar chain out  (T=4 vs four T=1)", c4[0, 3], rt[0, 0])
    ok &= check("invar chain state(T=4 vs four T=1)", cs4, st)

    # Invariance, fused path: fused T=4 == four fused T=1
    for t in range(4):
        ft, ftst = fused(single_row(i4, t), 1)
        if t < 3:
            advance_cs(i4)
    ok &= check("invar fused out  (T=4 vs four T=1)", f4[0, 3], ft[0, 0])
    ok &= check("invar fused state(T=4 vs four T=1)", fs4, ftst)

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
