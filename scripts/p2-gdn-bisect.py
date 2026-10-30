#!/usr/bin/env python3
"""Phase 2 GDN-fuse bisect: in-process chain vs fused, fixed inputs.

Run with the fused venv (has both the fused op and the chain ops).
Answers: (1) is the single-step fused op byte-identical to the 4-op chain
on the same inputs, (2) over 8 sequential decode steps which arm's state
grows/blows up, (3) per-step first divergence, (4) host-side replica of
the r4d g read (eg distribution) so the decay-gate quirk is quantified
directly from the chain's g.

Env: P2B_STEPS (default 8), P2B_DEBUG (default 1) -> P2_GDN_DEBUG=1 so the
fused op dumps /tmp/p2_gdn_dbg.bin (snapshot per step).
"""

import os

os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"

import mlx.core as mx
import numpy as np

STEPS = int(os.environ.get("P2B_STEPS", "8"))
Hk, Hv, Dk, Dv = 16, 32, 128, 128
key_dim = Hk * Dk
value_dim = Hv * Dv
conv_dim = 2 * key_dim + value_dim
KS = 4
INV_SCALE = Dk**-0.5


def fresh_inputs(rng):
    def bf16(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)

    q = bf16((1, 1, key_dim))
    k = bf16((1, 1, key_dim))
    v = bf16((1, 1, value_dim))
    b = bf16((1, 1, Hv))
    a = bf16((1, 1, Hv))
    al = mx.array(rng.standard_normal(Hv) + 1.0)  # f32 [Hv]
    db = mx.array(rng.standard_normal(Hv))  # f32 [Hv]
    cs = bf16((KS - 1, conv_dim))
    cw = bf16((conv_dim, KS))
    z = bf16((1, 1, Hv, Dv))
    w = bf16((Dv,))
    st = mx.zeros((1, Hv, Dv, Dk), dtype=mx.float32)
    return (q, k, v, b, a, al, db, cs, cw, z, w, st)


def chain_step(q, k, v, b, a, al, db, cs, cw, z, w, st):
    qn, kn, vc, beta, g = mx.fast.gdn_prep(q, k, v, b, a, al, db, cs, cw, inv_scale=INV_SCALE, eps=1e-6)
    out2, st2 = mx.fast.gated_delta_update(
        qn.reshape(1, 1, Hk, Dk),
        kn.reshape(1, 1, Hk, Dk),
        vc.reshape(1, 1, Hv, Dv),
        g.reshape(1, 1, Hv),
        beta.reshape(1, 1, Hv),
        st,
        None,
    )
    out3 = mx.fast.gdn_norm_gated(out2, z, w, eps=1e-6)[0]
    return out3, st2, (qn, kn, vc, beta, g)


def fused_step(q, k, v, b, a, al, db, cs, cw, z, w, st):
    return mx.fast.gdn_fused_step(
        q,
        k,
        v,
        b,
        a,
        al,
        db,
        cs,
        cw,
        z,
        st,
        w,
        inv_scale=INV_SCALE,
        prep_eps=1e-6,
        norm_eps=1e-6,
    )


def to_np(a, np_dtype):
    return np.frombuffer(bytes(bytearray(a)), np.uint8).view(np_dtype).reshape(-1)


def bf16_f32(a):
    return (to_np(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def report(tag, out, st):
    o = bf16_f32(out)
    s = to_np(st, np.float32)
    print(
        f"{tag} out max={np.abs(o).max():.6e} "
        f"inf={np.isinf(o).sum()} nan={np.isnan(o).sum()} "
        f"state max={np.abs(s).max():.6e} "
        f"inf={np.isinf(s).sum()} nan={np.isnan(s).sum()}",
        flush=True,
    )


def host_eg(g_arr):
    """Host-side replica of the r4d g read: bf16(low-u16-of-f32) -> eg."""
    g = to_np(g_arr, np.float32).copy()
    b = g.view(np.uint32)
    u16 = (b & 0xFFFF).astype(np.uint32) << 16
    v = u16.view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.abs(v))
    eg = np.exp(np.maximum(logv, -80.0))
    neg = int((v < 0).sum())
    zero = int((v == 0).sum())
    n = len(v)
    print(
        f"  host-eg n={n} neg={neg} zero={zero} "
        f"eg<1e-30={np.sum(eg < 1e-30)} eg>1e10={np.sum(eg > 1e10)} "
        f"eg==inf={np.sum(np.isinf(eg))} eg_max={np.nanmax(eg):.3e} "
        f"eg_p50={np.nanpercentile(eg, 50):.3e}",
        flush=True,
    )


def first_div(a, b):
    ab = np.frombuffer(bytes(bytearray(a)), np.uint8)
    bb = np.frombuffer(bytes(bytearray(b)), np.uint8)
    if ab.shape != bb.shape:
        return (0, int(min(ab.size, bb.size)))
    bad = np.nonzero(ab != bb)[0]
    if len(bad) == 0:
        return None
    return int(bad[0]), int(len(bad))


def main():
    assert hasattr(mx.fast, "gdn_fused_step"), "venv without the fused op"
    if os.environ.get("P2B_DEBUG", "1") == "1":
        os.environ["P2_GDN_DEBUG"] = "1"
    os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    rng = np.random.default_rng(7)

    # ---- single-step, state = zeros, identical inputs both arms ----
    (q, k, v, b, a, al, db, cs, cw, z, w, st) = fresh_inputs(rng)
    (q2, k2, v2, b2, a2, al2, db2, cs2, cw2, z2, w2, st2) = fresh_inputs(np.random.default_rng(7))
    out_f, st_f = fused_step(q, k, v, b, a, al, db, cs, cw, z, w, st)
    out_c, st_c, (qn, kn, vc, beta, g) = chain_step(q2, k2, v2, b2, a2, al2, db2, cs2, cw2, z2, w2, st2)
    mx.eval(out_f, st_f, out_c, st_c, qn, kn, vc, beta, g)
    print(
        f"SINGLE-STEP out_div={first_div(out_f, out_c)} state_div={first_div(st_f, st_c)}",
        flush=True,
    )
    report("SINGLE-STEP fused ", out_f, st_f)
    report("SINGLE-STEP chain ", out_c, st_c)
    host_eg(g)

    # per-phase intermediates: chain prep outputs vs the fused dbg buffer
    # (head = 1042 bytes: s_qn[0:256) s_kn[256:512) s_vconv[512:768)
    # s_beta[768:770) s_g[770:774) s_out[774:1030))
    if os.path.exists("/tmp/p2_gdn_dbg.bin"):
        with open("/tmp/p2_gdn_dbg.bin", "rb") as f:
            raw = f.read()
        head = 1042
        h0 = raw[:head]
        s_qn = np.frombuffer(h0[0:256], np.uint16)
        s_kn = np.frombuffer(h0[256:512], np.uint16)
        s_vc = np.frombuffer(h0[512:768], np.uint16)
        s_beta = int(np.frombuffer(h0[768:770], np.uint16)[0])
        s_g = float(np.frombuffer(h0[770:774], np.float32)[0])
        qn_f = to_np(qn, np.uint16)
        kn_f = to_np(kn, np.uint16)
        vc_f = to_np(vc, np.uint16)
        beta_c = int(to_np(beta, np.uint16)[0])
        g_c = float(to_np(g, np.float32)[0])
        print(
            f"  dbg-head0 qn={np.array_equal(s_qn, qn_f[:128])} "
            f"kn={np.array_equal(s_kn, kn_f[:128])} "
            f"vconv={np.array_equal(s_vc, vc_f[:128])} "
            f"beta_f={s_beta:#06x} beta_c={beta_c:#06x} "
            f"g_f={s_g:.8e} g_c={g_c:.8e}",
            flush=True,
        )

    # ---- 8 sequential steps (same token re-fed), zero state per arm ----
    for arm, step_fn in (("chain", chain_step), ("fused", fused_step)):
        print(f"== {arm} multi-step ==", flush=True)
        (q, k, v, b, a, al, db, cs, cw, z, w, st) = fresh_inputs(np.random.default_rng(7))
        for t in range(STEPS):
            if arm == "chain":
                out, st, _ = step_fn(q, k, v, b, a, al, db, cs, cw, z, w, st)
            else:
                out, st = step_fn(q, k, v, b, a, al, db, cs, cw, z, w, st)
            mx.eval(out, st)
            report(f"  step{t}", out, st)
            if arm == "fused" and os.path.exists("/tmp/p2_gdn_dbg.bin"):
                import shutil

                shutil.copy("/tmp/p2_gdn_dbg.bin", f"/tmp/p2_gdn_dbg_step{t}.bin")
    print("BISECT-DONE", flush=True)


if __name__ == "__main__":
    main()
