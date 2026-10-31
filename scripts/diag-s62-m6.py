#!/usr/bin/env python3
"""m6: focused rope_q A/B on the captured tensors (post RowInvariantRoPE patch).

Goal: verify the m6 patch (explicit row-invariant rotation replacing the
fused mx.fast.rope in qwen3_next.py) at the m5 S62 layer-3 inputs:

  1. vs-capture (bit-exact): the patched rope's output vs the captured
     fused-kernel output. T=1 (the 1D single-token kernel, the AR path):
     0/4096 for rope_q and 0/1024 for rope_k. T=2 (the general kernel):
     row 0 0/4096; row 1 is the documented 1-ULP outlier (the m5
     artifact - the old T=2 kernel's row 1 is 1 ULP off from its own
     T=1 sibling, so no single explicit formula can match both kernels
     on that element; the patched rope matches T=1 and is
     row-invariant, which is what the G4 gate requires).
  2. row-invariance (bit-exact): T=2 row 1 (pos 88) vs T=1 row 0
     (pos 88) - 0/4096 for rope_q and 0/1024 for rope_k.
  3. full chain (m5 Part 2 re-run, but with the patched rope): every
     component must be row-invariant - the first diverging component
     must be NONE (m5: rope_q was the first; downstream DIFFs were
     propagation only).

Setup: cap-m5-s62.pkl (the m5 capture: capA/capB = the layer-3 block
hooked for the T=2 verify [279, 3545] and the T=1 [3545] with the
offset advanced; sdpaA/sdpaB = the SDPA call records; weights). The
rope re-run feeds the CAPTURED rope inputs (bit-exact, no recompute);
the full chain re-runs the m5 chain with the patched rope.

Exit codes: 0 = all pass-criteria checks MATCH (M6-AB-OK); 1 = a
check DIFFed; 2 = capture mismatch; 3 = unexpected failure.
"""

import os
import pickle
import sys

import mlx.core as mx
import mlx.nn as mnn
import numpy as np

MODELS = os.environ.get("MODELS", "/models")

D = os.environ.get("CAP_DIR", f"{MODELS}/tmp/run-785")
M5_PKL = os.path.join(D, "cap-m5-s62.pkl")
M3_PKL = os.path.join(D, "cap-m3-s62.pkl")


def npf(a):
    """mx array -> f32 numpy (avoids the bf16 PEP 3118 buffer issue)."""
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def of(npv, dt):
    """f32 numpy -> mx array of dtype dt (bit-exact round-trip)."""
    if npv is None:
        return None
    return mx.array(np.asarray(npv, dtype=np.float32)).astype(dt)


def of_w(v, dt):
    """(numpy, tag) -> mx array (exact; the m5 snap_w inverse)."""
    if v is None:
        return None
    npv, tag = v
    if tag == "bf16":
        f = (npv.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f).astype(mx.bfloat16)
    if tag == "u32":
        return mx.array(npv)
    return mx.array(np.asarray(npv, dtype=np.float32))


def cmp_status(a, b):
    """Bit-exact comparison of two f32 arrays -> 'MATCH' / 'DIFF'."""
    if a is None or b is None:
        return "SKIP"
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return "SHAPE"
    return "MATCH" if np.array_equal(a, b) else "DIFF"


def cmp_report(name, a, b):
    st = cmp_status(a, b)
    if st in ("MATCH", "SKIP", "SHAPE"):
        extra = f"n={np.asarray(a).size}" if st == "MATCH" else ""
        print(f"  {name:<38} {st:<6} {extra}", flush=True)
        return st
    d = np.asarray(a, dtype=np.float32).ravel() - np.asarray(b, dtype=np.float32).ravel()
    n = int((d != 0).sum())
    idx = int(np.argmax(d != 0))
    extra = f" (h={idx // 512}, t={(idx % 512) // 256}, d={idx % 256})" if d.size == 8192 else ""
    print(
        f"  {name:<38} {st:<6} n={n}/{d.size} maxabs={np.abs(d).max():.3e} first_idx={idx}{extra}",
        flush=True,
    )
    return st


def row1(a):
    """A row 1 (token 1).

    (1,2,...) -> [0,1,...]; (1,H,2,...) -> [0,:,1,...] (rope layout,
    time axis 2); (1,2,H,...) -> [0,1,:,:] (q_norm layout, time axis 1).
    """
    if a.ndim == 3:
        return a[0, 1, :]
    if a.shape[1] in (1, 2):
        return a[0, 1, :, :]
    return a[0, :, 1, :]


def row0(a):
    """B single row (token 0); same layouts as row1."""
    if a.ndim == 3:
        return a[0, 0, :]
    if a.shape[1] in (1, 2):
        return a[0, 0, :, :]
    return a[0, :, 0, :]


def mk_linear(w):
    if w.get("quantized"):
        lin = mnn.QuantizedLinear(w["in"], w["out"], bias=False, bits=w["bits"], group_size=w["group"], mode=w["mode"])
        lin.weight = of_w(w["weight"], w["weight"][1])
        lin.scales = of_w(w["scales"], w["scales"][1])
        if w["biases"][0] is not None:
            lin.biases = of_w(w["biases"], w["biases"][1])
        return lin
    npv, dt = w["weight"]
    lin = mnn.Linear(npv.shape[1], npv.shape[0], bias=w["bias"][0] is not None)
    lin.weight = of_w(w["weight"], dt)
    if w["bias"][0] is not None:
        lin.bias = of_w(w["bias"], w["bias"][1])
    return lin


def mk_rms(w):
    npv, dt = w["weight"]
    n = mnn.RMSNorm(npv.shape[0], eps=w["eps"])
    n.weight = of_w(w["weight"], dt)
    return n


def main():
    import mlx_lm.models.qwen3_next as qn

    with open(qn.__file__) as f:
        src = f.read()
    if "class RowInvariantRoPE" not in src or "self.rope = RowInvariantRoPE(" not in src:
        print(f"NOT-PATCHED: {qn.__file__} (run scripts/apply-m6-patch.py first)", flush=True)
        sys.exit(3)
    print(f"patched module: {qn.__file__}", flush=True)

    with open(M5_PKL, "rb") as f:
        c = pickle.load(f)
    capA, capB = c["capA"], c["capB"]
    sdpaA, sdpaB = c["sdpaA"], c["sdpaB"]
    w = c["weights"]
    rd = w["attn.rope"]
    assert capA["rope"][0]["kw"]["offset"] == 87, capA["rope"][0]["kw"]
    assert capB["rope"][0]["kw"]["offset"] == 88, capB["rope"][0]["kw"]
    assert sdpaA["mask"] == "causal" and sdpaB["mask"] is None
    print(
        f"rope params: dims={rd['dims']} base={rd['base']} scale={rd['scale']} traditional={rd['traditional']}",
        flush=True,
    )

    # The exact rope the model uses (same construction as
    # Qwen3NextAttention.__init__ for the default path).
    rope = qn.RowInvariantRoPE(rd["dims"], traditional=rd["traditional"], base=rd["base"], scale=rd["scale"])

    ok = True

    # ------------------------------------------------ focused rope A/B
    print("== focused rope re-run (patched rope, captured rope inputs) ==", flush=True)
    # rope_q: capA["rope"][0] = queries, capA["rope"][1] = keys
    rqA = rope(of(capA["rope"][0]["in"][0], mx.bfloat16), offset=capA["rope"][0]["kw"]["offset"])
    rqB = rope(of(capB["rope"][0]["in"][0], mx.bfloat16), offset=capB["rope"][0]["kw"]["offset"])
    mx.eval([rqA, rqB])
    # A vs capture is informational: the old fused T=2 kernel's row 1 is the
    # 1-ULP outlier (m5 artifact); the explicit rope matches T=1 bit-exact and
    # is row-invariant, so it cannot match both kernels on that element.
    cmp_report(
        "rope_q A row0 re-run vs capture (fused T=2 row0)",
        npf(rqA[0, :, 0, :]),
        npf(capA["rope"][0]["out"])[0, :, 0, :],
    )
    cmp_report(
        "rope_q A row1 re-run vs capture (fused T=2 row1; expect 1/4096, the m5 artifact)",
        npf(rqA[0, :, 1, :]),
        npf(capA["rope"][0]["out"])[0, :, 1, :],
    )
    ok &= cmp_report("rope_q B re-run vs capture (fused T=1)", npf(rqB), capB["rope"][0]["out"]) == "MATCH"
    ok &= cmp_report("rope_q row-invariance (A row1 vs B)", npf(row1(rqA)), npf(row0(rqB))) == "MATCH"

    rkA = rope(of(capA["rope"][1]["in"][0], mx.bfloat16), offset=capA["rope"][1]["kw"]["offset"])
    rkB = rope(of(capB["rope"][1]["in"][0], mx.bfloat16), offset=capB["rope"][1]["kw"]["offset"])
    mx.eval([rkA, rkB])
    ok &= cmp_report("rope_k A re-run vs capture (fused T=2)", npf(rkA), capA["rope"][1]["out"]) == "MATCH"
    ok &= cmp_report("rope_k B re-run vs capture (fused T=1)", npf(rkB), capB["rope"][1]["out"]) == "MATCH"
    ok &= cmp_report("rope_k row-invariance (A row1 vs B)", npf(row1(rkA)), npf(row0(rkB))) == "MATCH"

    # ------------------------------------------------ full chain re-run
    print("== full chain re-run (m5 Part 2, patched rope) ==", flush=True)
    N_H, N_KV = 16, 4

    xA = of(capA["block"][0]["in"][0], mx.bfloat16)
    xB = of(capB["block"][0]["in"][0], mx.bfloat16)
    ln = mk_rms(w["input_ln"])
    pln = mk_rms(w["post_ln"])
    qproj, kproj, vproj, oproj = (
        mk_linear(w["attn.q_proj"]),
        mk_linear(w["attn.k_proj"]),
        mk_linear(w["attn.v_proj"]),
        mk_linear(w["attn.o_proj"]),
    )
    qnorm, knorm = mk_rms(w["attn.q_norm"]), mk_rms(w["attn.k_norm"])
    gp_l, up_l, down_l = (
        mk_linear(w["mlp.gate_proj"]),
        mk_linear(w["mlp.up_proj"]),
        mk_linear(w["mlp.down_proj"]),
    )

    def chain(x, off, ksdpa, vsdpa, scale, mask):
        ln_x = ln(x)
        qkv = qproj(ln_x)
        B, L, _ = qkv.shape
        queries, gate = mx.split(qkv.reshape(B, L, N_H, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = kproj(ln_x), vproj(ln_x)
        qn = qnorm(queries)
        kn = knorm(keys.reshape(B, L, N_KV, -1))
        qt, kt = qn.transpose(0, 2, 1, 3), kn.transpose(0, 2, 1, 3)
        vt = values.reshape(B, L, N_KV, -1).transpose(0, 2, 1, 3)
        # m6: the patched rope (RowInvariantRoPE) instead of mx.fast.rope
        rq = rope(qt, offset=off)
        rk = rope(kt, offset=off)
        sdpa_out = mx.fast.scaled_dot_product_attention(rq, ksdpa, vsdpa, scale=scale, mask=mask)
        oin = sdpa_out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)
        r = oproj(oin)
        h = x + r
        p = pln(h)
        gp, up = gp_l(p), up_l(p)
        sw = mnn.silu(gp) * up
        dp = down_l(sw)
        out = h + dp
        return {
            "ln": ln_x,
            "qkv": qkv,
            "keys": keys,
            "values": values,
            "qn": qn,
            "kn": kn,
            "rq": rq,
            "rk": rk,
            "vt": vt,
            "sdpa": sdpa_out,
            "gate": mx.sigmoid(gate),
            "oin": oin,
            "r": r,
            "h": h,
            "pln": p,
            "gp": gp,
            "up": up,
            "sw": sw,
            "dp": dp,
            "out": out,
        }

    A2 = chain(
        xA,
        capA["rope"][0]["kw"]["offset"],
        of(sdpaA["k"], mx.bfloat16),
        of(sdpaA["v"], mx.bfloat16),
        sdpaA["scale"],
        "causal",
    )
    B2 = chain(
        xB,
        capB["rope"][0]["kw"]["offset"],
        of(sdpaB["k"], mx.bfloat16),
        of(sdpaB["v"], mx.bfloat16),
        sdpaB["scale"],
        None,
    )
    mx.eval([A2[k] for k in A2] + [B2[k] for k in B2])

    print("-- full chain re-run vs in-model capture (sanity: same kernels as the capture) --", flush=True)
    for nm, a2, b2, ca, cb in [
        ("ln out", A2["ln"], B2["ln"], capA["input_ln"][0]["out"], capB["input_ln"][0]["out"]),
        ("q_proj out", A2["qkv"], B2["qkv"], capA["q_proj"][0]["out"], capB["q_proj"][0]["out"]),
        ("q_norm out", A2["qn"], B2["qn"], capA["q_norm"][0]["out"], capB["q_norm"][0]["out"]),
        ("rope_q out", A2["rq"], B2["rq"], capA["rope"][0]["out"], capB["rope"][0]["out"]),
        ("sdpa out", A2["sdpa"], B2["sdpa"], sdpaA["out"], sdpaB["out"]),
        ("o_proj in", A2["oin"], B2["oin"], capA["o_proj"][0]["in"][0], capB["o_proj"][0]["in"][0]),
        ("o_proj out", A2["r"], B2["r"], capA["o_proj"][0]["out"], capB["o_proj"][0]["out"]),
        ("post_ln out", A2["pln"], B2["pln"], capA["post_ln"][0]["out"], capB["post_ln"][0]["out"]),
        ("gate_proj out", A2["gp"], B2["gp"], capA["gate_proj"][0]["out"], capB["gate_proj"][0]["out"]),
        ("down_proj out", A2["dp"], B2["dp"], capA["down_proj"][0]["out"], capB["down_proj"][0]["out"]),
        ("block out", A2["out"], B2["out"], capA["block"][0]["out"], capB["block"][0]["out"]),
    ]:
        cmp_report(f"{nm} A re-run vs capture", npf(a2), ca)
        cmp_report(f"{nm} B re-run vs capture", npf(b2), cb)

    print("== full chain row-invariance (A row1 vs B; the m5 gate) ==", flush=True)
    re = [
        ("2  input_layernorm", A2["ln"], B2["ln"]),
        ("3a q_proj", A2["qkv"], B2["qkv"]),
        ("3b k_proj", A2["keys"], B2["keys"]),
        ("3c v_proj", A2["values"], B2["values"]),
        ("4a q_norm", A2["qn"], B2["qn"]),
        ("4b k_norm", A2["kn"], B2["kn"]),
        ("5a rope_q", A2["rq"], B2["rq"]),
        ("5b rope_k", A2["rk"], B2["rk"]),
        ("6  sdpa", A2["sdpa"], B2["sdpa"]),
        ("7  gate sigmoid", A2["gate"], B2["gate"]),
        ("8  o_proj in", A2["oin"], B2["oin"]),
        ("9  o_proj", A2["r"], B2["r"]),
        ("10 h = x + r", A2["h"], B2["h"]),
        ("11 post_attention_layernorm", A2["pln"], B2["pln"]),
        ("12a gate_proj", A2["gp"], B2["gp"]),
        ("12b up_proj", A2["up"], B2["up"]),
        ("13 swiglu", A2["sw"], B2["sw"]),
        ("14 down_proj", A2["dp"], B2["dp"]),
        ("15 block_out", A2["out"], B2["out"]),
    ]
    first_diff = None
    for nm, a, b in re:
        st = cmp_status(npf(row1(a)), npf(row0(b)))
        if st == "MATCH":
            print(f"  {nm:<40} MATCH    n={npf(row1(a)).size}", flush=True)
        elif st == "DIFF":
            d = npf(row1(a)).ravel() - npf(row0(b)).ravel()
            print(
                f"  {nm:<40} DIFF     n={int((d != 0).sum())}/{d.size} maxabs={np.abs(d).max():.3e}",
                flush=True,
            )
            if first_diff is None:
                first_diff = nm
        else:
            print(
                f"  {nm:<40} {st}  shapes={npf(row1(a)).shape} vs {npf(row0(b)).shape}",
                flush=True,
            )
    print(f"first diverging component: {first_diff if first_diff else 'NONE'}", flush=True)
    ok &= first_diff is None

    # m3 cross-check: validate the pkl is the real m3 capture (layer-3
    # attn k clean 0/91136; layer-4 GDN ssm = m3's first divergence
    # 515560/524288; rope_q is not cached, which is why m3 missed it)
    with open(M3_PKL, "rb") as f:
        m3 = pickle.load(f)
    m3_k_done = False
    for i, (kind, off, k, v) in enumerate(m3["ha"]):
        if kind == "attn" and off == 89 and not m3_k_done:
            for kind2, off2, k2, _v2 in m3["hb"]:
                if kind2 == "attn" and off2 == 89:
                    d = np.asarray(k, dtype=np.float32).ravel() - np.asarray(k2, dtype=np.float32).ravel()
                    print(
                        f"m3 cross-check: layer-3 k n_diff={int((d != 0).sum())}/{d.size} "
                        "(expect 0/91136, clean in m3)",
                        flush=True,
                    )
                    m3_k_done = True
                    break
        if i == 4 and kind == "gdn":
            _, _, _, v2 = m3["hb"][4]
            ds = np.asarray(v, dtype=np.float32).ravel() - np.asarray(v2, dtype=np.float32).ravel()
            print(
                f"m3 cross-check: layer-4 gdn ssm n_diff={int((ds != 0).sum())}/{ds.size} "
                "(expect 515560/524288, m3's first divergence)",
                flush=True,
            )

    if not ok:
        sys.exit(1)
    print("M6-AB-OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(3)
