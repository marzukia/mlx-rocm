#!/usr/bin/env python3
"""t1014 gate (a): direct k_proj T-invariance probe (audit repro on the lane wheel).

The #20 m8 audit (P3) localized the MTP flow 1-ulp to the k_proj matmul
T=2-vs-T=1 dispatch (direct test: random bf16 1024x4096, T2-row1 vs T1
n=1/4096, 1-ulp 1.221e-04). m8 round 2 (t976) corrected the flow
attribution (the flow 1-ulp was in the fused mx.fast.rope; the model's
k_proj is 4-bit QMM - qmv_wide_kernel for M=1 and M=2, T-invariant per
row by construction) and the m8rebase lane wheel re-verified D1 n=0 (t992:
direct-kernels.out). This probe re-runs the audit's direct test on the
current lane wheel (t1012-29k/venv-m8, libmlx.so sha256[:12]=de6de5f154c0).

Parts:
  D1: the model's ACTUAL k_proj (QuantizedLinear 4bit/64, all 8 FA layers):
      T=2 [a, b] vs T=1 [b] with identical row values (bf16), 20 random
      trials. Gate (a): T2-row1 vs T1-row0 must be bit-exact (n=0).
  D1b: FA15 q_proj / v_proj control (same QMM path).
  D2: the P3 audit's plain-bf16-matmul direct test: x (T, 4096) @ W.T with
      W (1024, 4096) (the model k_proj shape) and the transposed variant
      x (T, 1024) @ W.T with W (4096, 1024) (the audit's "1024x4096"
      orientation). Per-trial T2-row1 vs T1-row0. Informational: the plain
      M=1 gemv vs M=2 GEMM split may flip 1 ulp value-specifically on
      some .so builds (t976 open item); the flow's k_proj is QMM, so D2
      does not gate the lossless path.
  D3: determinism: one representative comparison per part is re-run; the
      re-run must be bit-identical to the first (n=0).

Gate (a) PASS iff D1 worst n=0 and D3 n=0 (D1b/D2 reported, not gating).
"""

import os

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
TRIALS = 20
SEED = 20261030


def npf(a):
    return np.asarray(a.astype(mx.float32)).copy()


def main():
    mx.random.seed(SEED)
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    layers = lm.model.layers
    fa_ids = [i for i, ly in enumerate(layers) if hasattr(ly, "self_attn")]
    print(f"FA layers: {fa_ids}", flush=True)

    # ---- D1: model k_proj (QMM) T-invariance ----
    worst1 = 0
    first_diff = None
    for trial in range(TRIALS):
        x2 = mx.random.normal((2, 4096)).astype(mx.bfloat16)
        for i in fa_ids:
            k = layers[i].self_attn.k_proj
            o2 = k(x2)
            o1 = k(x2[1:2])
            mx.eval(o2, o1)
            d = npf(o2[1]) - npf(o1[0])
            n = int(np.count_nonzero(d != 0))
            if n:
                worst1 = max(worst1, n)
                if first_diff is None and n <= 8:
                    idx = np.flatnonzero(d != 0)[:8]
                    first_diff = (
                        f"FA{i} trial{trial}: n={n} at {idx.tolist()} "
                        f"a={npf(o2[1])[idx[0]]:.6f} b={npf(o1[0])[idx[0]]:.6f}"
                    )
    print(
        f"D1 k_proj T-invariance (model QMM, {len(fa_ids)} FA layers x {TRIALS} trials): worst n={worst1} (expect 0)",
        flush=True,
    )
    if first_diff:
        print(f"D1 first diff: {first_diff}", flush=True)

    # ---- D1b: FA15 q/v proj control ----
    at15 = layers[15].self_attn
    worst_q = 0
    worst_v = 0
    for _ in range(TRIALS):
        x2 = mx.random.normal((2, 4096)).astype(mx.bfloat16)
        oq2 = at15.q_proj(x2)
        oq1 = at15.q_proj(x2[1:2])
        ov2 = at15.v_proj(x2)
        ov1 = at15.v_proj(x2[1:2])
        mx.eval(oq2, oq1, ov2, ov1)
        dq = npf(oq2[1]) - npf(oq1[0])
        dv = npf(ov2[1]) - npf(ov1[0])
        worst_q = max(worst_q, int(np.count_nonzero(dq != 0)))
        worst_v = max(worst_v, int(np.count_nonzero(dv != 0)))
    print(f"D1b FA15 q_proj T-invariance worst n={worst_q} (expect 0); v_proj worst n={worst_v} (expect 0)", flush=True)

    # ---- D2: plain bf16 matmul (the P3 audit's direct test) ----
    for tag, rows, cols in [
        ("kproj-shape x(2,4096)@W(1024,4096).T", 4096, 1024),
        ("t-shape x(2,1024)@W(4096,1024).T", 1024, 4096),
    ]:
        worst2 = 0
        ulps = 0
        first2 = None
        for trial in range(TRIALS):
            x2 = mx.random.normal((2, rows)).astype(mx.bfloat16)
            w = mx.random.normal((cols, rows)).astype(mx.bfloat16)
            o2 = x2 @ w.T
            o1 = x2[1:2] @ w.T
            mx.eval(o2, o1)
            d = npf(o2[1]) - npf(o1[0])
            n = int(np.count_nonzero(d != 0))
            if n:
                worst2 = max(worst2, n)
                ulps += n
                if first2 is None and n <= 8:
                    idx = np.flatnonzero(d != 0)[:8]
                    first2 = f"trial{trial}: n={n} at {idx.tolist()} " + ", ".join(
                        f"a={npf(o2[1])[j]:.6f}/b={npf(o1[0])[j]:.6f}" for j in idx[:3]
                    )
        print(
            f"D2 {tag}: total flips {ulps}/{TRIALS} trials, worst n={worst2} "
            f"(informational; value-specific 1-ulp possible)",
            flush=True,
        )
        if first2:
            print(f"D2 first diff ({tag}): {first2}", flush=True)

    # ---- D3: determinism (same inputs re-fed; must be bit-identical) ----
    x2 = mx.random.normal((2, 4096)).astype(mx.bfloat16)
    w = mx.random.normal((1024, 4096)).astype(mx.bfloat16)
    k = layers[15].self_attn.k_proj
    a2 = k(x2)
    a1 = k(x2[1:2])
    r2 = x2 @ w.T
    r1 = x2[1:2] @ w.T
    mx.eval(a2, a1, r2, r1)
    ref = (a2, a1, r2, r1)
    a2b = k(x2)
    a1b = k(x2[1:2])
    r2b = x2 @ w.T
    r1b = x2[1:2] @ w.T
    mx.eval(a2b, a1b, r2b, r1b)
    d3 = max(
        int(np.count_nonzero(npf(a2b) - npf(ref[0]))),
        int(np.count_nonzero(npf(a1b) - npf(ref[1]))),
        int(np.count_nonzero(npf(r2b) - npf(ref[2]))),
        int(np.count_nonzero(npf(r1b) - npf(ref[3]))),
    )
    print(f"D3 determinism re-run n={d3} (expect 0)", flush=True)
    print(
        f"KPROJ-TINVARIANCE: D1={'PASS' if worst1 == 0 else 'FAIL'} D3={'PASS' if d3 == 0 else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
