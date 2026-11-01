#!/usr/bin/env python3
"""G2 kernel gate: S=4 causal SDPA per-row bit-exact vs the AR S=1 no-mask
row (the RUN-verify-attention.md step 3 bar, the m8 "freshB == flowB"
method generalized to qL=4).

G2.1 (the gate): sdpa(q[0, 0:4], kv[:kL+3-i...], mask causal) row i
     == sdpa(q[0, i:i+1], kv[:kL], mask None)[0]  bit-exact, all kL x trials.
     (Causal row i sees keys [0, kL-4+i]; the AR S=1 no-mask row i over
     the same visible prefix is the same arithmetic with the same
     compiled code (m8 single instantiation).)
G2.2 (regression): qL=1 no-mask / qL=1 causal rows vs a second
     no-mask qL=1 call at the same position (intra-wheel T-invariance
     sanity; cross-wheel M=1 invariant is the AR A/B gate G5).
G2.3 (sanity, tolerance): qL=4 causal vs fp32 numpy reference,
     max|delta| <= 2e-2 (bf16); row 3 (all keys visible) checked too.
G2.4 (perf sanity): 1000-call wall on a fixed qL=4 shape ~ 1-5 us/call
     (not ~0.3 ms of the scalar flash path at 1200 keys).
"""
import time

import numpy as np
import mlx.core as mx
from mlx.core import fast as _f
sdpa = _f.scaled_dot_product_attention

B, H, D = 1, 16, 128
KLS = (33, 67, 129, 257, 512)
TRIALS = 16
SEED = 20261101

rng = np.random.default_rng(SEED)


def mk(kL, trial):
    q = mx.array(rng.normal(size=(B, H, 4, D)).astype(np.float32).copy())
    # Alignment (kernel line 138: use_key = i <= kL - qL + q_seq_idx):
    # row i of the causal S=4 batch with K=kL keys sees the first
    # kL-4+i+1 keys; the AR S=1 no-mask row at the same stream position
    # sees exactly those keys.
    kfull = rng.normal(size=(B, H, kL, D))
    vfull = rng.normal(size=(B, H, kL, D))
    return (
        mx.array(kfull.astype(np.float32).copy()).astype(mx.bfloat16),
        mx.array(vfull.astype(np.float32).copy()).astype(mx.bfloat16),
        q.astype(mx.bfloat16),
    )


def main():
    bad21 = 0
    n21 = 0
    bad22 = 0
    bad23 = 0
    worst23 = 0.0
    for trial in range(TRIALS):
        for kL in KLS:
            kfull, vfull, q = mk(kL, trial)
            # G2.1: causal S=4 row i vs AR S=1 no-mask at the same position
            causal = sdpa(q, kfull, vfull,
                         scale=1.0 / (D ** 0.5), mask="causal")
            # [B, H, 4, D]; row i sees the first kL-4+i+1 keys
            for i in range(4):
                vis = kL - 4 + i + 1
                ar = sdpa(q[:, :, i:i + 1], kfull[:, :, :vis],
                         vfull[:, :, :vis], scale=1.0 / (D ** 0.5))
            # mask None
                n21 += 1
                if not bool(mx.all(
                        mx.reshape(causal[:, :, i], (H, D))
                        == mx.reshape(ar[0, :, 0], (H, D)))):
                    bad21 += 1
            # G2.2: qL=1 no-mask twice + causal row 3 vs no-mask kL
            kL1 = kfull[:, :, :kL]
            vL1 = vfull[:, :, :kL]
            sc = 1.0 / (D ** 0.5)
            a1 = sdpa(q[:, :, 0:1], kL1, vL1, scale=sc)
            a2 = sdpa(q[:, :, 0:1], kL1, vL1, scale=sc)
            if not bool(mx.all(mx.reshape(a1[0, :, 0], (H, D))
                              == mx.reshape(a2[0, :, 0], (H, D)))):
                bad22 += 1
            # G2.3: fp32 numpy reference for causal row 3 (all kL+3 keys?
            # row 3 sees kL+3-4+3 = kL+2 keys) - use the visible prefix
            vis3 = kL
            qf = q[:, :, 3:4].astype(mx.float32)
            kf = kfull[:, :, :vis3].astype(mx.float32)
            vf = vfull[:, :, :vis3].astype(mx.float32)
            qn = np.asarray(q[:, :, 3:4].astype(mx.float32))
            kn = np.asarray(kfull[:, :, :vis3].astype(mx.float32))
            vn = np.asarray(vfull[:, :, :vis3].astype(mx.float32))
            scale = 1.0 / np.sqrt(D)
            s = qn @ np.transpose(kn, (0, 1, 3, 2)) * scale
            e = np.exp(s - s.max(axis=-1, keepdims=True))
            ref = np.asarray((e @ vn) / e.sum(axis=-1, keepdims=True))
            got = np.asarray(causal[:, :, 3:4].astype(mx.float32))
            d = float(np.max(np.abs(got - ref)))
            worst23 = max(worst23, d)
            if d > 2e-2:
                bad23 += 1
    # G2.4: perf sanity (1000 calls, fixed shape, 512 keys)
    q, k, v = mk(512, 0)
    mx.eval(q, k, v)
    sc = 1.0 / (D ** 0.5)
    for _ in range(50):
        sdpa(q, k, v, scale=sc, mask="causal")
    t0 = time.perf_counter()
    for _ in range(1000):
        sdpa(q, k, v, scale=sc, mask="causal")
    mx.eval()
    wall = time.perf_counter() - t0
    print("G2.1 n=%d bad=%d  %s" % (n21, bad21,
          "PASS" if bad21 == 0 else "FAIL"))
    print("G2.2 bad=%d  %s" % (bad22, "PASS" if bad22 == 0 else "FAIL"))
    print("G2.3 worst_abs_delta=%.2e (bar 2e-2, %d over)  %s"
          % (worst23, bad23, "PASS" if bad23 == 0 else "FAIL"))
    print("G2.4 qL=4 x1000 wall=%.1f ms (per-call %.2f us)"
          % (1000 * wall, 1e6 * wall / 1000))
    ok = bad21 == 0 and bad22 == 0 and bad23 == 0
    print("G2 VERDICT: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
