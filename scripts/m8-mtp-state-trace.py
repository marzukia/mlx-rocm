#!/usr/bin/env python3
"""m8 P3: per-verify-step state trace (S=2 vs 2xS=1, every step).

If the token-148 fork is fully clean (row 0/1 exact, state exact at
147), the G4 first_div=148 must come from an EARLIER step: the first
verify step k where the S=2 post-state != the 2xS=1 post-state (from
the same pre-state). This probe runs the exact mtp-bench3 mode-2 flow
and at EVERY verify step:

  saved = save_gdn (ref, pre-S=2)
  A: S=2 [prev, d]  -> snap_Apost (numpy, dtype-preserving)
  rollback: restore_gdn + trim(2) + restore_mtp
  B: S=1 [prev], S=1 [d]  -> snap_Bpost
  compare snap_Apost vs snap_Bpost (GDN conv/ssm/lengths, KV keys/
  values/offset) - print the first leaking step + buffers
  restore snap_Apost; continue the flow (accept/reject as mtp-bench3)

Cost: 2 extra target forwards per verify step (~85 verifies).
Also prints per-step: t2, accept, and any maxabs > 0.
"""
import os

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = ("Explain in great detail how a modern smartphone works, from the "
          "hardware components to the operating system, covering every "
          "subsystem. Be exhaustive.")
MAX_TOKENS = 170
EOS = 248044


def npf(a):
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def snap_all(cache, mtp_cache):
    gdn, kv, mtp = [], [], None
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            kv.append((i, c.keys.dtype, npf(c.keys), c.values.dtype,
                      npf(c.values), c.offset))
        else:
            ln = c.lengths
            gdn.append((i, c[0].dtype, npf(c[0]), c[1].dtype, npf(c[1]),
                       npf(ln) if ln is not None else None))
    if mtp_cache:
        c = mtp_cache[0]
        if c.keys is not None:
            mtp = (c.keys.dtype, npf(c.keys), c.values.dtype,
                   npf(c.values), c.offset)
    return gdn, kv, mtp


def restore_all(cache, mtp_cache, snap):
    gdn, kv, mtp = snap
    for i, dt0, cc, dt1, ss, ln in gdn:
        cache[i][0] = mx.array(cc, dtype=dt0)
        cache[i][1] = mx.array(ss, dtype=dt1)
        if ln is not None:
            cache[i].lengths = mx.array(ln, dtype=cache[i].lengths.dtype)
        else:
            cache[i].lengths = None
    for i, dtk, kk, dtv, vv, off in kv:
        cache[i].keys = mx.array(kk, dtype=dtk)
        cache[i].values = mx.array(vv, dtype=dtv)
        cache[i].offset = off
    if mtp is not None and mtp_cache:
        dtk, kk, dtv, vv, off = mtp
        mtp_cache[0].keys = mx.array(kk, dtype=dtk)
        mtp_cache[0].values = mx.array(vv, dtype=dtv)
        mtp_cache[0].offset = off


def compare_states(name, sa, sb):
    """returns (leak, worst) - worst = max maxabs over leaking buffers"""
    leak = False
    worst = 0.0
    for gA, gB in zip(sa[0], sb[0]):
        for x, y, nm in ((gA[2], gB[2], "conv"), (gA[4], gB[4], "ssm")):
            if x is None or y is None or x.shape != y.shape:
                continue
            dd = x - y
            n = int(np.count_nonzero(dd != 0))
            if n > 0:
                leak = True
                m = float(np.abs(dd).max())
                worst = max(worst, m)
                print(f"  [{name}] GDN layer {gA[0]} {nm}: n={n}/{dd.size} "
                      f"maxabs={m:.3e}")
    for kA, kB in zip(sa[1], sb[1]):
        for x, y, nm in ((kA[2], kB[2], "keys"), (kA[4], kB[4], "values")):
            if x is None or y is None:
                continue
            if x.shape != y.shape:
                m = min(x.shape[0], y.shape[0]) if x.ndim >= 1 else 0
                x, y = x[:m], y[:m]
            dd = x - y
            n = int(np.count_nonzero(dd != 0))
            if n > 0:
                leak = True
                ma = float(np.abs(dd).max())
                worst = max(worst, ma)
                print(f"  [{name}] KV layer {kA[0]} {nm}: n={n}/{dd.size} "
                      f"maxabs={ma:.3e}")
    for lnA, lnB in zip([g[5] for g in sa[0]], [g[5] for g in sb[0]]):
        if lnA is None or lnB is None:
            continue
        dd = lnA - lnB
        n = int(np.count_nonzero(dd != 0))
        if n > 0:
            leak = True
            ma = float(np.abs(dd).max())
            worst = max(worst, ma)
            print(f"  [{name}] GDN lengths: n={n}/{dd.size} "
                  f"maxabs={ma:.3e}")
    return leak, worst


def main():
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache,
                       return_hidden=True, return_prenorm=True)
    l = logits[0, -1]
    h = hidden[0, -1]

    def save_gdn(cache):
        return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache)
                if hasattr(c, "cache")]

    def restore_gdn(cache, saved):
        for i, conv, ssm, lengths in saved:
            cache[i][0] = conv
            cache[i][1] = ssm
            cache[i].lengths = lengths

    def save_mtp(mtp_cache):
        c = mtp_cache[0]
        if c.keys is None:
            return (None, None, c.offset)
        return (npf(c.keys), npf(c.values), c.offset)

    def restore_mtp(mtp_cache, saved):
        keys, values, offset = saved
        c = mtp_cache[0]
        if keys is not None:
            c.keys = mx.array(keys)
            c.values = mx.array(values)
        c.offset = offset

    out = []
    accepted = 0
    verifies = 0
    draft = None
    prev_token = None
    mtp_saved = None
    first_leak_step = None
    n_leak = 0

    for _ in range(MAX_TOKENS):
        if draft is None:
            t = int(mx.argmax(l))
            if t == EOS:
                break
            out.append(t)
            mtp_saved = save_mtp(mtp_cache)
            emb = lm.model.embed_tokens(mx.array([[t]]))
            dh = lm.mtp(h, emb, mtp_cache)
            draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
            prev_token = t
        else:
            d = draft
            draft = None
            verifies += 1
            step = len(out)  # verify appends token index `step`
            saved = save_gdn(cache)
            # --- A: S=2 [prev, d] ---
            logits, hidden = lm(mx.array([[prev_token, d]]), cache=cache,
                               return_hidden=True, return_prenorm=True)
            l0, h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            t2 = int(mx.argmax(l0))
            snap_Apost = snap_all(cache, mtp_cache)
            # --- rollback + B: S=1 [prev], S=1 [d] ---
            restore_gdn(cache, saved)
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(2)
            restore_mtp(mtp_cache, mtp_saved)
            lm(mx.array([[prev_token]]), cache=cache,
               return_hidden=True, return_prenorm=True)
            lm(mx.array([[d]]), cache=cache,
               return_hidden=True, return_prenorm=True)
            snap_Bpost = snap_all(cache, mtp_cache)
            # --- compare + restore A post-state ---
            leak, worst = compare_states(f"step{step}", snap_Apost,
                                        snap_Bpost)
            restore_all(cache, mtp_cache, snap_Apost)
            if leak:
                n_leak += 1
                if first_leak_step is None:
                    first_leak_step = (step, worst)
                    print(f"FIRST STATE LEAK at verify step {step} "
                          f"(token index {step}, t2={t2}, d={d}) "
                          f"worst={worst:.3e}", flush=True)
            elif worst > 0:
                pass
            # --- continue flow as mtp-bench3 ---
            if t2 == EOS:
                break
            if t2 == d:
                accepted += 1
                out.append(t2)
                l, h = l1, h1
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                restore_mtp(mtp_cache, mtp_saved)
                out.append(t2)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                l, h = logits[0, -1], hidden[0, -1]
            draft = None
    print(f"done: n_out={len(out)} accepted={accepted}/{verifies} "
          f"n_leak_steps={n_leak} first_leak={first_leak_step}")
    print(f"token@148={out[148] if len(out) > 148 else 'n/a'}")


if __name__ == "__main__":
    main()
