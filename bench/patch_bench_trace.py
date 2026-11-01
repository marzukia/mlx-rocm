#!/usr/bin/env python3
"""Add GDN/KV state checksum tracing to a copy of mtp-bench3-side.py.""" import sys p = sys.argv[1] s = open(p).read()
Trace ="""
import hashlib
import mlx.core as _mxtrace

def _md5(a):
    import numpy as _np
    a = _np.asarray(a.astype(_mxtrace.float32))
    return hashlib.md5(a.tobytes()).hexdigest()[:10]

def gdn_trace(tag, cache, mtp_cache=None):
    s = []
    for c in cache:
        if hasattr(c, "cache"):
            s.append(_md5(c[0]))
            s.append(_md5(c[1]))
    if mtp_cache is not None and len(mtp_cache) > 0 and mtp_cache[0].keys is not None:
        s.append(_md5(mtp_cache[0].keys))
        s.append(_md5(mtp_cache[0].values))
    print("TRACE " + tag + " " + "-".join(s), flush=True)

"""anchor = "_MTP4_GUARD = None" assert anchor in s s = s.replace(anchor, trace + anchor, 1) old =
anchor = "_MTP4_GUARD = None" assert anchor in s s = s.replace(anchor, trace + anchor, 1) old ="""
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, 0], hidden[0, 0]     # after prev_token
            l1, h1 = logits[0, 1], hidden[0, 1]     # after d
            t2 = int(mx.argmax(l0))                 # true greedy token after prev_token
            if t2 == EOS:                          # lossless: EOS not appended (mirror plain mode)
                break
            if t2 == d:                          # accept
                accepted += 1
                out.append(t2)
                l, h = l1, h1                   # state S' (after [prev_token, d]); no rollback
            else:                                 # reject: roll the S=2 forward back,
                restore_gdn(cache, saved)         # GDN state -> S_k (before batched forward)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)                # remove prev_token AND d from attention"""
new = """            saved = save_gdn(cache)                 # GDN state BEFORE batched forward (S_k)
            gdn_trace("V%d pre   p=%d d=%d" % (verifies + 1, prev_token, d), cache, mtp_cache)
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache, return_hidden=True, return_prenorm=True
            )
            gdn_trace("V%d postf p=%d d=%d" % (verifies + 1, prev_token, d), cache, mtp_cache)
            l0, h0 = logits[0, 0], hidden[0, 0]     # after prev_token
            l1, h1 = logits[0, 1], hidden[0, 1]     # after d
            t2 = int(mx.argmax(l0))                 # true greedy token after prev_token
            print("TRACE V%d dec   t2=%d accept=%s" % (verifies + 1, t2, t2 == d), flush=True)
            if t2 == EOS:                          # lossless: EOS not appended (mirror plain mode)
                break
            if t2 == d:                          # accept
                accepted += 1
                out.append(t2)
                l, h = l1, h1                   # state S' (after [prev_token, d]); no rollback
            else:                                 # reject: roll the S=2 forward back,
                restore_gdn(cache, saved)         # GDN state -> S_k (before batched forward)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)                # remove prev_token AND d from attention
                gdn_trace("V%d rest  p=%d d=%d" % (verifies + 1, prev_token, d), cache, mtp_cache)"""
assert s.count(old) == 1
s = s.replace(old, new)

old2 = """                mtp_saved = save_mtp(mtp_cache)  # m7: snapshot BEFORE the draft entry is written
                emb = lm.model.embed_tokens(mx.array([[t]]))
                dh = lm.mtp(h, emb, mtp_cache)
                draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
                prev_token = t"""
new2 = """                mtp_saved = save_mtp(mtp_cache)  # m7: snapshot BEFORE the draft entry is written
                emb = lm.model.embed_tokens(mx.array([[t]]))
                dh = lm.mtp(h, emb, mtp_cache)
                draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
                prev_token = t
                gdn_trace("S%d step t=%d draft=%d" % (len(out), t, draft), cache, mtp_cache)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)

open(p, "w").write(s)
print("patched", p)
