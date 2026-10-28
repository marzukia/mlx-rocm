#!/usr/bin/env python3
"""Un-sandwiched per-component timing: T=1 fwd, S=3 fwd, K=2 draft step.
Forces eval of the actual outputs (like the real decode loop)."""
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = "/mnt/models/qwen3-9b-mlx"
PROMPT = "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive."

model, tok = load(MODEL)
lm = model.language_model
mx.eval(model.parameters())
mtp_cache = [KVCache()]

prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
cache = lm.make_cache()

logits, hidden = lm(
    mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True
)
l = logits[0, -1]
h = hidden[0, -1]
t = int(mx.argmax(l))
emb = lm.model.embed_tokens(mx.array([[t]]))
dh1 = lm.mtp(h, emb, mtp_cache)
draft1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
dh2 = lm.mtp(dh1, emb2, mtp_cache)
draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
mx.eval(l, h, emb, dh1, emb2, dh2)

# Warm
for _ in range(3):
    out = lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
    mx.eval(*out)

def timeit(fn, n=9):
    out = fn()  # warm
    mx.eval(*out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(*out)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]

def t1_fwd():
    return lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)

def s3_fwd():
    return lm(mx.array([[t, draft1, draft2]]), cache=cache, return_hidden=True, return_prenorm=True)

def draft_step():
    e = lm.model.embed_tokens(mx.array([[t]]))
    d1 = lm.mtp(h, e, mtp_cache)
    x1 = int(mx.argmax(lm.lm_head(d1)[0, -1]))  # argmax forces eval
    e2 = lm.model.embed_tokens(mx.array([[x1]]))
    d2 = lm.mtp(d1, e2, mtp_cache)
    x2 = int(mx.argmax(lm.lm_head(d2)[0, -1]))
    return (e, d1, e2, d2)

T1 = timeit(t1_fwd)
S3 = timeit(s3_fwd)
DR = timeit(draft_step)
print("T1_FWD_MS=%.2f" % T1, flush=True)
print("S3_FWD_MS=%.2f" % S3, flush=True)
print("S3_OVER_T1=%.2fx" % (S3 / T1), flush=True)
print("DRAFT2_MS=%.2f" % DR, flush=True)
print("PROBE-DONE", flush=True)
