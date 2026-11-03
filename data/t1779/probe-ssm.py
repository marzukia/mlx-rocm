#!/usr/bin/env python3
"""Probe A (t1779): re-run of the v2.1 per-step PA-vs-PG GDN state md5 probe.

Identical methodology to the v2.1 round probe: per-step md5 of the GDN ssm state (first 4096 f32 values) + tokens, arm set by MLX_ROCM_GRAPH_DECODE
(0 = PA eager, 1 = PG graph). Question: is the GDN ssm state now step-exact
at replay after the 9c82e4c in-place fix, or still period-4?
"""

import hashlib
import os
import struct

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = os.environ.get("MTP_MODEL", os.environ.get("MODELS", "/models") + "/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the hardware "
    "components to the operating system, covering every subsystem. Be exhaustive."
)
N = int(os.environ.get("PROBE_N", "24"))
ARM = os.environ.get("MLX_ROCM_GRAPH_DECODE", "0")


def md5b(b):
    return hashlib.md5(b).hexdigest()[:12]


def sl(a, n=4096):
    x = a.reshape(-1).astype(mx.float32)[0:n].tolist()
    return md5b(struct.pack(f"{len(x)}d", *x))


def main():
    print(f"ARM={ARM} N={N}", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    lg = logits[0, -1].astype(mx.float32)
    print(f"[probe] s-1 (prefill) L={len(prompt_ids)}", flush=True)
    toks = []
    for step in range(N):
        t = int(mx.argmax(lg))
        print(f"[probe] s{step} tok={t}", flush=True)
        toks.append(t)
        sline = []
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                continue
            if hasattr(c, "cache"):
                c1 = c[1]
                sline.append(f"g{i}={sl(c1)}")
        joined = " ".join(sline)
        print(f"[probe] s{step} state n={len(sline)} {joined}", flush=True)
        logits, hidden = lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
        lg = logits[0, -1].astype(mx.float32)
    print("PROBE_TOKS:", " ".join(map(str, toks)), flush=True)


main()
