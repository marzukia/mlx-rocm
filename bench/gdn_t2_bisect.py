#!/usr/bin/env python3
"""Isolate the S=2 F-branch divergence (gate-1 FAIL at token 5). One process: prefill the prompt once (FUSE=1), snapshot
The GDN caches, then run ONE S=2 verify forward twice from bit-identical restored caches with MLX_ROCM_GDN_FUSE=1 and
=0 (use_fallback reads the env at call time). Compares final logits and per-layer linear_attn outputs; reports the
First divergent layer."""

import glob
import os

import mlx.core as mx
import numpy as np

MODEL = os.environ.get("MTP_MODEL") or (os.environ["MODELS"] + "/qwen3-9b-mlx")

# park all but the canonical sidecar (bench mtp-bench3-side.py pattern)
for _f in (
    "model-mtp-fixed.safetensors",
    "model-mtp.safetensors",
    "model-mtp-4bit.safetensors",
    "model-mtp-8bit.safetensors",
):
    _p = os.path.join(MODEL, _f)
    if _f != "model-mtp.safetensors" and os.path.exists(_p):
        os.rename(_p, _p + ".disabled")
print("GLOB:", sorted(glob.glob(os.path.join(MODEL, "model*.safetensors"))), flush=True)

from mlx_lm.utils import load

model, tok = load(MODEL)
import mlx.nn as nn  # noqa

lm = model.language_model
mx.eval(model.parameters())

PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
prompt_ids = tok.encode(PROMPT, add_special_tokens=False)


class Hook:
    def __init__(self):
        self.on = False
        self.recs = {}

    def wrap(self, la, idx):
        orig = type(la).__call__

        def call(x, mask=None, cache=None, _la=la, _orig=orig):
            o = _orig(_la, x, mask, cache)
            if self.on:
                self.recs[_idx] = (np.asarray(mx.eval(x)), np.asarray(mx.eval(o)))
            return o

        la.__call__ = call

    def clear(self):
        self.recs = {}


hooks = {}
for name, mod in lm.layers.items():
    la = getattr(mod, "linear_attn", None)
    if la is None:
        continue
    h = Hook()
    h.wrap(la, int(name.split(".")[-1]))
    hooks[int(name.split(".")[-1])] = h

cache = lm.make_cache()
os.environ["MLX_ROCM_GDN_FUSE"] = "1"
logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
l = logits[0, -1]
t1 = int(np.argmax(l.numpy()))
t2 = 3  # arbitrary second verify token
print("prefill done; t1 =", t1, flush=True)


def gdn_snap(c):
    return [
        (i, np.asarray(mx.eval(x[0])), np.asarray(mx.eval(x[1])), mx.eval(x.lengths))
        for i, x in enumerate(c)
        if hasattr(x, "cache")
    ]


def gdn_restore(c, snap):
    for i, conv, ssm, lengths in snap:
        c[i][0] = mx.array(conv)
        c[i][1] = mx.array(ssm)
        c[i].lengths = lengths


def verify(snap):
    for h in hooks.values():
        h.on = True
        h.clear()
    gdn_restore(cache, snap)
    os.environ["MLX_ROCM_GDN_FUSE"] = MODE
    logits, _ = lm(mx.array([[t1, t2]]), cache=cache, return_prenorm=True)
    for h in hooks.values():
        h.on = False
    return np.asarray(mx.eval(logits)), {i: h.recs for i, h in hooks.items()}


snap0 = gdn_snap(cache)
print("snapshot taken", flush=True)

MODE = "1"
out1, rec1 = verify(snap0)
MODE = "0"
out0, rec0 = verify(snap0)

print("logits row0 maxdiff:", float(np.abs(out1[0, 0] - out0[0, 0]).max()), flush=True)
print("logits row1 maxdiff:", float(np.abs(out1[0, 1] - out0[0, 1]).max()), flush=True)
print("argmax row0:", int(np.argmax(out1[0, 0])), int(np.argmax(out0[0, 0])), flush=True)
print("argmax row1:", int(np.argmax(out1[0, 1])), int(np.argmax(out0[0, 1])), flush=True)

div = None
for i in sorted(rec1):
    r1 = rec1[i].get(i)
    r0 = rec0[i].get(i)
    if r1 is None or r0 is None:
        print("layer", i, "missing rec", flush=True)
        continue
    if not (np.array_equal(r1[1], r0[1])):
        md = float(np.abs(r1[1] - r0[1]).max())
        n = int((r1[1] != r0[1]).sum())
        print("LAYER DIVERGENCE:", i, "n=", n, "maxdiff=", md, flush=True)
        div = i
        if i >= 3:
            break
if div is None:
    print("NO per-layer divergence detected (all layers bit-equal)", flush=True)
# also check conv/ssm cache states after the verify
s1 = gdn_snap(cache)
gdn_restore(cache, snap0)
MODE = "1"
verify(snap0)
s_on = gdn_snap(cache)
gdn_restore(cache, snap0)
MODE = "0"
verify(snap0)
s_off = gdn_snap(cache)
for (i1, c1, s1v, l1), (i2, c2, s2v, l2) in zip(s_on, s_off):
    cd = int((c1 != c2).sum()) if c1.shape == c2.shape else -1
    sd = int((s1v != s2v).sum()) if s1v.shape == s2v.shape else -1
    if cd or sd:
        print(
            f"STATE DIVERGENCE layer {i1}: conv {cd} ssm {sd} "
            f"convmax {float(np.abs(c1 - c2).max())} "
            f"ssmmax {float(np.abs(s1v - s2v).max())}",
            flush=True,
        )
print("DONE", flush=True)
