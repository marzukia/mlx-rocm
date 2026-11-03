#!/usr/bin/env python3
"""Probe B (t1779): D2H read of the per-step idx buffer DEVICE contents at
LAUNCH time (not op time), to confirm/refute the cross-stream visibility hypothesis (the replayed kernel reads pre-launch buffer contents).

Method: PG (MLX_ROCM_GRAPH_DECODE=1). Each step creates the per-step idx
buffer exactly as the bench does (fresh mx.array([[t]])); before the step's
forward (i.e. before the graph launch, with nothing else writing the buffer in between) a D2H read (.item()) forces the creation H2D to complete and
reports the buffer's device contents at launch time. match=1 means the
kernel's read point sees the current token; match=0 means it sees a stale
(pre-launch) value. MLX_G1Y_DEBUG=1 gives the [gdidx] gather lines
(VA + op-time idxval) for cross-reference.
"""

import os

import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("MTP_MODEL", os.environ.get("MODELS", "/models") + "/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the hardware "
    "components to the operating system, covering every subsystem. Be exhaustive."
)
N = int(os.environ.get("PROBE_N", "24"))


def main():
    print(f"ARM=PG N={N}", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    lg = logits[0, -1].astype(mx.float32)
    prev = prompt_ids[-1]
    toks = []
    for step in range(N):
        t = int(mx.argmax(lg))
        arr = mx.array([[t]])
        # D2H read at launch time: forces the creation H2D to complete before
        # the step's forward (the graph launch); nothing else writes arr in
        # between, so this is the buffer's device contents at the kernel's
        # read point.
        readback = int(arr.reshape(-1).item())
        print(
            f"[probeB] s{step} tok={t} readback={readback} match={1 if readback == t else 0} prev={prev}",
            flush=True,
        )
        toks.append(t)
        prev = t
        logits, hidden = lm(arr, cache=cache, return_hidden=True, return_prenorm=True)
        lg = logits[0, -1].astype(mx.float32)
    print("PROBE_TOKS:", " ".join(map(str, toks)), flush=True)


main()
