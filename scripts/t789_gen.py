#!/usr/bin/env python3
"""t789 G1/G2 gate generator: greedy 1024-token generation on the 4bit model,
same protocol as the m0 reference (mtp-bench3.py mode 0: smartphone prompt,
greedy, MAX_TOKENS=1024, fresh process per run). MLX_GDN_TAIL_FUSION read
from env (t608 lesson: never positional). Prints TOKS/TOKS_S/ALLTOKS."""

import os
import time

import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("T789_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
MAX_TOKENS = 1024
EOS = 248044


def main():
    fusion = os.environ.get("MLX_GDN_TAIL_FUSION", "<unset>")
    assert fusion in ("0", "1"), f"MLX_GDN_TAIL_FUSION={fusion} (expected 0 or 1)"
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits = lm(mx.array([prompt_ids]), cache=cache)
    lg = logits
    out = []
    t1 = time.time()
    for _ in range(MAX_TOKENS):
        t = int(mx.argmax(lg))
        if t == EOS:
            break
        out.append(t)
        logits = lm(mx.array([[t]]), cache=cache)
        lg = logits
    dt = time.time() - t1
    n = len(out)
    print(
        f"FUSION={fusion} LOAD={t_load:.2f} GEN={dt:.2f} TOKS={n} TOKS_S={n / max(dt, 1e-9):.2f}",
        flush=True,
    )
    print("FIRST16:", " ".join(map(str, out[:16])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)


if __name__ == "__main__":
    main()
