#!/usr/bin/env python3
"""t2231 seam-measurement driver: warm up the bench, then ONE marker-bracketed
decode step, replicating the mtp-bench6.py loop body exactly (same prompt,
same load path, same env-mode mapping).

Prints one line per field:
  STEP_KIND   <ar|spec>
  STEP_WALL_MS <float>   wall of the marker-bracketed step (marker A launch -> sync after final argmax)
  WARMUP_ITERS <int>
  GEN_MS <float>  (prefill + warmup + step)

Marker: one small fp32 elementwise sigmoid over a (17,23) buffer, eval'd on the
default stream. Under the profiler, the first and last marker kernel rows delimit
the profiled step; all kernels between them are the step's kernels.

AR (MLX_MTP=0): one step = one 1-token forward + one argmax (one bench loop
iteration in steady state).
Spec (MLX_MTP=3, K=2): one step = one bench verify-branch iteration (the S=3
batched forward + argmax + rejection cascade, exactly as in the bench loop body);
a draft-branch iteration is run as warmup if the next iteration would be a draft.
"""

import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the hardware components to the "
    "operating system, covering every subsystem. Be exhaustive."
)
WARMUP = int(os.environ.get("SEAM_WARMUP", "10"))
EOS = 248044
MARKER_SHAPE = (17, 23)


def save_gdn(cache):
    return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


def marker():
    mx.eval(mx.sigmoid(mx.full(MARKER_SHAPE, 1.0)))


def main():
    mode = os.environ.get("MLX_MTP", "0")
    assert mode in ("0", "3"), mode
    mtp_on = mode == "3"
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    if mtp_on:
        assert lm.mtp is not None
        mtp_cache = [KVCache()]

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    t_gen0 = time.time()

    # Prefill (identical to the bench).
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    lg = logits[0, -1]
    h = hidden[0, -1]
    draft1 = None
    draft2 = None
    prev_token = None

    # Warmup iterations (identical loop body to the bench).
    warm = 0
    for _ in range(WARMUP * 2):
        if warm >= WARMUP:
            break
        if draft1 is None:
            t = int(mx.argmax(lg))
            if t == EOS:
                break
            if mtp_on:
                prev_token = t
                emb1 = lm.model.embed_tokens(mx.array([[t]]))
                dh1 = lm.mtp(h, emb1, mtp_cache)
                draft1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
                emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                dh2 = lm.mtp(dh1, emb2, mtp_cache)
                draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
            else:
                logits, hidden = lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
                lg = logits[0, -1]
                h = hidden[0, -1]
            warm += 1
        elif mode == "3":
            d1, d2 = draft1, draft2
            draft1 = draft2 = None
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, _h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            l2, h2 = logits[0, 2], hidden[0, 2]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d1:
                lg, h = l1, h1
                t3 = int(mx.argmax(l1))
                if t3 == EOS:
                    break
                if t3 == d2:
                    lg, h = l2, h2
                else:
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3)
                    logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                    logits, hidden = lm(mx.array([[d1]]), cache=cache, return_hidden=True, return_prenorm=True)
                    logits, hidden = lm(mx.array([[t3]]), cache=cache, return_hidden=True, return_prenorm=True)
                    lg, h = logits[0, -1], hidden[0, -1]
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(3)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                lg, h = logits[0, -1], hidden[0, -1]
            warm += 1

    # The profiled step.
    if mode == "0":
        t = int(mx.argmax(lg))
        if t == EOS:
            raise SystemExit("EOS in warmup; rerun")
        t_wall0 = time.perf_counter()
        marker()
        logits, hidden = lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
        lg = logits[0, -1]
        h = hidden[0, -1]
        _ = int(mx.argmax(lg))
        marker()
    else:
        assert draft1 is not None, "warmup did not reach a verify step; raise SEAM_WARMUP"
        d1, d2 = draft1, draft2
        draft1 = draft2 = None
        saved = save_gdn(cache)
        t_wall0 = time.perf_counter()
        marker()
        logits, hidden = lm(
            mx.array([[prev_token, d1, d2]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        l0, _h0 = logits[0, 0], hidden[0, 0]
        l1, h1 = logits[0, 1], hidden[0, 1]
        l2, h2 = logits[0, 2], hidden[0, 2]
        t2 = int(mx.argmax(l0))
        if t2 == EOS:
            raise SystemExit("EOS at profiled step; rerun")
        if t2 == d1:
            lg, h = l1, h1
            t3 = int(mx.argmax(l1))
            if t3 == EOS:
                raise SystemExit("EOS at profiled step; rerun")
            if t3 == d2:
                lg, h = l2, h2
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(3)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[d1]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t3]]), cache=cache, return_hidden=True, return_prenorm=True)
                lg, h = logits[0, -1], hidden[0, -1]
        else:
            restore_gdn(cache, saved)
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(3)
            logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
            logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
            lg, h = logits[0, -1], hidden[0, -1]
        marker()
    t_wall1 = time.perf_counter()

    wall_ms = (t_wall1 - t_wall0) * 1000.0
    gen_ms = (time.time() - t_gen0) * 1000.0
    print(f"STEP_KIND {'ar' if mode == '0' else 'spec'}", flush=True)
    print(f"STEP_WALL_MS {wall_ms:.3f}", flush=True)
    print(f"WARMUP_ITERS {warm}", flush=True)
    print(f"GEN_MS {gen_ms:.3f}", flush=True)
    mx.eval(lg)  # drain before exit


if __name__ == "__main__":
    main()
