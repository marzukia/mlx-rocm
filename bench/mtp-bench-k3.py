#!/usr/bin/env python3
"""t1228 instrumented K=3 MTP bench v2: disjoint-checkpoint timing.

Wall-clock decomposition of the K=3 verify loop. Classes (disjoint):
  single  = interleaved single-token step (emit + 3 drafts + 3 draft lm_heads)
  verify  = batched S=4 target forward
  refwd   = rejection re-forwards (S=1)
  other   = residual (argmax/restore/trim/gaps) = wall - single - verify - refwd
Sub-components inside single: draft (3 mtp module calls), dlmh (3 lm_head).
Computation identical to bench/mtp-bench6.py mode 4.
"""

import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
MAX_TOKENS = int(os.environ.get("MTP_MAX_TOKENS", "256"))
EOS = 248044


def save_gdn(cache):
    return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


class T:
    def __init__(self):
        self.single = 0.0
        self.verify = 0.0
        self.refwd = 0.0
        self.draft = 0.0
        self.dlmh = 0.0
        self.n_refwd = 0
        self.n_single = 0
        self.n_verify = 0


def main():
    mode = os.environ.get("MLX_MTP", "4")
    assert mode == "4", mode
    tp = T()
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    assert lm.mtp is not None, "no mtp module"
    mtp_cache = [KVCache()]

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    mx.eval(logits, hidden)
    lgrow = logits[0, -1]
    h = hidden[0, -1]
    draft1 = None
    draft2 = None
    draft3 = None
    prev_token = None
    out = []
    verifies = 0
    d1_acc = 0
    d2_acc = 0
    d3_acc = 0
    t1 = time.perf_counter()
    for _ in range(MAX_TOKENS):
        if draft1 is None:
            # Single-token step
            ts = time.perf_counter()
            t = int(mx.argmax(lgrow))
            if t == EOS:
                break
            out.append(t)
            prev_token = t
            emb1 = lm.model.embed_tokens(mx.array([[t]]))
            ta = time.perf_counter()
            dh1 = lm.mtp(h, emb1, mtp_cache)
            mx.eval(dh1)
            tp.draft += time.perf_counter() - ta
            ta = time.perf_counter()
            draft1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
            tp.dlmh += time.perf_counter() - ta
            emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
            ta = time.perf_counter()
            dh2 = lm.mtp(dh1, emb2, mtp_cache)
            mx.eval(dh2)
            tp.draft += time.perf_counter() - ta
            ta = time.perf_counter()
            draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
            tp.dlmh += time.perf_counter() - ta
            emb3 = lm.model.embed_tokens(mx.array([[draft2]]))
            ta = time.perf_counter()
            dh3 = lm.mtp(dh2, emb3, mtp_cache)
            mx.eval(dh3)
            tp.draft += time.perf_counter() - ta
            ta = time.perf_counter()
            draft3 = int(mx.argmax(lm.lm_head(dh3)[0, -1]))
            tp.dlmh += time.perf_counter() - ta
            tp.single += time.perf_counter() - ts
            tp.n_single += 1
        else:
            # Verify step: batched S=4
            d1, d2, d3 = draft1, draft2, draft3
            draft1 = draft2 = draft3 = None
            verifies += 1
            tp.n_verify += 1
            saved = save_gdn(cache)
            ta = time.perf_counter()
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2, d3]]), cache=cache, return_hidden=True, return_prenorm=True
            )
            mx.eval(logits, hidden)
            tp.verify += time.perf_counter() - ta
            l0, _ = logits[0, 0], hidden[0, 0]
            l1, _ = logits[0, 1], hidden[0, 1]
            l2, _ = logits[0, 2], hidden[0, 2]
            l3, h3 = logits[0, 3], hidden[0, 3]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d1:
                d1_acc += 1
                out.append(t2)
                t3 = int(mx.argmax(l1))
                if t3 == EOS:
                    break
                if t3 == d2:
                    d2_acc += 1
                    out.append(t3)
                    t4 = int(mx.argmax(l2))
                    if t4 == EOS:
                        break
                    if t4 == d3:
                        d3_acc += 1
                        out.append(t4)
                        lgrow, h = l3, h3
                    else:
                        out.append(t4)
                        restore_gdn(cache, saved)
                        for c in cache:
                            if isinstance(c, KVCache):
                                c.trim(4)
                        for tok_ in (prev_token, d1, d2, t4):
                            ta = time.perf_counter()
                            logits, hidden = lm(
                                mx.array([[tok_]]), cache=cache, return_hidden=True, return_prenorm=True
                            )
                            mx.eval(logits, hidden)
                            tp.refwd += time.perf_counter() - ta
                            tp.n_refwd += 1
                        lgrow, h = logits[0, -1], hidden[0, -1]
                else:
                    out.append(t3)
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(4)
                    for tok_ in (prev_token, d1, t3):
                        ta = time.perf_counter()
                        logits, hidden = lm(mx.array([[tok_]]), cache=cache, return_hidden=True, return_prenorm=True)
                        mx.eval(logits, hidden)
                        tp.refwd += time.perf_counter() - ta
                        tp.n_refwd += 1
                    lgrow, h = logits[0, -1], hidden[0, -1]
            else:
                out.append(t2)
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(4)
                for tok_ in (prev_token, t2):
                    ta = time.perf_counter()
                    logits, hidden = lm(mx.array([[tok_]]), cache=cache, return_hidden=True, return_prenorm=True)
                    mx.eval(logits, hidden)
                    tp.refwd += time.perf_counter() - ta
                    tp.n_refwd += 1
                lgrow, h = logits[0, -1], hidden[0, -1]
    wall = time.perf_counter() - t1
    n = len(out)
    d1pct = 100.0 * d1_acc / verifies if verifies else 0.0
    d2pct = 100.0 * d2_acc / d1_acc if d1_acc else 0.0
    d3pct = 100.0 * d3_acc / d2_acc if d2_acc else 0.0
    nv = max(tp.n_verify, 1)
    print(
        f"MTP={mode} LOAD={t_load:.2f} GEN={wall:.2f} TOKS={n} "
        f"TOKS_S={n / wall:.2f} VERIFIES={verifies} D1ACC={d1_acc} "
        f"D2ACC={d2_acc} D3ACC={d3_acc} D1PCT={d1pct:.1f} "
        f"D2PCT={d2pct:.1f} D3PCT={d3pct:.1f} TOK_PER_2ITER={n / nv:.3f}",
        flush=True,
    )
    other = wall - tp.single - tp.verify - tp.refwd
    print(
        f"WALL_MS single={1000 * tp.single / nv:.2f} "
        f"verify={1000 * tp.verify / nv:.2f} refwd={1000 * tp.refwd / nv:.2f} "
        f"other={1000 * other / nv:.2f} (n_refwd={tp.n_refwd})",
        flush=True,
    )
    print(
        f"SUB_MS draft={1000 * tp.draft / nv:.2f} dlmh={1000 * tp.dlmh / nv:.2f} (inside single)",
        flush=True,
    )
    print(
        f"PER_PAIR_MS wall={1000 * wall / nv:.2f} fwd_only={1000 * (tp.single + tp.verify + tp.refwd) / nv:.2f}",
        flush=True,
    )
    print("FIRST20:", " ".join(map(str, out[:20])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)


if __name__ == "__main__":
    main()
