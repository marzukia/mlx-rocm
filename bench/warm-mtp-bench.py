#!/usr/bin/env python3
"""run-1014 BENCH: AR / MTP K=1 / MTP K=2 tok/s at a 14-token AND a 29K prompt,
WARMUP FIRST, on the lane wheel (run-1012-29k/venv-m8, .so de6de5f154c0).

Harness = the mtp-bench5.py MTP flow (mode 0 = AR greedy, mode 2 = K=1 batched
S=2 verify, mode 3 = K=2 batched S=3 verify; draft chain / reject rollbacks
verbatim from mtp-bench5) + the run-1012-29k.py deterministic 29K prompt (fixed
1264-char corpus tiled, truncated to exactly T29K_TARGET tokens) and chunked
prefill (T29K_CHUNK=1024; for the 14-token prompt the chunk == the whole
prompt, i.e. single-pass prefill - the stock mtp-bench3/5 flow).

Op-order WARMUP FIRST: with T14_WARM=1 (default) one full prefill+decode pass
at the same context is run and DISCARDED (fresh cache for the measured pass),
so kernels / hipBLASLt heuristic selection / L2 / KV allocation are steady
before measurement. Warm numbers are logged separately (WARM line).

Steady-state window: the first T14_SKIP (default 56) decode iterations are
excluded from the window tok/s (same convention as run-1012: steps 57-256). For
MTP modes each iteration is a single-token (draft) step or a verify step
(emits 1-3 tokens); window tok/s = tokens emitted inside the window / window
wall-time (iteration-based, documented in the report).

Prints: MEASURE (toks, prefill ms/tok_s, decode tok_s overall + window,
iteration stats, acceptance D1ACC/D2ACC/verifies/tok_per_step, .so sha),
WARM (same fields, discarded), ALLTOKS.
"""

import hashlib
import os

os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("GD_MODEL", f"{MODELS}/qwen3-9b-mlx")
TARGET = int(os.environ.get("T29K_TARGET", "14"))
MAX_TOKENS = int(os.environ.get("T14_MAX_TOKENS", "256"))
SKIP = int(os.environ.get("T14_SKIP", "56"))
CHUNK = int(os.environ.get("T29K_CHUNK", "1024"))
WARM = os.environ.get("T14_WARM", "1") == "1"
EOS = 248044

# The stock 14-token prompt (mtp-bench3/5 lineage; byte-identical to the
# baseline bench set).
PROMPT_14 = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
# run-1012's fixed corpus (byte-identical), tiled + truncated to TARGET tokens.
CORPUS = (
    "A modern smartphone is a small general-purpose computer. Its main processor "
    "and graphics run on a system-on-chip, with the CPU cores, GPU, and neural "
    "processing unit fabricated on the same die. The operating system manages "
    "memory, scheduling, and drivers. The display controller drives an LCD or "
    "OLED panel through a high-bandwidth interface. The baseband processor "
    "handles cellular radio, while separate or integrated radios handle Wi-Fi, "
    "Bluetooth, and GPS. The modem chain converts between digital baseband "
    "symbols and the analog RF signal, applying filtering, amplification, and "
    "modem algorithms such as channel estimation and equalization. Power is "
    "managed by a fuel-gauge and PMIC that balance battery charging, "
    "voltage rails, and thermal limits. The camera pipeline demosaics and "
    "applies noise reduction and tone mapping before compression. The audio "
    "path handles codecs, amplifiers, and spatial audio. Storage is UFS or "
    "NV flash managed by file systems and wear-leveling. Sensors such as the "
    "accelerometer, gyroscope, barometer, and proximity sensor feed drivers "
    "that publish to user space. The security model uses a trusted execution "
    "environment, secure boot, and hardware key storage. The radio firmware "
    "runs in a separate address space from the main OS."
)


def build_prompt_ids(tok):
    if TARGET <= 100:
        return tok.encode(PROMPT_14, add_special_tokens=False)
    base = tok.encode(CORPUS, add_special_tokens=False)
    n = len(base)
    reps = (TARGET + n - 1) // n
    ids = list(base) * reps
    return ids[:TARGET]


def save_gdn(cache):
    return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


def forward(lm, tok_arr, cache):
    logits, hidden = lm(tok_arr, cache=cache, return_hidden=True, return_prenorm=True)
    return logits[0, -1], hidden[0, -1]


def generate(lm, prompt_ids, mode):
    """One full pass: prefill + <= MAX_TOKENS decode iterations.

    Returns (out, prefill_ms, it_times, it_toks, decode_ms, verifies,
    d1_acc, d2_acc). it_times/it_toks are per decode iteration (a draft step
    or a verify step)."""
    mtp_on = mode in ("2", "3")
    if mtp_on:
        assert lm.mtp is not None
        mtp_cache = [KVCache()]
    else:
        mtp_cache = None
    cache = lm.make_cache()
    t0 = time.perf_counter()
    for i in range(0, len(prompt_ids), CHUNK):
        chunk = prompt_ids[i : i + CHUNK]
        lg, h = forward(lm, mx.array([chunk]), cache)
    t1 = time.perf_counter()
    draft1 = draft2 = None
    prev_token = None
    out = []
    verifies = 0
    d1_acc = 0
    d2_acc = 0
    it_times = []
    it_toks = []
    done = False
    while len(out) < MAX_TOKENS and not done:
        ts = time.perf_counter()
        if draft1 is None:
            t = int(mx.argmax(lg))
            if t == EOS:
                break
            out.append(t)
            if mtp_on:
                # Draft step (state stays S_k; no target forward).
                prev_token = t
                emb1 = lm.model.embed_tokens(mx.array([[t]]))
                dh1 = lm.mtp(h, emb1, mtp_cache)
                draft1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
                if mode == "3":
                    emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                    dh2 = lm.mtp(dh1, emb2, mtp_cache)
                    draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
            else:
                # AR: advance the state with t (mtp-bench5 mode-0 path).
                lg, h = forward(lm, mx.array([[t]]), cache)
        else:
            d1 = draft1
            d2 = draft2
            draft1 = draft2 = None
            verifies += 1
            saved = save_gdn(cache)
            if mode == "2":
                # K=1: batched S=2 verify (mtp-bench5 mode-2 path).
                logits, hidden = lm(
                    mx.array([[prev_token, d1]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                t2 = int(mx.argmax(logits[0, 0]))
                if t2 == EOS:
                    done = True
                elif t2 == d1:
                    d1_acc += 1
                    out.append(t2)
                    lg, h = logits[0, 1], hidden[0, 1]
                else:
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(2)
                    out.append(t2)
                    lg, h = forward(lm, mx.array([[prev_token]]), cache)
                    lg, h = forward(lm, mx.array([[t2]]), cache)
            else:
                # K=2: batched S=3 verify (mtp-bench5 mode-3 path).
                logits, hidden = lm(
                    mx.array([[prev_token, d1, d2]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                t2 = int(mx.argmax(logits[0, 0]))
                if t2 == EOS:
                    done = True
                elif t2 == d1:
                    d1_acc += 1
                    out.append(t2)
                    t3 = int(mx.argmax(logits[0, 1]))
                    if t3 == EOS:
                        done = True
                    elif t3 == d2:
                        d2_acc += 1
                        out.append(t3)
                        lg, h = logits[0, 2], hidden[0, 2]
                    else:
                        out.append(t3)
                        restore_gdn(cache, saved)
                        for c in cache:
                            if isinstance(c, KVCache):
                                c.trim(3)
                        lg, h = forward(lm, mx.array([[prev_token]]), cache)
                        lg, h = forward(lm, mx.array([[d1]]), cache)
                        lg, h = forward(lm, mx.array([[t3]]), cache)
                else:
                    out.append(t2)
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3)
                    lg, h = forward(lm, mx.array([[prev_token]]), cache)
                    lg, h = forward(lm, mx.array([[t2]]), cache)
        it_times.append(time.perf_counter() - ts)
        it_toks.append(len(out) - (sum(it_toks)))
    return (
        out,
        (t1 - t0) * 1e3,
        it_times,
        it_toks,
        (time.perf_counter() - t1) * 1e3,
        verifies,
        d1_acc,
        d2_acc,
    )


def so_sha():
    p = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def fmt_run(tag, mode, npt, r):
    out, prefill_ms, it_times, it_toks, decode_ms, verifies, d1_acc, d2_acc = r
    n = len(out)
    w_t = it_times[SKIP:]
    w_tok = it_toks[SKIP:]
    dec_s = decode_ms / 1e3
    win_s = sum(w_t)
    acc = ""
    if mode in ("2", "3"):
        tok_step = 1.0 + d1_acc / verifies if verifies else 0.0
        acc = (
            f" VERIFIES={verifies} D1ACC={d1_acc} D2ACC={d2_acc} "
            f"D1PCT={100.0 * d1_acc / verifies if verifies else 0.0:.1f} "
            f"D2PCT={100.0 * d2_acc / d1_acc if d1_acc else 0.0:.1f} "
            f"TOK_PER_STEP={tok_step:.3f}"
        )
    s = sorted(it_times)
    n_s = len(s)
    med = s[n_s // 2]
    mean = sum(it_times) / n_s
    return (
        f"{tag} MLX_MTP={mode} PROMPT_TOKS={npt} toks={n} "
        f"prefill={prefill_ms:.1f}ms prefill_tok_s={npt / max(prefill_ms / 1e3, 1e-9):.0f} "
        f"decode_tok_s_all={n / max(dec_s, 1e-9):.2f} "
        f"decode_tok_s_win={sum(w_tok) / max(win_s, 1e-9):.2f} "
        f"(win={len(w_t)} its, skip={SKIP}) "
        f"it median={med * 1e6:.0f}us mean={mean * 1e6:.0f}us "
        f"min={s[0] * 1e6:.0f}us max={s[-1] * 1e6:.0f}us {acc}"
    )


def main():
    mode = os.environ.get("MLX_MTP", "0")
    assert mode in ("0", "2", "3"), mode
    t0 = time.time()
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    t_load = time.time() - t0
    prompt_ids = build_prompt_ids(tok)
    npt = len(prompt_ids)
    print(
        f"ENV python={__import__('sys').version.split()[0]} mlx={mx.__file__} "
        f"so_sha256_12={so_sha()} model={MODEL} target={TARGET} "
        f"max_tokens={MAX_TOKENS} skip={SKIP} chunk={CHUNK} warm={WARM}",
        flush=True,
    )
    warm_line = None
    if WARM:
        warm_line = fmt_run("WARM", mode, npt, generate(lm, prompt_ids, mode))
    r = generate(lm, prompt_ids, mode)
    print(
        f"MEASURE LOAD_S={t_load:.1f} " + fmt_run("", mode, npt, r),
        flush=True,
    )
    if warm_line:
        print(warm_line, flush=True)
    print("ALLTOKS: " + " ".join(map(str, r[0])), flush=True)


if __name__ == "__main__":
    main()
