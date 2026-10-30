#!/usr/bin/env python3
"""Phase 2 GDN-fuse harness: A/B bench + lossless gate data capture.

One process per measured path (runner spawns fresh processes). Mirrors the
t1012-29k.py protocol exactly (WARM pass discarded, MEASURE pass, decode
tok/s ALL steps + steady-state window with first N=56 steps excluded, median
step time), plus:
  * stream mode: ALLTOKS token-id stream for byte-identity diff
  * state mode: per-GDN-layer state sha256 at selected decode steps, for the
    cross-process fused-vs-unfused state match (n is asserted)
  * P2PATH marker + fused-path call counter (non-vacuity: proves the fused
    op actually ran, not a silent fallback to the chain)

Env:
  GD_MODEL            model dir (default $MODELS/qwen3-9b-mlx; MODELS env
                      defaults to /models, set it to the real root on the guest)
  P2_PROMPT_TOKENS    prompt length (14 for the short gate, 29000 for the
                      long gate; default 29000)
  P2_MAX_TOKENS       decode step cap (default 256)
  P2_SKIP             steady-state window: exclude first N steps (default 56)
  P2_CHUNK            prefill chunk (default 1024)
  P2_WARM             do the discarded warm pass first (default 1)
  P2_STATE_STEPS      comma list of decode-step indexes to capture state
                      (default: first 8 + final)
  P2_STATE_OUT        state JSON-lines output path
  P2_STREAM_OUT       ALLTOKS stream output path (default stdout line)
  P2_MODE             bench | oneshot (default oneshot = bench + stream + state)
  MLX_ROCM_GDN_FUSE   the gate under test (1 = fused op path)
"""

import hashlib
import json
import os

import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import time

import mlx.core as mx
from mlx_lm import load

# Fused-path call counter lives in the patched model module (GDN_FUSE_CALLS);
# absent in the unpatched (default-path) venv -> 0.
try:
    from mlx_lm.models import qwen3_5 as _q35
except Exception:
    _q35 = None


def fused_calls():
    if _q35 is None:
        return 0
    return int(getattr(_q35, "GDN_FUSE_CALLS", 0))


def reset_fused_calls():
    if _q35 is not None:
        _q35.GDN_FUSE_CALLS = 0


MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("GD_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT_TOKENS = int(os.environ.get("P2_PROMPT_TOKENS", "29000"))
MAX_TOKENS = int(os.environ.get("P2_MAX_TOKENS", "256"))
SKIP = int(os.environ.get("P2_SKIP", "56"))
CHUNK = int(os.environ.get("P2_CHUNK", "1024"))
EOS = 248044
WARM = os.environ.get("P2_WARM", "1") == "1"
MODE = os.environ.get("P2_MODE", "oneshot")
STATE_OUT = os.environ.get("P2_STATE_OUT", "")
STREAM_OUT = os.environ.get("P2_STREAM_OUT", "")
_state_steps_env = os.environ.get("P2_STATE_STEPS", "")

# Fixed corpus (deterministic, single source) -- same text as t1012-29k.py.
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

_state_steps = None
if _state_steps_env:
    _state_steps = [int(x) for x in _state_steps_env.split(",") if x]
elif MAX_TOKENS > 8:
    _state_steps = list(range(8))
else:
    _state_steps = list(range(MAX_TOKENS))


def build_prompt_ids(tok):
    base = tok.encode(CORPUS, add_special_tokens=False)
    n = len(base)
    reps = (PROMPT_TOKENS + n - 1) // n
    ids = list(base) * reps
    ids = ids[:PROMPT_TOKENS]
    return ids


def gdn_layer_states(cache):
    """Per-GDN-layer recurrent state (cache[i][1], (1, Hv, Dv, Dk) f32) as
    (layer_idx, sha256, nbytes, any_nonzero). Non-GDN (FA) layers are skipped
    by shape check.
    """
    out = []
    for i, entry in enumerate(cache):
        try:
            st = entry[1]
        except (TypeError, IndexError, AttributeError):
            continue
        if st is None or not hasattr(st, "shape"):
            continue
        shape = tuple(st.shape)
        if len(shape) != 4 or shape[1:] != (32, 128, 128):
            continue
        mx.eval(st)
        b = bytes(bytearray(st))
        arr = np.frombuffer(b, np.uint8)
        out.append((i, hashlib.sha256(b).hexdigest(), len(b), bool(arr.any())))
    return out


def generate(lm, tok, prompt_ids, capture_state=False):
    cache = lm.make_cache()
    t0 = time.perf_counter()
    model = lm.language_model
    for i in range(0, len(prompt_ids), CHUNK):
        chunk = prompt_ids[i : i + CHUNK]
        logits, _hidden = model(
            mx.array([chunk]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        mx.eval(logits)
    lg = logits[0, -1]
    t1 = time.perf_counter()
    out = []
    step_times = []
    states = []
    while len(out) < MAX_TOKENS:
        ts = time.perf_counter()
        tok = int(mx.argmax(lg))
        if tok == EOS:
            break
        out.append(tok)
        logits, _hidden = model(mx.array([[tok]]), cache=cache, return_hidden=True, return_prenorm=True)
        lg = logits[0, -1]
        step_times.append(time.perf_counter() - ts)
        step_idx = len(out) - 1
        if capture_state and step_idx in _state_steps:
            states.append((step_idx, gdn_layer_states(cache)))
    # final-step capture (state AFTER the last processed decode step)
    if capture_state and out:
        states.append((len(out), gdn_layer_states(cache)))
    t2 = time.perf_counter()
    return out, (t1 - t0) * 1e3, step_times, (t2 - t1) * 1e3, states


def stats(step_times):
    if not step_times:
        return "no-steps"
    s = sorted(step_times)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    mean = sum(s) / n
    return f"n={n} med_ms={1e3 * med:.3f} mean_ms={1e3 * mean:.3f}"


def main():
    t_load = time.perf_counter()
    lm, tok = load(MODEL)
    t_load = time.perf_counter() - t_load
    so = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
    try:
        with open(so, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        h = "unknown"
    print(
        f"ENV-PROVENANCE venv={os.environ.get('VENV_ID', '?')} "
        f"so_sha256={h} gdn_fuse={os.environ.get('MLX_ROCM_GDN_FUSE', '0')} "
        f"gdn_tail_fusion={os.environ.get('MLX_GDN_TAIL_FUSION', 'unset')}",
        flush=True,
    )

    prompt_ids = build_prompt_ids(tok)
    print(f"LOAD_S={t_load:.2f} PROMPT_TOKS={len(prompt_ids)}", flush=True)

    def run_pass(tag, capture_state):
        reset_fused_calls()
        out, prefill_ms, step_times, decode_ms, states = generate(lm, tok, prompt_ids, capture_state=capture_state)
        st = stats(step_times)
        n = len(step_times)
        # tok/s from the sum of per-step measured times (host argmax +
        # forward + eval); the t2-t1 wall includes state-capture/conversions
        # and is not comparable, so it is only reported as decode_ms.
        dec = (n / sum(step_times)) if n else float("nan")
        win = step_times[SKIP:]
        dec_win = (len(win) / sum(win)) if len(win) else float("nan")
        print(
            f"{tag} prefill_ms={prefill_ms:.1f} "
            f"prefill_tok_s={1e3 * len(prompt_ids) / max(prefill_ms, 1e-9):.1f} "
            f"decode_ms={decode_ms * 1e3:.1f} decode_tok_s_all={dec:.2f} "
            f"decode_tok_s_window={dec_win:.2f} step={st} "
            f"fused_calls={fused_calls()}",
            flush=True,
        )
        return out, states

    if WARM:
        run_pass("WARM", capture_state=False)
    out, states = run_pass("MEASURE", capture_state=(MODE == "oneshot"))

    if MODE in ("oneshot", "stream"):
        s = " ".join(str(t) for t in out)
        if STREAM_OUT:
            with open(STREAM_OUT, "w") as f:
                f.write(s + "\n")
        else:
            print(f"ALLTOKS {s}", flush=True)

    if MODE in ("oneshot", "state") and STATE_OUT:
        with open(STATE_OUT, "w") as f:
            for step_idx, layers in states:
                for li, sha, nb, nonzero in layers:
                    f.write(
                        json.dumps({"step": step_idx, "layer": li, "sha": sha, "nbytes": nb, "nonzero": bool(nonzero)})
                        + "\n"
                    )

    # Non-vacuity asserts (F2 lesson: empty/short comparisons must fail).
    assert len(out) > 0, "empty token stream"
    if MODE == "oneshot" and states:
        n_states = sum(len(st) for _, st in states)
        assert n_states > 0, "no GDN states captured"
        assert any(nz for _, st in states for (_, _, _, nz) in st), "all GDN states zero -- vacuous gate"
        print(f"STATE-CAPTURE n={n_states}", flush=True)
    print("P2-DONE", flush=True)


if __name__ == "__main__":
    main()
