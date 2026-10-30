#!/usr/bin/env python3
"""Quantify fused-vs-chain GDN recurrent-state drift at the final decode step.

Runs BOTH paths in ONE process (venv-fused has both the chain and the fused
op; the F branch toggles on MLX_ROCM_GDN_FUSE per call). Same model, same
prompt, greedy. Reports per-GDN-layer max/mean abs diff and max relative diff
of the f32 (1, Hv, Dv, Dk) state. A bounded, sub-bf16 drift (tiny max_abs,
tiny max_rel) confirms the fused kernel is a numerically faithful replacement
whose token stream is byte-identical (the lossless signal).
"""

import os

os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import mlx.core as mx
import numpy as np
from mlx_lm import load

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("GD_MODEL", f"{MODELS}/qwen3-9b-mlx")
N = int(os.environ.get("P2_N", "256"))
CHUNK = 1024
EOS = 248044
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


def build_prompt_ids(tok, n):
    base = tok.encode(CORPUS, add_special_tokens=False)
    reps = (n + len(base) - 1) // len(base)
    return (list(base) * reps)[:n]


def run_pass(lm, tok, prompt_ids, fuse):
    if fuse:
        os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    else:
        os.environ.pop("MLX_ROCM_GDN_FUSE", None)
    model = lm.language_model
    cache = lm.make_cache()
    for i in range(0, len(prompt_ids), CHUNK):
        logits, _h = model(mx.array([prompt_ids[i : i + CHUNK]]), cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval(logits)
    lg = logits[0, -1]
    toks = []
    while len(toks) < N:
        t = int(mx.argmax(lg))
        if t == EOS:
            break
        toks.append(t)
        logits, _h = model(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
        lg = logits[0, -1]
    states = {}
    for i, entry in enumerate(cache):
        try:
            st = entry[1]
        except (TypeError, IndexError, AttributeError):
            continue
        if st is None or not hasattr(st, "shape"):
            continue
        if tuple(st.shape) != (1, 32, 128, 128):
            continue
        mx.eval(st)
        states[i] = np.frombuffer(bytes(bytearray(st)), np.uint8).copy()
    return toks, states


def main():
    lm, tok = load(MODEL)
    pid = build_prompt_ids(tok, 14)
    chain_toks, chain_st = run_pass(lm, tok, pid, fuse=False)
    fused_toks, fused_st = run_pass(lm, tok, pid, fuse=True)
    stream_match = chain_toks == fused_toks
    print(f"STREAM-IDENTICAL={stream_match} n={len(chain_toks)}")
    print("layer  max_abs     mean_abs    max_rel     a_max")
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    for layer in sorted(chain_st):
        a = np.frombuffer(chain_st[layer], np.float32)
        b = np.frombuffer(fused_st[layer], np.float32)
        diff = np.abs(a - b)
        rel = diff / (np.abs(a) + 1e-6)
        ma = float(diff.max())
        mr = float(rel.max())
        overall_max_abs = max(overall_max_abs, ma)
        overall_max_rel = max(overall_max_rel, mr)
        print(f"{layer:5d}  {ma:.6f}  {float(diff.mean()):.6f}  {mr:.6f}  {float(np.abs(a).max()):.3f}")
    print(f"OVERALL max_abs={overall_max_abs:.6f} max_rel={overall_max_rel:.6f}")
    print("STATE-DIFF-DONE")


if __name__ == "__main__":
    main()
