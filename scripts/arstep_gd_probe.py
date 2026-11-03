# AR step change round 2: graph-decode probe.
#
# Mirrors the canonical mtp-bench6.py MTP=0 arm's per-step structure (one
# 1-token forward per step, argmax on host, persistent cache) on a SHORT
# window, so the graph-decode pilot arms (prefill chunk + two small stable
# steps), captures, and replays N steps. Prints ALLTOKS + the pilot's own
# [g1] log on stderr.
#
# Env: GPROBE_N (default 24) - decode steps in the measured window.
#      MLX_ROCM_GRAPH_DECODE=1 arms the graph (pilot gate); unset = eager.
import os
import time

import mlx.core as mx
import mlx_lm as mlxlm

PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
N = int(os.environ.get("GPROBE_N", "24"))
HID = os.environ.get("GPROBE_HID", "") != ""


def main():
    t0 = time.time()
    model, tok = mlxlm.load(os.path.join(os.environ.get("MODELS", "<models-mount>"), "qwen3-9b-mlx"))
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()

    res = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    t = int(mx.argmax(res[0][0, -1]))
    t2 = time.time()
    out = []
    for i in range(N):
        out.append(t)
        print(f"PYTOK step={i + 1} tok={t}", flush=True)
        res = lm(mx.array([[t]]), cache=cache, return_hidden=True, return_prenorm=True)
        t = int(mx.argmax(res[0][0, -1]))
        print(f"PYTOK step={i + 1} next={t}", flush=True)
        if HID:
            h = res[0][0, -1]
            p = res[1][0, -1]
            hn = float(mx.linalg.norm(h))
            pn = float(mx.linalg.norm(p))
            hv = [round(float(x), 3) for x in mx.array(h)[:8].tolist()]
            print(f"HID step={i + 1} hnorm={hn:.4f} pnorm={pn:.4f} h0_8={hv}", flush=True)
    t3 = time.time()
    print(
        f"PROBE N={N} LOAD={t_load:.2f} "
        f"DECODE={t3 - t2:.3f} MS/STEP={(t3 - t2) * 1000.0 / N:.2f} "
        f"TOKS_S={N / (t3 - t2):.2f}",
        flush=True,
    )
    print("FIRST20:", " ".join(map(str, out[:20])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)


if __name__ == "__main__":
    main()
