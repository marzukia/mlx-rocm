"""K2 dispatch probe: N decode steps on a warm model, for rocprofv3 kernel-sum.

Usage: k2-dispatch-probe.py MODEL_PATH N_STEPS
Prints JSON {n_steps, step_ms}. Run under:
  LD_LIBRARY_PATH=... rocprofv3 --collect kernel-sum python k2-dispatch-probe.py M N
"""

import json
import sys
import time

import mlx.core as mx
from mlx_lm import generate, load

mx.set_default_device(mx.gpu)


def main() -> None:
    model_path = sys.argv[1]
    n_steps = int(sys.argv[2])

    model, tokenizer = load(model_path)
    generate(model, tokenizer, prompt="Hello", max_tokens=16, verbose=False)
    mx.clear_cache()

    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    y = mx.array([100])
    for _ in range(5):
        logits = model(y[None], cache=cache)
        mx.eval(logits)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        logits = model(y[None], cache=cache)
        mx.eval(logits)
    t1 = time.perf_counter()
    print(
        json.dumps(
            {
                "n_steps": n_steps,
                "step_ms": round((t1 - t0) / n_steps * 1000, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
