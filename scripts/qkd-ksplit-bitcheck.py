#!/usr/bin/env python3
"""t2050 qgd-ksplit bit-identity + correctness check (qmv_gdn class).

One process per arm (the parent session script relaunches this file with
the arm env; the kernel dispatch reads MLX_ROCM_QGD_KSPLIT once per
process, so the stock and ksplit arms cannot coexist in one process):

  MODE=stock   (MLX_ROCM_QGD_KSPLIT unset) - the stock tiled path must run
               verbatim. Prints one JSON line per op: name, sha256 of the
               exact output values, first 4 values, and the full value
               list (for the compare mode).
  MODE=ksplit  (MLX_ROCM_QGD_KSPLIT=1, REF_FILE=<stock jsonl>) - same
               print; each line is annotated with the same-named stock
               hash (hash_match).
  MODE=compare REF_FILE=<stock jsonl> NEW_FILE=<ksplit jsonl> - per op:
               hash_match, maxdiff, maxrel, ndiff; asserts the unset-env
               guarantee (stock vs stock re-run hash equal - the parent
               runs two stock processes) and the K = 1024 fallback shape
               (K/4 = 256 < 512 -> the gate falls back to stock, so it
               must be bit-identical even with the env set).

Ops: random (fixed seed 42, q4 group-64, bf16 x, no bias - the class
config) over the chartered K = 8192 shapes + neighbors + the fallback
K = 1024 shape, plus the real GDN projection weight tensors of the 9B
model (first bias-free 4-bit group-64 QuantizedLinears, up to 6, random
x fixed seed).
"""

import hashlib
import json
import os
import sys

import mlx.core as mx


def quant_op(x, w):
    wq, s, b = mx.quantize(w, 64, 4)
    return mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)


def to_float_list(y):
    return y.astype(mx.float32).reshape(-1).tolist()


def op_line(name, y, refs=None, include_vals=False):
    vals = to_float_list(y)
    h = hashlib.sha256(json.dumps([repr(v) for v in vals]).encode("utf-8")).hexdigest()
    line = {"name": name, "hash": h, "first": [round(v, 6) for v in vals[:4]]}
    if include_vals:
        line["vals"] = vals
    if refs is not None and name in refs:
        line["ref_hash"] = refs[name]["hash"]
        line["hash_match"] = refs[name]["hash"] == h
    return line


def gen_ops(model_path):
    mx.random.seed(42)
    for K, N in [
        (8192, 8192),
        (8192, 4096),
        (8192, 2048),
        (8192, 1024),
        (8192, 32),
        (2048, 8192),
        (1024, 8192),
        (1024, 32),
    ]:
        x = mx.random.uniform(-1.0, 1.0, (1, K)).astype(mx.bfloat16)
        w = mx.random.uniform(-1.0, 1.0, (N, K))
        yield f"rand K={K} N={N}", quant_op(x, w)
    if model_path:
        import mlx.nn as mnn
        from mlx_lm import load

        model, _ = load(model_path)
        # mlx_lm Qwen3: model.language_model.model.layers (diag-s62 pattern).
        layers = None
        for attr in ("model", ""):
            sub = getattr(model.language_model, attr, None) if attr else model.language_model
            if sub is not None and hasattr(sub, "layers"):
                layers = sub.layers
                break
        if layers is None:
            print("WARN: no .layers under language_model", file=sys.stderr)
            return
        layer = layers[0]
        # mlx nn.Module is a mapping here (submodules are not in __dict__
        # or dir()); walk with the .items() mapping API, bounded depth.
        found = []

        def walk(obj, prefix, depth):
            if depth > 3:
                return
            for name, m in obj.items():
                if isinstance(m, mnn.QuantizedLinear):
                    found.append((prefix + name, m))
                elif isinstance(m, mnn.Module):
                    walk(m, prefix + name + ".", depth + 1)

        walk(layer, "", 0)
        seen = set()
        uniq = []
        for name, mod in found:
            if id(mod) not in seen:
                seen.add(id(mod))
                uniq.append((name, mod))
        found = uniq
        for name, lin in found[:6]:
            if (int(lin.bits), int(lin.group_size)) != (4, 64):
                continue
            # 4-bit pack: K values = weight cols * 32 // bits (diag-s62 pattern).
            K = int(lin.weight.shape[1] * 32 // lin.bits)
            x = mx.random.uniform(-1.0, 1.0, (1, K)).astype(mx.bfloat16)
            yield (
                f"model {name} K={K}",
                mx.quantized_matmul(
                    x,
                    lin.weight,
                    lin.scales,
                    lin.biases,
                    group_size=int(lin.group_size),
                    bits=int(lin.bits),
                ),
            )
        if not found:
            print("WARN: no QuantizedLinear found under layer 0", file=sys.stderr)


def read_jsonl(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                d_ = json.loads(line)
                d[d_["name"]] = d_
    return d


def main():
    mode = os.environ.get("MODE", "stock")
    mx.random.seed(42)
    if mode in ("stock", "ksplit"):
        refs = {}
        if mode == "ksplit":
            refs = read_jsonl(os.environ["REF_FILE"])
        for name, y in gen_ops(os.environ.get("MTP_MODEL", "")):
            line = op_line(
                name,
                y,
                refs=refs or None,
                include_vals=(mode in ("stock", "ksplit")),
            )
            print(json.dumps(line))
        print(f"OPS_DONE mode={mode}")
    elif mode == "compare":
        ref = read_jsonl(os.environ["REF_FILE"])
        new = read_jsonl(os.environ["NEW_FILE"])
        ok = True
        for name in sorted(set(ref) | set(new)):
            r, n = ref.get(name), new.get(name)
            if r is None or n is None:
                print(f"COMPARE name={name} MISSING side")
                ok = False
                continue
            if "vals" not in r or "vals" not in n:
                print(f"COMPARE name={name} no vals (hash only)")
                continue
            rv, nv = r["vals"], n["vals"]
            ndiff = sum(1 for a, b in zip(rv, nv, strict=False) if a != b)
            maxd = max((abs(a - b) for a, b in zip(rv, nv, strict=False)), default=0.0)
            mag = max((abs(v) for v in nv), default=0.0)
            print(
                f"COMPARE name={name} hash_match="
                f"{r['hash'] == n['hash']} ndiff={ndiff}/{len(nv)} "
                f"maxdiff={maxd:.3e} maxrel={maxd / mag if mag else 0:.3e}"
            )
            if "K=1024" in name and r["hash"] != n["hash"]:
                print(f"FALLBACK-NOT-IDENTICAL name={name}")
                ok = False
        print(f"COMPARE_DONE ok={ok}")
    else:
        raise SystemExit(f"bad MODE {mode}")


if __name__ == "__main__":
    main()
