#!/usr/bin/env python3
"""BIT-EXACT A/B GATE: t1858 rows-msg decode-hoist kernel vs the base wheel.

Runs the M=2..4 QMM rows path (the op the t1858 kernel patch changes; the
M=1 per-row path is untouched) on fixed-seed QuantizedLinear shapes and
prints md5 of each output. Run once in venv-r4v (base wheel) and once in
venv-kfix (patched wheel); the outputs must md5-match exactly (row-exact
contract: output row r == quantized_matmul(x[r:r+1]) bit for bit). 4-bit
g128, f32 scales (the model path: the op promotes bf16 x to f32), with and
without bias, bf16 x input (promotion exercised), M in {2,3,4}, N/K from
the real model shapes (MLP 12288/4096 + 4096/12288, GDN in/out 8192-32/4096,
lm_head 248320/4096, small-N 32/16).
"""

import hashlib
import os

import mlx.core as mx
import mlx.nn as mxnn
import numpy as np

os.environ.setdefault("MLX_ROCM_QMV_ROWS", "1")
os.environ.setdefault("MLX_ROCM_QMV_ROWS_MIN_N", "1")


def tag_bytes(tag, arr):
    h = hashlib.md5(np.asarray(arr.astype(mx.float32)).tobytes())
    mx_max = float(mx.max(arr))
    print(f"{tag} md5={h.hexdigest()}", tuple(arr.shape), f"{mx_max:.6f}", flush=True)


def one(in_f, out_f, M, with_bias, rep_seed):
    mx.random.seed(rep_seed)
    lin = mxnn.QuantizedLinear(in_f, out_f, bias=with_bias, bits=4, group_size=128)
    mx.random.seed(rep_seed + 7)
    x = mx.random.normal((M, in_f))  # bf16 input; f32 scales promote x in-op
    y = lin(x)
    tag_bytes(f"M{M}_B{int(with_bias)}_N{out_f}_K{in_f}", y)


def main():
    for rep_seed in (0, 1):
        for M in (2, 3, 4):
            for in_f, out_f in [
                (4096, 12288),  # mlp gate/up
                (12288, 4096),  # mlp down
                (4096, 8192),  # gdn in_proj_qkv
                (4096, 4096),  # gdn in_proj_z / out_proj / attn
                (4096, 1024),  # k/v proj
                (4096, 32),  # gdn in_proj_b
                (4096, 16),  # gdn in_proj_a
                (4096, 248320),  # lm_head
            ]:
                one(in_f, out_f, M, True, rep_seed)
                one(in_f, out_f, M, False, rep_seed)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
