#!/usr/bin/env python3
"""K2: apply non-GEMM fusion patches to mlx_lm model files in a venv.

Usage: python3 patch-models-k2.py /path/to/venv

Patches (idempotent):
  1. cache.py: KVCache.update_and_fetch_fused (one fused K+V append kernel)
  2. qwen3_next.py: Attention uses update_and_fetch_fused + compiled gated output
  3. qwen3_5.py: DecoderLayer fuses residual add + post-attention RMSNorm;
     GatedDeltaNet conv state stays a view (no contiguous copy)
  4. qwen3.py: TransformerBlock fuses residual add + post-attention RMSNorm;
     Attention uses update_and_fetch_fused
"""

import sys
from pathlib import Path


def patch(path: Path, old: str, new: str, tag: str) -> bool:
    s = path.read_text()
    if tag == "SKIP" or new in s:
        return False
    if old not in s:
        raise SystemExit(f"PATCH FAIL ({tag}): pattern not found in {path}\n---\n{old}\n---")
    path.write_text(s.replace(old, new, 1))
    return True


def main() -> None:
    venv = Path(sys.argv[1])
    models = venv / "lib" / "python3.12" / "site-packages" / "mlx_lm" / "models"

    # ---------------------------------------------------------------- cache.py
    p = models / "cache.py"
    patch(
        p,
        """        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    def size(self):
        return self.offset""",
        """        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def update_and_fetch_fused(self, keys, values):
        \"\"\"K2: fused in-place KV append (one kernel for K+V via mx.rocm.kv_append).\"\"\"
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        if keys.shape[2] == 1:
            self.keys, self.values = mx.rocm.kv_append(
                keys, values, self.keys, self.values, prev
            )
        else:
            self.keys[..., prev : self.offset, :] = keys
            self.values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    def size(self):
        return self.offset""",
        "cache.py: KVCache.update_and_fetch_fused",
    )

    # ---------------------------------------------------------- qwen3_next.py
    p = models / "qwen3_next.py"
    patch(
        p,
        """from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU""",
        """from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@partial(mx.compile, shapeless=True)
def gated_output(output, gate):
    # K2: fuse output * sigmoid(gate) (2 kernels -> 1)
    return output * mx.sigmoid(gate)""",
        "qwen3_next.py: gated_output",
    )
    # the file needs `from functools import partial`
    patch(
        p,
        """# Copyright \u00a9 2025 Apple Inc.

from __future__ import annotations""",
        """# Copyright \u00a9 2025 Apple Inc.

from __future__ import annotations

from functools import partial""",
        "qwen3_next.py: partial import",
    )
    patch(
        p,
        """        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))""",
        """        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch_fused(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(gated_output(output, gate))""",
        "qwen3_next.py: Attention fused kv + gated output",
    )

    # ------------------------------------------------------------ qwen3_5.py
    p = models / "qwen3_5.py"
    patch(
        p,
        """        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out""",
        """        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        # K2: fused residual add + post-attention RMSNorm (1 kernel, not 2)
        p_norm = mx.rocm.rms_norm_residual(
            x,
            r,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        out = x + r + self.mlp(p_norm)
        return out""",
        "qwen3_5.py: DecoderLayer fused rms_norm_residual",
    )
    patch(
        p,
        """            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])""",
        """            else:
                # K2: keep the conv state as a view (no contiguous copy);
                # the next step's mx.concatenate materializes it.
                cache[0] = conv_input[:, -n_keep:, :]""",
        "qwen3_5.py: GatedDeltaNet conv state view",
    )

    # ------------------------------------------------------------- qwen3.py
    p = models / "qwen3.py"
    patch(
        p,
        """        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out""",
        """        r = self.self_attn(self.input_layernorm(x), mask, cache)
        # K2: fused residual add + post-attention RMSNorm (1 kernel, not 2)
        p_norm = mx.rocm.rms_norm_residual(
            x,
            r,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        out = x + r + self.mlp(p_norm)
        return out""",
        "qwen3.py: TransformerBlock fused rms_norm_residual",
    )
    patch(
        p,
        """        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)""",
        """        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch_fused(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)""",
        "qwen3.py: Attention fused kv append",
    )

    print("all patches applied")


if __name__ == "__main__":
    main()
