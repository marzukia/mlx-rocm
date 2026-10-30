#!/usr/bin/env python3
"""Phase 2 GDN-fuse: patch the venv's mlx_lm qwen3_5.py with the F branch.

Adds:
  * a module-level GDN_FUSE_CALLS counter (read by the bench for the
    non-vacuity gate)
  * in the tail-fusion decode branch (S == 1), an early-return F branch
    selected by MLX_ROCM_GDN_FUSE=1: one mx.fast.gdn_fused_step call
    (conv+silu, beta, g, qk norm, recurrent state update, norm_gated)
    replacing the gdn_prep + gated_delta_update + gdn_norm_gated chain.
    The F branch mirrors the chain's cache/state handling and out_proj
    exactly, so the two paths differ only in the GDN op itself.

Idempotent (skips if the marker is present). Anchors are the run-827
tail-fusion qwen3_5.py from the run-1012 wheel.
"""

import sys
from pathlib import Path


def main() -> None:
    venv = Path(sys.argv[1])
    p = venv / "lib" / "python3.12" / "site-packages" / "mlx_lm" / "models" / "qwen3_5.py"
    src = p.read_text()
    if "GDN_FUSE_CALLS" in src:
        print(f"{p.name}: already patched, skipping")
        return

    # 1. module-level counter after the gated_delta import
    anchor = "from .gated_delta import gated_delta_update, normalize_qk"
    assert anchor in src, "import anchor missing"
    src = src.replace(
        anchor,
        anchor + "\n\nGDN_FUSE_CALLS = 0  # phase2 fused-op dispatch counter",
        1,
    )

    # 2. F branch: early return before the chain (no reindent of the chain)
    old = """            qv = qkv.reshape(B, S, -1)
            q = qv[:, :, : self.key_dim]
            k = qv[:, :, self.key_dim : 2 * self.key_dim]
            v = qv[:, :, 2 * self.key_dim :]
            qn, kn, vc, beta, g = mx.fast.gdn_prep("""
    new = """            qv = qkv.reshape(B, S, -1)
            q = qv[:, :, : self.key_dim]
            k = qv[:, :, self.key_dim : 2 * self.key_dim]
            v = qv[:, :, 2 * self.key_dim :]
            if os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1":
                global GDN_FUSE_CALLS
                GDN_FUSE_CALLS += 1
                out, state = mx.fast.gdn_fused_step(
                    q,
                    k,
                    v,
                    b,
                    a,
                    self.A_log,
                    self.dt_bias,
                    conv_state.reshape(
                        self.conv_kernel_size - 1, self.conv_dim
                    ),
                    self.conv1d.weight.reshape(
                        self.conv_dim, self.conv_kernel_size
                    ),
                    z,
                    state,
                    self.norm.weight,
                    inv_scale=self.head_k_dim**-0.5,
                    prep_eps=1e-6,
                    norm_eps=self.layer_norm_epsilon,
                )
                if cache is not None:
                    cache[1] = state
                    cache.advance(S)
                out = self.out_proj(out.reshape(B, S, -1))
                if self.sharding_group is not None:
                    out = mx.distributed.all_sum(out, group=self.sharding_group)
                return out
            qn, kn, vc, beta, g = mx.fast.gdn_prep("""
    assert old in src, "tail-fusion anchor missing"
    src = src.replace(old, new, 1)
    p.write_text(src)
    print(f"{p.name}: F branch + counter patched")


if __name__ == "__main__":
    main()
