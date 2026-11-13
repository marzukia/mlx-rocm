import os
# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    gather_last_axis,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update, normalize_qk


# MTP_INJECT_PATCH: MTP drafter module (mlx-lm PR #990 MTPModule, vendored via
# vllm_mlx/spec_decode/mtp/head.py, verbatim upstream). Predicts token t+2 from
# (prenorm hidden at t, embedding of token t+1); shared lm_head applies after.
class MTPDecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, quant: bool = False):
        super().__init__()
        self.self_attn = Attention(args)
        if quant:
            a = self.self_attn
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                old = getattr(a, n)
                setattr(a, n, nn.QuantizedLinear(old.weight.shape[1], old.weight.shape[0], bias=False, bits=int(os.environ.get("MTP_QBITS", "4"))))
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if getattr(args, "num_experts", 0) > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            m = MLP(args.hidden_size, args.intermediate_size)
            if quant:
                for n in ("gate_proj", "up_proj", "down_proj"):
                    old = getattr(m, n)
                    nl = nn.QuantizedLinear(old.weight.shape[1], old.weight.shape[0], bias=False, bits=int(os.environ.get("MTP_QBITS", "4")))
                    setattr(m, n, nl)
            self.mlp = m

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class MTPModule(nn.Module):
    # MTP6_MOD_BF16 (W1 phase 1, change B): one-shot lazy-cast guard
    # (class-level: one mtp instance per model, one model per process).
    _MTP6_BF16_CAST_DONE = False

    def __init__(self, args: TextModelArgs, n_layers: int, quant: bool = False):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if quant:
            self.fc = nn.QuantizedLinear(args.hidden_size * 2, args.hidden_size, bias=False, bits=int(os.environ.get("MTP_QBITS", "4")))
        else:
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPDecoderLayer(args, quant=quant) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden_states: mx.array, embedding: mx.array, cache=None):
        # MTP6_MOD_BF16: one-time lazy cast on first use (after load's
        # mx.eval has pinned the trunk array layout on the f32 sidecar
        # state - the load-site cast placement is layout-perturbing, see
        # cast_bf16).
        if (
            os.environ.get("MTP6_MOD_BF16", "0") == "1"
            and not MTPModule._MTP6_BF16_CAST_DONE
        ):
            MTPModule._MTP6_BF16_CAST_DONE = True
            self.cast_bf16()
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.reshape(1, 1, -1)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, 1, -1)
        e = self.pre_fc_norm_embedding(embedding)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(mx.concatenate([e, h], axis=-1))
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(fused, cache[0])
        for layer, c in zip(self.layers, cache):
            fused = layer(fused, mask, c)
        return self.norm(fused)

    def cast_bf16(self):
        # MTP6_MOD_BF16 (W1 phase 1, change B): cast all 15 sidecar f32
        # module keys to bf16 (module weights bf16 at first use). LAZY
        # placement is load-bearing (measured 11-12): the load-site
        # (sanitize) cast changes the heap layout before the trunk arrays
        # are pinned by mx.eval -> the trunk numerics perturb (prompt-
        # forward logits non-deterministic under the bf16 sidecar layout).
        # Post-eval: trunk arrays keep their addresses; the 15 module
        # leaves are rebound to fresh bf16 arrays (f32 originals freed).
        # DEFAULT OFF (no call site) -> module stays f32, byte-identical
        # base behavior. The MTP_4BIT path (quant=True) is untouched (B'
        # out of scope). Change C (MTP6_SLOT_SKIPCS) is PARKED for this
        # session (cloudy ruling 01:58Z).
        cast = 0

        def walk(m):
            nonlocal cast
            if isinstance(m, (list, tuple)):
                for v in m:
                    if isinstance(v, (nn.Module, list, tuple, mx.array)):
                        walk(v)
                return
            for k, v in m.items():
                if isinstance(v, mx.array):
                    setattr(m, k, v.astype(mx.bfloat16))
                    cast += 1
                elif isinstance(v, (nn.Module, list, tuple)):
                    walk(v)

        walk(self)
        mx.eval(self.parameters())
        return cast

GDN_FUSE_CALLS = 0  # F-branch fused-op dispatch counter
GDN_TAPE_CALLS = 0  # TAPE_BRANCH_PATCH dispatch counter
GDN_SLOT_CALLS = 0  # SLOT_BRANCH_PATCH (Phase B) dispatch counter
GDN_SLOT2_CALLS = 0  # SLOT2_BRANCH_PATCH (R1b, the fused sequential slot design) dispatch counter
GDN_SEQ_CALLS = 0  # SEQ_BRANCH_PATCH (Phase 2, the sequential GDN kernel - the llama.cpp GDN design port) dispatch counter
from .pipeline import PipelineMixin
from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # MTP fields (MTP_INJECT_PATCH)
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        if os.environ.get("MLX_ROCM_W1_INBA_FUSE", "0") == "1":
            fused = getattr(self, "_inba_fused", None)
            if fused is None:
                fused = nn.QuantizedLinear(
                    self.hidden_size,
                    2 * self.num_v_heads,
                    bits=self.in_proj_b.bits,
                    group_size=self.in_proj_b.group_size,
                    bias=False,
                )
                fused.weight = mx.concatenate(
                    [self.in_proj_b.weight, self.in_proj_a.weight], axis=0
                )
                fused.scales = mx.concatenate(
                    [self.in_proj_b.scales, self.in_proj_a.scales], axis=0
                )
                # W1 ZP-FIX 2026-11-13 (cloudy ruling, axis read): the
                # QuantizedLinear constructor's group zero-points come from
                # mx.quantize of an mx.random.uniform weight = per-process
                # RANDOM. The fused GEMM must use the model's true
                # zero-points or its output is non-deterministic per process.
                fused.biases = mx.concatenate(
                    [self.in_proj_b.biases, self.in_proj_a.biases], axis=0
                )
                self._inba_fused = fused
            ba = fused(inputs)
            b = mx.contiguous(ba[..., : self.num_v_heads])
            a = mx.contiguous(ba[..., self.num_v_heads:])
            if os.environ.get("MLX_W1_INBA_DUMP", "0") == "1" and getattr(self, "_layer_idx", -1) == 0:
                import numpy as _np
                b_ref = self.in_proj_b(inputs)
                a_ref = self.in_proj_a(inputs)
                _dump_n = getattr(self, "_inba_dump_n", 0)
                self._inba_dump_n = _dump_n + 1
                if _dump_n < 4:
                    _fn = "/tmp/inba_l0_%d_%s.npz" % (_dump_n, os.environ.get("W1_DUMP_TAG", "x"))
                    _d = lambda x: _np.asarray(mx.astype(x, mx.float32))
                    _np.savez(_fn, inp=_d(inputs), ba=_d(ba), b=_d(b),
                              a=_d(a), b_ref=_d(b_ref), a_ref=_d(a_ref))
        else:
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        # SEQ_BRANCH_PATCH (Phase 2, the sequential GDN kernel - the llama.cpp GDN design port):
        # gdn_fused_step_seq for 2<=S<=4 = the slot op contract (per-row
        # state slots for the commit rows 0..S-2; row S-1 is state_new) +
        # the conv_state_new output (the slot2 launch integration rides
        # along: this branch consumes it for the conv-state cache update
        # instead of dispatching concatenate + take_along_axis: launch
        # integration, -2 ops per GDN layer per verify) on the
        # COLUMN-PARALLEL sequential kernel pair (2 launches per GDN layer
        # per step; the attn intermediate is op-internal). MLX_ROCM_GDN_SEQ=1,
        # DEFAULT OFF -> zero behavior change. Runs before the conv-state
        # update and the SLOT2/SLOT/TAPE/F branches when selected.
        # last_tape is set to the (state_slot, conv_slot) pair,
        # row-indexed like the tape, so the bench commit() rebind code is
        # unchanged. S=1 falls through to the F-branch (the seq op has no
        # commit rows at T=1).
        _seq_state = cache[1] if cache is not None else None
        if (os.environ.get("MLX_ROCM_GDN_SEQ", "0") == "1"
                and os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1"
                and 2 <= S <= 4 and B == 1 and _seq_state is not None):
            global GDN_SEQ_CALLS
            GDN_SEQ_CALLS += 1
            q3 = qkv[:, :, : self.key_dim]
            k3 = qkv[:, :, self.key_dim : 2 * self.key_dim]
            v3 = qkv[:, :, 2 * self.key_dim :]
            out, seq_state, state_slot, conv_slot, conv_state_new = (
                mx.fast.gdn_fused_step_seq(
                    q3, k3, v3, b, a, self.A_log, self.dt_bias,
                    conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                    self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size),
                    z, _seq_state, self.norm.weight,
                    inv_scale=self.head_k_dim**-0.5,
                    prep_eps=1e-6,
                    norm_eps=self.layer_norm_epsilon,
                )
            )
            if cache is not None:
                cache[0] = conv_state_new.reshape(
                    B, self.conv_kernel_size - 1, self.conv_dim)
                cache[1] = seq_state
                cache.advance(S)
            self.last_tape = (state_slot, conv_slot)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        # SLOT2_BRANCH_PATCH (R1b, the fused sequential slot design):
        # gdn_fused_step_slot2 for 2<=S<=4 = the slot op (per-row state
        # slots for the commit rows 0..S-2; row S-1 is state_new) on the
        # NON-UNROLLED slot2 kernel (one per-row phase for all M rows; the
        # per-row re-execution of the unrolled 4x phase copies is amortized
        # into one pass over the rows; the per-thread weight loads are
        # loop-invariant and hoisted by LICM), plus the conv_state_new
        # output (the post-pass conv state, a pure selection over the input
        # window and the raw rows) which this branch consumes for the
        # conv-state cache update instead of dispatching concatenate +
        # take_along_axis (launch integration: -2 ops per GDN layer per
        # verify). MLX_ROCM_GDN_SLOT2=1, DEFAULT OFF -> zero behavior change.
        # Runs before the conv-state update and the SLOT/TAPE/F branches
        # when selected. last_tape is set to the (state_slot, conv_slot)
        # pair, row-indexed like the tape, so the bench commit() rebind
        # code is unchanged. S=1 falls through to the F-branch (the slot2
        # op has no commit rows at T=1).
        _state2 = cache[1] if cache is not None else None
        if (os.environ.get("MLX_ROCM_GDN_SLOT2", "0") == "1"
                and os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1"
                and 2 <= S <= 4 and B == 1 and _state2 is not None):
            global GDN_SLOT2_CALLS
            GDN_SLOT2_CALLS += 1
            q3 = qkv[:, :, : self.key_dim]
            k3 = qkv[:, :, self.key_dim : 2 * self.key_dim]
            v3 = qkv[:, :, 2 * self.key_dim :]
            out, state2, state_slot, conv_slot, conv_state_new = (
                mx.fast.gdn_fused_step_slot2(
                    q3, k3, v3, b, a, self.A_log, self.dt_bias,
                    conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                    self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size),
                    z, _state2, self.norm.weight,
                    inv_scale=self.head_k_dim**-0.5,
                    prep_eps=1e-6,
                    norm_eps=self.layer_norm_epsilon,
                )
            )
            if cache is not None:
                cache[0] = conv_state_new.reshape(
                    B, self.conv_kernel_size - 1, self.conv_dim)
                cache[1] = state2
                cache.advance(S)
            self.last_tape = (state_slot, conv_slot)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        # SLOT_BRANCH_PATCH (Phase B, the re-forward remover):
        # gdn_fused_step_slot for 2<=S<=4 (emits per-row state slots for the
        # commit rows 0..S-2; row S-1 is state_new). MLX_ROCM_GDN_SLOT=1,
        # DEFAULT OFF -> zero behavior change. Runs before the tape/F
        # branches when selected. last_tape is set to the (state_slot,
        # conv_slot) pair, row-indexed like the tape, so the bench
        # commit() rebind code is unchanged. S=1 falls through to the
        # F-branch (the slot op has no commit rows at T=1).
        _state = cache[1] if cache is not None else None
        if (os.environ.get("MLX_ROCM_GDN_SLOT", "0") == "1"
                and os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1"
                and 2 <= S <= 4 and B == 1 and _state is not None):
            global GDN_SLOT_CALLS
            GDN_SLOT_CALLS += 1
            q3 = qkv[:, :, : self.key_dim]
            k3 = qkv[:, :, self.key_dim : 2 * self.key_dim]
            v3 = qkv[:, :, 2 * self.key_dim :]
            state = cache[1] if cache is not None else None
            out, state, state_slot, conv_slot = (
                mx.fast.gdn_fused_step_slot(
                    q3, k3, v3, b, a, self.A_log, self.dt_bias,
                    conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                    self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size),
                    z, state, self.norm.weight,
                    inv_scale=self.head_k_dim**-0.5,
                    prep_eps=1e-6,
                    norm_eps=self.layer_norm_epsilon,
                )
            )
            if cache is not None:
                cache[1] = state
                cache.advance(S)
            self.last_tape = (state_slot, conv_slot)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        # TAPE_BRANCH_PATCH: gdn_fused_step_tape for S<=4 (emits per-row state
        # tapes; MLX_ROCM_GDN_TAPE=1, DEFAULT OFF -> zero behavior change). Runs
        # before the F-branch so the tape op (a superset) is used when selected.
        # STATE_NONE_GUARD: the fused/tape op pybinds take a non-optional
        # state; on the first S<=4 forward of a fresh cache (state is
        # None) fall through to the default chain (which handles a None
        # state) instead of crashing on the pybind conversion.
        if (os.environ.get("MLX_ROCM_GDN_TAPE", "0") == "1"
                and os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1"
                and S <= 4 and B == 1 and _state is not None):
            global GDN_TAPE_CALLS
            GDN_TAPE_CALLS += 1
            q3 = qkv[:, :, : self.key_dim]
            k3 = qkv[:, :, self.key_dim : 2 * self.key_dim]
            v3 = qkv[:, :, 2 * self.key_dim :]
            state = cache[1] if cache is not None else None
            out, state, state_tape, conv_state_tape = (
                mx.fast.gdn_fused_step_tape(
                    q3, k3, v3, b, a, self.A_log, self.dt_bias,
                    conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                    self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size),
                    z, state, self.norm.weight,
                    inv_scale=self.head_k_dim**-0.5,
                    prep_eps=1e-6,
                    norm_eps=self.layer_norm_epsilon,
                )
            )
            if cache is not None:
                cache[1] = state
                cache.advance(S)
            self.last_tape = (state_tape, conv_state_tape)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        # FUSE_BRANCH_PATCH: single gdn_fused_step for S<=4 decode/verify (81-build F-branch port, T=1..4 kernel)
        if (os.environ.get("MLX_ROCM_GDN_FUSE", "0") == "1" and S <= 4
                and B == 1 and _state is not None):
            global GDN_FUSE_CALLS
            GDN_FUSE_CALLS += 1
            q3 = qkv[:, :, : self.key_dim]
            k3 = qkv[:, :, self.key_dim : 2 * self.key_dim]
            v3 = qkv[:, :, 2 * self.key_dim :]
            state = cache[1] if cache is not None else None
            out, state, conv_state_new = mx.fast.gdn_fused_step(
                q3, k3, v3, b, a, self.A_log, self.dt_bias,
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size),
                z, state, self.norm.weight,
                inv_scale=self.head_k_dim**-0.5,
                prep_eps=1e-6,
                norm_eps=self.layer_norm_epsilon,
            )
            if cache is not None:
                # conv state from the op (graph-decode: the op writes it
                # in-place on the conv-state buffer; the python-side
                # mx.contiguous update above is shadowed by this)
                cache[0] = conv_state_new.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
                cache[1] = state
                cache.advance(S)
            out = self.out_proj(out.reshape(B, S, -1))
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            return out
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
            self.linear_attn._layer_idx = layer_idx
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_prenorm: bool = False,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        # PRENORM_PATCH: expose pre-final-norm hidden for MTP drafter
        post = self.norm(hidden_states)
        if return_prenorm:
            return hidden_states, post
        return post


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        # MTP_INJECT_PATCH: build the MTP drafter module when the config asks for it
        mtp_n = getattr(args, "mtp_num_hidden_layers", 0) or 0
        # MTP_4BIT=1 -> quantized MTP module (model-mtp-4bit.safetensors sidecar)
        _q4 = os.environ.get("MTP_4BIT", "0") == "1"
        self.mtp = MTPModule(args, mtp_n, quant=_q4) if mtp_n > 0 else None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
        return_prenorm: bool = False,
    ) -> mx.array:
        out = self.model(
            inputs, cache, input_embeddings=input_embeddings,
            return_prenorm=return_prenorm,
        )
        hidden_pre = None
        if return_prenorm:
            hidden_pre, out = out
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
            if self.sharding_group is not None:
                out = gather_last_axis(out, self.sharding_group)
        if return_hidden:  # PRENORM_PATCH
            return out, hidden_pre
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        # MTP_INJECT_PATCH: keep mtp.* keys when the module exists so they load
        if getattr(self, "mtp", None) is None:
            weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # Each rank computes a slice of the vocabulary
        lm = self.language_model
        if not lm.args.tie_word_embeddings:
            vocab = lm.args.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            lm.lm_head = shard_linear(lm.lm_head, "all-to-sharded", group=group)
            lm.sharding_group = group

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
