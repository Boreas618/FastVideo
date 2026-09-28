# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native MiniMax H3 joint audio-video diffusion transformer."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo import envs
from fastvideo.attention import DistributedAttention
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.selector import get_attn_backend
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.mxfp8_config import MXFP8QuantizeMethod
from fastvideo.layers.mlp import MLP
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.minimax_h3_fusions import (
    HAVE_TRITON,
    fused_qknorm_rope,
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
    minimax_h3_swiglu,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.profiler import nvtx_range
from fastvideo.utils import get_compute_dtype

logger = init_logger(__name__)

MINIMAX_H3_MODALITY_NUM = 3
_CFG = MiniMaxH3Config()
_MINIMAX_H3_FUSION_NAMES = frozenset({"modulate", "qknorm_rope", "swiglu"})
# Checkpoint tensors of the per-block and final AdaLN projections. Group 2 is
# the block index; it is absent for ``norm_out``.
_ADALN_PROJECTION_KEY = re.compile(r"^(transformer_blocks\.(\d+)\.adaln_proj|norm_out)\.linear\.(weight|bias)$")


def _enabled_minimax_h3_fusions(value: str | None = None) -> frozenset[str]:
    """Parse the independently switchable inference fusion set."""
    raw = envs.FASTVIDEO_MINIMAX_H3_FUSIONS if value is None else value
    normalized = raw.strip().lower()
    if normalized in {"", "0", "none"}:
        return frozenset()
    if normalized in {"1", "all"}:
        return _MINIMAX_H3_FUSION_NAMES
    enabled = frozenset(item.strip() for item in normalized.split(",") if item.strip())
    unknown = enabled - _MINIMAX_H3_FUSION_NAMES
    if unknown:
        supported = ",".join(sorted(_MINIMAX_H3_FUSION_NAMES))
        raise ValueError(f"Unknown MiniMax H3 fusion(s) {sorted(unknown)}; expected a subset of {supported}.")
    return enabled


def _can_run_minimax_h3_fusion(tensor: torch.Tensor) -> bool:
    """Triton kernels are inference-only and trace as opaque custom ops.

    The ``HAVE_TRITON`` check makes the eager fallback exact: on a CUDA build
    whose Triton failed to import, an enabled fusion falls back instead of
    hitting the strict wrappers' hard RuntimeError mid-forward.
    """
    return HAVE_TRITON and tensor.is_cuda and not torch.is_grad_enabled()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis rotary frequencies over packed `(t, h, w)` coordinates."""

    def __init__(self, rope_freq_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (rope_theta**(torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) /
                                       (2 * rope_freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rotary tensors on the device that owns the packed positions."""
        position_ids = position_ids.to(torch.float32)
        # Analytic rotary positional embedding (RoPE) state is non-persistent,
        # so runtime coordinates own the device after loading or state offload.
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3FeedForward(nn.Module):
    """Bias-free H3 SwiGLU with value-first packed halves."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        fuse_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.fc_in = ReplicatedLinear(
            hidden_size,
            2 * ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = ReplicatedLinear(
            ffn_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )
        self.fuse_swiglu = fuse_swiglu
        self.use_mxfp8 = isinstance(self.fc_in.quant_method, MXFP8QuantizeMethod) and isinstance(
            self.fc_out.quant_method, MXFP8QuantizeMethod)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_mxfp8:
            from fastvideo.layers.mxfp8linear import mxfp8_swiglu_feed_forward

            return mxfp8_swiglu_feed_forward(hidden_states, self.fc_in, self.fc_out)
        hidden_states, _ = self.fc_in(hidden_states)
        if self.fuse_swiglu and _can_run_minimax_h3_fusion(hidden_states):
            hidden_states = minimax_h3_swiglu(hidden_states)
        else:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            hidden_states = hidden_states * F.silu(gate)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module):
    """Full self-attention over one sequence-parallel packed document."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
        fuse_qknorm_rope: bool = False,
        fa4_packed_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.norm_q = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.to_out = ReplicatedLinear(
            inner_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )
        self.fuse_qknorm_rope = fuse_qknorm_rope
        # VSA carries a learned gate on its pooled-compression branch. The H3
        # checkpoint has no such weight, so the loader zero-initializes it
        # (ALLOWED_NEW_PARAM_PATTERNS) and the branch is exactly disabled
        # until finetuned. Built only when VSA-H3 actually resolves, keeping
        # the FLASH/SDPA paths and their state_dict untouched.
        resolved_backend = get_attn_backend(attention_head_dim,
                                            get_compute_dtype(),
                                            supported_attention_backends=supported_attention_backends)
        use_vsa = resolved_backend.get_name() == "VIDEO_SPARSE_ATTN_H3"
        attention_cls = DistributedAttention_VSA if use_vsa else DistributedAttention
        self.distributed_attention = attention_cls(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix,
            fa4_packed_varlen=fa4_packed_varlen,
        )
        self.to_gate_compress: ReplicatedLinear | None = None
        # None = unchecked; the first forward tests the loaded weight once and
        # skips the gate branch entirely while it is structurally zero.
        self._gate_compress_active: bool | None = None
        if use_vsa:
            self.to_gate_compress = ReplicatedLinear(
                hidden_size,
                inner_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_gate_compress",
            )

    def _gate_active(self) -> bool:
        """True when the compression gate can contribute.

        The zero-initialized (not-yet-finetuned) gate makes the whole branch
        a guaranteed zero — a full GEMM plus a third more all-to-all/tile
        traffic per layer for nothing — so test the loaded weight once and
        cache the answer. Under grad the branch always runs (gradients must
        reach a zero gate for it to ever train).
        """
        if torch.is_grad_enabled():
            return True
        if self._gate_compress_active is None:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "MiniMax H3 VSA compression gate was not resolved before torch.compile; "
                    "call prepare_for_compile() after loading weights.")
            self._resolve_gate_compress_for_compile()
        assert self._gate_compress_active is not None
        return self._gate_compress_active

    def _resolve_gate_compress_for_compile(self) -> None:
        """Resolve the inference-only compression-gate branch eagerly.

        Regional compilation wraps each transformer block with
        ``fullgraph=True``. Resolving the loaded weight before capture keeps
        the GPU-to-host bool conversion and the cache mutation out of every
        compiled block graph. Grad-enabled forwards ignore this cached value
        in ``_gate_active`` so training can still learn from a zero gate.
        """
        if self.to_gate_compress is None:
            return
        if self._gate_compress_active is None:
            weight = self.to_gate_compress.weight
            # bool() on a DTensor reduction resolves collectively, so every
            # rank caches the same answer.
            self._gate_compress_active = bool((weight != 0).any())

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Rotate the RoPE prefix while preserving the rest of each attention head."""
        cos, sin = rotary_emb
        rotary_dim = cos.shape[-1]
        hidden_states_rotary = hidden_states[..., :rotary_dim]
        hidden_states_pass = hidden_states[..., rotary_dim:]
        cos = cos.to(hidden_states.dtype)[None, :, None, :]
        sin = sin.to(hidden_states.dtype)[None, :, None, :]
        first_half, second_half = hidden_states_rotary.chunk(2, dim=-1)
        hidden_states_rotated = torch.cat((-second_half, first_half), dim=-1)
        hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
        return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
    ) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        query = query.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        key = key.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        value = value.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        if (self.fuse_qknorm_rope and rotary_emb is not None and _can_run_minimax_h3_fusion(query)):
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = fused_qknorm_rope(query, self.norm_q.weight, cos, sin, self.norm_q.eps)
            key = fused_qknorm_rope(key, self.norm_k.weight, cos, sin, self.norm_k.eps)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)
            if rotary_emb is not None:
                query = self._apply_rotary_emb(query, rotary_emb)
                key = self._apply_rotary_emb(key, rotary_emb)

        # H3 rotates only 96/128 channels, which the generic `freqs_cis`
        # branch cannot express. Apply it above, then pass no RoPE here.
        extra_attention_kwargs = {}
        if self.to_gate_compress is not None and self._gate_active():
            gate_compress, _ = self.to_gate_compress(hidden_states)
            extra_attention_kwargs["gate_compress"] = gate_compress.unflatten(
                -1, (self.num_attention_heads, self.attention_head_dim))
        hidden_states, _ = self.distributed_attention(
            query,
            key,
            value,
            original_seq_len=original_seq_len,
            freqs_cis=None,
            **extra_attention_kwargs,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm Transformer block for the projected text stream."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
        )

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), None, original_seq_len)
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    """Two-block text refiner used before packing the modalities."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([
            MiniMaxH3TokenRefinerBlock(
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                ffn_dim,
                norm_eps,
                qk_norm_eps,
                supported_attention_backends,
                quant_config,
                prefix=f"{prefix}.refiner_blocks.{index}",
            ) for index in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, original_seq_len)
        return self.final_norm(hidden_states)


def _adaln_projection(linear: ReplicatedLinear, temb: torch.Tensor, apply_silu: bool) -> torch.Tensor:
    """Project the time embedding of every distinct timestep through one AdaLN linear.

    The resident AdaLN modules and the AdaLN table build share this function,
    so a table entry comes from the same operations, dtypes and row count as
    the live projection it replaces.
    """
    if apply_silu:
        temb = F.silu(temb)
    projection, _ = linear(temb.to(linear.weight.dtype))
    return projection


def _split_block_modulation(projection: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, ...]:
    """View an ``[n, 6 * 3 * hidden]`` block projection as six ``[3n, hidden]`` row tables.

    Row ``3 * t + m`` of each table belongs to distinct timestep ``t`` and
    modality tag ``m``, the layout ``adaln_indices`` addresses.
    """
    return projection.view(-1, 6 * hidden_size).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Produce six modulation tables for every `(timestep, modality)` pair."""

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_silu: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.apply_silu = apply_silu
        self.linear = ReplicatedLinear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def project(self, temb: torch.Tensor) -> torch.Tensor:
        """Return the packed ``[n, 6 * 3 * hidden]`` projection of ``n`` distinct timesteps."""
        return _adaln_projection(self.linear, temb, self.apply_silu)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _split_block_modulation(self.project(temb), self.hidden_size)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final RMSNorm with per-timestep row modulation.

    With ``adaln_table`` the shift/scale projection is not built; callers pass
    its precomputed output to :meth:`modulate`.
    """

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_silu: bool = True,
        adaln_table: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.apply_silu = apply_silu
        self.linear = None if adaln_table else ReplicatedLinear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def project(self, temb: torch.Tensor) -> torch.Tensor:
        """Return the packed ``[n, 2 * hidden]`` shift/scale projection of ``n`` distinct timesteps."""
        assert self.linear is not None, "norm_out was built without its AdaLN projection"
        return _adaln_projection(self.linear, temb, self.apply_silu)

    def modulate(
        self,
        hidden_states: torch.Tensor,
        shift_scale: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize and modulate each row with its timestep's projected shift and scale."""
        shift, scale = shift_scale.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.modulate(hidden_states, self.project(temb), timestep_indices)


@dataclass(frozen=True)
class MiniMaxH3AdaLNStep:
    """AdaLN projections of one schedule step, in the live projection layout."""

    # One contiguous ``[n, 6 * 3 * hidden]`` projection per transformer block.
    blocks: tuple[torch.Tensor, ...]
    # The contiguous ``[n, 2 * hidden]`` final shift/scale projection.
    norm_out: torch.Tensor


@dataclass(frozen=True)
class MiniMaxH3AdaLNTable:
    """AdaLN projections for every step of the checkpoint's fixed schedule.

    ``keys[i]`` holds the distinct FP32 timesteps that step ``i`` feeds the
    transformer, in the order the denoising stage builds them. Step ``i``'s
    projections are valid for exactly those timesteps and nothing else.
    """

    keys: tuple[torch.Tensor, ...]
    steps: tuple[MiniMaxH3AdaLNStep, ...]

    def select(self, step_index: int, timestep: torch.Tensor) -> MiniMaxH3AdaLNStep:
        """Return step ``step_index`` after checking ``timestep`` against its key.

        The check compares FP32 bit patterns on the host and raises on any
        difference. A device ``timestep`` is copied to the host first, which
        synchronizes with its stream, so callers pass the host tensor.
        """
        if not 0 <= step_index < len(self.steps):
            raise ValueError(f"MiniMax-H3 AdaLN table covers {len(self.steps)} schedule steps, got step {step_index}.")
        key = self.keys[step_index]
        timestep = timestep.detach().to("cpu")
        if (timestep.dtype != torch.float32 or timestep.shape != key.shape
                or not torch.equal(timestep.view(torch.int32), key.view(torch.int32))):
            raise ValueError(f"MiniMax-H3 step {step_index} timesteps {timestep.tolist()} do not match the AdaLN "
                             f"table key {key.tolist()}; the table covers only the checkpoint's fixed "
                             "text-to-video/audio schedule.")
        return self.steps[step_index]


class MiniMaxH3TransformerBlock(nn.Module):
    """Packed self-attention and feed-forward branches with row-indexed AdaLN.

    With ``adaln_table`` the block has no AdaLN projection; the caller passes
    the step's precomputed projection as ``adaln_projection``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
        adaln_apply_silu: bool = True,
        fuse_modulate: bool = False,
        fuse_qknorm_rope: bool = False,
        fuse_swiglu: bool = False,
        fa4_packed_varlen: bool = False,
        adaln_table: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
            fuse_qknorm_rope=fuse_qknorm_rope,
            fa4_packed_varlen=fa4_packed_varlen,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
            fuse_swiglu=fuse_swiglu,
        )
        self.adaln_proj = None if adaln_table else MiniMaxH3AdaLayerNormModulation(
            time_embed_dim,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.adaln_proj",
            apply_silu=adaln_apply_silu,
        )
        self.fuse_modulate = fuse_modulate

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        adaln_projection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one block; ``adaln_projection`` replaces projecting ``temb`` in AdaLN-table mode."""
        with nvtx_range("minimax_h3.transformer_block.adaln_projection"):
            if adaln_projection is None:
                assert self.adaln_proj is not None, "an AdaLN-table block needs its step's adaln_projection"
                modulation = self.adaln_proj(temb)
            else:
                # The precomputed projection is split into the same six views
                # the live projection yields, so every consumer below sees
                # identical values, shapes and strides in both modes.
                modulation = _split_block_modulation(adaln_projection, self.hidden_size)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (t.to(hidden_states.dtype)
                                                                              for t in modulation)

        use_modulate_fusion = self.fuse_modulate and _can_run_minimax_h3_fusion(hidden_states)
        if use_modulate_fusion:
            with nvtx_range("minimax_h3.transformer_block.modulate_fusion"):
                norm_hidden_states = fused_rmsnorm_modulate(
                    hidden_states,
                    self.norm1.weight,
                    scale_msa,
                    shift_msa,
                    adaln_indices,
                    self.norm1.eps,
                )
        else:
            with nvtx_range("minimax_h3.transformer_block.no_modulate_fusion"):
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (
                    1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
        with nvtx_range("minimax_h3.transformer_block.self_attention"):
            attention_output = self.attn(norm_hidden_states, rotary_emb, original_seq_len)
        if use_modulate_fusion:
            with nvtx_range("minimax_h3.transformer_block.modulate_fusion"):
                hidden_states, norm_hidden_states = fused_residual_gate_rmsnorm_modulate(
                    hidden_states,
                    attention_output,
                    gate_msa,
                    self.norm2.weight,
                    scale_mlp,
                    shift_mlp,
                    adaln_indices,
                    self.norm2.eps,
                )
        else:
            with nvtx_range("minimax_h3.transformer_block.no_modulate_fusion"):
                hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attention_output
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (
                    1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
        with nvtx_range("minimax_h3.transformer_block.feed_forward"):
            feed_forward_output = self.ff(norm_hidden_states)
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * feed_forward_output


class MiniMaxH3Transformer3DModel(BaseDiT):
    """Joint H3 Transformer over one padless text/audio/video document.

    The layout builder validates semantic rows before denoising. Sequence-
    parallel padding is transport-only and `DistributedAttention` trims it
    before attention.
    """

    _fsdp_shard_conditions = _CFG._fsdp_shard_conditions
    _compile_conditions = _CFG._compile_conditions
    _supported_attention_backends = _CFG._supported_attention_backends
    param_names_mapping = _CFG.param_names_mapping
    reverse_param_names_mapping = _CFG.reverse_param_names_mapping
    lora_param_names_mapping = _CFG.lora_param_names_mapping
    _keep_in_fp32_modules = frozenset({
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    })
    # torch.compile kwargs of the inference loader's regional block compile;
    # the loader sets them when it compiles the blocks and they stay None for
    # an eager model.
    regional_compile_kwargs: dict[str, Any] | None = None

    @staticmethod
    def is_adaln_projection_key(name: str) -> bool:
        """Whether checkpoint key ``name`` belongs to a block or final AdaLN projection."""
        return _ADALN_PROJECTION_KEY.match(name) is not None

    def is_omitted_checkpoint_key(self, name: str) -> bool:
        """Whether checkpoint key ``name`` belongs to a module this instance does not build.

        AdaLN-table mode builds no AdaLN projections, so the loader must not
        read their tensors; every other checkpoint key still loads strictly.
        """
        return self.config.adaln_table and self.is_adaln_projection_key(name)

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the released input, timestep, and output projections in FP32.

        Factorized AdaLN uses FP16; BF16 is ~1.7x worse there.
        """
        # Precedence: the factorized-AdaLN FP16 pin wins over
        # uniform_parameter_dtype on purpose. Under FSDP's one-dtype rule the
        # resulting mix hard-fails at load time, which beats silently training
        # AdaLN in BF16. Rank-reduced checkpoints are inference artifacts --
        # train from the full-rank release.
        if getattr(self, "adaln_rank", None) is not None and (
                ".adaln_proj." in name or name.startswith(("norm_out.linear.", "adaln_basis."))):
            return torch.float16
        if self.config.uniform_parameter_dtype:
            return default_dtype
        return torch.float32 if name.split(".", 1)[0] in self._keep_in_fp32_modules else default_dtype

    def __init__(self, config: MiniMaxH3Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config, hf_config)
        arch = config.arch_config
        if config.adaln_table:
            # The table reproduces the full-rank, unquantized projection
            # exactly; other AdaLN representations have no table equivalent.
            if arch.adaln_rank is not None:
                raise ValueError("MiniMax H3 adaln_table requires a full-rank AdaLN checkpoint, "
                                 f"got adaln_rank={arch.adaln_rank}.")
            if config.quant_config is not None:
                raise ValueError("MiniMax H3 adaln_table requires an unquantized transformer, "
                                 f"got {type(config.quant_config).__name__}.")
        # Filled by build_adaln_table() in AdaLN-table mode; a plain attribute,
        # never in state_dict.
        self.adaln_table: MiniMaxH3AdaLNTable | None = None
        self.enabled_fusions = _enabled_minimax_h3_fusions()
        if self.enabled_fusions:
            if HAVE_TRITON:
                logger.info(
                    "MiniMax H3 inference fusions enabled: %s (CUDA inference-only; grad-enabled forwards "
                    "fall back to eager; torch.compile captures opaque custom-op boundaries).",
                    ",".join(sorted(self.enabled_fusions)))
            else:
                logger.warning(
                    "FASTVIDEO_MINIMAX_H3_FUSIONS requested %s but Triton is unavailable; "
                    "every forward stays on the eager path.", ",".join(sorted(self.enabled_fusions)))
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if arch.num_attention_heads % sp_world_size:
            raise ValueError(f"MiniMax H3 attention heads ({arch.num_attention_heads}) must be divisible by "
                             f"sequence parallel size ({sp_world_size}).")

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels
        self.patch_size = tuple(int(value) for value in arch.patch_size)
        video_patch_dim = arch.in_channels * math.prod(arch.patch_size)

        self.proj_in = ReplicatedLinear(
            video_patch_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_in",
        )
        self.audio_proj_in = ReplicatedLinear(
            arch.audio_in_channels,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_in",
        )
        self.context_embedder = ReplicatedLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.context_embedder",
        )
        self.time_proj = Timesteps(
            num_channels=arch.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = MLP(
            arch.freq_dim,
            arch.time_embed_hidden_dim,
            arch.time_embed_dim,
            act_type="silu",
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.time_embedder",
        )
        self.adaln_rank: int | None = arch.adaln_rank
        if self.adaln_rank is not None and config.uniform_parameter_dtype:
            raise ValueError(
                "Rank-reduced AdaLN checkpoints (adaln_rank set) cannot be trained: "
                "uniform_parameter_dtype needs one dtype for every trainable "
                "parameter, but factorized AdaLN weights are pinned to FP16 "
                "(BF16 reconstructs them ~1.7x worse). Fine-tune the full-rank "
                "checkpoint instead, then re-fit the basis with "
                "scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py.")
        adaln_dim = self.adaln_rank or arch.time_embed_dim
        self.adaln_basis = ReplicatedLinear(
            arch.time_embed_dim,
            self.adaln_rank,
            bias=False,
            quant_config=None,
            prefix=f"{config.prefix}.adaln_basis",
        ) if self.adaln_rank else None

        self.rope = MiniMaxH3RotaryPosEmbed(arch.rope_freq_dim, arch.rope_theta)
        # per-generation caches for loop-invariant work (see _rotary_for /
        # _refined_text); plain attrs, never in state_dict
        self._rope_cache: tuple | None = None
        self._text_cache: tuple | None = None
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch.hidden_size,
            arch.num_attention_heads,
            arch.attention_head_dim,
            arch.ffn_dim,
            arch.num_refiner_layers,
            arch.norm_eps,
            arch.qk_norm_eps,
            arch.final_norm_eps,
            # The refiner attends over the text stream only; the packed-sequence
            # VSA backend must never be selected for it.
            tuple(backend for backend in self.supported_attention_backends
                  if backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3),
            config.quant_config,
            prefix=f"{config.prefix}.token_refiner",
        )
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3TransformerBlock(
                arch.hidden_size,
                arch.num_attention_heads,
                arch.attention_head_dim,
                arch.ffn_dim,
                adaln_dim,
                arch.norm_eps,
                arch.qk_norm_eps,
                self.supported_attention_backends,
                config.quant_config,
                prefix=f"{config.prefix}.transformer_blocks.{index}",
                adaln_apply_silu=self.adaln_rank is None,
                fuse_modulate="modulate" in self.enabled_fusions,
                fuse_qknorm_rope="qknorm_rope" in self.enabled_fusions,
                fuse_swiglu="swiglu" in self.enabled_fusions,
                fa4_packed_varlen=envs.FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN,
                adaln_table=config.adaln_table,
            ) for index in range(arch.num_layers)
        ])
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            arch.hidden_size,
            adaln_dim,
            arch.final_norm_eps,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.norm_out",
            apply_silu=self.adaln_rank is None,
            adaln_table=config.adaln_table,
        )
        self.proj_out = ReplicatedLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_out",
        )
        self.audio_proj_out = ReplicatedLinear(
            arch.hidden_size,
            arch.audio_in_channels,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_out",
        )
        self.__post_init__()

    @staticmethod
    def _compile_setup_device(attention: MiniMaxH3Attention) -> torch.device:
        """Return the loaded device even when FP8 replaced the query weight."""
        query_state = next(attention.to_q.parameters(), None)
        if query_state is None:
            query_state = next(attention.to_q.buffers(), None)
        if query_state is None:
            raise RuntimeError("MiniMax H3 to_q has no materialized parameter or buffer for compile setup.")
        return query_state.device

    def prepare_for_compile(self) -> None:
        """Pipeline hook, called once right before torch.compile wraps the blocks.

        Resolve each loaded VSA compression gate eagerly and tensorize its
        layer identity so repeated blocks share one Dynamo graph. Generic and
        training compile retain their established attention dispatch; only
        the inference loader's separate ``prepare_for_regional_compile`` hook
        may preselect the inference-only sm_100a path.

        The inference-only Triton fusions expose fake-backed custom operators,
        so Dynamo can keep them active as opaque nodes inside each fullgraph
        block instead of tracing into their launcher implementation.
        """
        gate_states: list[bool] = []
        prepared_vsa_impls = 0
        for block in self.transformer_blocks:
            attention = block.attn
            if attention.to_gate_compress is not None:
                attention._resolve_gate_compress_for_compile()
                assert attention._gate_compress_active is not None
                gate_states.append(attention._gate_compress_active)
            prepare_vsa = getattr(attention.distributed_attention.attn_impl, "prepare_for_compile", None)
            if callable(prepare_vsa):
                prepare_vsa(self._compile_setup_device(attention))
                prepared_vsa_impls += 1
        if gate_states:
            logger.info(
                "Resolved MiniMax H3 VSA compression gates before torch.compile: %d active, %d inactive",
                sum(gate_states),
                len(gate_states) - sum(gate_states),
            )
        if prepared_vsa_impls:
            logger.info("Prepared %d MiniMax H3 VSA layer indices for torch.compile", prepared_vsa_impls)
        if self.enabled_fusions:
            logger.info(
                "MiniMax H3 inference fusions remain active under torch.compile through custom-op boundaries: %s",
                ",".join(sorted(self.enabled_fusions)),
            )

    def prepare_for_regional_compile(self) -> str | None:
        """Resolve state used only by inference regional fullgraph compile."""
        self.prepare_for_compile()
        prepared_vsa_impls = 0
        unsupported_reasons: set[str] = set()
        for block in self.transformer_blocks:
            attention = block.attn
            prepare_vsa = getattr(attention.distributed_attention.attn_impl, "prepare_for_regional_compile", None)
            if not callable(prepare_vsa):
                continue
            unsupported = prepare_vsa(self._compile_setup_device(attention))
            if unsupported:
                unsupported_reasons.add(str(unsupported))
            prepared_vsa_impls += 1
        if prepared_vsa_impls:
            logger.info("Prepared %d MiniMax H3 VSA attention implementations for regional torch.compile",
                        prepared_vsa_impls)
        if unsupported_reasons:
            return "; ".join(sorted(unsupported_reasons))
        return None

    def materialize_non_persistent_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Rebuild analytic RoPE state on the checkpoint loader device.

        RoPE frequencies are absent from the checkpoint, so meta-device model
        construction and device moves must derive the buffer from architecture
        fields before the first forward pass.
        """
        del dtype
        if self.rope.inv_freq.is_meta or self.rope.inv_freq.device != device:
            arch = self.config.arch_config
            inv_freq = 1.0 / (arch.rope_theta
                              **(torch.arange(0, 2 * arch.rope_freq_dim, 2, device=device, dtype=torch.float32) /
                                 (2 * arch.rope_freq_dim)))
            self.rope._buffers["inv_freq"] = inv_freq

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return the FP32 ``[n, time_embed_dim]`` embedding of ``n`` distinct timesteps."""
        temb = self.time_proj(timestep)
        return self.time_embedder(temb.to(self.time_embedder.fc_in.weight.dtype))

    @torch.no_grad()
    def build_adaln_table(
        self,
        step_timesteps: Sequence[torch.Tensor],
        projection_weights: Iterable[tuple[str, torch.Tensor]],
        param_dtype: torch.dtype,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Precompute every AdaLN projection for a fixed schedule.

        Args:
            step_timesteps: For each schedule step, the host FP32 distinct
                timesteps the denoising loop passes as ``timestep``; they
                become the table keys that ``forward`` checks.
            projection_weights: ``(checkpoint key, tensor)`` pairs covering
                exactly the block and final AdaLN projection tensors. Each
                layer is bound as soon as its weight and bias have arrived,
                evaluated for every step and released, so one layer is
                resident at a time.
            param_dtype: The loader's default parameter dtype; each tensor is
                cast as the resident loader would cast it.
            compile_kwargs: The regional compile kwargs when the blocks are
                compiled. Block projections then run through the same inductor
                lowering of the SiLU/cast prologue as the compiled blocks;
                ``norm_out`` is eager in both modes and stays eager here.

        Every projection runs on the time embedder's device with the resident
        path's operations, dtypes and row count, so eager table entries equal
        the eager live projections bit for bit.
        """
        if not self.config.adaln_table:
            raise RuntimeError("build_adaln_table requires a transformer built with dit_config.adaln_table=True.")
        arch = self.config.arch_config
        device = self.time_embedder.fc_in.weight.device
        keys: list[torch.Tensor] = []
        for timesteps in step_timesteps:
            if timesteps.ndim != 1 or timesteps.numel() == 0 or timesteps.dtype != torch.float32:
                raise ValueError("MiniMax H3 AdaLN table keys must be non-empty 1-D FP32 timestep tensors, got "
                                 f"shape={tuple(timesteps.shape)} dtype={timesteps.dtype}.")
            keys.append(timesteps.detach().to("cpu").clone())
        if not keys:
            raise ValueError("MiniMax H3 AdaLN table needs at least one schedule step.")
        # The device copy of each key is the tensor forward() would embed.
        tembs = [self._time_embedding(key.to(device)) for key in keys]

        # Two reusable meta-built linears take each checkpoint layer in turn.
        with torch.device("meta"):
            block_linear = ReplicatedLinear(
                arch.time_embed_dim,
                6 * arch.hidden_size * MINIMAX_H3_MODALITY_NUM,
                bias=True,
                quant_config=None,
                prefix=f"{self.config.prefix}.adaln_table.block",
            )
            out_linear = ReplicatedLinear(
                arch.time_embed_dim,
                2 * arch.hidden_size,
                bias=True,
                quant_config=None,
                prefix=f"{self.config.prefix}.adaln_table.norm_out",
            )
        block_projection = (_adaln_projection
                            if compile_kwargs is None else torch.compile(_adaln_projection, **compile_kwargs))

        num_layers = len(self.transformer_blocks)
        block_tables: list[list[torch.Tensor | None]] = [[None] * num_layers for _ in keys]
        norm_out_tables: list[torch.Tensor | None] = [None] * len(keys)
        pending: dict[str, dict[str, torch.Tensor]] = {}
        completed: set[str] = set()
        for name, tensor in projection_weights:
            match = _ADALN_PROJECTION_KEY.match(name)
            if match is None:
                raise ValueError(f"MiniMax H3 AdaLN table received a non-AdaLN checkpoint tensor: {name}")
            owner, layer, kind = match.group(1), match.group(2), match.group(3)
            if layer is not None and int(layer) >= num_layers:
                raise ValueError(f"MiniMax H3 AdaLN tensor {name} addresses block {layer} of {num_layers}.")
            layer_tensors = pending.setdefault(owner, {})
            if owner in completed or kind in layer_tensors:
                raise ValueError(f"MiniMax H3 AdaLN tensor {name} appears more than once.")
            layer_tensors[kind] = tensor.to(device=device, dtype=self._get_parameter_dtype(name, param_dtype))
            del tensor
            if len(layer_tensors) < 2:
                continue

            del pending[owner]
            completed.add(owner)
            linear = out_linear if layer is None else block_linear
            weight, bias = layer_tensors.pop("weight"), layer_tensors.pop("bias")
            if (tuple(weight.shape) != (linear.output_size, linear.input_size)
                    or tuple(bias.shape) != (linear.output_size, )):
                raise ValueError(f"MiniMax H3 AdaLN projection {owner} has weight {tuple(weight.shape)} and bias "
                                 f"{tuple(bias.shape)}, expected ({linear.output_size}, {linear.input_size}).")
            linear.weight = nn.Parameter(weight, requires_grad=False)
            linear.bias = nn.Parameter(bias, requires_grad=False)
            del weight, bias
            for step, temb in enumerate(tembs):
                if layer is None:
                    norm_out_tables[step] = _adaln_projection(linear, temb, True)
                else:
                    block_tables[step][int(layer)] = block_projection(linear, temb, True)

        expected = {f"transformer_blocks.{index}.adaln_proj" for index in range(num_layers)} | {"norm_out"}
        if completed != expected:
            raise ValueError("MiniMax H3 AdaLN table is missing checkpoint tensors for: "
                             f"{sorted(expected - completed)}")
        steps = tuple(
            MiniMaxH3AdaLNStep(blocks=tuple(blocks), norm_out=norm_out)  # type: ignore[arg-type]
            for blocks, norm_out in zip(block_tables, norm_out_tables, strict=True))
        self.adaln_table = MiniMaxH3AdaLNTable(keys=tuple(keys), steps=steps)
        table_bytes = sum(tensor.numel() * tensor.element_size() for step in steps
                          for tensor in (*step.blocks, step.norm_out))
        logger.info("Built MiniMax H3 AdaLN table: %d steps x (%d blocks + norm_out), %.1f MiB on %s (%s)", len(keys),
                    num_layers, table_bytes / 2**20, device, "compiled" if compile_kwargs is not None else "eager")

    def _rotary_for(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin depend only on positions; the denoising stage holds one
        position_ids tensor for the whole loop, so cache per layout and
        pre-cast once instead of rebuilding every step and re-casting in
        every block. Holding the source tensor keeps the identity check safe
        against id reuse."""
        if torch.compiler.is_compiling():
            cos, sin = self.rope(position_ids)
            return cos.to(dtype), sin.to(dtype)
        cached = self._rope_cache
        if (cached is not None and cached[0] is position_ids and cached[1] == dtype
                and cached[2][0].device == position_ids.device):
            return cached[2]
        cos, sin = self.rope(position_ids)
        value = (cos.to(dtype), sin.to(dtype))
        self._rope_cache = (position_ids, dtype, value)
        return value

    def _refined_text(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """The prompt embedding is constant across the denoising loop and the
        refiner blocks are timestep-free, so refine once per generation
        (saves two transformer blocks and four collectives per step). Skipped
        under grad (training needs the graph) and under compile."""
        cacheable = not torch.is_grad_enabled() and not torch.compiler.is_compiling()
        if (cacheable and self._text_cache is not None and self._text_cache[0] is encoder_hidden_states
                and self._text_cache[1].device == encoder_hidden_states.device):
            return self._text_cache[1]
        text_embeds, _ = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_original_seq_len = text_embeds.shape[1]
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_world_size > 1:
            text_embeds, _ = sequence_model_parallel_shard(text_embeds, dim=1)
        text_embeds = self.token_refiner(text_embeds, text_original_seq_len)
        if sp_world_size > 1:
            text_embeds = sequence_model_parallel_all_gather_with_unpad(
                text_embeds,
                text_original_seq_len,
                dim=1,
            )
        if cacheable:
            self._text_cache = (encoder_hidden_states, text_embeds)
        return text_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict video and audio velocities from one caller-defined packed layout.

        ``timestep`` holds the step's distinct FP32 timesteps. In AdaLN-table
        mode it is only checked against the table key of the forward
        context's step, on the host, so callers pass the host tensor.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must have shape (seq_len, 3), got {tuple(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length, ) or timestep_indices.shape != (sequence_length, ):
            raise ValueError("token_tags and timestep_indices must both match the packed sequence length.")
        if hidden_states.shape[1] != video_indices.numel():
            raise ValueError("hidden_states row count must match video_indices.")
        if audio_hidden_states.shape[1] != audio_indices.numel():
            raise ValueError("audio_hidden_states row count must match audio_indices.")
        if encoder_hidden_states.shape[1] != text_indices.numel():
            raise ValueError("encoder_hidden_states row count must match text_indices.")
        if self.config.adaln_table and self.adaln_table is None:
            raise RuntimeError("MiniMax H3 adaln_table mode needs build_adaln_table() before the first forward.")

        video_embeds, _ = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds, _ = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self._refined_text(encoder_hidden_states)
        rotary_emb = self._rotary_for(position_ids, text_embeds.dtype)
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1

        # text/video/audio indices partition [0, sequence_length), so the
        # uninitialized buffer is fully overwritten; in-place index_copy_ avoids
        # the three full-buffer clones out-of-place index_copy would make.
        packed_hidden_states = text_embeds.new_empty((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed_hidden_states.index_copy_(1, text_indices, text_embeds)
        packed_hidden_states.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed_hidden_states.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        adaln_step: MiniMaxH3AdaLNStep | None = None
        temb: torch.Tensor | None = None
        if self.config.adaln_table:
            # Eager lookup outside the compiled blocks: every block receives a
            # fixed-shape projection, and the step index comes from the
            # denoising loop's forward context.
            assert self.adaln_table is not None
            adaln_step = self.adaln_table.select(get_forward_context().current_timestep, timestep)
        else:
            temb = self._time_embedding(timestep)
            if self.adaln_basis is not None:
                temb, _ = self.adaln_basis(F.silu(temb).to(self.adaln_basis.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        local_timestep_indices = timestep_indices
        original_seq_len = sequence_length

        if sp_world_size > 1:
            packed_hidden_states, _ = sequence_model_parallel_shard(packed_hidden_states, dim=1)
            rotary_cos, _ = sequence_model_parallel_shard(rotary_emb[0], dim=0)
            rotary_sin, _ = sequence_model_parallel_shard(rotary_emb[1], dim=0)
            adaln_indices, _ = sequence_model_parallel_shard(adaln_indices, dim=0)
            local_timestep_indices, _ = sequence_model_parallel_shard(local_timestep_indices, dim=0)
            rotary_emb = (rotary_cos, rotary_sin)

        # The eager driver owns profiling markers while each block's compiled
        # forward owns the graph that the marker surrounds.
        block_projections = ((None, ) * len(self.transformer_blocks) if adaln_step is None else adaln_step.blocks)
        for block_index, (block, adaln_projection) in enumerate(
                zip(self.transformer_blocks, block_projections, strict=True)):
            with nvtx_range(f"minimax_h3.transformer_block.{block_index}"):
                packed_hidden_states = block(
                    packed_hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    original_seq_len,
                    adaln_projection=adaln_projection,
                )

        if adaln_step is None:
            packed_hidden_states = self.norm_out(packed_hidden_states, temb, local_timestep_indices)
        else:
            packed_hidden_states = self.norm_out.modulate(packed_hidden_states, adaln_step.norm_out,
                                                          local_timestep_indices)
        packed_hidden_states = packed_hidden_states.to(self.proj_out.weight.dtype)
        video_output, _ = self.proj_out(packed_hidden_states)
        audio_output, _ = self.audio_proj_out(packed_hidden_states)
        if sp_world_size > 1:
            video_output = sequence_model_parallel_all_gather_with_unpad(video_output, original_seq_len, dim=1)
            audio_output = sequence_model_parallel_all_gather_with_unpad(audio_output, original_seq_len, dim=1)
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)

        return video_output, audio_output


EntryClass = MiniMaxH3Transformer3DModel
