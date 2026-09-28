# SPDX-License-Identifier: Apache-2.0
"""Text-only MiniMax-H3 Qwen3-VL conditioner (``load_vision_model=False``)."""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (  # noqa: E402
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner  # noqa: E402


def _conditioner(load_vision_model: bool) -> MiniMaxH3Qwen3VLConditioner:
    config = MiniMaxH3Qwen3VLConfig(load_vision_model=load_vision_model)
    config.arch_config = MiniMaxH3Qwen3VLArchConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        output_hidden_state_index=3,
        num_hidden_layers_override=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
        image_token_id=60,
        vision_depth=2,
        vision_hidden_size=16,
        vision_intermediate_size=32,
        vision_num_heads=2,
        vision_patch_size=2,
        vision_num_position_embeddings=16,
        vision_out_hidden_size=16,
        vision_deepstack_visual_indexes=(0, ),
    )
    return MiniMaxH3Qwen3VLConditioner(config).eval()


def test_text_only_conditioner_skips_exactly_the_vision_tensors(distributed_setup) -> None:
    full = _conditioner(load_vision_model=True)
    text_only = _conditioner(load_vision_model=False)
    torch.manual_seed(20260928)
    checkpoint = {f"model.{name}": torch.randn_like(tensor) * 0.1 for name, tensor in full.state_dict().items()}
    vision_keys = {name for name in checkpoint if name.startswith("model.visual.")}

    full_loaded = full.load_weights(checkpoint.items())
    text_loaded = text_only.load_weights(checkpoint.items())

    assert text_only.visual is None and vision_keys
    assert full_loaded == {name for name, _ in full.named_parameters()}
    # The loader's strict check: every built parameter came from the checkpoint.
    assert text_loaded == {name for name, _ in text_only.named_parameters()}
    assert text_loaded == full_loaded - {name[len("model."):] for name in vision_keys}

    input_ids = torch.tensor([1, 5, 9, 13, 2])
    assert torch.equal(text_only.encode_ids(input_ids), full.encode_ids(input_ids))


def test_text_only_conditioner_rejects_image_and_video_inputs(distributed_setup) -> None:
    text_only = _conditioner(load_vision_model=False)
    input_ids = torch.tensor([1, 60, 2])
    grid = torch.tensor([[1, 2, 2]])

    with pytest.raises(RuntimeError, match="load_vision_model=False"):
        text_only.encode_ids(input_ids, pixel_values=torch.zeros(4, 24), image_grid_thw=grid)
    with pytest.raises(RuntimeError, match="load_vision_model=False"):
        text_only.encode_ids(input_ids, pixel_values_videos=torch.zeros(4, 24), video_grid_thw=grid)
