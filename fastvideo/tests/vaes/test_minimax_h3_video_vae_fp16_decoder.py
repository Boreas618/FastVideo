# SPDX-License-Identifier: Apache-2.0
"""FP16 storage of the MiniMax-H3 video VAE decoder weights (``fp16_decoder_weights``)."""

import pytest
import torch
import torch.nn as nn

from fastvideo.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3


def _tiny_vae(**config_overrides) -> AutoencoderKLMiniMaxH3:
    arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    return AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(arch_config=arch, use_tiling=False,
                                                          **config_overrides)).eval()


@pytest.fixture(scope="module")
def state_dict() -> dict[str, torch.Tensor]:
    """A released-layout FP32 checkpoint with non-trivial scales and register tokens."""
    torch.manual_seed(20260928)
    return {name: torch.randn_like(tensor) * 0.2 for name, tensor in _tiny_vae().state_dict().items()}


def _loaded(state_dict: dict[str, torch.Tensor], **config_overrides) -> AutoencoderKLMiniMaxH3:
    vae = _tiny_vae(**config_overrides)
    vae.load_state_dict(state_dict, strict=True)
    return vae


def test_fp16_decoder_weights_store_only_decoder_linear_weights_in_fp16(state_dict) -> None:
    vae = _loaded(state_dict, fp16_decoder_weights=True)
    linear_weights = {
        f"decoder.{name}.weight"
        for name, module in vae.decoder.named_modules() if isinstance(module, nn.Linear)
    }

    assert len(linear_weights) == 2 + 2 * 6  # proj_in, proj_out, and six linears per block
    for name, tensor in vae.state_dict().items():
        if name in linear_weights:
            assert tensor.dtype == torch.float16, name
            assert torch.equal(tensor, state_dict[name].to(torch.float16)), name
        else:
            assert tensor.dtype == torch.float32, name
            assert torch.equal(tensor, state_dict[name]), name
    assert all(buffer.dtype == torch.float32 for buffer in vae.buffers())
    assert all(tensor.dtype == torch.float32 for tensor in _loaded(state_dict).state_dict().values())


def test_fp16_decoder_weights_survive_dtype_requests(state_dict) -> None:
    vae = _loaded(state_dict, fp16_decoder_weights=True)
    expected = {name: tensor.dtype for name, tensor in vae.state_dict().items()}

    for request in (lambda: vae.to(torch.float32), lambda: vae.to("cpu", torch.bfloat16), vae.half, vae.bfloat16):
        assert request() is vae
        assert {name: tensor.dtype for name, tensor in vae.state_dict().items()} == expected


@torch.no_grad()
def test_fp16_decoder_weights_decode_identically_under_fp16_autocast(state_dict) -> None:
    reference = _loaded(state_dict)
    vae = _loaded(state_dict, fp16_decoder_weights=True)
    latents = torch.randn(1, 4, 5, 4, 4, generator=torch.Generator().manual_seed(5))

    with torch.autocast("cpu", dtype=torch.float16):
        expected = reference.decode(latents).sample
        actual = vae.decode(latents).sample

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_fp16_decoder_weights_require_fp16_autocast(state_dict) -> None:
    vae = _loaded(state_dict, fp16_decoder_weights=True)
    latents = torch.zeros(1, 4, 5, 4, 4)

    with pytest.raises(RuntimeError, match="requires float16 autocast"):
        vae.decode(latents)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError, match="requires float16 autocast"):
        vae.decode(latents)
