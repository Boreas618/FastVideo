# SPDX-License-Identifier: Apache-2.0
"""Decode-only MiniMax-H3 video and audio VAEs (``load_encoder=False``) through the component loaders."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import AudioDecoderLoader, VAELoader
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_parallel import encode_pixels_parallel
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

VIDEO_ARCH = dict(
    latent_channels=4,
    block_out_channels=[32, 32],
    layers_per_block=1,
    spatial_downsample_factors=[2, 2],
    temporal_downsample_factors=[2, 2],
    decoder_num_layers=1,
    decoder_num_attention_heads=1,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=1,
    latents_mean=[0.0] * 4,
    latents_std=[1.0] * 4,
)
AUDIO_ARCH = dict(
    encoder_dim=8,
    encoder_rates=[2, 2],
    latent_dim=32,
    latent_channels=4,
    num_attention_heads=2,
    decoder_dim=16,
    decoder_rates=[2, 2],
    decoder_kernel_sizes=[5, 4],
    resblock_kernel_sizes=[3],
    resblock_dilation_sizes=[[1, 2]],
)


def _write_component(directory: Path, class_name: str, arch: dict, module: torch.nn.Module) -> dict:
    torch.manual_seed(20260928)
    checkpoint = {name: torch.randn_like(tensor) * 0.1 for name, tensor in module.state_dict().items()}
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"_class_name": class_name, **arch}))
    save_file(checkpoint, str(directory / "diffusion_pytorch_model.safetensors"))
    return checkpoint


def _args(*, video_encoder: bool = True, audio_encoder: bool = True) -> FastVideoArgs:
    pipeline_config = MiniMaxH3PipelineConfig()
    pipeline_config.vae_config.load_encoder = video_encoder
    pipeline_config.audio_vae_config.load_encoder = audio_encoder
    return FastVideoArgs(model_path="unused/for-this-test", pipeline_config=pipeline_config, vae_cpu_offload=True)


@pytest.fixture(scope="module")
def video_vae_dir(tmp_path_factory) -> tuple[Path, dict]:
    full = AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(arch_config=MiniMaxH3VideoVAEArchConfig(**VIDEO_ARCH)))
    directory = tmp_path_factory.mktemp("minimax_h3_vae") / "vae"
    return directory, _write_component(directory, "AutoencoderKLMiniMaxH3", VIDEO_ARCH, full)


@pytest.fixture(scope="module")
def audio_vae_dir(tmp_path_factory) -> tuple[Path, dict]:
    full = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=MiniMaxH3AudioVAEArchConfig(**AUDIO_ARCH)))
    directory = tmp_path_factory.mktemp("minimax_h3_audio_vae") / "audio_vae"
    return directory, _write_component(directory, "AutoencoderKLMiniMaxH3Audio", AUDIO_ARCH, full)


@torch.no_grad()
def test_decode_only_video_vae_skips_exactly_the_encoder_tensors(video_vae_dir) -> None:
    directory, checkpoint = video_vae_dir
    full = VAELoader().load(str(directory), _args())
    vae = VAELoader().load(str(directory), _args(video_encoder=False))

    assert vae.encoder is None and vae.quant_conv is None
    encoder_keys = {name for name in checkpoint if name.startswith(("encoder.", "quant_conv."))}
    assert encoder_keys and encoder_keys == {name for name in checkpoint if vae.is_omitted_checkpoint_key(name)}
    assert set(vae.state_dict()) == set(checkpoint) - encoder_keys
    assert all(torch.equal(tensor, checkpoint[name]) for name, tensor in vae.state_dict().items())
    assert not any(full.is_omitted_checkpoint_key(name) for name in checkpoint)

    latents = torch.randn(1, 4, 5, 4, 4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(vae.decode(latents).sample, full.decode(latents).sample)


@torch.no_grad()
def test_decode_only_video_vae_rejects_every_encode_path(video_vae_dir) -> None:
    vae = VAELoader().load(str(video_vae_dir[0]), _args(video_encoder=False))

    for encode in (
            lambda: vae.encode(torch.zeros(1, 3, 17, 16, 16)),
            lambda: vae.encode_pixels(torch.zeros(1, 3, 17, 16, 16, dtype=torch.uint8)),
            lambda: vae.encode_keyframe(torch.zeros(1, 3, 1, 16, 16)),
            # Every rank raises before the first collective of a parallel encode.
            lambda: encode_pixels_parallel(vae, torch.zeros(1, 3, 17, 16, 16), SimpleNamespace(world_size=2)),
    ):
        with pytest.raises(RuntimeError, match="vae_config.load_encoder=False"):
            encode()


@torch.no_grad()
def test_decode_only_audio_vae_skips_exactly_the_encoder_tensors(audio_vae_dir) -> None:
    directory, checkpoint = audio_vae_dir
    full = AudioDecoderLoader().load(str(directory), _args())
    vae = AudioDecoderLoader().load(str(directory), _args(audio_encoder=False))

    assert all(module is None for module in (vae.encoder, vae.pre_block, vae.mean_proj, vae.logs_proj))
    encoder_prefixes = ("encoder.", "pre_block.", "mean_proj.", "logs_proj.")
    encoder_keys = {name for name in checkpoint if name.startswith(encoder_prefixes)}
    assert encoder_keys and encoder_keys == {name for name in checkpoint if vae.is_omitted_checkpoint_key(name)}
    assert set(vae.state_dict()) == set(checkpoint) - encoder_keys
    assert not any(full.is_omitted_checkpoint_key(name) for name in checkpoint)

    latents = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(2))
    assert torch.equal(vae.decode(latents).sample, full.decode(latents).sample)
    with pytest.raises(RuntimeError, match="audio_vae_config.load_encoder=False"):
        vae.encode(torch.zeros(1, 1, 64))
