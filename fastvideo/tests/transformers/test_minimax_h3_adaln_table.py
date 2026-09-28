# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the MiniMax-H3 AdaLN table (``dit_config.adaln_table``)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config  # noqa: E402
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig  # noqa: E402
from fastvideo.fastvideo_args import FastVideoArgs  # noqa: E402
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.layers.quantization.fp8_config import FP8Config  # noqa: E402
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model  # noqa: E402
from fastvideo.models.loader.weight_utils import safetensors_weights_iterator  # noqa: E402
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import adaln_table_conflicts  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.packing import (  # noqa: E402
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as denoising  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import dmd_target_step_timesteps  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (  # noqa: E402
    MINIMAX_H3_LAYOUT_KEY, )
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402

DMD_STEPS = [999, 874, 749, 624, 500, 375, 250, 125]
PATCH_SIZE = (1, 2, 2)
TINY_ARCH = dict(
    num_attention_heads=2,
    attention_head_dim=64,
    hidden_size=64,
    num_layers=3,
    num_refiner_layers=1,
    ffn_dim=96,
    in_channels=4,
    audio_in_channels=4,
    patch_size=PATCH_SIZE,
    text_dim=32,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=48,
    rope_freq_dim=8,
)
# Released checkpoint names for the module names FastVideo remaps on load.
CHECKPOINT_NAMES = (
    ("time_embedder.fc_in.", "time_embedder.linear_1."),
    ("time_embedder.fc_out.", "time_embedder.linear_2."),
    (".attn.to_out.", ".attn.to_out.0."),
    (".ff.fc_in.", ".ff.net.0.proj."),
    (".ff.fc_out.", ".ff.net.2."),
)


def _config(**overrides) -> MiniMaxH3Config:
    arch = MiniMaxH3ArchConfig(**{**TINY_ARCH, **overrides.pop("arch", {})})
    return MiniMaxH3Config(arch_config=arch, **overrides)


def _checkpoint_name(name: str) -> str:
    for model_part, checkpoint_part in CHECKPOINT_NAMES:
        name = name.replace(model_part, checkpoint_part)
    return name


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A random BF16 tiny H3 transformer checkpoint in the released layout."""
    torch.manual_seed(20260928)
    model = MiniMaxH3Transformer3DModel(_config(), {})
    tensors = {}
    for name, parameter in model.state_dict().items():
        if name.startswith("rope."):
            continue
        tensors[_checkpoint_name(name)] = (torch.randn(parameter.shape) * 0.05).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("minimax_h3_adaln_table") / "diffusion_pytorch_model.safetensors"
    save_file(tensors, str(path))
    return path


def _load(checkpoint: Path, **config_overrides) -> MiniMaxH3Transformer3DModel:
    """Load through the inference loader the pipeline uses, on CPU."""
    return maybe_load_fsdp_model(
        model_cls=MiniMaxH3Transformer3DModel,
        init_params={
            "config": _config(**config_overrides),
            "hf_config": {}
        },
        weight_dir_list=[str(checkpoint)],
        device=torch.device("cpu"),
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=torch.bfloat16,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cpu_offload=True,
        fsdp_inference=False,
        training_mode=False,
    ).eval()


def _step_timesteps() -> list[torch.Tensor]:
    return dmd_target_step_timesteps(MiniMaxH3Scheduler(shift=10.0), MiniMaxH3Scheduler(shift=3.0), DMD_STEPS)


def _projection_weights(checkpoint: Path, model: MiniMaxH3Transformer3DModel):
    return safetensors_weights_iterator([str(checkpoint)], to_cpu=True, name_filter=model.is_adaln_projection_key)


def _build_table(checkpoint: Path) -> MiniMaxH3Transformer3DModel:
    model = _load(checkpoint, adaln_table=True)
    model.build_adaln_table(_step_timesteps(), _projection_weights(checkpoint, model), torch.bfloat16)
    return model


def test_adaln_table_loading_leaves_no_adaln_projection_parameters(checkpoint: Path) -> None:
    model = _load(checkpoint, adaln_table=True)
    adaln_keys = [
        name for name, _ in safetensors_weights_iterator([str(checkpoint)], to_cpu=True)
        if model.is_adaln_projection_key(name)
    ]

    assert len(adaln_keys) == 2 * (TINY_ARCH["num_layers"] + 1)
    assert all(model.is_omitted_checkpoint_key(name) for name in adaln_keys)
    assert not any(".adaln_proj." in name or name.startswith("norm_out.linear.") for name in model.state_dict())
    # Everything else in the checkpoint still loads.
    assert "norm_out.norm.weight" in model.state_dict()
    assert not _load(checkpoint).is_omitted_checkpoint_key(adaln_keys[0])


def test_adaln_table_matches_per_layer_projection(checkpoint: Path) -> None:
    resident = _load(checkpoint)
    model = _build_table(checkpoint)
    tensors = dict(safetensors_weights_iterator([str(checkpoint)], to_cpu=True))
    assert model.adaln_table is not None

    for step, timesteps in enumerate(_step_timesteps()):
        table_step = model.adaln_table.select(step, timesteps)
        temb = resident._time_embedding(timesteps)
        activated = F.silu(temb).to(torch.bfloat16)
        for index, block in enumerate(resident.transformer_blocks):
            prefix = f"transformer_blocks.{index}.adaln_proj.linear"
            expected = F.linear(activated, tensors[f"{prefix}.weight"], tensors[f"{prefix}.bias"])
            assert torch.equal(table_step.blocks[index], expected)
            assert torch.equal(table_step.blocks[index], block.adaln_proj.project(temb))
        expected = F.linear(activated, tensors["norm_out.linear.weight"], tensors["norm_out.linear.bias"])
        assert torch.equal(table_step.norm_out, expected)
        assert torch.equal(table_step.norm_out, resident.norm_out.project(temb))


def test_adaln_table_forward_matches_resident_forward(checkpoint: Path, distributed_setup) -> None:
    resident = _load(checkpoint)
    model = _build_table(checkpoint)
    generator = torch.Generator().manual_seed(7)
    num_text, num_frames, height, width, num_audio = 5, 2, 4, 4, 3
    layout = build_packed_sequence(torch.full((num_text, ), MINIMAX_H3_TEXT_TAG), num_frames, height, width, num_audio,
                                   PATCH_SIZE)
    latents = torch.randn(1, TINY_ARCH["in_channels"], num_frames, height, width, generator=generator)
    inputs = dict(
        hidden_states=patchify_video_latents(latents, PATCH_SIZE)[None],
        audio_hidden_states=torch.randn(1, layout.audio_indices.numel(), 4, generator=generator),
        encoder_hidden_states=torch.randn(1, num_text, TINY_ARCH["text_dim"], generator=generator).bfloat16(),
        token_tags=layout.token_tags,
        position_ids=layout.position_ids,
        video_indices=layout.video_indices,
        audio_indices=layout.audio_indices,
        text_indices=layout.text_indices,
    )

    video_scheduler, audio_scheduler = MiniMaxH3Scheduler(shift=10.0), MiniMaxH3Scheduler(shift=3.0)
    for scheduler in (video_scheduler, audio_scheduler):
        base = torch.tensor([step / 1000.0 for step in DMD_STEPS] + [0.0])
        scheduler.set_timesteps(sigmas=scheduler.shift * base / (1 + (scheduler.shift - 1) * base))
    plan = zip(video_scheduler.timesteps, audio_scheduler.timesteps, strict=True)
    with torch.no_grad():
        for step, (video_timestep, audio_timestep) in enumerate(plan):
            video_value, audio_value = float(video_timestep), float(audio_timestep)
            unique, inverse = build_row_timesteps(layout, video_value, audio_value,
                                                  max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG), 1.0)
            with set_forward_context(current_timestep=step,
                                     attn_metadata=None,
                                     forward_batch=ForwardBatch(data_type="dummy")):
                expected = resident(timestep=unique, timestep_indices=inverse, **inputs)
                actual = model(timestep=unique, timestep_indices=inverse, **inputs)
            for actual_output, expected_output in zip(actual, expected, strict=True):
                assert torch.equal(actual_output, expected_output)


def test_denoising_stage_matches_resident_denoising(checkpoint: Path, distributed_setup, monkeypatch) -> None:
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(denoising, "_h3_vsa_metadata_builder", lambda *_: None)
    layout = build_packed_sequence(torch.full((4, ), MINIMAX_H3_TEXT_TAG), 2, 4, 4, 2, PATCH_SIZE)
    generator = torch.Generator().manual_seed(3)
    video_rows = torch.randn(layout.video_indices.numel(), 16, generator=generator)
    audio_rows = torch.randn(layout.audio_indices.numel(), 4, generator=generator)
    prompt_embeds = torch.randn(1, 4, TINY_ARCH["text_dim"], generator=generator)
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig(dmd_denoising_steps=DMD_STEPS),
                           dit_cpu_offload=False,
                           dit_layerwise_offload=False,
                           use_fsdp_inference=False)

    def denoise(transformer: MiniMaxH3Transformer3DModel) -> ForwardBatch:
        stage = denoising.MiniMaxH3DenoisingStage(transformer, MiniMaxH3Scheduler(shift=10.0),
                                                  MiniMaxH3Scheduler(shift=3.0))
        batch = ForwardBatch(data_type="video",
                             prompt_embeds=[prompt_embeds],
                             latents=video_rows.clone(),
                             audio_latents=audio_rows.clone(),
                             num_inference_steps=len(DMD_STEPS) + 1,
                             extra={MINIMAX_H3_LAYOUT_KEY: layout})
        return stage.forward(batch, args)

    expected = denoise(_load(checkpoint))
    actual = denoise(_build_table(checkpoint))

    assert actual.step_index == len(DMD_STEPS) - 1
    assert torch.equal(actual.latents, expected.latents)
    assert torch.equal(actual.audio_latents, expected.audio_latents)


def test_adaln_table_rejects_timesteps_outside_its_schedule(checkpoint: Path) -> None:
    model = _build_table(checkpoint)
    assert model.adaln_table is not None
    keys = _step_timesteps()

    model.adaln_table.select(3, keys[3].clone())
    with pytest.raises(ValueError, match="do not match the AdaLN table key"):
        model.adaln_table.select(3, keys[4])
    with pytest.raises(ValueError, match="do not match the AdaLN table key"):
        model.adaln_table.select(3, torch.nextafter(keys[3], torch.ones_like(keys[3])))
    with pytest.raises(ValueError, match="do not match the AdaLN table key"):
        # A condition row adds a third distinct timestep.
        model.adaln_table.select(0, torch.cat((keys[0], torch.ones(1))))
    with pytest.raises(ValueError, match="do not match the AdaLN table key"):
        model.adaln_table.select(0, keys[0].double())
    with pytest.raises(ValueError, match="covers 8 schedule steps"):
        model.adaln_table.select(8, keys[0])


def test_adaln_table_keys_are_the_denoising_loop_timesteps() -> None:
    layout = build_packed_sequence(torch.full((3, ), MINIMAX_H3_TEXT_TAG), 1, 2, 2, 2, PATCH_SIZE)
    video_scheduler, audio_scheduler = MiniMaxH3Scheduler(shift=10.0), MiniMaxH3Scheduler(shift=3.0)
    keys = dmd_target_step_timesteps(video_scheduler, audio_scheduler, DMD_STEPS)
    for scheduler in (video_scheduler, audio_scheduler):
        base = torch.tensor([step / 1000.0 for step in DMD_STEPS] + [0.0])
        scheduler.set_timesteps(sigmas=scheduler.shift * base / (1 + (scheduler.shift - 1) * base))

    assert len(keys) == len(DMD_STEPS)
    for key, video_timestep, audio_timestep in zip(keys, video_scheduler.timesteps, audio_scheduler.timesteps,
                                                   strict=True):
        video_value, audio_value = float(video_timestep), float(audio_timestep)
        unique, _ = build_row_timesteps(layout, video_value, audio_value,
                                        max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG), 1.0)
        assert key.dtype == torch.float32 and key.shape == (2, )
        assert torch.equal(key.view(torch.int32), unique.view(torch.int32))


def test_adaln_table_build_requires_every_projection_exactly_once(checkpoint: Path) -> None:
    model = _load(checkpoint, adaln_table=True)
    weights = list(_projection_weights(checkpoint, model))
    without_block_1 = [(name, tensor) for name, tensor in weights if not name.startswith("transformer_blocks.1.")]

    with pytest.raises(ValueError, match=r"missing checkpoint tensors for: \['transformer_blocks.1.adaln_proj'\]"):
        model.build_adaln_table(_step_timesteps(), without_block_1, torch.bfloat16)
    with pytest.raises(ValueError, match="appears more than once"):
        model.build_adaln_table(_step_timesteps(), weights + weights[:1], torch.bfloat16)
    with pytest.raises(ValueError, match="non-AdaLN checkpoint tensor"):
        model.build_adaln_table(_step_timesteps(), [("proj_in.weight", weights[0][1])], torch.bfloat16)
    assert model.adaln_table is None

    with set_forward_context(current_timestep=0, attn_metadata=None), pytest.raises(RuntimeError,
                                                                                    match="build_adaln_table"):
        model(hidden_states=torch.zeros(1, 0, 16),
              audio_hidden_states=torch.zeros(1, 0, 4),
              encoder_hidden_states=torch.zeros(1, 0, 32),
              timestep=_step_timesteps()[0],
              timestep_indices=torch.zeros(0, dtype=torch.long),
              token_tags=torch.zeros(0, dtype=torch.long),
              position_ids=torch.zeros(0, 3),
              video_indices=torch.zeros(0, dtype=torch.long),
              audio_indices=torch.zeros(0, dtype=torch.long),
              text_indices=torch.zeros(0, dtype=torch.long))


@pytest.mark.parametrize("overrides, match", [
    (dict(arch=dict(adaln_rank=16)), "full-rank AdaLN"),
    (dict(quant_config=FP8Config()), "unquantized transformer"),
])
def test_adaln_table_rejects_other_adaln_representations(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MiniMaxH3Transformer3DModel(_config(adaln_table=True, **overrides), {})


def _args(**overrides) -> FastVideoArgs:
    pipeline_config = MiniMaxH3PipelineConfig()
    pipeline_config.dit_config.adaln_table = True
    kwargs = dict(
        model_path="unused/for-this-test",
        pipeline_config=pipeline_config,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
    )
    kwargs.update(overrides)
    return FastVideoArgs(**kwargs)


def test_adaln_table_accepts_the_resident_text_to_video_deployment() -> None:
    assert adaln_table_conflicts(_args(), ref2va=False, deferred_load=False) == []
    assert adaln_table_conflicts(_args(inference_torch_compile=True, enable_torch_compile=True),
                                 ref2va=False,
                                 deferred_load=False) == []


@pytest.mark.parametrize("overrides, ref2va, deferred_load, reason", [
    (dict(use_fsdp_inference=True), False, False, "use_fsdp_inference"),
    (dict(dit_cpu_offload=True), False, False, "dit_cpu_offload"),
    (dict(dit_layerwise_offload=True), False, False, "dit_layerwise_offload"),
    (dict(lora_path="adapter.safetensors"), False, False, "lora_path"),
    (dict(enable_torch_compile=True), False, False, "pipeline-level transformer torch.compile"),
    ({}, False, True, "deferred transformer loading"),
    ({}, True, False, "Ref2VA"),
])
def test_adaln_table_rejects_unsupported_deployments(overrides: dict, ref2va: bool, deferred_load: bool,
                                                      reason: str) -> None:
    conflicts = adaln_table_conflicts(_args(**overrides), ref2va=ref2va, deferred_load=deferred_load)

    assert len(conflicts) == 1 and conflicts[0].startswith(reason)


def test_adaln_table_rejects_transformer_quantization() -> None:
    args = _args()
    args.pipeline_config.dit_config.quant_config = FP8Config()

    assert adaln_table_conflicts(args, ref2va=False, deferred_load=False) == ["transformer quantization"]
