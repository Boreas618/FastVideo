# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for how the MiniMax-H3 pipeline builds and guards its AdaLN table."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

import fastvideo.distributed.parallel_state as parallel_state  # noqa: E402
import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base  # noqa: E402
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config  # noqa: E402
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig  # noqa: E402
from fastvideo.fastvideo_args import FastVideoArgs  # noqa: E402
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model  # noqa: E402
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import dmd_target_step_timesteps  # noqa: E402
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase  # noqa: E402

DMD_STEPS = [999, 874, 749, 624, 500, 375, 250, 125]
TINY_ARCH = dict(
    num_attention_heads=2,
    attention_head_dim=64,
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=96,
    in_channels=4,
    audio_in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=32,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=48,
    rope_freq_dim=8,
)


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


def _patch_pipeline_construction(monkeypatch) -> None:
    monkeypatch.setattr(composed_pipeline_base, "maybe_init_distributed_environment_and_model_parallel",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())
    monkeypatch.setattr(composed_pipeline_base, "warmup_sequence_parallel_communication", lambda: None)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)


def _args(model_path: Path | str, **overrides) -> FastVideoArgs:
    pipeline_config = MiniMaxH3PipelineConfig()
    pipeline_config.dit_config = MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(**TINY_ARCH), adaln_table=True)
    kwargs = dict(
        model_path=str(model_path),
        pipeline_config=pipeline_config,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        enable_stage_verification=False,
    )
    kwargs.update(overrides)
    return FastVideoArgs(**kwargs)


def test_adaln_table_rejects_the_deployment_before_loading_components(monkeypatch) -> None:
    _patch_pipeline_construction(monkeypatch)
    loads: list[list[str]] = []

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)

    with pytest.raises(ValueError, match="adaln_table is incompatible with: use_fsdp_inference; dit_cpu_offload"):
        MiniMaxH3Pipeline("unused/for-this-test",
                          _args("unused/for-this-test", dit_cpu_offload=True, use_fsdp_inference=True))
    assert loads == []


def test_pipeline_builds_the_table_for_the_checkpoint_schedule(monkeypatch, tmp_path: Path) -> None:
    _patch_pipeline_construction(monkeypatch)
    # The table streams the AdaLN tensors onto the transformer's device.
    monkeypatch.setattr(parallel_state, "get_local_torch_device", lambda: torch.device("cpu"))

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    torch.manual_seed(11)
    reference = MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(**TINY_ARCH)), {})
    checkpoint = {}
    for name, parameter in reference.state_dict().items():
        if name.startswith("rope."):
            continue
        name = name.replace("time_embedder.fc_in.", "time_embedder.linear_1.")
        name = name.replace("time_embedder.fc_out.", "time_embedder.linear_2.")
        name = name.replace(".attn.to_out.", ".attn.to_out.0.")
        name = name.replace(".ff.fc_in.", ".ff.net.0.proj.").replace(".ff.fc_out.", ".ff.net.2.")
        checkpoint[name] = (torch.randn(parameter.shape) * 0.05).to(torch.bfloat16)
    save_file(checkpoint, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    (tmp_path / "fastvideo_inference.json").write_text(
        json.dumps({
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": DMD_STEPS,
            "num_inference_steps": len(DMD_STEPS) + 1,
            "transformer_forwards": len(DMD_STEPS),
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
        }))

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del loaded_modules
        fastvideo_args.model_paths["transformer"] = str(transformer_dir)
        transformer = maybe_load_fsdp_model(
            model_cls=MiniMaxH3Transformer3DModel,
            init_params={
                "config": fastvideo_args.pipeline_config.dit_config,
                "hf_config": {}
            },
            weight_dir_list=[str(transformer_dir / "diffusion_pytorch_model.safetensors")],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            default_dtype=torch.bfloat16,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cpu_offload=True,
            training_mode=False,
        ).eval()
        modules = {name: SimpleNamespace(name=name) for name in self.required_config_modules}
        modules.update(transformer=transformer,
                       scheduler=MiniMaxH3Scheduler(shift=10.0),
                       audio_scheduler=MiniMaxH3Scheduler(shift=3.0))
        return modules

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)
    args = _args(tmp_path)
    pipeline = MiniMaxH3Pipeline(str(tmp_path), args)
    transformer = pipeline.get_module("transformer")
    assert transformer.adaln_table is None
    pipeline.post_init()

    table = transformer.adaln_table
    keys = dmd_target_step_timesteps(MiniMaxH3Scheduler(shift=10.0), MiniMaxH3Scheduler(shift=3.0), DMD_STEPS)
    assert table is not None and len(table.keys) == len(DMD_STEPS)
    for step, key in enumerate(keys):
        assert torch.equal(table.keys[step].view(torch.int32), key.view(torch.int32))
        activated = F.silu(transformer._time_embedding(key)).to(torch.bfloat16)
        weight = checkpoint["transformer_blocks.1.adaln_proj.linear.weight"]
        bias = checkpoint["transformer_blocks.1.adaln_proj.linear.bias"]
        assert torch.equal(table.select(step, key).blocks[1], F.linear(activated, weight, bias))
    assert pipeline._stage_name_mapping["denoising_stage"].transformer is transformer

    with pytest.raises(RuntimeError, match="LoRA adapters are unsupported"):
        pipeline.set_lora_adapter("default", "adapter.safetensors")
