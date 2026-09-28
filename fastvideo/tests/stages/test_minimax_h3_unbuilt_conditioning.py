# SPDX-License-Identifier: Apache-2.0
"""Startup contract of the MiniMax-H3 text-to-video component switches."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
    MiniMaxH3RefPipeline,
    unbuilt_conditioning_components,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


def _patch_pipeline_construction(monkeypatch, loads: list[list[str]]) -> None:
    monkeypatch.setattr(composed_pipeline_base, "maybe_init_distributed_environment_and_model_parallel",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    def fake_load(self, fastvideo_args, loaded_modules=None):
        del fastvideo_args, loaded_modules
        loads.append(list(self.required_config_modules))
        return {}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_load)


def _args(*, video_encoder: bool = True, audio_encoder: bool = True, vision: bool = True) -> FastVideoArgs:
    pipeline_config = MiniMaxH3PipelineConfig()
    pipeline_config.vae_config.load_encoder = video_encoder
    pipeline_config.audio_vae_config.load_encoder = audio_encoder
    pipeline_config.text_encoder_configs[0].load_vision_model = vision
    return FastVideoArgs(model_path="unused/for-this-test", pipeline_config=pipeline_config)


def test_unbuilt_conditioning_components_names_each_switch() -> None:
    assert unbuilt_conditioning_components(_args()) == []
    assert unbuilt_conditioning_components(_args(video_encoder=False, audio_encoder=False, vision=False)) == [
        "vae_config.load_encoder=False",
        "audio_vae_config.load_encoder=False",
        "text_encoder_configs[0].load_vision_model=False",
    ]


@pytest.mark.parametrize("switch", ["video_encoder", "audio_encoder", "vision"])
def test_ref2va_rejects_unbuilt_conditioning_components_before_loading(monkeypatch, switch: str) -> None:
    loads: list[list[str]] = []
    _patch_pipeline_construction(monkeypatch, loads)

    with pytest.raises(ValueError, match="Ref2VA encodes image, video and audio references"):
        MiniMaxH3RefPipeline("unused/for-this-test", _args(**{switch: False}))
    assert loads == []


def test_text_to_video_pipeline_loads_without_conditioning_components(monkeypatch) -> None:
    loads: list[list[str]] = []
    _patch_pipeline_construction(monkeypatch, loads)

    MiniMaxH3Pipeline("unused/for-this-test", _args(video_encoder=False, audio_encoder=False, vision=False))

    assert loads == [list(MiniMaxH3Pipeline._required_config_modules)]
