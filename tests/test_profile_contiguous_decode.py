"""Tests for profiler-only model instrumentation."""

import importlib.util
from pathlib import Path

import torch

from moe_engine.model import Model, ModelConfig

_spec = importlib.util.spec_from_file_location(
    "profile_contiguous_decode",
    Path(__file__).parent.parent / "benchmarks" / "profile_contiguous_decode.py",
)
profile_decode = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_decode)


def test_profiler_instrumentation_preserves_outputs_and_restores_forward_path():
    torch.manual_seed(0)
    config = ModelConfig(
        max_seq_length=32,
        vocab_size=96,
        num_layers=2,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
    )
    model = Model(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        expected = model(ids, model.new_cache())
        with profile_decode.instrument_model(model):
            actual = model(ids, model.new_cache())
        restored = model(ids, model.new_cache())

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
