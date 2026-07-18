"""Load miniMoE ``.pt`` checkpoints into either model implementation.

Checkpoint layout (produced by miniMoE's training loop):
    {"model": state_dict, "model_config": dict, "step": int, "tokens_seen": int}

The same state_dict loads into the vendored reference model and the engine
model: the engine deliberately uses identical parameter names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from moe_engine import model as engine
from moe_engine.vendored import minimoe_model as reference


def _load(model_module, path: str | Path, device: str | torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = model_module.ModelConfig(**checkpoint["model_config"])
    model = model_module.Model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    metadata = {
        "path": str(Path(path)),
        "step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
    }
    return model, config, metadata


def load_reference_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[reference.Model, reference.ModelConfig, dict[str, Any]]:
    return _load(reference, path, device)


def load_engine_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[engine.Model, engine.ModelConfig, dict[str, Any]]:
    return _load(engine, path, device)
