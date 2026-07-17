"""Load miniMoE ``.pt`` checkpoints into the vendored reference model.

Checkpoint layout (produced by miniMoE's training loop):
    {"model": state_dict, "model_config": dict, "step": int, "tokens_seen": int}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from moe_engine.vendored.minimoe_model import Model, ModelConfig


def load_reference_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Model, ModelConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = Model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    metadata = {
        "path": str(Path(path)),
        "step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
    }
    return model, config, metadata
