"""Deterministic conversion from the training checkpoint to a local HF model."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import tiktoken

from .configuration_minimoe import MiniMoEConfig
from .modeling_minimoe import MiniMoEForCausalLM
from .tokenization_minimoe import MiniMoETokenizer

SOURCE_TO_HF_PREFIX = {
    "token_embedding.": "model.token_embedding.",
    "positional_embedding.": "model.positional_embedding.",
    "MoEBlocks.": "model.MoEBlocks.",
    "final_norm.": "model.final_norm.",
    "output_projection.": "lm_head.",
}

AUTO_MAP = {
    "AutoConfig": "configuration_minimoe.MiniMoEConfig",
    "AutoModel": "modeling_minimoe.MiniMoEModel",
    "AutoModelForCausalLM": "modeling_minimoe.MiniMoEForCausalLM",
    "AutoTokenizer": ["tokenization_minimoe.MiniMoETokenizer", None],
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hf_name(source_name: str) -> str:
    for source_prefix, target_prefix in SOURCE_TO_HF_PREFIX.items():
        if source_name.startswith(source_prefix):
            return target_prefix + source_name.removeprefix(source_prefix)
    raise KeyError(f"unrecognized miniMoE checkpoint key: {source_name}")


def convert_state_dict(
    source_state: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Map every source tensor once, in lexical source-key order, without casts."""
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for source_name in sorted(source_state):
        value = source_state[source_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint value {source_name!r} is not a tensor")
        target_name = hf_name(source_name)
        if target_name in converted:
            raise ValueError(f"duplicate converted key: {target_name}")
        converted[target_name] = value.detach().cpu().contiguous().clone()
    return converted


def config_from_checkpoint(model_config: Mapping[str, Any]) -> MiniMoEConfig:
    known = {
        "max_seq_length",
        "vocab_size",
        "num_layers",
        "hidden_dim",
        "moe_multiplier",
        "num_experts",
        "top_k",
        "router_top_k",
        "num_heads",
    }
    unexpected = set(model_config) - known
    if unexpected:
        raise ValueError(f"unrecognized model_config fields: {sorted(unexpected)}")
    values = dict(model_config)
    values.setdefault("num_heads", 8)
    config = MiniMoEConfig(**values)
    config.architectures = ["MiniMoEForCausalLM"]
    config.auto_map = AUTO_MAP
    return config


def _validate_tied_source_weights(source_state: Mapping[str, torch.Tensor]) -> None:
    token_weight = source_state.get("token_embedding.weight")
    output_weight = source_state.get("output_projection.weight")
    if token_weight is None or output_weight is None:
        raise KeyError("checkpoint must contain tied token and output weights")
    if not torch.equal(token_weight, output_weight):
        raise ValueError("checkpoint token and output projection weights are not tied")


def _prepare_output_directory(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_remote_code(output_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    for filename in (
        "configuration_minimoe.py",
        "modeling_minimoe.py",
        "tokenization_minimoe.py",
    ):
        shutil.copyfile(source_dir / filename, output_dir / filename)


def _write_tokenizer(output_dir: Path, max_length: int) -> None:
    encoding = tiktoken.get_encoding("gpt2")
    vocab_path = output_dir / "minimoe.tiktoken"
    serialized = b"".join(
        base64.b64encode(token) + b" " + str(rank).encode("ascii") + b"\n"
        for token, rank in sorted(
            encoding._mergeable_ranks.items(), key=lambda item: item[1]
        )
    )
    vocab_path.write_bytes(serialized)
    tokenizer = MiniMoETokenizer(
        str(vocab_path),
        model_max_length=max_length,
        auto_map={"AutoTokenizer": AUTO_MAP["AutoTokenizer"]},
    )
    tokenizer.save_pretrained(output_dir)


def convert_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Convert one checkpoint entirely on CPU and return its manifest."""
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise KeyError("checkpoint needs 'model' and 'model_config' entries")

    source_state = checkpoint["model"]
    _validate_tied_source_weights(source_state)
    converted_state = convert_state_dict(source_state)
    config = config_from_checkpoint(checkpoint["model_config"])
    model = MiniMoEForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    # Expert routing is stored as ``router_top_k`` and ``num_experts_per_tok``.
    # Keep the separate generation sampling flag unset for deterministic greedy use.
    model.generation_config.top_k = None
    incompatible = model.load_state_dict(converted_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict converted load failed: {incompatible}")
    if model.lm_head.weight.data_ptr() != model.model.token_embedding.weight.data_ptr():
        raise RuntimeError("Hugging Face output and token embeddings are not tied")

    _prepare_output_directory(output_dir, force)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        state_dict=converted_state,
    )
    _copy_remote_code(output_dir)
    _write_tokenizer(output_dir, config.max_seq_length)

    manifest = {
        "format": "minimoe-huggingface-v1",
        "source_checkpoint": str(checkpoint_path),
        "source_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "dtype": "float32",
        "state_dict_key_count": len(source_state),
        "state_dict_mapping": [
            {"source": source_name, "target": hf_name(source_name)}
            for source_name in sorted(source_state)
        ],
        "tied_weights": [
            "model.token_embedding.weight",
            "lm_head.weight",
        ],
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
