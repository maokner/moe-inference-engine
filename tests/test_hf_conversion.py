"""CPU-only structural and numerical tests for HF and vLLM compatibility."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import tiktoken
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from moe_engine.benchmarking import PROMPT
from moe_engine.hf import MiniMoEConfig, MiniMoEForCausalLM
from moe_engine.hf.convert import AUTO_MAP, convert_checkpoint, convert_state_dict
from moe_engine.model import Model as EngineModel
from moe_engine.model import ModelConfig as EngineConfig
from moe_engine.vendored.minimoe_model import Model as ReferenceModel
from moe_engine.vendored.minimoe_model import ModelConfig as ReferenceConfig

TINY = {
    "max_seq_length": 80,
    "vocab_size": 96,
    "num_layers": 2,
    "hidden_dim": 32,
    "num_experts": 4,
    "top_k": 2,
}


def _models():
    torch.manual_seed(0)
    reference = ReferenceModel(ReferenceConfig(**TINY)).eval()
    hf = MiniMoEForCausalLM(MiniMoEConfig(**TINY)).eval()
    hf.load_state_dict(convert_state_dict(reference.state_dict()), strict=True)
    engine = EngineModel(EngineConfig(**TINY)).eval()
    engine.load_state_dict(reference.state_dict())
    engine.set_moe_mode("reference")
    return reference, hf, engine


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_checkpoint(path: Path) -> ReferenceModel:
    torch.manual_seed(0)
    config = {**TINY, "vocab_size": 50304}
    reference = ReferenceModel(ReferenceConfig(**config)).eval()
    torch.save(
        {
            "model": reference.state_dict(),
            "model_config": config,
            "step": 7,
            "tokens_seen": 1234,
        },
        path,
    )
    return reference


def test_hf_logits_match_original_and_engine_on_cpu():
    reference, hf, engine = _models()
    torch.manual_seed(1)
    token_ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        reference_logits, _ = reference(token_ids)
        hf_logits = hf(token_ids).logits
        engine_logits = engine(token_ids, engine.new_cache())
    torch.testing.assert_close(hf_logits, reference_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(engine_logits, reference_logits, atol=1e-4, rtol=1e-4)


def test_all_64_greedy_tokens_match_on_cpu():
    reference, hf, engine = _models()
    torch.manual_seed(2)
    prompt = torch.randint(0, TINY["vocab_size"], (1, 8))
    full_reference = prompt
    full_hf = prompt
    engine_input = prompt
    engine_cache = engine.new_cache()
    reference_tokens = []
    hf_tokens = []
    engine_tokens = []
    with torch.no_grad():
        for _ in range(64):
            reference_logits, _ = reference(full_reference)
            reference_token = reference_logits[:, -1].argmax(dim=-1, keepdim=True)
            hf_token = hf(full_hf).logits[:, -1].argmax(dim=-1, keepdim=True)
            engine_token = engine(engine_input, engine_cache)[:, -1].argmax(
                dim=-1, keepdim=True
            )
            reference_tokens.append(int(reference_token.item()))
            hf_tokens.append(int(hf_token.item()))
            engine_tokens.append(int(engine_token.item()))
            full_reference = torch.cat((full_reference, reference_token), dim=1)
            full_hf = torch.cat((full_hf, hf_token), dim=1)
            engine_input = engine_token
    assert len(reference_tokens) == 64
    assert reference_tokens == hf_tokens == engine_tokens


def test_conversion_is_exact_deterministic_and_auto_loadable(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    reference = _synthetic_checkpoint(checkpoint)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = convert_checkpoint(checkpoint, first)
    convert_checkpoint(checkpoint, second)

    expected = convert_state_dict(reference.state_dict())
    loaded = AutoModelForCausalLM.from_pretrained(first, trust_remote_code=True).eval()
    for name, tensor in expected.items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, atol=0, rtol=0)
    assert (
        loaded.lm_head.weight.data_ptr()
        == loaded.model.token_embedding.weight.data_ptr()
    )
    assert manifest["state_dict_key_count"] == len(reference.state_dict())

    filenames = sorted(path.name for path in first.iterdir())
    assert filenames == sorted(path.name for path in second.iterdir())
    assert {name: _sha256(first / name) for name in filenames} == {
        name: _sha256(second / name) for name in filenames
    }

    config = AutoConfig.from_pretrained(first, trust_remote_code=True)
    base = AutoModel.from_pretrained(first, trust_remote_code=True).eval()
    assert type(config).__name__ == "MiniMoEConfig"
    assert type(base).__name__ == "MiniMoEModel"
    assert config.architectures == ["MiniMoEForCausalLM"]
    assert config.auto_map == AUTO_MAP
    assert config.hidden_act == "gelu"
    assert config.num_attention_heads == 8
    assert config.num_experts_per_tok == 2
    assert config.tie_word_embeddings is True


def test_converted_tokenizer_exactly_matches_tiktoken(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    _synthetic_checkpoint(checkpoint)
    model_dir = tmp_path / "model"
    convert_checkpoint(checkpoint, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    expected = tiktoken.get_encoding("gpt2").encode(PROMPT)
    actual = tokenizer.encode(PROMPT, add_special_tokens=False)
    assert len(actual) == 62
    assert actual == expected
    assert tokenizer.decode(actual) == PROMPT
    assert tokenizer.vocab_size == tiktoken.get_encoding("gpt2").n_vocab == 50257
    assert (
        tokenizer.vocab_size
        != AutoConfig.from_pretrained(model_dir, trust_remote_code=True).vocab_size
    )
    assert tokenizer.model_max_length == TINY["max_seq_length"]


def test_config_contains_no_architecture_substitution():
    config = MiniMoEConfig()
    config.architectures = ["MiniMoEForCausalLM"]
    serialized = json.dumps(config.to_dict()).lower()
    assert "mixtral" not in serialized
    assert "grok" not in serialized
    assert config.hidden_act == "gelu"
    assert config.num_attention_heads == 8
    assert config.max_position_embeddings == 1024


def test_vllm_plugin_uses_only_vllm_runtime_kernels():
    source_path = Path(__file__).parents[1] / "src/moe_engine/vllm_model.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "moe_engine.fused_moe" not in imports
    assert "moe_engine.paged_attention" not in imports
    assert "from vllm.attention.layer import Attention" in source
    assert "from vllm.model_executor.layers.fused_moe import FusedMoE" in source
    assert 'activation="gelu"' in source
    assert "is_act_and_mul=False" in source
    assert "has_bias=True" in source
    assert "embedding_bias=self.lm_head.bias" in source
