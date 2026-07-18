"""Parity: the engine must produce the same numbers as the reference.

Everything the engine will ever claim rests on this file: same weights +
same input => same logits, so any speedup is pure engineering, not a
quietly different model. Tests use a tiny random-weight config (fast, no
checkpoint needed); the same state_dict is loaded into both
implementations, which also proves the parameter names line up.
"""

import pytest
import torch

from moe_engine.model import Model as EngineModel
from moe_engine.model import ModelConfig as EngineConfig
from moe_engine.vendored.minimoe_model import Model as ReferenceModel
from moe_engine.vendored.minimoe_model import ModelConfig as ReferenceConfig

TINY = dict(max_seq_length=32, vocab_size=96, num_layers=2, hidden_dim=32, num_experts=4, top_k=2)


def make_models():
    torch.manual_seed(0)
    reference = ReferenceModel(ReferenceConfig(**TINY)).eval()
    engine = EngineModel(EngineConfig(**TINY)).eval()
    engine.load_state_dict(reference.state_dict())
    return reference, engine


def test_full_forward_matches_reference():
    reference, engine = make_models()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        reference_logits, _ = reference(ids)
        engine_logits = engine(ids, engine.new_cache())
    torch.testing.assert_close(engine_logits, reference_logits, atol=1e-4, rtol=1e-4)


def test_cached_decode_matches_full_forward():
    """The whole point of the cache: prefill 4 tokens, then feed the rest one
    at a time. Each step's logits must equal the full-context forward at the
    same position - if they don't, the cache is corrupting attention."""
    _, engine = make_models()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        full = engine(ids, engine.new_cache())

        cache = engine.new_cache()
        stepwise = [engine(ids[:, :4], cache)[:, -1]]
        for t in range(4, 16):
            stepwise.append(engine(ids[:, t : t + 1], cache)[:, -1])
        stepwise = torch.stack(stepwise, dim=1)  # logits at positions 3..15
    torch.testing.assert_close(stepwise, full[:, 3:], atol=1e-4, rtol=1e-4)


def test_greedy_generation_matches_reference():
    reference, engine = make_models()
    prompt = torch.randint(0, TINY["vocab_size"], (8,))
    reference_out = reference.generate(prompt, max_new_tokens=12, temperature=0)
    engine_out = engine.generate(prompt, max_new_tokens=12, temperature=0)
    assert torch.equal(engine_out, reference_out)


def test_cache_follows_model_dtype():
    """A converted model (here float64) must still decode: new_cache() has to
    adopt the model's dtype, or attention rejects mixed-precision q/k/v."""
    _, engine = make_models()
    engine.double()
    ids = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        full = engine(ids, engine.new_cache())
        cache = engine.new_cache()
        engine(ids[:, :4], cache)
        step = engine(ids[:, 4:5], cache)
    assert full.dtype == torch.float64
    torch.testing.assert_close(step[:, -1], full[:, 4])


def test_generate_validates_arguments():
    """The reference rejects nonsense sampling arguments; the engine claims
    the same semantics, so it must too."""
    _, engine = make_models()
    prompt = torch.randint(0, TINY["vocab_size"], (4,))
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=4, temperature=-1)
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=4, top_k=0)
