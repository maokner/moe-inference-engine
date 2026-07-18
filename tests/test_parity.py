"""Parity: the engine must produce the same numbers as the reference.

Everything the engine will ever claim rests on this file: same weights +
same input => same logits, so any speedup is pure engineering, not a
quietly different model. Tests use a tiny random-weight config (fast, no
checkpoint needed); the same state_dict is loaded into both
implementations, which also proves the parameter names line up.
"""

import torch

from moe_engine.model import KVCache
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
        engine_logits = engine(ids, KVCache(engine.config, batch_size=1, device=ids.device))
    torch.testing.assert_close(engine_logits, reference_logits, atol=1e-4, rtol=1e-4)


def test_cached_decode_matches_full_forward():
    """The whole point of the cache: prefill 4 tokens, then feed the rest one
    at a time. Each step's logits must equal the full-context forward at the
    same position - if they don't, the cache is corrupting attention."""
    _, engine = make_models()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        full = engine(ids, KVCache(engine.config, batch_size=1, device=ids.device))

        cache = KVCache(engine.config, batch_size=1, device=ids.device)
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
