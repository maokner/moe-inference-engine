"""Parity tests for the engine and reference model."""

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
    """Cached token logits must match a full-context forward pass."""
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


def test_paged_decode_matches_full_forward():
    """Paged token logits must match a full-context forward pass."""
    _, engine = make_models()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        full = engine(ids, engine.new_cache())

        allocator = engine.new_block_allocator(num_blocks=8, block_size=4)
        cache = allocator.new_cache()
        stepwise = [engine(ids[:, :4], cache)[:, -1]]
        for t in range(4, 16):
            stepwise.append(engine(ids[:, t : t + 1], cache)[:, -1])
        stepwise = torch.stack(stepwise, dim=1)
    torch.testing.assert_close(stepwise, full[:, 3:], atol=1e-4, rtol=1e-4)


def test_paged_cache_shares_pool_without_leaking():
    """Sequences sharing a block pool must keep their attention state separate."""
    _, engine = make_models()
    ids_a = torch.randint(0, TINY["vocab_size"], (1, 16))
    ids_b = torch.randint(0, TINY["vocab_size"], (1, 16))
    with torch.no_grad():
        full_a = engine(ids_a, engine.new_cache())
        full_b = engine(ids_b, engine.new_cache())

        allocator = engine.new_block_allocator(num_blocks=8, block_size=4)
        cache_a = allocator.new_cache()
        cache_b = allocator.new_cache()
        logits_a = engine(ids_a[:, :4], cache_a)
        logits_b = engine(ids_b[:, :4], cache_b)
        for t in range(4, 16):
            logits_a = engine(ids_a[:, t : t + 1], cache_a)
            logits_b = engine(ids_b[:, t : t + 1], cache_b)

    torch.testing.assert_close(logits_a[:, -1], full_a[:, -1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits_b[:, -1], full_b[:, -1], atol=1e-4, rtol=1e-4)


def test_block_allocator_reclaims_freed_blocks():
    _, engine = make_models()
    ids = torch.randint(0, TINY["vocab_size"], (1, 8))
    allocator = engine.new_block_allocator(num_blocks=2, block_size=4)
    cache_a = allocator.new_cache()
    cache_b = allocator.new_cache()

    with torch.no_grad():
        engine(ids, cache_a)
        with pytest.raises(RuntimeError, match="no free KV cache blocks left"):
            engine(ids[:, :1], cache_b)

        cache_a.free()
        engine(ids[:, :1], cache_b)


def test_failed_allocation_rolls_back():
    """A prefill that outgrows the pool must leave the pool and cache untouched."""
    _, engine = make_models()
    allocator = engine.new_block_allocator(num_blocks=1, block_size=4)
    cache = allocator.new_cache()
    ids = torch.randint(0, TINY["vocab_size"], (1, 8))  # Needs 2 blocks.

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="no free KV cache blocks left"):
            engine(ids, cache)
        assert cache.block_table == []
        assert cache.position == 0
        assert allocator.free_blocks == [0]

        engine(ids[:, :4], cache)  # The cache is still usable after the failure.


def test_paged_cache_rejects_batches():
    """The paged cache stores one sequence, so batch size must be 1."""
    _, engine = make_models()
    allocator = engine.new_block_allocator(num_blocks=8, block_size=4)
    ids = torch.randint(0, TINY["vocab_size"], (2, 4))
    with torch.no_grad(), pytest.raises(ValueError, match="batch size 1"):
        engine(ids, allocator.new_cache())


def test_generate_stops_at_context_limit():
    """Generation must stop cleanly when the context window fills."""
    _, engine = make_models()
    prompt = torch.randint(0, TINY["vocab_size"], (TINY["max_seq_length"] - 2,))
    out = engine.generate(prompt, max_new_tokens=10, temperature=0)
    # The last emitted token never enters the cache, so max - prompt + 1 fit.
    assert out.shape[1] == TINY["max_seq_length"] + 1


def test_generate_validates_prompt_and_cache():
    """Reject oversized prompts and non-fresh caches before touching state."""
    _, engine = make_models()
    too_long = torch.randint(0, TINY["vocab_size"], (TINY["max_seq_length"] + 1,))
    with pytest.raises(ValueError, match="max_seq_length"):
        engine.generate(too_long, max_new_tokens=1)

    allocator = engine.new_block_allocator(num_blocks=8, block_size=4)
    used = allocator.new_cache()
    prompt = torch.randint(0, TINY["vocab_size"], (1, 4))
    with torch.no_grad():
        engine(prompt, used)
    blocks_before = list(used.block_table)
    with pytest.raises(ValueError, match="fresh cache"):
        engine.generate(prompt, max_new_tokens=1, cache=used)
    assert used.block_table == blocks_before  # The rejected call touched nothing.


def test_greedy_generation_matches_reference():
    reference, engine = make_models()
    prompt = torch.randint(0, TINY["vocab_size"], (8,))
    reference_out = reference.generate(prompt, max_new_tokens=12, temperature=0)
    engine_out = engine.generate(prompt, max_new_tokens=12, temperature=0)
    assert torch.equal(engine_out, reference_out)


def test_cache_follows_model_dtype():
    """Cache dtype must follow the model dtype."""
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


def test_paged_cache_follows_model_dtype():
    """Paged cache dtype must follow the model dtype."""
    _, engine = make_models()
    engine.double()
    ids = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        full = engine(ids, engine.new_cache())
        allocator = engine.new_block_allocator(num_blocks=2, block_size=4)
        cache = allocator.new_cache()
        engine(ids[:, :4], cache)
        step = engine(ids[:, 4:5], cache)
    assert step.dtype == torch.float64
    torch.testing.assert_close(step[:, -1], full[:, 4], atol=1e-4, rtol=1e-4)


def test_stacked_expert_cache_invalidated_on_reload():
    """Reloading weights must clear the fused path's stacked copy."""
    _, engine = make_models()
    moe = engine.MoEBlocks[0].moe
    moe._stacked_experts()
    assert moe._stacked is not None
    engine.load_state_dict(engine.state_dict())
    assert moe._stacked is None  # A stale copy would silently serve old weights.


def test_generate_validates_arguments():
    """Reject invalid sampling arguments."""
    _, engine = make_models()
    prompt = torch.randint(0, TINY["vocab_size"], (4,))
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=4, temperature=-1)
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=4, top_k=0)
