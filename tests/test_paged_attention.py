"""Tests for the direct paged-attention path.

The gather path (read blocks into contiguous K/V, then PyTorch attention) is
the correctness oracle everywhere. CUDA/Triton kernel tests skip cleanly on
machines without CUDA; the fallback tests run on CPU (and MPS where present).

Tolerances: everything here is float32. The kernel accumulates the softmax
online over 128-token tiles while SDPA reduces in a different order, so
results differ only by float32 rounding, which stays orders of magnitude
below atol=2e-5/rtol=1e-5; anything larger indicates a real addressing or
masking bug. Model-level checks use the suite-wide 1e-4, matching the
existing parity tests, because stacked LayerNorm/MoE layers compound
rounding differences.
"""

import random

import pytest
import torch
import torch.nn.functional as F

from moe_engine import paged_attention
from moe_engine.model import BlockAllocator, Model, ModelConfig

CUDA_KERNEL = torch.cuda.is_available() and paged_attention.HAS_TRITON
cuda_kernel = pytest.mark.skipif(not CUDA_KERNEL, reason="needs CUDA and Triton")

# Production attention shape: 8 heads of dim 96, KV blocks of 16 tokens.
PROD = dict(max_seq_length=1024, vocab_size=96, num_layers=1, hidden_dim=768, num_heads=8)
TINY = dict(max_seq_length=128, vocab_size=96, num_layers=2, hidden_dim=32, num_experts=4, top_k=2)


def fallback_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def make_filled_cache(config, device, context_len, block_size, shuffle_blocks=False, seed=0):
    """Write `context_len` tokens of random K/V into a fresh paged cache."""
    torch.manual_seed(seed)
    cfg = ModelConfig(**config)
    num_blocks = -(-cfg.max_seq_length // block_size)
    allocator = BlockAllocator(cfg, num_blocks, block_size, device)
    if shuffle_blocks:
        # Force deliberately non-contiguous, non-monotonic physical block ids.
        random.Random(seed).shuffle(allocator.free_blocks)
    cache = allocator.new_cache()
    head_dim = cfg.hidden_dim // cfg.num_heads
    shape = (1, cfg.num_heads, context_len, head_dim)
    # Distinct K/V per layer, so layer-indexing bugs cannot cancel out.
    for layer in range(cfg.num_layers):
        cache.write(layer, 0, torch.randn(shape, device=device), torch.randn(shape, device=device))
    cache.position = context_len
    return cache


def oracle_attention(q, cache, layer, context_len):
    keys, values = cache.read(layer, context_len)
    return F.scaled_dot_product_attention(q, keys, values)


# --- Kernel vs gather oracle (CUDA only) ---


@cuda_kernel
@pytest.mark.parametrize("context_len", [1, 15, 16, 17, 31, 32, 63, 64, 190, 1024])
def test_kernel_matches_oracle_production_shape(context_len):
    """Direct kernel output must match gather + SDPA at head_dim 96."""
    cache = make_filled_cache(PROD, "cuda", context_len, block_size=16)
    q = torch.randn(1, PROD["num_heads"], 1, 96, device="cuda")
    direct = paged_attention.decode(q, cache, layer_idx=0, context_len=context_len)
    torch.testing.assert_close(
        direct, oracle_attention(q, cache, 0, context_len), atol=2e-5, rtol=1e-5
    )


@cuda_kernel
@pytest.mark.parametrize("context_len", [17, 190])
def test_kernel_handles_noncontiguous_blocks(context_len):
    """Scattered physical block ids must not change the result."""
    cache = make_filled_cache(PROD, "cuda", context_len, block_size=16, shuffle_blocks=True)
    table = cache.block_table
    assert any(b != a + 1 for a, b in zip(table, table[1:])), "table accidentally contiguous"
    q = torch.randn(1, PROD["num_heads"], 1, 96, device="cuda")
    direct = paged_attention.decode(q, cache, 0, context_len)
    torch.testing.assert_close(
        direct, oracle_attention(q, cache, 0, context_len), atol=2e-5, rtol=1e-5
    )


@cuda_kernel
@pytest.mark.parametrize("block_size", [4, 16, 32])
def test_kernel_handles_other_block_sizes(block_size):
    """Partial final blocks at several block sizes, small head_dim (8 heads of 4)."""
    for context_len in [1, block_size - 1, block_size, block_size + 1, 3 * block_size + 2]:
        cache = make_filled_cache(TINY, "cuda", context_len, block_size)
        q = torch.randn(1, 8, 1, 4, device="cuda")
        direct = paged_attention.decode(q, cache, 0, context_len)
        torch.testing.assert_close(
            direct, oracle_attention(q, cache, 0, context_len), atol=2e-5, rtol=1e-5
        )


@cuda_kernel
def test_kernel_matches_oracle_on_every_layer():
    """Layer indexing into the pools must hit the right layer.

    make_filled_cache writes distinct random K/V into every layer, so a
    kernel that read the wrong layer would match some other layer's oracle
    instead of its own; both directions are asserted here.
    """
    config = dict(PROD, num_layers=3)
    cache = make_filled_cache(config, "cuda", 40, block_size=16, seed=3)
    q = torch.randn(1, 8, 1, 96, device="cuda")
    oracles = [oracle_attention(q, cache, layer, 40) for layer in range(3)]
    assert not torch.allclose(oracles[0], oracles[1]), "layers accidentally identical"
    for layer in range(3):
        direct = paged_attention.decode(q, cache, layer, 40)
        torch.testing.assert_close(direct, oracles[layer], atol=2e-5, rtol=1e-5)
        for other in range(3):
            if other != layer:
                assert not torch.allclose(direct, oracles[other], atol=1e-3, rtol=1e-3)


# --- Model-level parity (CUDA only) ---


@cuda_kernel
def test_direct_decode_logits_match_gather_cuda():
    """Full-model decode logits: direct kernel vs gather oracle, production dims."""
    torch.manual_seed(0)
    config = ModelConfig(max_seq_length=256, num_layers=2)
    model = Model(config).to("cuda").eval()
    ids = torch.randint(0, 1000, (1, 62), device="cuda")

    logits = {}
    for mode in ["gather", "direct"]:
        allocator = model.new_block_allocator(num_blocks=16, block_size=16, attention_mode=mode)
        cache = allocator.new_cache()
        with torch.no_grad():
            steps = [model(ids[:, :50], cache)[:, -1]]
            for t in range(50, 62):
                steps.append(model(ids[:, t : t + 1], cache)[:, -1])
        logits[mode] = torch.stack(steps, dim=1)
        cache.free()
    torch.testing.assert_close(logits["direct"], logits["gather"], atol=1e-4, rtol=1e-4)


@cuda_kernel
def test_greedy_generation_matches_contiguous_cuda():
    """Whole greedy generations must be token-identical to the contiguous cache."""
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to("cuda").eval()
    prompt = torch.randint(0, TINY["vocab_size"], (8,), device="cuda")

    contiguous_out = model.generate(prompt, max_new_tokens=40, temperature=0)
    allocator = model.new_block_allocator(num_blocks=16, block_size=4, attention_mode="direct")
    cache = allocator.new_cache()
    direct_out = model.generate(prompt, max_new_tokens=40, temperature=0, cache=cache)
    cache.free()
    assert torch.equal(direct_out, contiguous_out)


@cuda_kernel
def test_blocks_reused_after_generation_cuda():
    """Freed blocks must be reusable for a second, identical generation."""
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to("cuda").eval()
    prompt = torch.randint(0, TINY["vocab_size"], (8,), device="cuda")
    allocator = model.new_block_allocator(num_blocks=16, block_size=4, attention_mode="direct")

    cache = allocator.new_cache()
    first = model.generate(prompt, max_new_tokens=20, temperature=0, cache=cache)
    cache.free()
    assert sorted(allocator.free_blocks) == list(range(16))

    cache = allocator.new_cache()
    second = model.generate(prompt, max_new_tokens=20, temperature=0, cache=cache)
    cache.free()
    assert torch.equal(first, second)


# --- Prefill fast path and fallback (all devices) ---


@pytest.mark.parametrize("device", fallback_devices())
def test_auto_mode_matches_gather_mode(device):
    """auto (fast prefill + fallback decode off-CUDA) must match the oracle path."""
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to(device).eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16), device=device)

    logits = {}
    for mode in ["gather", "auto"]:
        allocator = model.new_block_allocator(num_blocks=16, block_size=4, attention_mode=mode)
        cache = allocator.new_cache()
        with torch.no_grad():
            steps = [model(ids[:, :8], cache)[:, -1]]
            for t in range(8, 16):
                steps.append(model(ids[:, t : t + 1], cache)[:, -1])
        logits[mode] = torch.stack(steps, dim=1)
        cache.free()
    # Identical float32 math on identical values; only SDPA's layout differs.
    torch.testing.assert_close(logits["auto"], logits["gather"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", fallback_devices())
def test_prefill_fast_path_matches_contiguous(device):
    """Prompt prefill logits must match the contiguous cache exactly per-token."""
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to(device).eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, 19), device=device)  # partial last block

    with torch.no_grad():
        contiguous = model(ids, model.new_cache())
        allocator = model.new_block_allocator(num_blocks=16, block_size=4)
        cache = allocator.new_cache()
        paged = model(ids, cache)
        cache.free()
    torch.testing.assert_close(paged, contiguous, atol=1e-5, rtol=1e-5)


def assert_cache_untouched(cache, allocator, num_blocks):
    """A rejected forward must leave no trace in the cache or the pool."""
    assert cache.position == 0
    assert cache.block_table == []
    assert sorted(allocator.free_blocks) == list(range(num_blocks))
    # The device mirror was never appended to and no write indices were cached.
    assert (cache.block_table_device == 0).all()
    assert cache._write_index_key is None


def test_direct_mode_raises_without_cuda():
    """Forcing 'direct' where the kernel cannot run must fail loudly, not silently,
    and must fail before any block is allocated or K/V is written."""
    if CUDA_KERNEL:
        pytest.skip("this machine can run the kernel")
    model = Model(ModelConfig(**TINY)).eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, 5))
    allocator = model.new_block_allocator(num_blocks=8, block_size=4, attention_mode="direct")
    cache = allocator.new_cache()
    with torch.no_grad(), pytest.raises(RuntimeError, match="direct"):
        model(ids, cache)
    assert_cache_untouched(cache, allocator, num_blocks=8)


@cuda_kernel
def test_direct_mode_raises_on_unsupported_dtype():
    """The kernel is float32-only; forced 'direct' in float64 must reject the
    forward before mutating the cache, even on CUDA."""
    model = Model(ModelConfig(**TINY)).to("cuda").double().eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, 5), device="cuda")
    allocator = model.new_block_allocator(num_blocks=8, block_size=4, attention_mode="direct")
    cache = allocator.new_cache()
    with torch.no_grad(), pytest.raises(RuntimeError, match="direct"):
        model(ids, cache)
    assert_cache_untouched(cache, allocator, num_blocks=8)


def test_invalid_attention_mode_rejected():
    model = Model(ModelConfig(**TINY))
    with pytest.raises(ValueError, match="attention_mode"):
        model.new_block_allocator(num_blocks=4, block_size=4, attention_mode="turbo")


def test_device_block_table_mirrors_python_table():
    """The device mirror must track the Python block table incrementally."""
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).eval()
    allocator = model.new_block_allocator(num_blocks=16, block_size=4)
    random.Random(1).shuffle(allocator.free_blocks)
    cache = allocator.new_cache()
    ids = torch.randint(0, TINY["vocab_size"], (1, 10))
    with torch.no_grad():
        model(ids, cache)  # 10 tokens -> 3 blocks
        for t in range(3):  # cross into a 4th block
            model(ids[:, t : t + 1], cache)
    n = len(cache.block_table)
    assert n == 4
    assert cache.block_table_device[:n].tolist() == cache.block_table
