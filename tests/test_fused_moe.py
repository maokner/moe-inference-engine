"""CUDA correctness preparation for the fixed-shape batch-one MoE path.

These tests intentionally exercise production dimensions and are not part of
the stage-1 local run.  The CUDA/Triton cases must run on the gated A6000.
Float32 reductions and transcendental implementations differ between Triton
and PyTorch, so kernel comparisons use explicit tolerances while expert ids
and greedy tokens must match exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import tiktoken
import torch
from torch.nn import functional as F

from moe_engine import fused_moe
from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model
from moe_engine.model import Model, ModelConfig, MoEFeedForward

CUDA_KERNEL = torch.cuda.is_available() and fused_moe.HAS_TRITON
cuda_kernel = pytest.mark.skipif(not CUDA_KERNEL, reason="needs CUDA and Triton")
CHECKPOINT = Path(os.environ.get("MINIMOE_CHECKPOINT", "checkpoints/minimoe_sft.pt"))
TINY = {
    "max_seq_length": 32,
    "vocab_size": 96,
    "num_layers": 2,
    "hidden_dim": 32,
    "num_experts": 4,
    "top_k": 2,
}


def make_production_moe(mode="direct"):
    torch.manual_seed(0)
    return MoEFeedForward(768, 3072, 8, 2, mode=mode).to("cuda").eval()


def route_with_workspace(moe, x):
    fused_moe.route(
        x,
        moe.router.weight,
        moe._router_logits,
        moe._expert_ids,
        moe._route_weights,
    )
    return (
        moe._router_logits.clone(),
        moe._expert_ids.clone(),
        moe._route_weights.clone(),
    )


@cuda_kernel
def test_router_logits_top2_ids_and_normalized_weights_match_pytorch():
    moe = make_production_moe()
    x = torch.randn(1, 1, 768, device="cuda")
    logits, expert_ids, route_weights = route_with_workspace(moe, x)

    expected_logits = F.linear(x.reshape(1, 768), moe.router.weight).squeeze(0)
    expected_top2, expected_ids = torch.topk(expected_logits, 2)
    expected_weights = F.softmax(expected_top2, dim=0)

    torch.testing.assert_close(logits, expected_logits, atol=2e-5, rtol=1e-5)
    assert torch.equal(expert_ids.long(), expected_ids)
    torch.testing.assert_close(route_weights, expected_weights, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(route_weights.sum(), torch.ones((), device="cuda"))


@cuda_kernel
def test_every_expert_pointer_works_in_both_routes():
    moe = make_production_moe()
    x = torch.ones(1, 1, 768, device="cuda")
    first_routes = []
    second_routes = []
    with torch.no_grad():
        # Every expert gets unique W1, B1, W2, and B2 values.  Addressing the
        # wrong expert in either projection or bias therefore changes output.
        for expert_id, expert in enumerate(moe.experts):
            expert.net[0].weight.zero_()
            expert.net[0].weight[0, 0] = 0.2 * (expert_id + 1)
            expert.net[0].bias.copy_(
                torch.linspace(-0.3, 0.3, 3072, device="cuda") + 0.05 * expert_id
            )
            expert.net[2].weight.zero_()
            expert.net[2].weight[:, 0].copy_(
                torch.linspace(0.01, 0.02, 768, device="cuda") * (expert_id + 1)
            )
            expert.net[2].bias.copy_(
                torch.linspace(-0.2, 0.2, 768, device="cuda") + 0.1 * expert_id
            )

        for first_id in range(8):
            second_id = (first_id + 3) % 8
            # All eight logits are distinct and both selected routes have
            # material weight, so route swapping cannot hide behind a tie.
            logits = -2.0 - 0.1 * torch.arange(8, device="cuda")
            logits[first_id] = 1.0
            logits[second_id] = 0.25
            moe.router.weight.copy_(logits[:, None].expand(8, 768) / 768)

            _, ids, _ = route_with_workspace(moe, x)
            first_routes.append(ids[0])
            second_routes.append(ids[1])
            expected = moe._forward_reference(x)
            actual = moe(x, decode=True)
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=2e-4)

    assert torch.equal(torch.stack(first_routes), torch.arange(8, device="cuda"))
    assert torch.equal(
        torch.stack(second_routes),
        (torch.arange(8, device="cuda") + 3) % 8,
    )


@cuda_kernel
def test_selected_expert_output_matches_pytorch_oracle():
    moe = make_production_moe()
    x = torch.randn(1, 1, 768, device="cuda")
    with torch.no_grad():
        expected = moe._forward_reference(x)
        actual = moe(x, decode=True)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=2e-4)


@cuda_kernel
def test_both_expert_biases_and_exact_gelu_semantics_match_oracle():
    moe = make_production_moe()
    x = torch.linspace(-1.5, 1.5, 768, device="cuda").view(1, 1, 768)
    with torch.no_grad():
        # Nonzero, sign-varying biases make omission of either bias observable.
        for expert_id, expert in enumerate(moe.experts):
            expert.net[0].bias.copy_(
                torch.linspace(-2.0, 2.0, 3072, device="cuda") + expert_id * 0.01
            )
            expert.net[2].bias.fill_((expert_id + 1) * 0.025)
        expected = moe._forward_reference(x)
        actual = moe(x, decode=True)
    # This covers nn.GELU(approximate="none") over negative and positive input.
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=2e-4)


def test_auto_falls_back_on_unsupported_cpu_and_forced_direct_fails():
    torch.manual_seed(0)
    moe = MoEFeedForward(32, 128, 4, 2, mode="auto").eval()
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        expected = moe._forward_reference(x)
        actual = moe(x, decode=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    moe.set_mode("direct")
    with pytest.raises(RuntimeError, match="direct"):
        moe(x, decode=True)
    # Forced direct never applies to prefill, even on an unsupported device.
    prefill_x = x.expand(1, 2, 32)
    with torch.no_grad():
        expected_prefill = moe._forward_reference(prefill_x)
        actual_prefill = moe(prefill_x, decode=False)
    torch.testing.assert_close(actual_prefill, expected_prefill, atol=0, rtol=0)


@cuda_kernel
def test_forced_direct_preflight_preserves_paged_cache_at_block_boundary():
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to("cuda").eval()
    allocator = model.new_block_allocator(num_blocks=4, block_size=4)
    cache = allocator.new_cache()
    ids = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
    with torch.no_grad():
        model(ids, cache)

    assert cache.position == 4
    assert len(cache.block_table) == 1
    position_before = cache.position
    table_before = list(cache.block_table)
    free_before = sorted(allocator.free_blocks)
    write_index_key_before = cache._write_index_key
    mirror_before = cache.block_table_device[:2].clone()
    k_before = allocator.k_pool.clone()
    v_before = allocator.v_pool.clone()

    model.set_moe_mode("direct")
    with torch.no_grad(), pytest.raises(RuntimeError, match="direct"):
        model(ids[:, :1], cache)

    assert cache.position == position_before
    assert cache.block_table == table_before
    assert sorted(allocator.free_blocks) == free_before
    assert cache._write_index_key == write_index_key_before
    assert torch.equal(cache.block_table_device[:2], mirror_before)
    assert torch.equal(allocator.k_pool, k_before)
    assert torch.equal(allocator.v_pool, v_before)


@cuda_kernel
def test_forced_direct_preflight_preserves_contiguous_cache():
    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).to("cuda").eval()
    cache = model.new_cache()
    ids = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
    with torch.no_grad():
        model(ids, cache)

    position_before = cache.position
    k_before = cache.k.clone()
    v_before = cache.v.clone()
    model.set_moe_mode("direct")
    with torch.no_grad(), pytest.raises(RuntimeError, match="direct"):
        model(ids[:, :1], cache)

    assert cache.position == position_before
    assert torch.equal(cache.k, k_before)
    assert torch.equal(cache.v, v_before)


@cuda_kernel
def test_auto_falls_back_on_float64_and_forced_direct_fails():
    moe = make_production_moe(mode="auto").double()
    x = torch.randn(1, 1, 768, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        expected = moe._forward_reference(x)
        actual = moe(x, decode=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    moe.set_mode("direct")
    with pytest.raises(RuntimeError, match="float32"):
        moe(x, decode=True)


def assert_canonical_storage(moe, *, device_type, dtype):
    for storage_name, layer_index, parameter_name in (
        ("w1", 0, "weight"),
        ("b1", 0, "bias"),
        ("w2", 2, "weight"),
        ("b2", 2, "bias"),
    ):
        storage = moe._expert_storage[storage_name]
        parameters = [
            getattr(expert.net[layer_index], parameter_name) for expert in moe.experts
        ]
        stride_bytes = parameters[0].numel() * parameters[0].element_size()
        assert storage.device.type == device_type
        assert storage.dtype == dtype
        assert storage.is_contiguous()
        assert all(parameter.is_contiguous() for parameter in parameters)
        assert storage.data_ptr() == parameters[0].data_ptr()
        assert [parameter.data_ptr() for parameter in parameters] == [
            parameters[0].data_ptr() + expert_id * stride_bytes
            for expert_id in range(moe.num_experts)
        ]


def assert_checkpoint_values(moe, checkpoint):
    for name, parameter in moe.state_dict().items():
        torch.testing.assert_close(parameter, checkpoint[name], atol=0, rtol=0)


def test_canonical_storage_survives_load_and_dtype_apply():
    moe = MoEFeedForward(32, 128, 4, 2)
    expected = {
        f"experts.{expert}.{layer}.{parameter}"
        for expert in range(4)
        for layer in ("net.0", "net.2")
        for parameter in ("weight", "bias")
    }
    expected.add("router.weight")
    assert set(moe.state_dict()) == expected
    checkpoint = {
        name: torch.randn_like(parameter)
        for name, parameter in moe.state_dict().items()
    }
    moe.load_state_dict(checkpoint)
    assert_checkpoint_values(moe, checkpoint)
    assert_canonical_storage(moe, device_type="cpu", dtype=torch.float32)

    assigned = {
        name: torch.randn_like(parameter)
        for name, parameter in moe.state_dict().items()
    }
    moe.load_state_dict(assigned, assign=True)
    assert_checkpoint_values(moe, assigned)
    assert_canonical_storage(moe, device_type="cpu", dtype=torch.float32)

    moe.double()
    assert_canonical_storage(moe, device_type="cpu", dtype=torch.float64)
    moe.float()
    assert_canonical_storage(moe, device_type="cpu", dtype=torch.float32)


@cuda_kernel
def test_canonical_storage_survives_device_apply():
    moe = MoEFeedForward(32, 128, 4, 2).to("cuda")
    assert_canonical_storage(moe, device_type="cuda", dtype=torch.float32)
    moe.cpu()
    assert_canonical_storage(moe, device_type="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def checkpoint_model():
    if not CUDA_KERNEL:
        pytest.skip("needs CUDA and Triton")
    if not CHECKPOINT.exists():
        pytest.skip(f"real checkpoint not found at {CHECKPOINT}")
    model, _, metadata = load_engine_model(CHECKPOINT, "cuda")
    return model, metadata


def test_real_checkpoint_full_model_one_token_decode_logits(checkpoint_model):
    model, metadata = checkpoint_model
    assert metadata["step"] is not None
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device="cuda").unsqueeze(0)

    logits = {}
    with torch.no_grad():
        for mode in ("reference", "direct"):
            model.set_moe_mode(mode)
            cache = model.new_cache()
            prefill_logits = model(prompt_ids, cache)
            step_input = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
            logits[mode] = model(step_input, cache)
    torch.testing.assert_close(
        logits["direct"], logits["reference"], atol=1e-3, rtol=5e-4
    )


def test_real_checkpoint_full_greedy_tokens_match(checkpoint_model):
    model, _ = checkpoint_model
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device="cuda")
    generations = {}
    with torch.no_grad():
        for mode in ("reference", "direct"):
            model.set_moe_mode(mode)
            generations[mode] = model.generate(
                prompt_ids, max_new_tokens=64, temperature=0
            )
    assert torch.equal(generations["direct"], generations["reference"])
