"""Parity tests for the fused MoE kernel against the expert loop.

These need CUDA and Triton, so they run on the GPU box and skip locally.
"""

import pytest
import torch

requires_triton = False
if torch.cuda.is_available():
    try:
        import triton  # noqa: F401

        requires_triton = True
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(not requires_triton, reason="needs CUDA and Triton")

TINY = dict(max_seq_length=32, vocab_size=96, num_layers=2, hidden_dim=32, num_experts=4, top_k=2)


def test_moe_align_block_size_layout():
    from moe_engine.fused_moe import moe_align_block_size

    # Pairs flatten to [0, 1, 2, 1, 1, 3, 0, 2]; counts per expert 2/3/2/1.
    topk_ids = torch.tensor([[0, 1], [2, 1], [1, 3], [0, 2]], device="cuda")
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 4, 4)

    assert num_post.item() == 16  # Four experts, each padded to one block of 4.
    sentinel = 8
    assert sorted_ids[:16].tolist() == [
        0, 6, sentinel, sentinel,  # expert 0
        1, 3, 4, sentinel,         # expert 1
        2, 7, sentinel, sentinel,  # expert 2
        5, sentinel, sentinel, sentinel,  # expert 3
    ]
    assert expert_ids[:4].tolist() == [0, 1, 2, 3]
    assert (sorted_ids[16:] == sentinel).all()  # Worst-case tail stays padding.


def loop_experts(x, w1, b1, w2, b2, topk_weights, topk_ids):
    """The same math as MoEFeedForward's expert loop, as a plain function."""
    out = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        token_ids, slot = torch.where(topk_ids == expert)
        if token_ids.numel() == 0:
            continue
        hidden = torch.nn.functional.gelu(x[token_ids] @ w1[expert].T + b1[expert])
        expert_out = hidden @ w2[expert].T + b2[expert]
        weight = topk_weights[token_ids, slot].unsqueeze(-1).to(x.dtype)
        out.index_add_(0, token_ids, weight * expert_out)
    return out


@pytest.mark.parametrize("tokens", [1, 2, 16, 63, 200])
def test_fused_matches_loop(tokens):
    from moe_engine.fused_moe import fused_moe

    torch.manual_seed(0)
    dim, hidden, num_experts, top_k = 32, 128, 4, 2
    x = torch.randn(tokens, dim, device="cuda")
    w1 = torch.randn(num_experts, hidden, dim, device="cuda") * 0.05
    b1 = torch.randn(num_experts, hidden, device="cuda") * 0.05
    w2 = torch.randn(num_experts, dim, hidden, device="cuda") * 0.05
    b2 = torch.randn(num_experts, dim, device="cuda") * 0.05
    logits = torch.randn(tokens, num_experts, device="cuda")
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), k=top_k, dim=-1)

    got = fused_moe(x, w1, b1, w2, b2, topk_weights, topk_ids)
    want = loop_experts(x, w1, b1, w2, b2, topk_weights, topk_ids)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", [1, 200])
def test_fused_matches_loop_at_production_shape(tokens):
    """The shipped shape (768/3072, 8 experts, top-2), random weights."""
    from moe_engine.model import MoEFeedForward

    torch.manual_seed(0)
    moe = MoEFeedForward(768, 3072, num_experts=8, top_k=2).eval().cuda()
    x = torch.randn(1, tokens, 768, device="cuda")
    with torch.no_grad():
        moe.fused_enabled = True
        fused = moe(x)
        moe.fused_enabled = False
        loop = moe(x)
    torch.testing.assert_close(fused, loop, atol=1e-4, rtol=1e-4)


def test_model_forward_fused_matches_loop():
    from moe_engine.model import Model, ModelConfig

    torch.manual_seed(0)
    model = Model(ModelConfig(**TINY)).eval().cuda()
    ids = torch.randint(0, TINY["vocab_size"], (1, 16), device="cuda")

    def set_fused(enabled):
        for block in model.MoEBlocks:
            block.moe.fused_enabled = enabled

    with torch.no_grad():
        set_fused(True)
        fused_logits = model(ids, model.new_cache())
        set_fused(False)
        loop_logits = model(ids, model.new_cache())
    torch.testing.assert_close(fused_logits, loop_logits, atol=1e-4, rtol=1e-4)

    set_fused(True)
    fused_out = model.generate(ids[0], max_new_tokens=12, temperature=0)
    set_fused(False)
    loop_out = model.generate(ids[0], max_new_tokens=12, temperature=0)
    assert torch.equal(fused_out, loop_out)
