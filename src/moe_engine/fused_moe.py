"""Fused MoE dispatch: one grouped GEMM per projection instead of a Python
loop over experts.

The design follows vLLM's fused_moe. Every (token, expert-slot) pair is one
row of work. Pairs are sorted by expert and each expert's segment is padded
to a multiple of BLOCK_M, so a [BLOCK_M, BLOCK_N] GEMM tile always serves
exactly one expert and can load that expert's weight tile directly. Two
kernel launches cover the expert MLP:

    1. up projection + bias + GELU        (per pair, no routing weight)
    2. down projection + bias * weight    (per pair, routing weight fused)

followed by a plain torch sum over each token's top-k pair outputs.

Simplifications against vLLM: no quantization, no expert parallelism, no
autotuned configs, no L2 swizzling of the launch grid. Additions: bias and
exact-erf GELU in the epilogues, because miniMoE's experts use nn.Linear
with bias and nn.GELU.
"""

import torch
import triton
import triton.language as tl


def moe_align_block_size(topk_ids, block_size: int, num_experts: int):
    """Sort (token, slot) pairs by expert and pad segments to block_size.

    Returns worst-case sized tensors plus a device-side pair count, so no
    value ever crosses back to the host (a .item() here would synchronize
    the GPU once per layer per step; vLLM uses a CUDA op for the same
    reason).

    - sorted_ids: pair indices grouped by expert; padding slots hold the
      sentinel value `num_pairs`, which the kernel masks out.
    - expert_ids: the expert served by each BLOCK_M-sized block.
    - num_post: 0-dim tensor, number of pair slots actually in use.
    """
    flat = topk_ids.reshape(-1)
    num_pairs = flat.numel()
    device = flat.device

    # Not bincount: its output size is data-dependent, which forces a
    # device-to-host sync per call. scatter_add keeps everything on-device.
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    counts.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.long))
    padded = -(-counts // block_size) * block_size
    ends = padded.cumsum(0)
    seg_starts = ends - padded
    num_post = ends[-1]

    # Worst case: every expert wastes block_size - 1 padding slots.
    max_len = num_pairs + num_experts * (block_size - 1)
    max_len = -(-max_len // block_size) * block_size
    sorted_ids = torch.full((max_len,), num_pairs, dtype=torch.int32, device=device)

    # Stable sort keeps pairs of one expert in token order.
    order = torch.argsort(flat, stable=True)
    experts_in_order = flat[order]
    rank_within_expert = (
        torch.arange(num_pairs, device=device)
        - (counts.cumsum(0) - counts)[experts_in_order]
    )
    positions = seg_starts[experts_in_order] + rank_within_expert
    sorted_ids[positions] = order.to(torch.int32)

    # Expert per block: how many padded segments end at or before the block.
    block_starts = torch.arange(max_len // block_size, device=device) * block_size
    expert_ids = torch.searchsorted(ends, block_starts, right=True).to(torch.int32)
    expert_ids = expert_ids.clamp(max=num_experts - 1)  # Blocks past num_post exit early.
    return sorted_ids, expert_ids, num_post


@triton.jit
def gelu(x):
    # Exact erf form, matching nn.GELU's default.
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def fused_moe_kernel(
    a_ptr,  # [num_rows, K] input rows
    b_ptr,  # [E, N, K] stacked expert weights (nn.Linear layout)
    bias_ptr,  # [E, N] stacked expert biases
    c_ptr,  # [num_pairs, N] output rows, one per pair
    topk_weights_ptr,  # [num_pairs] routing weights (flat, pair order)
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    N,
    K,
    num_pairs,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_ROWS_PER_PAIR: tl.constexpr,  # top_k when A holds tokens, 1 when per-pair
    APPLY_GELU: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_post = tl.load(num_post_ptr)
    if pid_m * BLOCK_M >= num_post:
        return  # This block is entirely worst-case padding.

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    pair_mask = pair_ids < num_pairs  # Padding slots hold the sentinel.
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # In the up projection A holds one row per token and top_k pairs share
    # it; in the down projection A already holds one row per pair.
    a_rows = pair_ids // A_ROWS_PER_PAIR

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        # "ieee" keeps fp32 GEMMs exactly fp32 (no tf32), matching torch.
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + expert * N + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    if APPLY_GELU:
        acc = gelu(acc)
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + pair_ids, mask=pair_mask, other=0.0)
        acc *= weight[:, None]

    c_ptrs = c_ptr + pair_ids[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=pair_mask[:, None] & (offs_n[None, :] < N),
    )


# One fixed config; vLLM autotunes these per shape from config files.
BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32


def fused_moe(x, w1, b1, w2, b2, topk_weights, topk_ids):
    """Run the expert MLPs for every (token, expert-slot) pair.

    x: [tokens, dim]; w1/b1: [E, hidden, dim] / [E, hidden];
    w2/b2: [E, dim, hidden] / [E, dim]; topk_weights/topk_ids: [tokens, top_k].
    Returns [tokens, dim]: the weighted sum over each token's experts.
    """
    tokens, dim = x.shape
    num_experts, hidden, _ = w1.shape
    top_k = topk_ids.shape[1]
    num_pairs = tokens * top_k

    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, BLOCK_M, num_experts
    )
    grid_m = sorted_ids.numel() // BLOCK_M

    def launch(a, b, bias, c, weights, a_rows_per_pair, apply_gelu, mul_weight):
        n = b.shape[1]
        grid = (grid_m * triton.cdiv(n, BLOCK_N),)
        fused_moe_kernel[grid](
            a, b, bias, c, weights, sorted_ids, expert_ids, num_post,
            n, b.shape[2], num_pairs,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            A_ROWS_PER_PAIR=a_rows_per_pair,
            APPLY_GELU=apply_gelu, MUL_ROUTED_WEIGHT=mul_weight,
        )

    flat_weights = topk_weights.reshape(-1).to(x.dtype)
    up = torch.empty((num_pairs, hidden), device=x.device, dtype=x.dtype)
    launch(x, w1, b1, up, flat_weights, top_k, apply_gelu=True, mul_weight=False)

    down = torch.empty((num_pairs, dim), device=x.device, dtype=x.dtype)
    launch(up, w2, b2, down, flat_weights, 1, apply_gelu=False, mul_weight=True)

    return down.view(tokens, top_k, dim).sum(dim=1)
