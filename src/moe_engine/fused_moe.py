"""Fixed-shape Triton MoE kernels for batch-one float32 CUDA decode.

The production decode shape is one token with hidden size 768, eight experts,
top-k two, and expert hidden size 3072.  Prefill and every unsupported shape
stay on the PyTorch oracle in :mod:`moe_engine.model`.

The kernels receive four canonical contiguous expert-storage groups whose
slices are the checkpoint-named Parameters.  An expert id therefore selects
weights with pointer arithmetic and remains on the device throughout routing
and both expert projections.  No route-dependent tensor shape, host read,
expert-discovery operation, or per-token weight pack is needed.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU/MPS machines use the PyTorch oracle.
    HAS_TRITON = False


HIDDEN_DIM = 768
NUM_EXPERTS = 8
TOP_K = 2
EXPERT_HIDDEN_DIM = 3072


def is_configuration_supported(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    expert_hidden_dim: int,
) -> bool:
    """Return whether model metadata can use the fixed direct decode path."""
    return (
        HAS_TRITON
        and device.type == "cuda"
        and dtype == torch.float32
        and batch_size == 1
        and hidden_dim == HIDDEN_DIM
        and num_experts == NUM_EXPERTS
        and top_k == TOP_K
        and expert_hidden_dim == EXPERT_HIDDEN_DIM
    )


def is_supported(
    x: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    expert_hidden_dim: int,
) -> bool:
    """Return whether ``x`` and the model shape can use the direct kernels."""
    if x.shape != (1, 1, HIDDEN_DIM) or not x.is_contiguous():
        return False
    return is_configuration_supported(
        device=x.device,
        dtype=x.dtype,
        batch_size=x.shape[0],
        hidden_dim=x.shape[-1],
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
    )


def has_canonical_layout(
    router_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    router_logits: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    hidden: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Validate direct-kernel shapes, contiguity, dtype, and device metadata."""
    expected = (
        (router_weight, (NUM_EXPERTS, HIDDEN_DIM)),
        (w1, (NUM_EXPERTS, EXPERT_HIDDEN_DIM, HIDDEN_DIM)),
        (b1, (NUM_EXPERTS, EXPERT_HIDDEN_DIM)),
        (w2, (NUM_EXPERTS, HIDDEN_DIM, EXPERT_HIDDEN_DIM)),
        (b2, (NUM_EXPERTS, HIDDEN_DIM)),
        (router_logits, (NUM_EXPERTS,)),
        (route_weights, (TOP_K,)),
        (hidden, (TOP_K, EXPERT_HIDDEN_DIM)),
        (out, (HIDDEN_DIM,)),
    )
    device = router_weight.device
    return (
        all(
            tensor.shape == shape
            and tensor.is_contiguous()
            and tensor.device == device
            and tensor.dtype == torch.float32
            for tensor, shape in expected
        )
        and expert_ids.shape == (TOP_K,)
        and expert_ids.is_contiguous()
        and expert_ids.device == device
        and expert_ids.dtype == torch.int32
    )


if HAS_TRITON:

    @triton.jit
    def _router_top2_kernel(
        x_ptr,  # [768]
        router_ptr,  # [8, 768], row-major
        logits_ptr,  # [8] float32 output
        expert_ids_ptr,  # [2] int32 output
        route_weights_ptr,  # [2] float32 output
        DIM: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Grid = (1,).  Eight router rows are reduced together in one program.
        d = tl.arange(0, BLOCK_D)
        experts = tl.arange(0, NUM_EXPERTS)
        d_mask = d < DIM
        x = tl.load(x_ptr + d, mask=d_mask, other=0.0)
        router = tl.load(
            router_ptr + experts[:, None] * DIM + d[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(router * x[None, :], axis=1)
        tl.store(logits_ptr + experts, logits)

        # tl.argmax breaks ties toward the lower lane.  Real checkpoint logits
        # are expected to be distinct; the CUDA stage includes an explicit
        # comparison with torch.topk to detect any relevant tie discrepancy.
        first_id = tl.argmax(logits, axis=0)
        without_first = tl.where(experts == first_id, float("-inf"), logits)
        second_id = tl.argmax(without_first, axis=0)
        first_logit = tl.max(logits, axis=0)
        second_logit = tl.max(without_first, axis=0)

        # Stable two-value softmax.  Routing weights remain float32 on device.
        second_exp = tl.exp(second_logit - first_logit)
        denom = 1.0 + second_exp
        tl.store(expert_ids_ptr, first_id)
        tl.store(expert_ids_ptr + 1, second_id)
        tl.store(route_weights_ptr, 1.0 / denom)
        tl.store(route_weights_ptr + 1, second_exp / denom)

    @triton.jit
    def _selected_w1_gelu_kernel(
        x_ptr,  # [768]
        expert_ids_ptr,  # [2] int32, device-resident
        w1_ptr,  # canonical [8, 3072, 768] storage
        b1_ptr,  # canonical [8, 3072] storage
        hidden_ptr,  # [2, 3072] float32 output
        DIM: tl.constexpr,
        EXPERT_HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Grid = (ceil(3072 / BLOCK_N), 2).  Axis 1 is the top-2 route.
        hidden_block = tl.program_id(0)
        route = tl.program_id(1)
        hidden = hidden_block * BLOCK_N + tl.arange(0, BLOCK_N)
        hidden_mask = hidden < EXPERT_HIDDEN
        expert_id = tl.load(expert_ids_ptr + route)

        # Each program computes BLOCK_N independent dot products of length 768.
        # The selected expert is addressed by an on-device storage offset.
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for start in range(0, DIM, BLOCK_K):
            d = start + tl.arange(0, BLOCK_K)
            d_mask = d < DIM
            x = tl.load(x_ptr + d, mask=d_mask, other=0.0)
            w_offset = (
                expert_id * EXPERT_HIDDEN * DIM + hidden[:, None] * DIM + d[None, :]
            )
            w = tl.load(
                w1_ptr + w_offset,
                mask=hidden_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(w * x[None, :], axis=1)

        pre_activation = acc + tl.load(
            b1_ptr + expert_id * EXPERT_HIDDEN + hidden,
            mask=hidden_mask,
            other=0.0,
        )
        # nn.GELU() defaults to approximation="none": x * Phi(x), using erf.
        activated = (
            0.5 * pre_activation * (1.0 + tl.erf(pre_activation * 0.7071067811865476))
        )
        tl.store(
            hidden_ptr + route * EXPERT_HIDDEN + hidden,
            activated,
            mask=hidden_mask,
        )

    @triton.jit
    def _selected_w2_reduce_kernel(
        hidden_ptr,  # [2, 3072]
        expert_ids_ptr,  # [2] int32, device-resident
        route_weights_ptr,  # [2] float32, device-resident
        w2_ptr,  # canonical [8, 768, 3072] storage
        b2_ptr,  # canonical [8, 768] storage
        out_ptr,  # [768] float32 output
        DIM: tl.constexpr,
        EXPERT_HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Grid = (ceil(768 / BLOCK_N),).  Each program produces one output tile
        # after evaluating both selected W2 rows and applying the route weights.
        output_block = tl.program_id(0)
        output = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
        output_mask = output < DIM
        combined = tl.zeros([BLOCK_N], dtype=tl.float32)

        for route in range(TOP_K):
            expert_id = tl.load(expert_ids_ptr + route)
            route_weight = tl.load(route_weights_ptr + route)
            acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for start in range(0, EXPERT_HIDDEN, BLOCK_K):
                hidden = start + tl.arange(0, BLOCK_K)
                hidden_mask = hidden < EXPERT_HIDDEN
                activation = tl.load(
                    hidden_ptr + route * EXPERT_HIDDEN + hidden,
                    mask=hidden_mask,
                    other=0.0,
                )
                w_offset = (
                    expert_id * DIM * EXPERT_HIDDEN
                    + output[:, None] * EXPERT_HIDDEN
                    + hidden[None, :]
                )
                w = tl.load(
                    w2_ptr + w_offset,
                    mask=output_mask[:, None] & hidden_mask[None, :],
                    other=0.0,
                )
                acc += tl.sum(w * activation[None, :], axis=1)
            bias = tl.load(
                b2_ptr + expert_id * DIM + output,
                mask=output_mask,
                other=0.0,
            )
            combined += route_weight * (acc + bias)

        tl.store(out_ptr + output, combined, mask=output_mask)


def _require_cuda_float32(*tensors: torch.Tensor) -> None:
    if not HAS_TRITON or any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("direct MoE needs CUDA and Triton")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("direct MoE needs float32 activations and parameters")
    if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("direct MoE tensors must be on the same CUDA device")


def route(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_logits: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> None:
    """Launch the fixed 8-way router and normalized top-2 kernel."""
    _require_cuda_float32(x, router_weight, router_logits, route_weights)
    if (
        x.numel() != HIDDEN_DIM
        or not x.is_contiguous()
        or router_weight.shape != (NUM_EXPERTS, HIDDEN_DIM)
        or not router_weight.is_contiguous()
    ):
        raise ValueError("direct MoE router needs x [768] and router weight [8, 768]")
    if (
        not expert_ids.is_cuda
        or expert_ids.device != x.device
        or expert_ids.dtype != torch.int32
        or expert_ids.shape != (TOP_K,)
    ):
        raise ValueError("direct MoE expert id workspace must be CUDA int32")
    if router_logits.shape != (NUM_EXPERTS,) or route_weights.shape != (TOP_K,):
        raise ValueError("direct MoE router workspaces have invalid shapes")
    _router_top2_kernel[(1,)](
        x,
        router_weight,
        router_logits,
        expert_ids,
        route_weights,
        DIM=HIDDEN_DIM,
        NUM_EXPERTS=NUM_EXPERTS,
        BLOCK_D=1024,
        num_warps=4,
    )


def decode(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    w1_base: torch.Tensor,
    b1_base: torch.Tensor,
    w2_base: torch.Tensor,
    b2_base: torch.Tensor,
    *,
    router_logits: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    hidden: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run router, the two selected W1+GELU paths, and fused W2 reduction."""
    _require_cuda_float32(
        x,
        router_weight,
        w1_base,
        b1_base,
        w2_base,
        b2_base,
        router_logits,
        route_weights,
        hidden,
        out,
    )
    if not has_canonical_layout(
        router_weight,
        w1_base,
        b1_base,
        w2_base,
        b2_base,
        router_logits,
        expert_ids,
        route_weights,
        hidden,
        out,
    ):
        raise ValueError("direct MoE weights and workspaces must use canonical layouts")
    route(x, router_weight, router_logits, expert_ids, route_weights)
    flat_x = x.reshape(HIDDEN_DIM)
    _selected_w1_gelu_kernel[(48, TOP_K)](
        flat_x,
        expert_ids,
        w1_base,
        b1_base,
        hidden,
        DIM=HIDDEN_DIM,
        EXPERT_HIDDEN=EXPERT_HIDDEN_DIM,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=4,
    )
    _selected_w2_reduce_kernel[(12,)](
        hidden,
        expert_ids,
        route_weights,
        w2_base,
        b2_base,
        out,
        DIM=HIDDEN_DIM,
        EXPERT_HIDDEN=EXPERT_HIDDEN_DIM,
        TOP_K=TOP_K,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=4,
    )
    return out.view(1, 1, HIDDEN_DIM)
