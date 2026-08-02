"""Direct paged attention: a Triton decode kernel that reads K/V straight
from the physical block pool through the sequence's block table.

The old paged path gathered every block into a fresh contiguous K and V
tensor at every layer of every decode step, then called PyTorch attention.
This kernel removes the gather: for each logical token position p it computes

    logical block  = p // block_size
    block offset   = p %  block_size
    physical block = block_table[logical block]

and loads K/V directly from the pool at that physical address. The gather
path stays available as the correctness oracle and the CPU/MPS fallback.

Scope: one sequence, one query token (decode). Prefill never needs this
kernel because during an initial prefill the full history *is* the current
contiguous projection output (see CachedAttention.forward).
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU/MPS machines: the gather fallback is used instead.
    HAS_TRITON = False


# Tokens processed per loop iteration inside the kernel. 128 tokens = 8 KV
# blocks per iteration at the default block size of 16; the 190-token
# benchmark context needs 2 iterations. Must be a power of two for tl.arange.
BLOCK_T = 128


def is_supported(x: torch.Tensor) -> bool:
    """The kernel path needs CUDA, Triton, and float32 activations."""
    return HAS_TRITON and x.is_cuda and x.dtype == torch.float32


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr,  # [heads, head_dim] query for the single new token
        k_ptr,  # [num_blocks, heads, block_size, head_dim] one layer's K pool
        v_ptr,  # same layout as k_ptr
        out_ptr,  # [heads, head_dim]
        block_table_ptr,  # [>= ceil(ctx_len / block_size)] int32 physical block ids
        ctx_len,  # number of cached tokens to attend over (includes the new token)
        scale,  # 1 / sqrt(head_dim)
        stride_qh,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_oh,
        BLOCK_SIZE: tl.constexpr,  # tokens per KV block
        BLOCK_T: tl.constexpr,  # tokens per loop iteration
        HEAD_DIM: tl.constexpr,  # true head dim (96 in production)
        BLOCK_D: tl.constexpr,  # head dim padded to a power of two (128)
    ):
        # Grid = (num_heads,): each program computes one head's output row.
        head = tl.program_id(0)

        # d_mask hides the pad columns 96..127 so they are never read/written.
        d = tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q = tl.load(q_ptr + head * stride_qh + d, mask=d_mask, other=0.0)

        # Online-softmax running state: max, normalizer, weighted-V accumulator.
        # Splitting the context into tiles this way is numerically identical to
        # one full softmax but never materializes all scores at once.
        m = float("-inf")
        norm = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(0, ctx_len, BLOCK_T):
            toks = start + tl.arange(0, BLOCK_T)  # logical token positions
            t_mask = toks < ctx_len
            # Logical position -> physical block id via the block table.
            phys = tl.load(block_table_ptr + toks // BLOCK_SIZE, mask=t_mask, other=0)
            slot = toks % BLOCK_SIZE
            # Element offset of token t, head h, column d inside one layer's
            # pool: phys * stride_kb + h * stride_kh + slot * stride_ks + d
            # (the head_dim axis is contiguous, checked by the wrapper).
            k_off = phys[:, None] * stride_kb + head * stride_kh + slot[:, None] * stride_ks
            kv_mask = t_mask[:, None] & d_mask[None, :]
            k = tl.load(k_ptr + k_off + d[None, :], mask=kv_mask, other=0.0)

            # One score per token; no tl.dot needed for a single query row.
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(t_mask, scores, float("-inf"))

            m_new = tl.maximum(m, tl.max(scores, axis=0))
            alpha = tl.exp(m - m_new)  # rescales previous tiles; 0 on the first
            p = tl.exp(scores - m_new)  # masked lanes: exp(-inf) = 0
            norm = norm * alpha + tl.sum(p, axis=0)

            v_off = phys[:, None] * stride_vb + head * stride_vh + slot[:, None] * stride_vs
            v = tl.load(v_ptr + v_off + d[None, :], mask=kv_mask, other=0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new

        tl.store(out_ptr + head * stride_oh + d, acc / norm, mask=d_mask)


def decode(q: torch.Tensor, cache, layer_idx: int, context_len: int) -> torch.Tensor:
    """Single-token paged attention over cached positions 0..context_len-1.

    q is [1, heads, 1, head_dim]; returns the same shape. context_len is a
    host-side Python int (the cache tracks its own position), so this path
    never synchronizes the device to read a length.
    """
    allocator = cache.allocator
    k_pool = allocator.k_pool[layer_idx]  # view: [num_blocks, heads, block_size, head_dim]
    v_pool = allocator.v_pool[layer_idx]
    num_heads, head_dim = q.shape[1], q.shape[3]

    q2 = q[0, :, 0]  # [heads, head_dim] view; row stride = head_dim
    out = torch.empty_like(q2)

    # The kernel computes element offsets in int32; one layer's pool must fit.
    assert k_pool.stride(0) * k_pool.shape[0] < 2**31, "KV pool too large for int32 offsets"
    # Offset arithmetic above assumes the head_dim axis is contiguous.
    assert k_pool.stride(3) == 1 and q2.stride(1) == 1

    _paged_decode_kernel[(num_heads,)](
        q2,
        k_pool,
        v_pool,
        out,
        cache.block_table_device,
        context_len,
        1.0 / math.sqrt(head_dim),
        q2.stride(0),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        out.stride(0),
        BLOCK_SIZE=allocator.block_size,
        BLOCK_T=BLOCK_T,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out.view(1, num_heads, 1, head_dim)
