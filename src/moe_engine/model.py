"""Inference-only miniMoE with external KV caches.

Parameter names match the reference model for direct checkpoint loading.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from moe_engine.fused_moe import fused_moe
except ImportError:  # Triton is CUDA-only; the expert loop covers CPU/MPS.
    fused_moe = None

EOS_TOKEN_ID = 50256


def sample_next_token(logits, temperature, top_k):
    """Pick the next token id from last-position logits shaped [batch, vocab]."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@dataclass
class ModelConfig:
    max_seq_length: int = 1024
    vocab_size: int = 50304
    num_layers: int = 6
    hidden_dim: int = 768
    moe_multiplier: float = 0.01  # Retained for checkpoint compatibility.
    num_experts: int = 8
    top_k: int = 2
    num_heads: int = 8  # Fixed by the reference architecture.


class KVCache:
    """Contiguous K/V storage shaped [layer, batch, head, sequence, head_dim].

    `position` is the number of cached tokens. Use `Model.new_cache()` to
    match the model's device and dtype.
    """

    def __init__(self, config: ModelConfig, batch_size: int, device, dtype=torch.float32):
        head_dim = config.hidden_dim // config.num_heads
        shape = (
            config.num_layers,
            batch_size,
            config.num_heads,
            config.max_seq_length,
            head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.position = 0

    def write(self, layer_idx, position, k, v):
        seq_len = k.shape[2]
        self.k[layer_idx, :, :, position : position + seq_len] = k
        self.v[layer_idx, :, :, position : position + seq_len] = v

    def read(self, layer_idx, length):
        return self.k[layer_idx, :, :, :length], self.v[layer_idx, :, :, :length]


class BlockAllocator:
    """Shared K/V block storage for paged sequence caches."""

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        device,
        dtype=torch.float32,
    ):
        self.block_size = block_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        shape = (
            config.num_layers,
            num_blocks,
            config.num_heads,
            block_size,
            self.head_dim,
        )
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("no free KV cache blocks left")
        return self.free_blocks.pop()

    def free(self, block_ids: list[int]) -> None:
        self.free_blocks.extend(block_ids)

    def new_cache(self) -> "PagedKVCache":
        return PagedKVCache(self)


class PagedKVCache:
    """Per-sequence K/V cache backed by blocks from a shared allocator."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_table: list[int] = []
        self.position = 0

    def _reserve(self, num_tokens: int) -> None:
        """Reserve enough blocks for `num_tokens` without partial allocation."""
        blocks_needed = -(-num_tokens // self.allocator.block_size)
        taken: list[int] = []
        try:
            while len(self.block_table) + len(taken) < blocks_needed:
                taken.append(self.allocator.allocate())
        except RuntimeError:
            self.allocator.free(taken)
            raise
        self.block_table.extend(taken)

    def _block_for(self, token_pos: int) -> tuple[int, int]:
        block_size = self.allocator.block_size
        return self.block_table[token_pos // block_size], token_pos % block_size

    def write(self, layer_idx, position, k, v):
        # Each cache has one block table and represents one sequence.
        if k.shape[0] != 1:
            raise ValueError("PagedKVCache holds one sequence; use batch size 1")
        seq_len = k.shape[2]
        self._reserve(position + seq_len)
        for offset in range(seq_len):
            block_id, slot = self._block_for(position + offset)
            self.allocator.k_pool[layer_idx, block_id, :, slot] = k[0, :, offset]
            self.allocator.v_pool[layer_idx, block_id, :, slot] = v[0, :, offset]

    def read(self, layer_idx, length):
        block_size = self.allocator.block_size
        num_blocks = -(-length // block_size)
        block_ids = self.block_table[:num_blocks]

        def read_pool(pool):
            return (
                pool[layer_idx, block_ids]
                .transpose(0, 1)
                .reshape(self.allocator.num_heads, -1, self.allocator.head_dim)[:, :length]
                .unsqueeze(0)
            )

        return read_pool(self.allocator.k_pool), read_pool(self.allocator.v_pool)

    def free(self):
        """Return all blocks and reset the cache."""
        self.allocator.free(self.block_table)
        self.block_table = []
        self.position = 0


class CachedAttention(nn.Module):
    """Causal self-attention with external K/V storage."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        # Match nn.MultiheadAttention's reset_parameters. Checkpoints overwrite
        # these, but tests build raw models: torch.empty is uninitialized
        # memory and can hold inf/nan garbage that poisons every forward pass.
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, x, cache, layer_idx, position: int):
        batch, seq_len, dim = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # [batch, seq, dim] -> [batch, head, seq, head_dim]
        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        cache.write(layer_idx, position, k, v)
        keys, values = cache.read(layer_idx, position + seq_len)

        # Prefill needs a mask; a single decode query can use the full cache.
        assert position == 0 or seq_len == 1, "prefill must start at position 0"
        out = F.scaled_dot_product_attention(q, keys, values, is_causal=seq_len > 1)

        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out_proj(out)

    def forward_decode(self, x, caches, layer_idx):
        """One-token decode step for a batch of independent sequences.

        The projections run batched, but attention runs per sequence: each
        cache holds a different-length history in different pool blocks, so
        there is no single rectangular K/V tensor to attend over. A future
        paged-attention kernel replaces this Python loop.
        """
        batch, seq_len, dim = x.shape  # seq_len is always 1 during decode.
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        outs = []
        for i, cache in enumerate(caches):
            cache.write(layer_idx, cache.position, k[i : i + 1], v[i : i + 1])
            keys, values = cache.read(layer_idx, cache.position + 1)
            # A single query attends over its whole history: no mask needed.
            outs.append(F.scaled_dot_product_attention(q[i : i + 1], keys, values))

        out = torch.cat(outs).transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out_proj(out)


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Top-k expert routing with gather and scatter-add."""

    def __init__(self, dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList(Expert(dim, hidden_dim) for _ in range(num_experts))
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.fused_enabled = True  # Benchmarks flip this to compare paths.
        self._stacked = None
        # Reloading weights must drop the stacked copy, or the fused path
        # would silently keep serving the old parameters.
        self._register_load_state_dict_pre_hook(self._clear_stacked)

    def _clear_stacked(self, *args):
        self._stacked = None

    def _stacked_experts(self):
        """Stack expert weights once for the fused kernel.

        Built lazily on the first fused forward so it runs after checkpoint
        loading and device placement. This duplicates the expert weights,
        which are most of the model (about 900 MB for the real checkpoint).
        """
        if self._stacked is None:
            self._stacked = (
                torch.stack([e.net[0].weight for e in self.experts]),
                torch.stack([e.net[0].bias for e in self.experts]),
                torch.stack([e.net[2].weight for e in self.experts]),
                torch.stack([e.net[2].bias for e in self.experts]),
            )
        return self._stacked

    def forward(self, x):
        batch, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)

        # Keep routing decisions in float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(flat_x.float(), self.router.weight.float())
            topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)

        if fused_moe is not None and self.fused_enabled and flat_x.is_cuda:
            out = fused_moe(flat_x, *self._stacked_experts(), topk_weights, topk_indices)
            return out.reshape(batch, seq_len, dim)

        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slot = torch.where(topk_indices == expert_id)
            if token_ids.numel() == 0:
                continue
            weight = topk_weights[token_ids, slot].unsqueeze(-1).to(flat_x.dtype)
            out.index_add_(0, token_ids, weight * expert(flat_x[token_ids]))
        return out.reshape(batch, seq_len, dim)


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = CachedAttention(config.hidden_dim, config.num_heads)
        self.moe = MoEFeedForward(
            config.hidden_dim, config.hidden_dim * 4, config.num_experts, config.top_k
        )
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.moe_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, cache, layer_idx, position):
        x = x + self.attention(self.attn_norm(x), cache, layer_idx, position)
        x = x + self.moe(self.moe_norm(x))
        return x

    def forward_decode(self, x, caches, layer_idx):
        x = x + self.attention.forward_decode(self.attn_norm(x), caches, layer_idx)
        x = x + self.moe(self.moe_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        self.MoEBlocks = nn.ModuleList(
            TransformerMoEBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        self.output_projection.weight = self.token_embedding.weight  # Weight tying.

    def new_cache(self, batch_size: int = 1) -> KVCache:
        """Create a cache matching the model's device and dtype."""
        weight = self.token_embedding.weight
        return KVCache(self.config, batch_size, weight.device, weight.dtype)

    def new_block_allocator(self, num_blocks: int, block_size: int = 16) -> BlockAllocator:
        """Create a block pool matching the model's device and dtype."""
        weight = self.token_embedding.weight
        return BlockAllocator(self.config, num_blocks, block_size, weight.device, weight.dtype)

    def forward(self, token_ids, cache):
        """Run a forward pass using a contiguous or paged cache."""
        batch, seq_len = token_ids.shape
        position = cache.position
        assert position + seq_len <= self.max_seq_length, "KV cache is full"

        # New tokens start at the current cache position.
        pos = torch.arange(position, position + seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.positional_embedding(pos)

        for i, block in enumerate(self.MoEBlocks):
            x = block(x, cache, i, position)
        cache.position += seq_len

        x = self.final_norm(x)
        return self.output_projection(x)

    def forward_decode(self, token_ids, caches):
        """Decode one token for a batch of sequences at different positions.

        `token_ids` is [num_seqs, 1] and `caches` holds one PagedKVCache per
        row. Embeddings, norms, and the expert MLPs run genuinely batched;
        only attention loops over sequences (see CachedAttention). This is
        the step continuous batching repeats.
        """
        num_seqs, seq_len = token_ids.shape
        assert seq_len == 1, "forward_decode takes one token per sequence"
        assert len(caches) == num_seqs, "one cache per sequence"
        for cache in caches:
            assert cache.position + 1 <= self.max_seq_length, "KV cache is full"

        positions = torch.tensor(
            [cache.position for cache in caches], device=token_ids.device
        )
        x = self.token_embedding(token_ids) + self.positional_embedding(positions).unsqueeze(1)

        for i, block in enumerate(self.MoEBlocks):
            x = block.forward_decode(x, caches, i)
        for cache in caches:
            cache.position += 1

        x = self.final_norm(x)
        return self.output_projection(x)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, cache=None):
        """Generate one sequence. The caller owns any supplied cache."""
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.token_embedding.weight.device).long()
        if token_ids.shape[0] != 1:
            raise ValueError("generate() handles one sequence; use batch size 1")
        if token_ids.shape[1] == 0:
            raise ValueError("prompt must contain at least one token")
        if token_ids.shape[1] > self.max_seq_length:
            raise ValueError(
                f"prompt is {token_ids.shape[1]} tokens; max_seq_length is {self.max_seq_length}"
            )
        if cache is None:
            cache = self.new_cache(batch_size=1)
        # Reject used caches before a forward pass can mutate them.
        if cache.position != 0:
            raise ValueError("generate() needs a fresh cache")

        x = token_ids
        step_input = token_ids
        for _ in range(max_new_tokens):
            logits = self(step_input, cache)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k)

            if next_token == EOS_TOKEN_ID:
                return x
            x = torch.cat((x, next_token), dim=1)
            step_input = next_token
            if cache.position >= self.max_seq_length:
                break  # The sampled token has not entered the cache.
        return x
