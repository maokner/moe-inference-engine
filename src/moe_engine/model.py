"""Inference-only miniMoE with a contiguous KV cache.

Parameter names match the reference model for direct checkpoint loading.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

EOS_TOKEN_ID = 50256


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


class CachedAttention(nn.Module):
    """Causal self-attention with external K/V storage."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, k_cache, v_cache, position: int):
        batch, seq_len, dim = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # [batch, seq, dim] -> [batch, head, seq, head_dim]
        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        k_cache[:, :, position : position + seq_len] = k
        v_cache[:, :, position : position + seq_len] = v
        keys = k_cache[:, :, : position + seq_len]
        values = v_cache[:, :, : position + seq_len]

        # Prefill needs a mask; a single decode query can use the full cache.
        assert position == 0 or seq_len == 1, "prefill must start at position 0"
        out = F.scaled_dot_product_attention(q, keys, values, is_causal=seq_len > 1)

        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
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

    def forward(self, x):
        batch, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)

        # Keep routing decisions in float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(flat_x.float(), self.router.weight.float())
            topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)

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

    def forward(self, x, k_cache, v_cache, position):
        x = x + self.attention(self.attn_norm(x), k_cache, v_cache, position)
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

    def forward(self, token_ids, cache: KVCache):
        batch, seq_len = token_ids.shape
        position = cache.position
        assert position + seq_len <= self.max_seq_length, "KV cache is full"

        # New tokens start at the current cache position.
        pos = torch.arange(position, position + seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.positional_embedding(pos)

        for i, block in enumerate(self.MoEBlocks):
            x = block(x, cache.k[i], cache.v[i], position)
        cache.position += seq_len

        x = self.final_norm(x)
        return self.output_projection(x)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Generate one sequence with cached greedy or top-k sampling."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.token_embedding.weight.device).long()
        cache = self.new_cache(batch_size=1)

        x = token_ids
        step_input = token_ids
        for _ in range(max_new_tokens):
            logits = self(step_input, cache)
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    sample_top_k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, sample_top_k)
                    next_token_logits[next_token_logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if next_token == EOS_TOKEN_ID:
                return x
            x = torch.cat((x, next_token), dim=1)
            step_input = next_token
        return x
