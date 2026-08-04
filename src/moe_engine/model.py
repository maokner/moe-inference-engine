"""Inference-only miniMoE with external KV caches.

Parameter names match the reference model for direct checkpoint loading.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_engine import fused_moe, paged_attention

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

    def __init__(
        self, config: ModelConfig, batch_size: int, device, dtype=torch.float32
    ):
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
        attention_mode: str = "auto",
    ):
        # How caches from this pool run decode attention:
        #   "gather" - reconstruct contiguous K/V then call PyTorch attention
        #              (the original path, kept as oracle and fallback),
        #   "direct" - require the Triton kernel that reads blocks in place,
        #   "auto"   - direct when supported (CUDA + Triton), else gather.
        if attention_mode not in ("auto", "direct", "gather"):
            raise ValueError(f"unknown attention_mode {attention_mode!r}")
        self.attention_mode = attention_mode
        self.num_blocks = num_blocks
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
        self.attention_mode = allocator.attention_mode
        self.block_table: list[int] = []
        self.position = 0
        # Device-side mirror of block_table so the direct kernel can map
        # logical positions to physical blocks without a host copy per step.
        # Only the first len(block_table) entries are meaningful; the mirror
        # is updated incrementally as blocks are appended, never rebuilt.
        self.block_table_device = torch.zeros(
            allocator.num_blocks, dtype=torch.int32, device=allocator.k_pool.device
        )
        self._write_index_key = None  # (position, seq_len) of cached indices

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
        if taken:
            start = len(self.block_table)
            self.block_table.extend(taken)
            # Append only the new entries to the device mirror. A new block is
            # needed once per block_size tokens, so this tiny host-to-device
            # copy stays out of the per-layer hot path.
            self.block_table_device[start : start + len(taken)] = torch.tensor(
                taken, dtype=torch.int32, device=self.block_table_device.device
            )

    def _block_for(self, token_pos: int) -> tuple[int, int]:
        block_size = self.allocator.block_size
        return self.block_table[token_pos // block_size], token_pos % block_size

    def _write_indices(self, position, seq_len):
        """Physical (block, slot) index tensors for a multi-token write.

        Every layer writes the same positions during one forward pass, so the
        tensors are built once per step and reused across layers.
        """
        if self._write_index_key != (position, seq_len):
            block_size = self.allocator.block_size
            pos = torch.arange(
                position, position + seq_len, device=self.block_table_device.device
            )
            self._write_index = (
                self.block_table_device[pos // block_size].long(),  # physical block ids
                pos % block_size,  # slot within each block
            )
            self._write_index_key = (position, seq_len)
        return self._write_index

    def write(self, layer_idx, position, k, v):
        # Each cache has one block table and represents one sequence.
        if k.shape[0] != 1:
            raise ValueError("PagedKVCache holds one sequence; use batch size 1")
        seq_len = k.shape[2]
        self._reserve(position + seq_len)
        if seq_len == 1:
            # Decode: one indexed store per pool, addressed with host-side ints.
            block_id, slot = self._block_for(position)
            self.allocator.k_pool[layer_idx, block_id, :, slot] = k[0, :, 0]
            self.allocator.v_pool[layer_idx, block_id, :, slot] = v[0, :, 0]
            return
        # Prefill: scatter every position in one indexed copy instead of a
        # Python loop per token. Indexing [layer, blocks, :, slots] selects
        # one (block, slot) pair per token, giving [seq, head, head_dim].
        blocks, slots = self._write_indices(position, seq_len)
        self.allocator.k_pool[layer_idx, blocks, :, slots] = k[0].transpose(0, 1)
        self.allocator.v_pool[layer_idx, blocks, :, slots] = v[0].transpose(0, 1)

    def read(self, layer_idx, length):
        block_size = self.allocator.block_size
        num_blocks = -(-length // block_size)
        block_ids = self.block_table[:num_blocks]

        def read_pool(pool):
            return (
                pool[layer_idx, block_ids]
                .transpose(0, 1)
                .reshape(self.allocator.num_heads, -1, self.allocator.head_dim)[
                    :, :length
                ]
                .unsqueeze(0)
            )

        return read_pool(self.allocator.k_pool), read_pool(self.allocator.v_pool)

    def free(self):
        """Return all blocks and reset the cache."""
        self.allocator.free(self.block_table)
        self.block_table = []
        self.position = 0
        self._write_index_key = None  # Stale device-mirror entries are never read.


class CachedAttention(nn.Module):
    """Causal self-attention with external K/V storage."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        # Checkpoints overwrite these values, but tests and small standalone
        # models need deterministic, finite parameters before any load.
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, x, cache, layer_idx, position: int):
        batch, seq_len, dim = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # [batch, seq, dim] -> [batch, head, seq, head_dim]
        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Contiguous caches always read + attend; paged caches pick a mode.
        # A forced 'direct' that cannot run must fail before the write below
        # allocates blocks, so a rejected forward leaves the cache untouched.
        mode = getattr(cache, "attention_mode", "gather")
        use_kernel = mode != "gather" and paged_attention.is_supported(x)
        if mode == "direct" and not use_kernel:
            raise RuntimeError(
                "attention_mode='direct' needs CUDA, Triton, and float32; "
                "use 'auto' for automatic fallback or 'gather' for the oracle path"
            )

        cache.write(layer_idx, position, k, v)

        # Prefill needs a mask; a single decode query can use the full cache.
        assert position == 0 or seq_len == 1, "prefill must start at position 0"

        if mode != "gather" and position == 0:
            # Prefill fast path: at position 0 the entire history is exactly
            # the K/V just projected, so attend over those contiguous tensors
            # instead of writing blocks and immediately gathering them back.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=seq_len > 1)
        elif use_kernel:
            # Decode: the Triton kernel reads K/V in place from the physical
            # block pool through the block table - no contiguous rebuild.
            out = paged_attention.decode(q, cache, layer_idx, context_len=position + 1)
        else:
            # Gather path: reconstruct contiguous K/V, then PyTorch attention.
            # This is the original implementation, the correctness oracle, and
            # the CPU/MPS decode fallback.
            keys, values = cache.read(layer_idx, position + seq_len)
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
    """PyTorch MoE oracle plus fixed-shape Triton decode dispatch."""

    def __init__(self, dim, hidden_dim, num_experts, top_k, mode="auto"):
        super().__init__()
        if mode not in ("auto", "direct", "reference"):
            raise ValueError(f"unknown MoE mode {mode!r}")
        self.mode = mode
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            Expert(dim, hidden_dim) for _ in range(num_experts)
        )
        self.router = nn.Linear(dim, num_experts, bias=False)

        # These non-persistent buffers are one fixed workspace per layer.
        # They move with the module but never alter checkpoint keys or values.
        self.register_buffer(
            "_router_logits",
            torch.empty(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_expert_ids", torch.empty(top_k, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "_route_weights", torch.empty(top_k, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_expert_hidden",
            torch.empty(top_k, hidden_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_direct_out", torch.empty(dim, dtype=torch.float32), persistent=False
        )

        # The checkpoint exposes one parameter per expert.  Re-home those
        # parameters into four canonical contiguous storages once, preserving
        # every public state-dict name as a non-overlapping Parameter view.
        # Triton can then select expert e by a device-side canonical offset
        # without keeping a stacked duplicate or packing weights per token.
        self._pack_expert_parameters()
        self.register_load_state_dict_post_hook(self._repack_after_load)

    def _pack_parameter_group(
        self, layer_index: int, parameter_name: str
    ) -> torch.Tensor:
        parameters = [
            getattr(expert.net[layer_index], parameter_name) for expert in self.experts
        ]
        shape = parameters[0].shape
        if any(parameter.shape != shape for parameter in parameters):
            raise ValueError("all experts must have identical parameter shapes")
        packed = torch.empty(
            (self.num_experts, *shape),
            device=parameters[0].device,
            dtype=parameters[0].dtype,
        )
        with torch.no_grad():
            for expert_id, parameter in enumerate(parameters):
                packed[expert_id].copy_(parameter)
                setattr(
                    self.experts[expert_id].net[layer_index],
                    parameter_name,
                    nn.Parameter(
                        packed[expert_id], requires_grad=parameter.requires_grad
                    ),
                )
        return packed

    def _pack_expert_parameters(self) -> None:
        """Create canonical storages while retaining legacy parameter names."""
        self._expert_storage = {
            "w1": self._pack_parameter_group(0, "weight"),
            "b1": self._pack_parameter_group(0, "bias"),
            "w2": self._pack_parameter_group(2, "weight"),
            "b2": self._pack_parameter_group(2, "bias"),
        }

    def _repack_after_load(self, _module, _incompatible_keys) -> None:
        # Covers load_state_dict(assign=True) as well as the normal copy path.
        self._pack_expert_parameters()

    def _apply(self, fn, recurse=True):
        # Module._apply may replace each Parameter independently.  Repacking
        # after a device/dtype conversion restores the canonical layout.
        result = super()._apply(fn, recurse=recurse)
        self._pack_expert_parameters()
        return result

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "direct", "reference"):
            raise ValueError(f"unknown MoE mode {mode!r}")
        self.mode = mode

    def _direct_supported(self, x: torch.Tensor) -> bool:
        return (
            fused_moe.is_supported(
                x,
                num_experts=self.num_experts,
                top_k=self.top_k,
                expert_hidden_dim=self.hidden_dim,
            )
            and self._has_direct_layout()
        )

    def _has_direct_layout(self) -> bool:
        return fused_moe.has_canonical_layout(
            self.router.weight,
            self._expert_storage["w1"],
            self._expert_storage["b1"],
            self._expert_storage["w2"],
            self._expert_storage["b2"],
            self._router_logits,
            self._expert_ids,
            self._route_weights,
            self._expert_hidden,
            self._direct_out,
        )

    def preflight_direct_decode(self, *, batch_size: int) -> None:
        """Reject an unsupported forced decode before any cache mutation."""
        if self.mode != "direct":
            return
        router = self.router.weight
        supported = fused_moe.is_configuration_supported(
            device=router.device,
            dtype=router.dtype,
            batch_size=batch_size,
            hidden_dim=self.dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.hidden_dim,
        )
        supported = supported and self._has_direct_layout()
        if not supported:
            self._raise_direct_unsupported()

    @staticmethod
    def _raise_direct_unsupported() -> None:
        raise RuntimeError(
            "MoE mode 'direct' needs one-token batch-one CUDA float32 decode "
            "with shape 768x8x2x3072 and Triton; use 'auto' for automatic "
            "fallback or 'reference' for the PyTorch oracle"
        )

    def uses_direct(self, x: torch.Tensor, *, decode: bool) -> bool:
        """Resolve reference/direct/auto without reading device results."""
        if not decode or self.mode == "reference":
            return False
        supported = self._direct_supported(x)
        if self.mode == "direct" and not supported:
            self._raise_direct_unsupported()
        return supported

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized prefill and correctness oracle for direct decode."""
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

    def _forward_direct(self, x: torch.Tensor) -> torch.Tensor:
        return fused_moe.decode(
            x,
            self.router.weight,
            self._expert_storage["w1"],
            self._expert_storage["b1"],
            self._expert_storage["w2"],
            self._expert_storage["b2"],
            router_logits=self._router_logits,
            expert_ids=self._expert_ids,
            route_weights=self._route_weights,
            hidden=self._expert_hidden,
            out=self._direct_out,
        )

    def forward(self, x, *, decode=False):
        if self.uses_direct(x, decode=decode):
            return self._forward_direct(x)
        return self._forward_reference(x)


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
        normed = self.moe_norm(x)
        x = x + self.moe(normed, decode=position > 0 and normed.shape[1] == 1)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_embedding = nn.Embedding(
            config.max_seq_length, config.hidden_dim
        )
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

    def set_moe_mode(self, mode: str) -> "Model":
        """Set reference, forced-direct, or automatic MoE dispatch on all layers."""
        for block in self.MoEBlocks:
            block.moe.set_mode(mode)
        return self

    def new_block_allocator(
        self, num_blocks: int, block_size: int = 16, attention_mode: str = "auto"
    ) -> BlockAllocator:
        """Create a block pool matching the model's device and dtype."""
        weight = self.token_embedding.weight
        return BlockAllocator(
            self.config,
            num_blocks,
            block_size,
            weight.device,
            weight.dtype,
            attention_mode,
        )

    def forward(self, token_ids, cache):
        """Run a forward pass using a contiguous or paged cache."""
        batch, seq_len = token_ids.shape
        position = cache.position
        assert position + seq_len <= self.max_seq_length, "KV cache is full"

        # Forced direct MoE failures must occur before attention in layer zero
        # can write K/V or reserve a paged-cache block.  Position zero remains
        # prompt prefill and always uses the PyTorch MoE oracle.
        if position > 0 and seq_len == 1:
            for block in self.MoEBlocks:
                block.moe.preflight_direct_decode(batch_size=batch)

        # New tokens start at the current cache position.
        pos = torch.arange(position, position + seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.positional_embedding(pos)

        for i, block in enumerate(self.MoEBlocks):
            x = block(x, cache, i, position)
        cache.position += seq_len

        x = self.final_norm(x)
        return self.output_projection(x)

    @torch.no_grad()
    def generate(
        self, token_ids, max_new_tokens, temperature=1.0, top_k=None, cache=None
    ):
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
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    sample_top_k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, sample_top_k)
                    next_token_logits[next_token_logits < values[:, [-1]]] = -float(
                        "inf"
                    )
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if next_token == EOS_TOKEN_ID:
                return x
            x = torch.cat((x, next_token), dim=1)
            step_input = next_token
            if cache.position >= self.max_seq_length:
                break  # The sampled token has not entered the cache.
        return x
