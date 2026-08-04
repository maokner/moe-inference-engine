"""Faithful Transformers implementation of miniMoE.

The module deliberately contains only PyTorch and Transformers operations.
vLLM's Transformers backend can replace the attention interface, while the
native plugin uses vLLM's own attention and fused-MoE layers directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_minimoe import MiniMoEConfig


class MiniMoEAttention(nn.Module):
    """Eight-head causal attention with the checkpoint's packed QKV parameters."""

    def __init__(self, config: MiniMoEConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * config.hidden_dim, config.hidden_dim)
        )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * config.hidden_dim))
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        query, key, value = F.linear(
            hidden_states, self.in_proj_weight, self.in_proj_bias
        ).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        query, key, value = map(split_heads, (query, key, value))
        implementation = getattr(self.config, "_attn_implementation", "sdpa")
        attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]
        attention_output, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=attention_mask is None and seq_len > 1,
            **kwargs,
        )
        attention_output = attention_output.reshape(batch, seq_len, hidden_dim)
        return self.out_proj(attention_output)


class MiniMoEExpert(nn.Module):
    """The original biased Linear-GELU-Linear expert."""

    def __init__(self, config: MiniMoEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class MiniMoEExperts(nn.ModuleList):
    """Naive oracle whose signature permits vLLM expert replacement."""

    def __init__(self, config: MiniMoEConfig) -> None:
        super().__init__(MiniMoEExpert(config) for _ in range(config.num_experts))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        flat_indices = top_k_index.reshape(-1, top_k_index.shape[-1])
        flat_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
        output = torch.zeros_like(flat_states)
        for expert_id, expert in enumerate(self):
            token_ids, slots = torch.where(flat_indices == expert_id)
            if token_ids.numel() == 0:
                continue
            weights = flat_weights[token_ids, slots].to(flat_states.dtype).unsqueeze(-1)
            output.index_add_(0, token_ids, weights * expert(flat_states[token_ids]))
        return output.reshape(original_shape)


class MiniMoESparseBlock(nn.Module):
    """Learned top-2 router followed by the exact non-gated GELU experts."""

    def __init__(self, config: MiniMoEConfig) -> None:
        super().__init__()
        self.top_k = config.router_top_k
        self.experts = MiniMoEExperts(config)
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            router_logits = self.router(hidden_states.float())
            top_k_logits, top_k_index = torch.topk(router_logits, k=self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)
        return self.experts(hidden_states, top_k_index, top_k_weights)


class MiniMoEBlock(nn.Module):
    def __init__(self, config: MiniMoEConfig, layer_idx: int) -> None:
        super().__init__()
        self.attention = MiniMoEAttention(config, layer_idx)
        self.moe = MiniMoESparseBlock(config)
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.moe_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attn_norm(hidden_states), attention_mask=attention_mask, **kwargs
        )
        return hidden_states + self.moe(self.moe_norm(hidden_states))


class MiniMoEPreTrainedModel(PreTrainedModel):
    config_class = MiniMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_attention_backend = True
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, MiniMoEAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.in_proj_bias)


class MiniMoEModel(MiniMoEPreTrainedModel):
    """Base decoder model returning normalized hidden states."""

    def __init__(self, config: MiniMoEConfig) -> None:
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_embedding = nn.Embedding(
            config.max_seq_length, config.hidden_dim
        )
        self.MoEBlocks = nn.ModuleList(
            MiniMoEBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput | tuple[torch.Tensor]:
        del use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device)[None, :]
        if position_ids.shape[-1] != seq_len:
            raise ValueError("position_ids length must match the input sequence")
        if int(position_ids.max()) >= self.config.max_seq_length:
            raise ValueError("position_ids exceed max_seq_length")

        hidden_states = inputs_embeds + self.positional_embedding(position_ids)
        causal_mask = self._causal_mask(
            hidden_states, attention_mask, position_ids, kwargs
        )
        for block in self.MoEBlocks:
            hidden_states = block(hidden_states, attention_mask=causal_mask, **kwargs)
        hidden_states = self.final_norm(hidden_states)
        if return_dict is False:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)

    @staticmethod
    def _causal_mask(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        # vLLM owns masking and KV-cache addressing in its registered interface.
        if "attention_instances" in kwargs:
            return None
        batch, seq_len, _ = hidden_states.shape
        if seq_len == 1 and attention_mask is None:
            return None
        positions = position_ids.reshape(batch, seq_len)
        key_positions = torch.arange(seq_len, device=hidden_states.device)
        allowed = key_positions[None, None, :] <= positions[:, :, None]
        mask = torch.zeros(
            (batch, 1, seq_len, seq_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mask.masked_fill_(~allowed[:, None, :, :], torch.finfo(hidden_states.dtype).min)
        if attention_mask is not None:
            padding = attention_mask[:, None, None, :seq_len].to(torch.bool)
            mask.masked_fill_(~padding, torch.finfo(hidden_states.dtype).min)
        return mask


class MiniMoEForCausalLM(MiniMoEPreTrainedModel, GenerationMixin):
    """Causal LM wrapper with tied token/output weights and learned head bias."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MiniMoEConfig) -> None:
        super().__init__(config)
        self.model = MiniMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> dict[str, Any]:
        # miniMoE's Transformers oracle deliberately recomputes the full prefix.
        return {
            "input_ids": input_ids[:, -self.config.max_seq_length :],
            "attention_mask": kwargs.get("attention_mask"),
            "use_cache": False,
        }
