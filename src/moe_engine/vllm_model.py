"""Native vLLM implementation of miniMoE for the faithful comparison.

This module is intentionally isolated from the custom engine. It uses vLLM's
Attention, FusedMoE, linear, embedding, and logits components exclusively.
The target API is vLLM 0.14.x, as pinned by the ``vllm`` project extra.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors


class MiniMoEVllmAttention(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.in_proj = ReplicatedLinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        head_dim = config.hidden_size // config.num_attention_heads
        self.attn = Attention(
            num_heads=config.num_attention_heads,
            head_size=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.in_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        attention_output = self.attn(query, key, value)
        output, _ = self.out_proj(attention_output)
        return output


class MiniMoEVllmSparseBlock(nn.Module):
    """Exact top-2 routing backed by vLLM's non-gated GELU FusedMoE."""

    def __init__(self, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.router",
        )
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            activation="gelu",
            is_act_and_mul=False,
            has_bias=True,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits, _ = self.router(flat_states.float())
        output = self.experts(flat_states, router_logits)
        return output.reshape(original_shape)


class MiniMoEVllmBlock(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str) -> None:
        super().__init__()
        self.attention = MiniMoEVllmAttention(
            config, cache_config, quant_config, f"{prefix}.attention"
        )
        self.moe = MiniMoEVllmSparseBlock(config, quant_config, f"{prefix}.moe")
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.moe_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.attn_norm(hidden_states))
        return hidden_states + self.moe(self.moe_norm(hidden_states))


@support_torch_compile
class MiniMoEVllmModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.token_embedding",
        )
        self.positional_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.MoEBlocks = nn.ModuleList(
            MiniMoEVllmBlock(
                config,
                vllm_config.cache_config,
                vllm_config.quant_config,
                f"{prefix}.MoEBlocks.{layer_idx}",
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("miniMoE benchmark plugin supports one pipeline rank")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        hidden_states = hidden_states + self.positional_embedding(positions)
        for block in self.MoEBlocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMoEForCausalLM(nn.Module, MixtureOfExperts):
    """vLLM-native batch-one model retaining miniMoE's biased tied LM head."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        model_prefix = f"{prefix}.model" if prefix else "model"
        head_prefix = f"{prefix}.lm_head" if prefix else "lm_head"
        self.model = MiniMoEVllmModel(vllm_config=vllm_config, prefix=model_prefix)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            bias=True,
            quant_config=self.quant_config,
            prefix=head_prefix,
        )
        self.lm_head.weight = self.model.token_embedding.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.moe_layers = [block.moe.experts for block in self.model.MoEBlocks]
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = self.config.num_experts
        self.num_physical_experts = self.config.num_experts
        self.num_local_physical_experts = self.config.num_experts
        self.num_routed_experts = self.config.num_experts
        self.num_redundant_experts = 0
        self.num_expert_groups = 1
        self.num_shared_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(
            self.lm_head, hidden_states, embedding_bias=self.lm_head.bias
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="net.0",
            ckpt_down_proj_name="net.2",
            ckpt_up_proj_name="__unused__",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        expert_mapping = self.get_expert_mapping()
        direct_mapping = {
            ".attention.in_proj_weight": ".attention.in_proj.weight",
            ".attention.in_proj_bias": ".attention.in_proj.bias",
        }
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name == "lm_head.weight":
                # The identical model.token_embedding.weight tensor owns storage.
                continue
            for target_fragment, source_fragment, expert_id, shard_id in expert_mapping:
                if source_fragment not in checkpoint_name:
                    continue
                parameter_name = checkpoint_name.replace(
                    source_fragment, target_fragment
                )
                parameter = params[parameter_name]
                parameter.weight_loader(
                    parameter,
                    loaded_weight,
                    parameter_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded.add(parameter_name)
                break
            else:
                parameter_name = checkpoint_name
                for source_fragment, target_fragment in direct_mapping.items():
                    parameter_name = parameter_name.replace(
                        source_fragment, target_fragment
                    )
                parameter = params[parameter_name]
                weight_loader = getattr(
                    parameter, "weight_loader", default_weight_loader
                )
                weight_loader(parameter, loaded_weight)
                loaded.add(parameter_name)
        return loaded
