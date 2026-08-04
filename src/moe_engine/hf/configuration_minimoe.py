"""Hugging Face configuration for the original miniMoE architecture."""

from transformers import PretrainedConfig


class MiniMoEConfig(PretrainedConfig):
    """Configuration with both original and Transformers-standard field names."""

    model_type = "minimoe"
    keys_to_ignore_at_inference = ["router_logits"]

    def __init__(
        self,
        max_seq_length: int = 1024,
        vocab_size: int = 50304,
        num_layers: int = 6,
        hidden_dim: int = 768,
        moe_multiplier: float = 0.01,
        num_experts: int = 8,
        top_k: int = 2,
        router_top_k: int | None = None,
        num_heads: int = 8,
        **kwargs,
    ) -> None:
        if router_top_k is not None:
            top_k = router_top_k
        kwargs.setdefault("bos_token_id", 50256)
        kwargs.setdefault("eos_token_id", 50256)
        kwargs.setdefault("pad_token_id", 50256)
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)

        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")

        # Original checkpoint field names.
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.moe_multiplier = moe_multiplier
        self.num_experts = num_experts
        self.router_top_k = top_k
        self.num_heads = num_heads

        # Names consumed by Transformers and vLLM without architecture aliases.
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_seq_length
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_dim
        self.intermediate_size = hidden_dim * 4
        self.moe_intermediate_size = hidden_dim * 4
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads
        self.num_local_experts = num_experts
        self.num_experts_per_tok = top_k
        self.hidden_act = "gelu"
        self.norm_topk_prob = True
