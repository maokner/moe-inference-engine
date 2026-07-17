import inspect

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


EOS_TOKEN_ID = 50256


def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


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
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.layer_index = -1
        self.routing_mode = "learned_top2"
        self.routing_observer = None

    def set_routing(self, mode="learned_top2", observer=None):
        valid = {"learned_top2", "learned_top1", "random_top2", "uniform_all"}
        valid.update(f"ablate_expert_{i}" for i in range(self.num_experts))
        if mode not in valid:
            raise ValueError(f"Unknown routing mode {mode!r}; expected one of {sorted(valid)}")
        self.routing_mode = mode
        self.routing_observer = observer

    def forward(self, x, token_ids=None):
        # x = [batch_size, sequence_length, input_dim]
        batch_size, seq_len, dim = x.shape
        num_tokens = batch_size * seq_len

        # Router runs in float32 even under autocast: routing decisions and the
        # aux loss are sensitive to bf16 rounding (ST-MoE keeps the router in fp32).
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = self.router(x.float())  # [batch_size, sequence_length, num_experts]
            router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, sequence_length, num_experts]
            if self.training and self.routing_mode != "learned_top2":
                raise RuntimeError("Evaluation routing modes cannot be used while training")

            if self.routing_mode == "learned_top2":
                topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
                topk_weights = F.softmax(topk_logits, dim=-1)
            elif self.routing_mode == "learned_top1":
                topk_indices = torch.argmax(router_logits, dim=-1, keepdim=True)
                topk_weights = torch.ones_like(topk_indices, dtype=router_probs.dtype)
            elif self.routing_mode == "random_top2":
                random_scores = torch.rand_like(router_probs)
                topk_indices = torch.topk(random_scores, k=2, dim=-1).indices
                topk_weights = torch.full_like(topk_indices, 0.5, dtype=router_probs.dtype)
            elif self.routing_mode == "uniform_all":
                shape = (*router_logits.shape[:-1], self.num_experts)
                topk_indices = torch.arange(self.num_experts, device=x.device).expand(shape)
                topk_weights = torch.full(shape, 1.0 / self.num_experts, device=x.device)
            else:
                ablated = int(self.routing_mode.rsplit("_", 1)[1])
                learned_logits, learned_indices = torch.topk(router_logits, k=2, dim=-1)
                keep = learned_indices != ablated
                masked_logits = learned_logits.masked_fill(~keep, -float("inf"))
                topk_weights = F.softmax(masked_logits, dim=-1)
                both_removed = ~keep.any(dim=-1, keepdim=True)
                topk_weights = torch.where(both_removed, torch.zeros_like(topk_weights), topk_weights)
                topk_indices = learned_indices

        active_k = topk_indices.size(-1)

        if self.routing_observer is not None:
            observed_tokens = token_ids
            if observed_tokens is None:
                observed_tokens = torch.full(x.shape[:2], -1, device=x.device, dtype=torch.long)
            self.routing_observer(
                self.layer_index,
                observed_tokens.detach(),
                topk_indices.detach(),
                topk_weights.detach(),
                router_probs.detach(),
            )

        flat_x = x.reshape(num_tokens, dim)  # [batch_size * sequence_length, input_dim]

        flat_topk_indices = topk_indices.reshape(
            num_tokens, active_k
        )  # [batch_size * sequence_length, active routes]
        flat_topk_weights = topk_weights.reshape(
            num_tokens, active_k
        )  # [batch_size * sequence_length, active routes]

        flat_output = torch.zeros_like(flat_x)  # [batch_size * sequence_length, input_dim]

        for expert_id, expert in enumerate(self.experts):
            selected = (flat_topk_indices == expert_id)
            token_ids, topk_slots = torch.where(selected)

            # Run the expert even on zero tokens: skipping it would leave its
            # parameters out of the autograd graph, which makes DDP with
            # find_unused_parameters=False error out mid-training.
            expert_input = flat_x[token_ids]
            expert_output = expert(expert_input)
            routing_weights = flat_topk_weights[token_ids, topk_slots].unsqueeze(-1)
            weighted_output = routing_weights.to(expert_output.dtype) * expert_output

            flat_output = flat_output.index_add(0, token_ids, weighted_output.to(flat_output.dtype))

        selected_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        load = selected_mask.sum(dim=2).mean(dim=(0, 1)) / active_k
        importance = router_probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(load * importance)
        output = flat_output.reshape(batch_size, seq_len, dim)  # [batch_size, sequence_length, input_dim]

        return output, aux_loss


class TransformerMoEBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.moe = MoEFeedForward(hidden_dim, hidden_dim * 4, num_experts=num_experts, top_k=top_k)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.moe_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, causal_mask=None, token_ids=None):
        seq_len = x.size(1)
        if causal_mask is None:
            causal_mask = make_causal_mask(seq_len, x.device)

        x_norm = self.attn_norm(x)
        attn_output, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + attn_output  # Residual connection
        moe_output, aux_loss = self.moe(self.moe_norm(x), token_ids=token_ids)
        x = x + moe_output
        return x, aux_loss


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_embedding = nn.Embedding(self.max_seq_length, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        self.MoEBlocks = nn.ModuleList(
            [TransformerMoEBlock(config.hidden_dim, config.num_experts, config.top_k) for _ in range(config.num_layers)]
        )
        for layer_index, block in enumerate(self.MoEBlocks):
            block.moe.layer_index = layer_index
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.moe_multiplier = config.moe_multiplier

        self.output_projection.weight = self.token_embedding.weight  # Tie weights

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        # X = [Batch size, Sequence length ] need to convert to [batch size, Sequence length, hidden_dim] for attention
        token_ids = x.long()
        x = self.token_embedding(token_ids)  # [Batch size, Sequence length, hidden_dim]
        _, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device)

        x = x + self.positional_embedding(pos)  # [Batch size, Sequence length, hidden_dim]

        causal_mask = make_causal_mask(seq_len, x.device)
        aux_losses = []

        for block in self.MoEBlocks:
            x, aux_loss = block(x, causal_mask=causal_mask, token_ids=token_ids)
            aux_losses.append(aux_loss)
        moe_aux_loss = torch.stack(aux_losses).mean()
        x = self.final_norm(x)
        logits = self.output_projection(x)

        if y is not None:
            # Sum-reduce and divide by the valid (non-ignored) token count instead
            # of reduction="mean". This is numerically identical whenever any target
            # is supervised, but returns 0 (not NaN) if a batch happens to be fully
            # masked (all -100) - which can occur with prompt-masked SFT data. A NaN
            # here would spread through the DDP all-reduce and destroy the run.
            flat_logits = logits.view(-1, logits.size(-1))
            flat_y = y.view(-1)
            cross_loss = F.cross_entropy(flat_logits, flat_y, reduction="sum")
            valid = (flat_y != -100).sum().clamp(min=1)
            cross_loss = cross_loss / valid
            return logits, cross_loss + moe_aux_loss * self.moe_multiplier
        return logits, None

    def set_routing(self, mode="learned_top2", observer=None):
        """Configure evaluation-only routing and optional detached telemetry."""
        for block in self.MoEBlocks:
            block.moe.set_routing(mode, observer)

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        was_training = self.training
        self.eval()

        try:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            x = x.to(next(self.parameters()).device).long()

            for _ in range(max_new_tokens):
                x_context = x[:, -self.max_seq_length:]
                logits, _ = self(x_context)
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

            return x

        finally:
            if was_training:
                self.train()

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


@dataclass
class ModelConfig:
    max_seq_length: int = 1024  # max sequence length
    vocab_size: int = 50304  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 token
    num_layers: int = 6  # number of transformer/MoE blocks
    hidden_dim: int = 768  # embedding dimension
    moe_multiplier: float = 0.01  # moe aux loss multiplier
    num_experts: int = 8  # number of experts
    top_k: int = 2  # number of top-k experts to select
