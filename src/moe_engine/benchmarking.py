"""Shared workload definitions for reproducible miniMoE benchmarks."""

# This is 62 GPT-2 tokens and is the canonical benchmark/profile prompt.
PROMPT = (
    "The mixture-of-experts architecture replaces the dense feed-forward "
    "layer of a transformer with a set of expert networks and a router. "
    "For each token, the router selects a small number of experts, so the "
    "model gains parameters without a matching increase in compute. The "
    "hard part is serving it efficiently, because"
)
