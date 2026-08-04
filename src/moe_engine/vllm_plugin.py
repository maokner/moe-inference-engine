"""Lazy entry point for miniMoE's out-of-tree vLLM model."""


def register() -> None:
    """Register without importing CUDA-touching vLLM model modules eagerly."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MiniMoEForCausalLM",
        "moe_engine.vllm_model:MiniMoEForCausalLM",
    )
