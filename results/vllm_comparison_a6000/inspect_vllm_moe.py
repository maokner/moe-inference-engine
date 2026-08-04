import asyncio
import json

from moe_engine.vllm_runtime import async_engine_kwargs


def inspect_worker(worker):
    import torch
    from torch.nn import functional as F

    model = worker.model_runner.model
    moe = model.model.MoEBlocks[0].moe
    experts = moe.experts
    torch.manual_seed(1234)
    hidden = torch.randn(5, 768, device="cuda", dtype=torch.float32)
    router_logits, _ = moe.router(hidden)
    reference_hidden = hidden.clone()
    selected_weights, selected_indices = experts.router.select_experts(
        hidden_states=hidden,
        router_logits=router_logits,
    )
    experts.ensure_moe_quant_config_init()
    actual = experts.quant_method.apply(
        layer=experts,
        router=experts.router,
        x=hidden,
        router_logits=router_logits,
    )
    top_values, top_indices = torch.topk(router_logits, 2, dim=-1)
    top_weights = F.softmax(top_values, dim=-1)

    def reference(activation, indices=top_indices, weights=top_weights):
        output = torch.zeros_like(hidden)
        for token_id in range(hidden.shape[0]):
            for slot in range(2):
                expert_id = int(indices[token_id, slot])
                intermediate = F.linear(
                    reference_hidden[token_id],
                    experts.w13_weight[expert_id],
                    experts.w13_bias[expert_id],
                )
                activated = (
                    F.gelu(intermediate, approximate=activation)
                    if activation in ("none", "tanh")
                    else F.silu(intermediate)
                )
                expert_output = F.linear(
                    activated,
                    experts.w2_weight[expert_id],
                    experts.w2_bias[expert_id],
                )
                output[token_id] += weights[token_id, slot] * expert_output
        return output

    def error(expected):
        difference = (actual - expected).abs()
        return {
            "max": float(difference.max()),
            "mean": float(difference.mean()),
            "actual_norm": float(actual.norm()),
            "expected_norm": float(expected.norm()),
        }

    return {
        "exact_gelu": error(reference("none")),
        "tanh_gelu": error(reference("tanh")),
        "silu": error(reference("silu")),
        "selected_exact_gelu": error(
            reference("none", selected_indices, selected_weights)
        ),
        "actual_sample": actual[0, :8].tolist(),
        "routing_indices": top_indices.tolist(),
        "routing_weights": top_weights.tolist(),
        "selected_indices": selected_indices.tolist(),
        "selected_weights": selected_weights.tolist(),
    }


async def main():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    args = AsyncEngineArgs(
        **async_engine_kwargs(
            "checkpoints/minimoe-hf",
            enforce_eager=True,
            kv_cache_memory_bytes=75_497_472,
        )
    )
    engine = AsyncLLM.from_engine_args(args)
    try:
        report = await engine.collective_rpc(inspect_worker)
        print(json.dumps(report, indent=2))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
