import asyncio
import json

from moe_engine.vllm_runtime import async_engine_kwargs


def inspect_worker(worker):
    from pathlib import Path
    from safetensors import safe_open

    model = worker.model_runner.model
    params = dict(model.named_parameters())
    checkpoint_path = Path("checkpoints/minimoe-hf/model.safetensors")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:

        def difference(param_name, checkpoint_name, select=None):
            actual = params[param_name].detach().float().cpu()
            if select is not None:
                actual = actual[select]
            expected = checkpoint.get_tensor(checkpoint_name).float()
            error = (actual - expected).abs()
            return {
                "shape": list(actual.shape),
                "max": error.max().item(),
                "mean": error.mean().item(),
            }

        report = {
            "token_embedding": difference(
                "model.token_embedding.weight", "model.token_embedding.weight"
            ),
            "positional_embedding": difference(
                "model.positional_embedding.weight",
                "model.positional_embedding.weight",
            ),
            "lm_head_bias": difference("lm_head.bias", "lm_head.bias"),
            "attention_in_weight": difference(
                "model.MoEBlocks.0.attention.in_proj.weight",
                "model.MoEBlocks.0.attention.in_proj_weight",
            ),
            "attention_in_bias": difference(
                "model.MoEBlocks.0.attention.in_proj.bias",
                "model.MoEBlocks.0.attention.in_proj_bias",
            ),
            "attention_out_weight": difference(
                "model.MoEBlocks.0.attention.out_proj.weight",
                "model.MoEBlocks.0.attention.out_proj.weight",
            ),
            "router": difference(
                "model.MoEBlocks.0.moe.router.weight",
                "model.MoEBlocks.0.moe.router.weight",
            ),
            "expert0_w1": difference(
                "model.MoEBlocks.0.moe.experts.w13_weight",
                "model.MoEBlocks.0.moe.experts.0.net.0.weight",
                0,
            ),
            "expert0_b1": difference(
                "model.MoEBlocks.0.moe.experts.w13_bias",
                "model.MoEBlocks.0.moe.experts.0.net.0.bias",
                0,
            ),
            "expert0_w2": difference(
                "model.MoEBlocks.0.moe.experts.w2_weight",
                "model.MoEBlocks.0.moe.experts.0.net.2.weight",
                0,
            ),
            "expert0_b2": difference(
                "model.MoEBlocks.0.moe.experts.w2_bias",
                "model.MoEBlocks.0.moe.experts.0.net.2.bias",
                0,
            ),
            "tied_pointer": (
                params["model.token_embedding.weight"].data_ptr()
                == model.lm_head.weight.data_ptr()
            ),
            "parameter_names": sorted(params),
        }
    return report


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
