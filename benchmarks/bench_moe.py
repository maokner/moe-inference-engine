"""Benchmark the fused MoE kernel against the expert loop, layer-level.

Times one MoE layer from the real checkpoint at decode- and prefill-shaped
token counts. End-to-end effects show up in bench.py and bench_batch.py,
which use the fused path automatically on CUDA.

Usage:
    uv run python benchmarks/bench_moe.py --output results/moe_kernel_cuda.json
"""

import argparse
import json
import platform
import time

import torch

from moe_engine.checkpoint import load_engine_model

TOKEN_COUNTS = [1, 8, 32, 64, 256, 1024]


def time_call(fn, repeats=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e6  # microseconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--output", help="also write results to this JSON file")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "kernel benchmark needs CUDA"

    model, config, metadata = load_engine_model(args.checkpoint, "cuda")
    moe = model.MoEBlocks[0].moe  # One layer stands in for all six.

    runs = []
    with torch.no_grad():
        for tokens in TOKEN_COUNTS:
            x = torch.randn(1, tokens, config.hidden_dim, device="cuda")
            moe.fused_enabled = True
            fused_us = time_call(lambda: moe(x))
            moe.fused_enabled = False
            loop_us = time_call(lambda: moe(x))
            runs.append(
                {
                    "tokens": tokens,
                    "fused_us": round(fused_us, 1),
                    "loop_us": round(loop_us, 1),
                    "speedup": round(loop_us / fused_us, 2),
                }
            )
            print(runs[-1])

    results = {
        "system": "fused MoE kernel vs expert loop (one layer)",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "runs": runs,
    }
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
