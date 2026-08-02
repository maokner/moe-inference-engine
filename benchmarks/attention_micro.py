"""Microbenchmark one layer's decode attention in isolation.

End-to-end decode is dominated by the MoE expert MLPs, so cache-path
differences are small percentages there. This isolates the attention op to
show what the direct kernel actually replaces: the gather path launches a
chain of index/reshape/copy kernels before SDPA, the direct path launches
one Triton kernel. Calls are timed back to back without per-call syncs, so
launch overhead overlaps as it would in a real forward pass.

Usage:
    uv run python benchmarks/attention_micro.py --output results/attention_micro.json
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from moe_engine import paged_attention
from moe_engine.model import BlockAllocator, KVCache, ModelConfig

ITERS = 200


def time_calls(fn, sync) -> float:
    """Average microseconds per call over ITERS back-to-back calls."""
    for _ in range(20):  # warm up compilation and allocator caches
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync()
    return (time.perf_counter() - start) / ITERS * 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="16,64,256,1024")
    parser.add_argument("--output", help="write results to this JSON file")
    args = parser.parse_args()

    if not (torch.cuda.is_available() and paged_attention.HAS_TRITON):
        raise SystemExit("this microbenchmark needs CUDA + Triton")
    device, sync = "cuda", torch.cuda.synchronize

    # Production attention shape: 8 heads of dim 96, blocks of 16 tokens.
    config = ModelConfig()
    max_ctx = config.max_seq_length
    contiguous = KVCache(config, batch_size=1, device=device)
    allocator = BlockAllocator(config, num_blocks=max_ctx // 16, block_size=16, device=device)
    paged = allocator.new_cache()

    torch.manual_seed(0)
    shape = (1, config.num_heads, max_ctx, config.hidden_dim // config.num_heads)
    k, v = torch.randn(shape, device=device), torch.randn(shape, device=device)
    contiguous.write(0, 0, k, v)
    paged.write(0, 0, k, v)

    q = torch.randn(1, config.num_heads, 1, config.hidden_dim // config.num_heads, device=device)
    results = {}
    for ctx in (int(c) for c in args.contexts.split(",")):
        results[ctx] = {
            "contiguous_read_sdpa_us": round(
                time_calls(
                    lambda: F.scaled_dot_product_attention(q, *contiguous.read(0, ctx)), sync
                ),
                1,
            ),
            "paged_gather_sdpa_us": round(
                time_calls(lambda: F.scaled_dot_product_attention(q, *paged.read(0, ctx)), sync),
                1,
            ),
            "paged_direct_kernel_us": round(
                time_calls(lambda: paged_attention.decode(q, paged, 0, ctx), sync), 1
            ),
        }

    report = {"device": torch.cuda.get_device_name(), "iters": ITERS, "attention_us": results}
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
