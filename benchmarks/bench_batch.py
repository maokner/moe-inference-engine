"""Measure aggregate decode throughput of the engine at several batch sizes.

For each batch size N, submit N copies of the shared prompt and drive the
engine loop on the calling thread until every request finishes. Wall time
covers prefills and decodes, so the number is end-to-end serving
throughput, not a decode-only figure like bench.py reports.

Usage:
    uv run python benchmarks/bench_batch.py
    uv run python benchmarks/bench_batch.py --batch-sizes 1,8,32 --output results/batch.json
"""

import argparse
import json
import platform
import time

import tiktoken
import torch

# bench.py lives in this directory, which Python puts on sys.path.
from bench import PROMPT, pick_device, sync

from moe_engine.checkpoint import load_engine_model
from moe_engine.engine import Engine, Request


def run_batch(model, prompt_ids, device, batch_size, new_tokens, block_size=16):
    """Serve `batch_size` identical greedy requests and time the whole run."""
    blocks_per_seq = -(-model.max_seq_length // block_size)
    allocator = model.new_block_allocator(
        num_blocks=batch_size * blocks_per_seq, block_size=block_size
    )
    engine = Engine(model, allocator, max_batch_size=batch_size)
    requests = [
        Request(prompt_ids=prompt_ids, max_new_tokens=new_tokens, temperature=0)
        for _ in range(batch_size)
    ]

    sync(device)
    start = time.perf_counter()
    for request in requests:
        engine.submit(request)
    while not all(request.done.is_set() for request in requests):
        engine.step()
    sync(device)
    elapsed = time.perf_counter() - start

    for request in requests:
        if request.error is not None:
            raise request.error
    total_tokens = sum(len(request.new_ids) for request in requests)
    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "wall_s": round(elapsed, 2),
        "aggregate_tok_per_sec": round(total_tokens / elapsed, 1),
        "per_seq_tok_per_sec": round(total_tokens / elapsed / batch_size, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--output", help="also write results to this JSON file")
    args = parser.parse_args()
    batch_sizes = [int(n) for n in args.batch_sizes.split(",")]
    if any(n < 1 for n in batch_sizes):
        parser.error("--batch-sizes entries must be at least 1")
    if args.new_tokens < 1:
        parser.error("--new-tokens must be at least 1")

    device = pick_device()
    enc = tiktoken.get_encoding("gpt2")
    model, _, metadata = load_engine_model(args.checkpoint, device)
    prompt_ids = enc.encode(PROMPT)

    with torch.no_grad():
        run_batch(model, prompt_ids, device, batch_size=1, new_tokens=8)  # Warm up.
        runs = [
            run_batch(model, prompt_ids, device, n, args.new_tokens) for n in batch_sizes
        ]

    results = {
        "system": "engine (continuous batching)",
        "device": device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "prompt_tokens": len(prompt_ids),
        "requested_tokens": args.new_tokens,
        "runs": runs,
    }
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
