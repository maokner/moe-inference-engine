"""Measure single-token decode latency as the context grows.

For each target context length C the model prefills C-8 synthetic tokens,
then times 8 one-token decode steps, so the timed steps attend over roughly
C cached tokens. Feeding a fixed next token (instead of argmax + .item())
keeps host-device synchronization out of everything except the timing sync.

Systems run in the given order, then again reversed (the Thunder GPU is
virtualized and has run-to-run variance), and all raw step times land in the
output JSON.

Usage:
    uv run python benchmarks/context_sweep.py --output results/context_sweep.json
"""

import argparse
import json
import platform
import time

import torch

from moe_engine import paged_attention
from moe_engine.checkpoint import load_engine_model

STEPS = 8  # timed decode steps per context length
FILLER_TOKEN = 100  # arbitrary fixed token id for synthetic prompts


def median(values) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_context(model, device: str, system: str, ctx: int) -> list[float]:
    """Time STEPS one-token decodes at roughly `ctx` cached tokens."""
    block_size = 16
    num_blocks = -(-model.max_seq_length // block_size)
    prefill_len = max(1, ctx - STEPS)
    prompt = torch.full((1, prefill_len), FILLER_TOKEN, device=device, dtype=torch.long)
    step = torch.full((1, 1), FILLER_TOKEN, device=device, dtype=torch.long)

    if system == "contiguous":
        cache = model.new_cache()
    else:  # "paged-gather" or "paged-direct"
        mode = system.removeprefix("paged-")
        allocator = model.new_block_allocator(
            num_blocks=num_blocks, block_size=block_size, attention_mode=mode
        )
        cache = allocator.new_cache()

    step_times = []
    with torch.no_grad():
        model(prompt, cache)
        for _ in range(min(STEPS, model.max_seq_length - prefill_len)):
            sync(device)
            start = time.perf_counter()
            model(step, cache)
            sync(device)
            step_times.append(time.perf_counter() - start)
    if system != "contiguous":
        cache.free()
    return step_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--contexts", default="16,64,256,1024")
    parser.add_argument(
        "--systems", default="contiguous,paged-gather,paged-direct", help="comma-separated"
    )
    parser.add_argument("--passes", type=int, default=2, help="odd passes reverse system order")
    parser.add_argument("--output", help="write raw results to this JSON file")
    args = parser.parse_args()

    device = pick_device()
    model, _, metadata = load_engine_model(args.checkpoint, device)
    contexts = [int(c) for c in args.contexts.split(",")]
    systems = args.systems.split(",")
    if "paged-direct" in systems and not (paged_attention.HAS_TRITON and device == "cuda"):
        print(f"skipping paged-direct: needs CUDA + Triton, have {device}")
        systems.remove("paged-direct")

    # Warm up kernel compilation and allocator caches outside the timed region.
    for system in systems:
        run_context(model, device, system, contexts[0])

    # Interleave systems inside each context so slow GPU drift (large on
    # virtualized GPUs) hits every system equally instead of one chunk each.
    passes = []
    for i in range(args.passes):
        order = systems if i % 2 == 0 else list(reversed(systems))
        result = {system: {} for system in systems}
        for ctx in contexts:
            for system in order:
                result[system][ctx] = run_context(model, device, system, ctx)
        passes.append(result)

    report = {
        "device": device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "steps_per_context": STEPS,
        "median_step_ms": {
            system: {
                ctx: round(median(t for p in passes for t in p[system][ctx]) * 1000, 2)
                for ctx in contexts
            }
            for system in systems
        },
        "raw_step_times_ms": [
            {system: {ctx: [round(t * 1000, 2) for t in times] for ctx, times in sweeps.items()}
             for system, sweeps in p.items()}
            for p in passes
        ],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "raw_step_times_ms"}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
