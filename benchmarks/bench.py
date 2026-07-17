"""Milestone 1 benchmark harness: the numbers every later change is measured against.

Two things get measured, because inference has two distinct phases:

1. Prefill - one forward pass over the whole prompt. This bounds
   time-to-first-token (TTFT), what a user feels as "lag before it starts".
2. Decode - generating tokens one at a time. Steady-state tokens/sec,
   what a user feels as "how fast it types".

The reference model recomputes the ENTIRE context for every new token
(no KV cache), so its decode cost grows with sequence length. That is
exactly the flaw milestone 2 fixes - this harness exists to prove it.

Usage:
    uv run python benchmarks/bench.py
    uv run python benchmarks/bench.py --output results/baseline.json
"""

import argparse
import json
import platform
import time

import tiktoken
import torch

from moe_engine.checkpoint import load_reference_model

# A fixed prompt so every system we ever benchmark sees identical input.
PROMPT = (
    "The mixture-of-experts architecture replaces the dense feed-forward "
    "layer of a transformer with a set of expert networks and a router. "
    "For each token, the router selects a small number of experts, so the "
    "model gains parameters without a matching increase in compute. The "
    "hard part is serving it efficiently, because"
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    # GPU work is asynchronous: python moves on while the device is still
    # computing. Timing without a sync measures how fast we *queued* work,
    # not how fast it ran.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_prefill(model, prompt_ids, device: str, repeats: int = 5) -> float:
    """Median seconds for one forward pass over the full prompt (~TTFT)."""
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            model(prompt_ids.unsqueeze(0))
            sync(device)
            times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def time_decode(model, prompt_ids, device: str, new_tokens: int) -> float:
    """Seconds to generate new_tokens greedily (temperature=0 for determinism)."""
    sync(device)
    start = time.perf_counter()
    model.generate(prompt_ids, max_new_tokens=new_tokens, temperature=0)
    sync(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--output", help="also write results to this JSON file")
    args = parser.parse_args()

    device = pick_device()
    model, config, metadata = load_reference_model(args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)

    # Warmup: the first pass on a fresh device pays one-time setup costs
    # (kernel compilation, memory allocator growth) that aren't steady-state.
    model.generate(prompt_ids, max_new_tokens=8, temperature=0)

    prefill_s = time_prefill(model, prompt_ids, device)
    decode_s = time_decode(model, prompt_ids, device, args.new_tokens)

    results = {
        "system": "reference (no KV cache)",
        "device": device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "prompt_tokens": len(prompt_ids),
        "new_tokens": args.new_tokens,
        "prefill_ms": round(prefill_s * 1000, 1),
        "decode_tokens_per_sec": round(args.new_tokens / decode_s, 1),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
