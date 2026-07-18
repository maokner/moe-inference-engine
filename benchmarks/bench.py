"""Milestone 1 benchmark harness: the numbers every later change is measured against.

Two things get measured, because inference has two distinct phases:

1. Prefill - one forward pass over the whole prompt. This bounds
   time-to-first-token (TTFT), what a user feels as "lag before it starts".
2. Decode - generating tokens one at a time. Steady-state tokens/sec,
   what a user feels as "how fast it types".

The reference model recomputes the ENTIRE context for every new token
(no KV cache), so its decode cost grows with sequence length. That is
exactly the flaw milestone 2 fixes - this harness exists to prove it.

Measurement rules (each fixes a real bug in the first version of this file):
- Decode is timed per token with our own greedy loop instead of timing
  generate() as a black box. The loop does the same work, but per-token
  times let us see decode slow down as the sequence grows, where a single
  average would hide it.
- The first decode step runs the model over the whole prompt - that IS the
  prefill - so it is excluded from steady-state decode throughput.
- The loop stops at EOS exactly like generate(), so throughput divides by
  tokens actually produced, not tokens requested.
- Decode is repeated, and reported numbers are per-position MEDIANS across
  runs: greedy decode is deterministic, so position i is the same work every
  run, and the median discards one-off stalls (a background process freezing
  a single step for seconds) that would poison a wall-clock average.
- Every raw step time from every run is preserved in the --output artifact,
  so any statistic reported here can be recomputed - and challenged - later.

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

from moe_engine.checkpoint import load_engine_model, load_reference_model

EOS_TOKEN_ID = 50256  # GPT-2 <|endoftext|>; the reference generate() stops on it too

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


def time_prefill(forward_once, device: str, repeats: int) -> float:
    """Median seconds for one forward pass over the full prompt (~TTFT)."""
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            forward_once()
            sync(device)
            times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def timed_greedy_decode(model, prompt_ids, device: str, new_tokens: int) -> list[float]:
    """Greedy-decode up to new_tokens, returning seconds per generated token.

    Same computation as the reference generate() at temperature=0 - full
    recompute over the growing sequence each step - just with a stopwatch
    around every token.
    """
    x = prompt_ids.unsqueeze(0)
    step_times = []
    with torch.no_grad():
        for _ in range(new_tokens):
            sync(device)
            start = time.perf_counter()
            logits, _ = model(x[:, -model.max_seq_length:])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync(device)
            step_times.append(time.perf_counter() - start)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == EOS_TOKEN_ID:
                break
    return step_times


def timed_greedy_decode_engine(model, prompt_ids, device: str, new_tokens: int) -> list[float]:
    """Engine version of the loop above. The first iteration processes the
    whole prompt (the prefill fills the cache); every iteration after that
    feeds ONE token through the model - the cache remembers the rest.
    Compare the two loop bodies: that one-line difference in what gets fed
    to model() is the entire milestone."""
    cache = model.new_cache()
    step_input = prompt_ids.unsqueeze(0)
    step_times = []
    with torch.no_grad():
        for _ in range(new_tokens):
            sync(device)
            start = time.perf_counter()
            logits = model(step_input, cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync(device)
            step_times.append(time.perf_counter() - start)
            step_input = next_token
            if next_token.item() == EOS_TOKEN_ID:
                break
    return step_times


def tok_per_sec(step_times: list[float]) -> float:
    return round(len(step_times) / sum(step_times), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["reference", "engine"], default="reference")
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="also write results (with per-token times) to this JSON file")
    args = parser.parse_args()

    device = pick_device()
    enc = tiktoken.get_encoding("gpt2")

    if args.system == "reference":
        model, config, metadata = load_reference_model(args.checkpoint, device)
        prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
        prefill_once = lambda: model(prompt_ids.unsqueeze(0))
        decode = lambda n: timed_greedy_decode(model, prompt_ids, device, n)
        label = "reference (no KV cache)"
    else:
        model, config, metadata = load_engine_model(args.checkpoint, device)
        prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
        prefill_once = lambda: model(prompt_ids.unsqueeze(0), model.new_cache())
        decode = lambda n: timed_greedy_decode_engine(model, prompt_ids, device, n)
        label = "engine (KV cache)"

    # Warmup: the first pass on a fresh device pays one-time setup costs
    # (kernel compilation, memory allocator growth) that aren't steady-state.
    decode(8)

    prefill_s = time_prefill(prefill_once, device, repeats=5)

    runs = [decode(args.new_tokens) for _ in range(args.repeats)]

    # steps[0] of each run processed the whole prompt (the prefill), so
    # steady-state decode is everything after it. "typical" is the median
    # time across runs at each position - a stall-free reconstruction of one
    # run. Splitting it in half shows whether decode slows as context grows.
    decode_runs = [steps[1:] for steps in runs]
    typical = [sorted(times)[len(times) // 2] for times in zip(*decode_runs)]
    pooled = sorted(t for steps in decode_runs for t in steps)
    half = len(typical) // 2

    results = {
        "system": label,
        "device": device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "prompt_tokens": len(prompt_ids),
        "requested_tokens": args.new_tokens,
        "generated_tokens": min(len(steps) for steps in runs),
        "repeats": args.repeats,
        "prefill_ms": round(prefill_s * 1000, 1),
        "decode_median_ms": round(pooled[len(pooled) // 2] * 1000, 1),
        "decode_p90_ms": round(pooled[int(len(pooled) * 0.9)] * 1000, 1),
        "decode_tok_per_sec": tok_per_sec(typical),
        "decode_tok_per_sec_first_half": tok_per_sec(typical[:half]),
        "decode_tok_per_sec_second_half": tok_per_sec(typical[half:]),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        results["runs_step_times_ms"] = [
            [round(t * 1000, 2) for t in steps] for steps in runs
        ]
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
