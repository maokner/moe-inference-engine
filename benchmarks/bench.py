"""Benchmark prefill latency and greedy decode throughput.

Decode metrics exclude prefill and use per-position medians across
deterministic runs. JSON output includes all raw step times.

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

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model, load_reference_model

EOS_TOKEN_ID = 50256


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    # Device kernels are asynchronous.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_prefill(make_cache, forward, device: str, repeats: int, cleanup=None) -> float:
    """Return median forward-only prefill time in seconds."""
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            cache = make_cache()
            sync(device)
            start = time.perf_counter()
            forward(cache)
            sync(device)
            times.append(time.perf_counter() - start)
            if cleanup is not None:
                cleanup(cache)
    return sorted(times)[len(times) // 2]


def timed_greedy_decode(model, prompt_ids, device: str, new_tokens: int) -> list[float]:
    """Time reference-model greedy decode one token at a time."""
    x = prompt_ids.unsqueeze(0)
    step_times = []
    with torch.no_grad():
        for _ in range(new_tokens):
            sync(device)
            start = time.perf_counter()
            logits, _ = model(x[:, -model.max_seq_length :])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync(device)
            step_times.append(time.perf_counter() - start)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == EOS_TOKEN_ID:
                break
    return step_times


def timed_greedy_decode_engine(
    model, prompt_ids, device: str, new_tokens: int, cache=None
) -> list[float]:
    """Time cache-backed greedy decode one token at a time."""
    if cache is None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", choices=["reference", "engine", "engine-paged"], default="reference"
    )
    parser.add_argument(
        "--paged-attention",
        choices=["auto", "direct", "gather"],
        default="auto",
        help="engine-paged only: direct Triton kernel, legacy gather path, or auto",
    )
    parser.add_argument(
        "--moe",
        choices=["auto", "direct", "reference"],
        default="auto",
        help="engine only: fixed-shape Triton decode, PyTorch oracle, or auto",
    )
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu"
    )
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", help="also write results (with per-token times) to this JSON file"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error(
            "--new-tokens must be at least 2 (the first step is excluded as prefill)"
        )
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    device = pick_device() if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")
    enc = tiktoken.get_encoding("gpt2")

    kv_config = {}
    if args.system == "reference":
        model, _config, metadata = load_reference_model(args.checkpoint, device)
        prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
        cleanup = None

        def make_cache():
            return None

        def prefill(_):
            return model(prompt_ids.unsqueeze(0))

        def decode(n):
            return timed_greedy_decode(model, prompt_ids, device, n)

        label = "reference (no KV cache)"
    elif args.system == "engine":
        model, _config, metadata = load_engine_model(args.checkpoint, device)
        model.set_moe_mode(args.moe)
        prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
        make_cache, cleanup = model.new_cache, None

        def prefill(cache):
            return model(prompt_ids.unsqueeze(0), cache)

        def decode(n):
            return timed_greedy_decode_engine(model, prompt_ids, device, n)

        label = f"engine (KV cache, MoE {args.moe})"
        kv_config["moe"] = args.moe
    else:
        model, _config, metadata = load_engine_model(args.checkpoint, device)
        model.set_moe_mode(args.moe)
        prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
        # Match the server's one-sequence pool.
        block_size = 16
        num_blocks = -(-model.max_seq_length // block_size)
        allocator = model.new_block_allocator(
            num_blocks=num_blocks,
            block_size=block_size,
            attention_mode=args.paged_attention,
        )
        pool_bytes = 2 * allocator.k_pool.numel() * allocator.k_pool.element_size()
        kv_config = {
            "kv_block_size": block_size,
            "kv_num_blocks": num_blocks,
            "kv_pool_mb": round(pool_bytes / 2**20, 1),
            "paged_attention": args.paged_attention,
            "moe": args.moe,
        }

        def decode(n):
            cache = allocator.new_cache()
            try:
                return timed_greedy_decode_engine(
                    model, prompt_ids, device, n, cache=cache
                )
            finally:
                cache.free()

        def cleanup(cache):
            cache.free()

        make_cache = allocator.new_cache

        def prefill(cache):
            return model(prompt_ids.unsqueeze(0), cache)

        label = f"engine (paged KV cache, {args.paged_attention}, MoE {args.moe})"

    # Warm up kernels and device allocations.
    decode(8)

    prefill_s = time_prefill(make_cache, prefill, device, repeats=5, cleanup=cleanup)

    runs = [decode(args.new_tokens) for _ in range(args.repeats)]

    # Exclude prefill, then take the median at each decode position.
    decode_runs = [steps[1:] for steps in runs]
    typical = [sorted(times)[len(times) // 2] for times in zip(*decode_runs)]
    pooled = sorted(t for steps in decode_runs for t in steps)
    if not pooled:
        raise SystemExit(
            "no decode steps to aggregate: every run hit EOS on its first token"
        )
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
        **kv_config,
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
