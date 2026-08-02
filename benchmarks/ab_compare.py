"""Interleaved comparison of the KV-cache paths on one noisy GPU.

bench.py measures one system per process, which is fine on quiet hardware.
On Thunder's virtualized GPUs, throughput drifts by tens of percent over a
few minutes, which swamps the difference between cache paths measured
minutes apart. This script instead interleaves short rounds of every system
(rotating which one starts each round), so drift hits all systems equally
and paired per-round comparisons stay meaningful.

Usage:
    uv run python benchmarks/ab_compare.py --output results/ab_compare.json
"""

import argparse
import json
import platform
import time

import tiktoken
import torch

from moe_engine.checkpoint import load_engine_model

# Same workload as bench.py.
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
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_round(model, device, cache, prompt_ids, new_tokens):
    """One prefill + greedy decode; returns (prefill_seconds, step_seconds)."""
    with torch.no_grad():
        sync(device)
        start = time.perf_counter()
        logits = model(prompt_ids.unsqueeze(0), cache)
        sync(device)
        prefill_s = time.perf_counter() - start

        step_input = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_times = []
        for _ in range(new_tokens):
            sync(device)
            start = time.perf_counter()
            logits = model(step_input, cache)
            step_input = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync(device)
            step_times.append(time.perf_counter() - start)
    return prefill_s, step_times


def median(values) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Two-sided 95% critical values of Student's t by degrees of freedom.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def paired_t_summary(deltas):
    """Mean and 95% confidence interval for paired per-round deltas.

    Every system runs inside every round, so per-round differences are
    paired observations; a paired t-interval is the standard uncertainty
    summary for their mean. Rounds are treated as independent, which the
    rotating start order supports.
    """
    n = len(deltas)
    mean = sum(deltas) / n
    if n < 2:
        return {"per_round_delta_ms": deltas, "mean_delta_ms": round(mean, 3), "ci95_ms": None}
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    half_width = T95.get(n - 1, 1.96) * (var / n) ** 0.5
    return {
        "per_round_delta_ms": [round(d, 3) for d in deltas],
        "mean_delta_ms": round(mean, 3),
        "ci95_ms": [round(mean - half_width, 3), round(mean + half_width, 3)],
    }


def build_report(results, systems):
    """Aggregate raw per-round timings into summary, per-round, and paired stats.

    `results` maps system -> {"prefill_s": [...], "rounds": [[step_s, ...], ...]}.
    Throughput is aggregate: completed decode steps divided by total decode
    time, not the inverse of a median step.
    """
    summary, per_round = {}, {}
    for system in systems:
        rounds = results[system]["rounds"]
        pooled = sorted(t for steps in rounds for t in steps)
        total_s = sum(pooled)
        summary[system] = {
            "prefill_ms": round(median(results[system]["prefill_s"]) * 1000, 1),
            "total_decode_steps": len(pooled),
            "total_decode_s": round(total_s, 3),
            "decode_tok_per_sec": round(len(pooled) / total_s, 2),
            "decode_median_ms": round(pooled[len(pooled) // 2] * 1000, 2),
            "decode_p90_ms": round(pooled[int(len(pooled) * 0.9)] * 1000, 2),
        }
        per_round[system] = {
            "total_decode_ms": [round(sum(steps) * 1000, 2) for steps in rounds],
            "tok_per_sec": [round(len(steps) / sum(steps), 2) for steps in rounds],
            "mean_step_ms": [round(sum(steps) / len(steps) * 1000, 3) for steps in rounds],
        }

    # Paired per-round deltas of mean step latency (positive = first is slower).
    paired = {}
    def mean_steps_ms(system):
        return [sum(steps) / len(steps) * 1000 for steps in results[system]["rounds"]]

    if "paged-direct" in systems and "contiguous" in systems:
        paired["direct_minus_contiguous"] = paired_t_summary(
            [d - c for d, c in zip(mean_steps_ms("paged-direct"), mean_steps_ms("contiguous"))]
        )
    if "paged-direct" in systems and "paged-gather" in systems:
        paired["direct_minus_gather"] = paired_t_summary(
            [d - g for d, g in zip(mean_steps_ms("paged-direct"), mean_steps_ms("paged-gather"))]
        )
    return summary, per_round, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument(
        "--systems", default="contiguous,paged-gather,paged-direct", help="comma-separated"
    )
    parser.add_argument("--output", help="write raw results to this JSON file")
    args = parser.parse_args()

    device = pick_device()
    model, _, metadata = load_engine_model(args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device=device)
    systems = args.systems.split(",")

    block_size = 16
    num_blocks = -(-model.max_seq_length // block_size)

    def make_cache(system):
        if system == "contiguous":
            return model.new_cache()
        allocator = model.new_block_allocator(
            num_blocks=num_blocks,
            block_size=block_size,
            attention_mode=system.removeprefix("paged-"),
        )
        return allocator.new_cache()

    def run(system):
        cache = make_cache(system)
        try:
            return run_round(model, device, cache, prompt_ids, args.new_tokens)
        finally:
            if system != "contiguous":
                cache.free()

    # Warm up kernel compilation and allocations outside the timed rounds.
    for system in systems:
        run(system)

    results = {system: {"prefill_s": [], "rounds": []} for system in systems}
    for r in range(args.rounds):
        # Rotate the starting system so no path always runs first in a round.
        order = systems[r % len(systems) :] + systems[: r % len(systems)]
        for system in order:
            prefill_s, step_times = run(system)
            results[system]["prefill_s"].append(prefill_s)
            results[system]["rounds"].append(step_times)

    summary, per_round, paired = build_report(results, systems)
    report = {
        "device": device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "checkpoint_step": metadata["step"],
        "prompt_tokens": len(prompt_ids),
        "new_tokens_per_round": args.new_tokens,
        "rounds": args.rounds,
        "summary": summary,
        "per_round": per_round,
        "paired_deltas": paired,
        "raw": {
            system: {
                "prefill_ms": [round(t * 1000, 2) for t in results[system]["prefill_s"]],
                "step_times_ms": [
                    [round(t * 1000, 2) for t in steps] for steps in results[system]["rounds"]
                ],
            }
            for system in systems
        },
    }

    print(json.dumps({k: report[k] for k in ("summary", "per_round", "paired_deltas")}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
