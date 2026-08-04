"""Interleaved reference-versus-direct MoE decode benchmark.

Every round uses the same 62-token prompt, contiguous KV cache, checkpoint,
and repeated one-token greedy decode.  The starting mode rotates by round so
host or virtualized-GPU drift affects both paths comparably.

Usage:
    uv run python benchmarks/moe_ab_compare.py \
        --output results/moe_ab_compare_a6000.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time

import tiktoken
import torch

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model


def sync() -> None:
    torch.cuda.synchronize()


def environment_report():
    """Record the accelerator and exact tracked source state for the run."""
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "git_commit": commit,
        "git_tracked_status": tracked_status,
        "git_tracked_files_clean": not tracked_status,
    }


def run_round(model, mode, prompt_ids, new_tokens):
    """Return prefill time, one-token decode times, and exact greedy tokens."""
    model.set_moe_mode(mode)
    cache = model.new_cache()
    with torch.no_grad():
        sync()
        start = time.perf_counter()
        logits = model(prompt_ids.unsqueeze(0), cache)
        sync()
        prefill_s = time.perf_counter() - start

        step_input = logits[:, -1].argmax(dim=-1, keepdim=True)
        step_times = []
        tokens = []
        for _ in range(new_tokens):
            sync()
            start = time.perf_counter()
            logits = model(step_input, cache)
            step_input = logits[:, -1].argmax(dim=-1, keepdim=True)
            sync()
            step_times.append(time.perf_counter() - start)
            tokens.append(step_input)
    return prefill_s, step_times, torch.cat(tokens, dim=1).cpu()


def paired_t_summary(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    if n < 2:
        return {"mean_delta_ms": mean, "ci95_ms": None}
    # Student-t critical values for the usual 9 or 21 round runs.
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
    }.get(n - 1, 1.96)
    half_width = critical * statistics.stdev(deltas) / n**0.5
    return {
        "per_round_delta_ms": [round(delta, 3) for delta in deltas],
        "mean_delta_ms": round(mean, 3),
        "ci95_ms": [round(mean - half_width, 3), round(mean + half_width, 3)],
    }


def summarize(results):
    summary = {}
    per_round_means = {}
    for mode, measurements in results.items():
        pooled = [
            time_s for round_times in measurements["rounds"] for time_s in round_times
        ]
        ordered = sorted(pooled)
        total_s = sum(pooled)
        means = [
            statistics.mean(round_times) * 1000
            for round_times in measurements["rounds"]
        ]
        round_total_ms = [
            sum(round_times) * 1000 for round_times in measurements["rounds"]
        ]
        round_tokens_per_second = [
            len(round_times) / sum(round_times)
            for round_times in measurements["rounds"]
        ]
        per_round_means[mode] = means
        summary[mode] = {
            "prefill_median_ms": round(
                statistics.median(measurements["prefill_s"]) * 1000, 2
            ),
            "total_decode_steps": len(pooled),
            "total_decode_s": round(total_s, 6),
            "decode_median_ms": round(statistics.median(pooled) * 1000, 3),
            "decode_p90_ms": round(ordered[int(len(ordered) * 0.9)] * 1000, 3),
            "decode_tok_per_sec": round(len(pooled) / total_s, 3),
            "per_round_total_decode_ms": [round(value, 3) for value in round_total_ms],
            "per_round_decode_tok_per_sec": [
                round(value, 3) for value in round_tokens_per_second
            ],
            "per_round_mean_ms": [round(value, 3) for value in means],
        }
    deltas = [
        direct - reference
        for direct, reference in zip(
            per_round_means["direct"], per_round_means["reference"]
        )
    ]
    return summary, paired_t_summary(deltas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("the direct MoE A/B benchmark needs CUDA")
    if args.rounds < 2 or args.new_tokens < 1:
        parser.error("--rounds must be at least 2 and --new-tokens must be positive")

    model, _, metadata = load_engine_model(args.checkpoint, "cuda")
    environment = environment_report()
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device="cuda")
    if len(prompt_ids) != 62:
        raise RuntimeError(
            f"expected the fixed 62-token prompt, found {len(prompt_ids)}"
        )

    modes = ["reference", "direct"]
    for mode in modes:
        run_round(model, mode, prompt_ids, min(args.new_tokens, 8))

    results = {mode: {"prefill_s": [], "rounds": [], "tokens": []} for mode in modes}
    for round_index in range(args.rounds):
        order = modes[round_index % 2 :] + modes[: round_index % 2]
        round_tokens = {}
        for mode in order:
            prefill_s, step_times, tokens = run_round(
                model, mode, prompt_ids, args.new_tokens
            )
            results[mode]["prefill_s"].append(prefill_s)
            results[mode]["rounds"].append(step_times)
            results[mode]["tokens"].append(tokens.tolist()[0])
            round_tokens[mode] = tokens
        if not torch.equal(round_tokens["direct"], round_tokens["reference"]):
            raise RuntimeError(f"greedy tokens differ in round {round_index}")

    mode_summary, paired = summarize(results)
    summary = {
        "modes": mode_summary,
        "paired_direct_minus_reference_mean_step_ms": paired,
    }
    report = {
        "device": "cuda",
        "environment": environment,
        "checkpoint": args.checkpoint,
        "checkpoint_step": metadata["step"],
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "new_tokens_per_round": args.new_tokens,
        "rounds": args.rounds,
        "cache": "contiguous",
        "summary": summary,
        "raw": {
            "modes": {
                mode: {
                    "prefill_ms": [
                        round(value * 1000, 3) for value in results[mode]["prefill_s"]
                    ],
                    "step_times_ms": [
                        [round(value * 1000, 3) for value in round_times]
                        for round_times in results[mode]["rounds"]
                    ],
                    "generated_token_ids": results[mode]["tokens"],
                }
                for mode in modes
            },
            "paired_direct_minus_reference_mean_step_ms": paired["per_round_delta_ms"],
        },
    }
    print(json.dumps(summary, indent=2))
    with open(args.output, "w") as output_file:
        json.dump(report, output_file, indent=2)


if __name__ == "__main__":
    main()
