"""Compare engine and reference outputs on a checkpoint.

Usage:
    uv run python scripts/check_parity.py
    uv run python scripts/check_parity.py --device mps --output results/parity_mps.json

CPU is the default for deterministic results.
"""

import argparse
import json

import tiktoken
import torch

from moe_engine.checkpoint import load_engine_model, load_reference_model

PROMPT = "The key idea behind a mixture-of-experts model is"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument(
        "--engine-cache",
        choices=["contiguous", "paged-auto", "paged-gather", "paged-direct"],
        default="contiguous",
        help="which KV cache and attention path the engine uses",
    )
    parser.add_argument(
        "--moe",
        choices=["auto", "direct", "reference"],
        default="auto",
        help="which MoE decode path the engine uses",
    )
    parser.add_argument("--output", help="also write the report to this JSON file")
    args = parser.parse_args()

    reference, _, _ = load_reference_model(args.checkpoint, args.device)
    engine, _, _ = load_engine_model(args.checkpoint, args.device)
    engine.set_moe_mode(args.moe)

    def make_cache():
        if args.engine_cache == "contiguous":
            return engine.new_cache()
        block_size = 16
        num_blocks = -(-engine.max_seq_length // block_size)
        allocator = engine.new_block_allocator(
            num_blocks=num_blocks,
            block_size=block_size,
            attention_mode=args.engine_cache.removeprefix("paged-"),
        )
        return allocator.new_cache()

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode(PROMPT)], device=args.device)

    with torch.no_grad():
        reference_logits, _ = reference(ids)
        engine_logits = engine(ids, make_cache())

    abs_diff = (reference_logits - engine_logits).abs()
    # Clamp the scale to avoid unstable ratios near zero.
    rel_diff = abs_diff / reference_logits.abs().clamp(min=1.0)

    reference_tokens = reference.generate(ids[0], args.new_tokens, temperature=0)[
        0
    ].tolist()
    engine_tokens = engine.generate(
        ids[0], args.new_tokens, temperature=0, cache=make_cache()
    )[0].tolist()
    first_mismatch = next(
        (i for i, (a, b) in enumerate(zip(reference_tokens, engine_tokens)) if a != b),
        None,
    )
    if first_mismatch is None and len(reference_tokens) != len(engine_tokens):
        first_mismatch = min(len(reference_tokens), len(engine_tokens))

    report = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "engine_cache": args.engine_cache,
        "moe": args.moe,
        "dtype": str(next(engine.parameters()).dtype),
        "prompt": PROMPT,
        "prompt_tokens": ids.shape[1],
        "max_abs_logit_diff": abs_diff.max().item(),
        "max_rel_logit_diff": rel_diff.max().item(),
        "greedy_tokens_compared": len(reference_tokens) - ids.shape[1],
        "first_greedy_mismatch": first_mismatch,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
