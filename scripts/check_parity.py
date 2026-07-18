"""Reproduce the real-checkpoint parity claim: same weights, same numbers.

The tiny random-weight tests in tests/test_parity.py prove the mechanism;
this script proves it on the actual 280M checkpoint. It reports the maximum
absolute and relative logit differences between the reference and the
engine, and the first position (if any) where their greedy generations
disagree.

Usage:
    uv run python scripts/check_parity.py
    uv run python scripts/check_parity.py --device mps --output results/parity_mps.json

CPU is the default because it is deterministic; mps/cuda kernels are
allowed to differ slightly more.
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
    parser.add_argument("--output", help="also write the report to this JSON file")
    args = parser.parse_args()

    reference, _, _ = load_reference_model(args.checkpoint, args.device)
    engine, _, _ = load_engine_model(args.checkpoint, args.device)

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode(PROMPT)], device=args.device)

    with torch.no_grad():
        reference_logits, _ = reference(ids)
        engine_logits = engine(ids, engine.new_cache())

    abs_diff = (reference_logits - engine_logits).abs()
    # Element-wise relative error explodes wherever a logit crosses zero, so
    # measure against at-least-unit logit magnitude instead.
    rel_diff = abs_diff / reference_logits.abs().clamp(min=1.0)

    reference_tokens = reference.generate(ids[0], args.new_tokens, temperature=0)[0].tolist()
    engine_tokens = engine.generate(ids[0], args.new_tokens, temperature=0)[0].tolist()
    first_mismatch = next(
        (i for i, (a, b) in enumerate(zip(reference_tokens, engine_tokens)) if a != b),
        None,
    )
    if first_mismatch is None and len(reference_tokens) != len(engine_tokens):
        first_mismatch = min(len(reference_tokens), len(engine_tokens))

    report = {
        "checkpoint": args.checkpoint,
        "device": args.device,
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
