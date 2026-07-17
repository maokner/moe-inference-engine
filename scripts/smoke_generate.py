"""Step 0: prove the model loads and generates on this machine.

Usage:
    uv run python scripts/smoke_generate.py
"""

import argparse
import time

import tiktoken
import torch

from moe_engine.checkpoint import load_reference_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--prompt", default="The key idea behind a mixture-of-experts model is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = pick_device()
    model, config, metadata = load_reference_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded step {metadata['step']} ({n_params / 1e6:.0f}M params) on {device}")

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(args.prompt), device=device)

    start = time.perf_counter()
    out = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=50)
    elapsed = time.perf_counter() - start

    new_tokens = out.shape[-1] - prompt_ids.shape[-1]
    print(f"\n{enc.decode(out[0].tolist())}\n")
    print(f"{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
