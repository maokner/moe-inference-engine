"""Validate source, Hugging Face, and engine parity on an explicit device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model, load_reference_model
from moe_engine.hf.convert import convert_state_dict

ATOL_HF = 1e-5
RTOL_HF = 1e-5
ATOL_ENGINE = 1e-4
RTOL_ENGINE = 1e-4
TOKENS = 64


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = (actual - expected).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    return {
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "max_relative_error": float((error / denominator).max().item()),
    }


@torch.no_grad()
def _generate_reference(model, prompt: torch.Tensor) -> list[int]:
    sequence = prompt[None, :]
    generated = []
    for _ in range(TOKENS):
        logits, _ = model(sequence[:, -model.max_seq_length :])
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        sequence = torch.cat((sequence, token), dim=1)
    return generated


@torch.no_grad()
def _generate_hf(model, prompt: torch.Tensor) -> list[int]:
    sequence = prompt[None, :]
    generated = []
    for _ in range(TOKENS):
        logits = model(input_ids=sequence, use_cache=False).logits
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        sequence = torch.cat((sequence, token), dim=1)
    return generated


@torch.no_grad()
def _generate_engine(model, prompt: torch.Tensor) -> list[int]:
    cache = model.new_cache(batch_size=1)
    step_input = prompt[None, :]
    generated = []
    for _ in range(TOKENS):
        logits = model(step_input, cache)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        step_input = token
    return generated


def validate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "mps":
        raise ValueError("MPS validation is prohibited for this comparison")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_hf_state = convert_state_dict(checkpoint["model"])
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    loaded_hf_state = hf_model.state_dict()
    for name, expected in expected_hf_state.items():
        torch.testing.assert_close(
            loaded_hf_state[name].cpu(), expected, atol=0, rtol=0
        )

    reference, _, _ = load_reference_model(args.checkpoint, device)
    engine, _, _ = load_engine_model(args.checkpoint, device)
    engine.set_moe_mode("reference")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if len(prompt_ids) != 62:
        raise AssertionError(f"canonical prompt has {len(prompt_ids)} tokens, not 62")
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        reference_logits, _ = reference(prompt[None, :])
        hf_logits = hf_model(input_ids=prompt[None, :], use_cache=False).logits
        engine_logits = engine(prompt[None, :], engine.new_cache())
    torch.testing.assert_close(hf_logits, reference_logits, atol=ATOL_HF, rtol=RTOL_HF)
    torch.testing.assert_close(
        engine_logits, reference_logits, atol=ATOL_ENGINE, rtol=RTOL_ENGINE
    )

    reference_tokens = _generate_reference(reference, prompt)
    hf_tokens = _generate_hf(hf_model, prompt)
    engine_tokens = _generate_engine(engine, prompt)
    if not reference_tokens == hf_tokens == engine_tokens:
        raise AssertionError("64-token greedy outputs differ")

    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": str(device),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": TOKENS,
        "checkpoint_values_exact": True,
        "hf_vs_original_logits": {
            **_max_errors(hf_logits, reference_logits),
            "atol": ATOL_HF,
            "rtol": RTOL_HF,
            "passed": True,
        },
        "engine_vs_original_logits": {
            **_max_errors(engine_logits, reference_logits),
            "atol": ATOL_ENGINE,
            "rtol": RTOL_ENGINE,
            "passed": True,
        },
        "all_64_greedy_tokens_equal": True,
        "generated_token_ids": reference_tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/minimoe_sft.pt")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("checkpoints/minimoe-hf")
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate(args)
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
