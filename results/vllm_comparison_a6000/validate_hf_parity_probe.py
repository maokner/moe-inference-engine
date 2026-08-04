"""Validate source, Hugging Face, engine, and native vLLM parity."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model, load_reference_model
from moe_engine.hf.convert import convert_state_dict
from moe_engine.vllm_runtime import (
    DEFAULT_KV_CACHE_MEMORY_BYTES,
    async_engine_kwargs,
    validate_kv_cache_memory_bytes,
)

ATOL_HF = 1e-5
RTOL_HF = 1e-5
ATOL_ENGINE = 1e-4
RTOL_ENGINE = 1e-4
ATOL_CUDA_REFERENCE = 5e-4
RTOL_CUDA_REFERENCE = 1e-4
ATOL_VLLM_LOGPROBS = 1.0
RTOL_VLLM_LOGPROBS = 2e-4
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


async def _last_output(engine, prompt_ids: list[int], sampling):
    output = None
    async for update in engine.generate(
        {"prompt_token_ids": prompt_ids},
        sampling,
        request_id=uuid.uuid4().hex,
    ):
        output = update
    if output is None:
        raise RuntimeError("vLLM returned no output")
    return output


async def _native_vllm_oracle(
    model_dir: Path,
    prompt_ids: list[int],
    *,
    enforce_eager: bool,
    kv_cache_memory_bytes: int,
    vocab_size: int,
) -> tuple[torch.Tensor, list[int]]:
    """Return one full-vocabulary distribution and an untimed greedy sequence."""
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        **async_engine_kwargs(
            model_dir,
            enforce_eager=enforce_eager,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            max_logprobs=-1,
        )
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        distribution_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            logprobs=-1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        distribution_output = await _last_output(
            engine, prompt_ids, distribution_sampling
        )
        positions = distribution_output.outputs[0].logprobs
        if positions is None or len(positions) != 1:
            raise RuntimeError("vLLM did not return one generated-token distribution")
        token_logprobs = positions[0]
        if token_logprobs is None or len(token_logprobs) != vocab_size:
            actual = None if token_logprobs is None else len(token_logprobs)
            raise RuntimeError(
                f"vLLM returned {actual} log-probabilities, expected {vocab_size}"
            )
        logprobs = torch.full((vocab_size,), float("nan"), dtype=torch.float32)
        for token_id, value in token_logprobs.items():
            logprobs[int(token_id)] = float(value.logprob)
        if not torch.isfinite(logprobs).all():
            raise RuntimeError("vLLM full-vocabulary log-probabilities are incomplete")

        generation_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=TOKENS,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        generation_output = await _last_output(engine, prompt_ids, generation_sampling)
        generated = list(generation_output.outputs[0].token_ids)
        if len(generated) != TOKENS:
            raise RuntimeError(
                f"vLLM generated {len(generated)} tokens, expected {TOKENS}"
            )
        return logprobs, generated
    finally:
        engine.shutdown()


def _validate_native_vllm(
    model_dir: Path,
    prompt_ids: list[int],
    hf_logprobs: torch.Tensor,
    expected_tokens: list[int],
    kv_cache_memory_bytes: int,
) -> dict:
    modes = {}
    for name, enforce_eager in (("vllm", False), ("vllm-eager", True)):
        native_logprobs, native_tokens = asyncio.run(
            _native_vllm_oracle(
                model_dir,
                prompt_ids,
                enforce_eager=enforce_eager,
                kv_cache_memory_bytes=kv_cache_memory_bytes,
                vocab_size=hf_logprobs.numel(),
            )
        )
        torch.testing.assert_close(
            native_logprobs,
            hf_logprobs,
            atol=ATOL_VLLM_LOGPROBS,
            rtol=RTOL_VLLM_LOGPROBS,
        )
        if native_tokens != expected_tokens:
            raise AssertionError(f"64-token greedy output differs for {name}")
        modes[name] = {
            "enforce_eager": enforce_eager,
            "full_vocabulary_size": hf_logprobs.numel(),
            "normalized_logprob_errors": _max_errors(native_logprobs, hf_logprobs),
            "atol": ATOL_VLLM_LOGPROBS,
            "rtol": RTOL_VLLM_LOGPROBS,
            "all_64_greedy_tokens_equal": True,
            "generated_token_ids": native_tokens,
        }
    return modes


def validate(args: argparse.Namespace) -> dict:
    device = torch.device(
        f"cuda:{args.gpu_index}" if args.device == "cuda" else args.device
    )
    if device.type == "mps":
        raise ValueError("MPS validation is prohibited for this comparison")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_index)
    if args.validate_native_vllm and device.type != "cuda":
        raise ValueError("native vLLM validation requires --device cuda")
    if args.validate_native_vllm:
        gpu_name = torch.cuda.get_device_name(args.gpu_index)
        if "A6000" not in gpu_name:
            raise ValueError(
                f"native vLLM validation is restricted to an A6000, found {gpu_name}"
            )
        validate_kv_cache_memory_bytes(args.kv_cache_memory_bytes)

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
    reference_atol = ATOL_CUDA_REFERENCE if device.type == "cuda" else ATOL_HF
    reference_rtol = RTOL_CUDA_REFERENCE if device.type == "cuda" else RTOL_HF
    torch.testing.assert_close(
        hf_logits,
        reference_logits,
        atol=reference_atol,
        rtol=reference_rtol,
    )
    torch.testing.assert_close(
        engine_logits,
        reference_logits,
        atol=(ATOL_CUDA_REFERENCE if device.type == "cuda" else ATOL_ENGINE),
        rtol=(RTOL_CUDA_REFERENCE if device.type == "cuda" else RTOL_ENGINE),
    )
    torch.testing.assert_close(hf_logits, engine_logits, atol=ATOL_HF, rtol=RTOL_HF)

    reference_tokens = _generate_reference(reference, prompt)
    hf_tokens = _generate_hf(hf_model, prompt)
    engine_tokens = _generate_engine(engine, prompt)
    if not reference_tokens == hf_tokens == engine_tokens:
        raise AssertionError("64-token greedy outputs differ")

    native_vllm = None
    if args.validate_native_vllm:
        hf_logprobs = torch.log_softmax(hf_logits[0, -1].float(), dim=-1).cpu()
        native_vllm = _validate_native_vllm(
            args.model_dir,
            prompt_ids,
            hf_logprobs,
            reference_tokens,
            args.kv_cache_memory_bytes,
        )

    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(args.gpu_index)
            if device.type == "cuda"
            else None
        ),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": TOKENS,
        "checkpoint_values_exact": True,
        "hf_vs_original_logits": {
            **_max_errors(hf_logits, reference_logits),
            "atol": reference_atol,
            "rtol": reference_rtol,
            "passed": True,
        },
        "engine_vs_original_logits": {
            **_max_errors(engine_logits, reference_logits),
            "atol": (ATOL_CUDA_REFERENCE if device.type == "cuda" else ATOL_ENGINE),
            "rtol": (RTOL_CUDA_REFERENCE if device.type == "cuda" else RTOL_ENGINE),
            "passed": True,
        },
        "hf_vs_engine_logits": {
            **_max_errors(hf_logits, engine_logits),
            "atol": ATOL_HF,
            "rtol": RTOL_HF,
            "passed": True,
        },
        "cuda_reference_tolerance_basis": (
            "The vendored nn.MultiheadAttention CUDA path and the shared HF/engine "
            "SDPA path use different FP32 reduction orders; measured preflight "
            "max absolute error was 3.44038e-4 with identical argmaxes."
            if device.type == "cuda"
            else None
        ),
        "all_64_greedy_tokens_equal": True,
        "generated_token_ids": reference_tokens,
        "native_vllm_numerical_validation_passed": native_vllm is not None,
        "native_vllm": native_vllm,
        "native_vllm_validation_basis": (
            {
                "comparison": "full-vocabulary normalized log-probabilities",
                "reference": "Hugging Face FP32 logits after log_softmax",
                "reserved_kv_cache_bytes": args.kv_cache_memory_bytes,
                "timed": False,
            }
            if native_vllm is not None
            else None
        ),
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
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--validate-native-vllm", action="store_true")
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
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
