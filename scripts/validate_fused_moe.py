"""Produce an A6000 correctness and storage report for direct MoE decode.

This is an explicit CUDA validation entry point, not a portable smoke test.
It compares the three Triton kernels with the PyTorch oracle, exercises every
expert in both routing slots, validates checkpoint/storage lifecycle behavior,
and checks one-token logits plus 64-token greedy generation on the real model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import tiktoken
import torch
from torch.nn import functional as F

from moe_engine import fused_moe
from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model
from moe_engine.model import Model, ModelConfig, MoEFeedForward


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual - expected).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
    }


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def storage_layout(moe: MoEFeedForward) -> dict:
    groups = {}
    all_valid = True
    for storage_name, layer_index, parameter_name in (
        ("w1", 0, "weight"),
        ("b1", 0, "bias"),
        ("w2", 2, "weight"),
        ("b2", 2, "bias"),
    ):
        canonical = moe._expert_storage[storage_name]
        parameters = [
            getattr(expert.net[layer_index], parameter_name) for expert in moe.experts
        ]
        stride_bytes = parameters[0].numel() * parameters[0].element_size()
        expected_pointers = [
            parameters[0].data_ptr() + expert_id * stride_bytes
            for expert_id in range(moe.num_experts)
        ]
        actual_pointers = [parameter.data_ptr() for parameter in parameters]
        parameter_storage_pointers = {
            parameter.untyped_storage().data_ptr() for parameter in parameters
        }
        valid = (
            canonical.is_contiguous()
            and all(parameter.is_contiguous() for parameter in parameters)
            and canonical.data_ptr() == parameters[0].data_ptr()
            and actual_pointers == expected_pointers
            and parameter_storage_pointers == {canonical.untyped_storage().data_ptr()}
        )
        all_valid = all_valid and valid
        groups[storage_name] = {
            "shape": list(canonical.shape),
            "device": str(canonical.device),
            "dtype": str(canonical.dtype),
            "contiguous": canonical.is_contiguous(),
            "one_shared_storage": len(parameter_storage_pointers) == 1,
            "canonical_base_matches_expert_zero": (
                canonical.data_ptr() == parameters[0].data_ptr()
            ),
            "expert_pointer_stride_bytes": stride_bytes,
            "expert_pointer_offsets_bytes": [
                pointer - actual_pointers[0] for pointer in actual_pointers
            ],
            "layout_valid": valid,
            "storage_bytes": canonical.untyped_storage().nbytes(),
            "logical_parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            ),
        }

    unique_storages = {
        storage.untyped_storage().data_ptr(): storage.untyped_storage().nbytes()
        for storage in moe._expert_storage.values()
    }
    logical_bytes = sum(
        parameter.numel() * parameter.element_size()
        for expert in moe.experts
        for parameter in expert.parameters()
    )
    return {
        "all_groups_valid": all_valid,
        "canonical_storage_count": len(unique_storages),
        "canonical_unique_storage_bytes": sum(unique_storages.values()),
        "expert_logical_parameter_bytes": logical_bytes,
        "no_persistent_full_duplicate": sum(unique_storages.values()) == logical_bytes,
        "groups": groups,
    }


def model_storage_report(model: Model, expected_keys: set[str]) -> dict:
    layers = [storage_layout(block.moe) for block in model.MoEBlocks]
    state_keys = set(model.state_dict())
    return {
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "state_dict_keys_unchanged": state_keys == expected_keys,
        "layer_count": len(layers),
        "all_layers_valid": all(layer["all_groups_valid"] for layer in layers),
        "no_layer_has_persistent_full_duplicate": all(
            layer["no_persistent_full_duplicate"] for layer in layers
        ),
        "layers": layers,
    }


def lifecycle_report(checkpoint: dict) -> dict:
    config = ModelConfig(**checkpoint["model_config"])
    expected_keys = set(checkpoint["model"])

    normal = Model(config)
    normal.load_state_dict(checkpoint["model"])
    normal_cpu = model_storage_report(normal, expected_keys)
    normal.to("cuda")
    normal_cuda = model_storage_report(normal, expected_keys)
    del normal
    torch.cuda.empty_cache()

    assigned = Model(config)
    assigned.load_state_dict(checkpoint["model"], assign=True)
    assign_cpu = model_storage_report(assigned, expected_keys)
    assigned.to("cuda")
    assign_cuda = model_storage_report(assigned, expected_keys)
    del assigned
    torch.cuda.empty_cache()

    return {
        "ordinary_load_cpu": normal_cpu,
        "ordinary_load_then_cuda": normal_cuda,
        "assign_true_cpu": assign_cpu,
        "assign_true_then_cuda": assign_cuda,
    }


def kernel_report(model: Model) -> dict:
    moe = model.MoEBlocks[0].moe
    x = torch.linspace(-1.25, 1.25, 768, device="cuda").view(1, 1, 768)
    fused_moe.route(
        x,
        moe.router.weight,
        moe._router_logits,
        moe._expert_ids,
        moe._route_weights,
    )
    expected_logits = F.linear(x.reshape(1, 768), moe.router.weight).squeeze(0)
    expected_top2, expected_ids = torch.topk(expected_logits, 2)
    expected_weights = F.softmax(expected_top2, dim=0)
    router = {
        "logits_error": tensor_error(moe._router_logits, expected_logits),
        "top2_ids_equal": torch.equal(moe._expert_ids.long(), expected_ids),
        "direct_top2_ids": moe._expert_ids.tolist(),
        "reference_top2_ids": expected_ids.tolist(),
        "weights_error": tensor_error(moe._route_weights, expected_weights),
        "direct_weights": moe._route_weights.tolist(),
        "reference_weights": expected_weights.tolist(),
        "direct_weights_sum": moe._route_weights.sum().item(),
    }

    pointer_moe = MoEFeedForward(768, 3072, 8, 2, mode="direct").to("cuda").eval()
    pointer_x = torch.ones(1, 1, 768, device="cuda")
    route_cases = []
    with torch.no_grad():
        for expert_id, expert in enumerate(pointer_moe.experts):
            expert.net[0].weight.zero_()
            expert.net[0].weight[0, 0] = 0.2 * (expert_id + 1)
            expert.net[0].bias.copy_(
                torch.linspace(-2.0, 2.0, 3072, device="cuda") + 0.05 * expert_id
            )
            expert.net[2].weight.zero_()
            expert.net[2].weight[:, 0].copy_(
                torch.linspace(0.01, 0.02, 768, device="cuda") * (expert_id + 1)
            )
            expert.net[2].bias.copy_(
                torch.linspace(-0.2, 0.2, 768, device="cuda") + 0.1 * expert_id
            )

        for first_id in range(8):
            second_id = (first_id + 3) % 8
            logits = -2.0 - 0.1 * torch.arange(8, device="cuda")
            logits[first_id] = 1.0
            logits[second_id] = 0.25
            pointer_moe.router.weight.copy_(logits[:, None].expand(8, 768) / 768)
            expected = pointer_moe._forward_reference(pointer_x)
            actual = pointer_moe(pointer_x, decode=True)
            route_cases.append(
                {
                    "requested_first": first_id,
                    "requested_second": second_id,
                    "actual_ids": pointer_moe._expert_ids.tolist(),
                    "output_error": tensor_error(actual, expected),
                }
            )

    return {
        "router": router,
        "all_experts_both_slots_with_distinct_weights_and_biases": route_cases,
        "all_route_ids_correct": all(
            case["actual_ids"] == [case["requested_first"], case["requested_second"]]
            for case in route_cases
        ),
        "worst_selected_expert_output_max_abs": max(
            case["output_error"]["max_abs"] for case in route_cases
        ),
        "worst_selected_expert_output_mean_abs": max(
            case["output_error"]["mean_abs"] for case in route_cases
        ),
        "gelu": "torch.nn.GELU approximate='none' oracle over sign-varying W1 bias",
    }


def full_model_report(model: Model) -> dict:
    encoder = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(encoder.encode(PROMPT), device="cuda")
    logits = {}
    generations = {}
    with torch.no_grad():
        for mode in ("reference", "direct"):
            model.set_moe_mode(mode)
            cache = model.new_cache()
            prefill_logits = model(prompt_ids.unsqueeze(0), cache)
            step_input = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
            logits[mode] = model(step_input, cache)
            generations[mode] = model.generate(
                prompt_ids, max_new_tokens=64, temperature=0
            )
    direct_tokens = generations["direct"][:, len(prompt_ids) :]
    reference_tokens = generations["reference"][:, len(prompt_ids) :]
    return {
        "prompt_tokens": len(prompt_ids),
        "one_token_logit_error": tensor_error(logits["direct"], logits["reference"]),
        "one_token_greedy_id_equal": torch.equal(
            logits["direct"].argmax(dim=-1), logits["reference"].argmax(dim=-1)
        ),
        "greedy_tokens_compared": direct_tokens.numel(),
        "all_greedy_tokens_equal": torch.equal(direct_tokens, reference_tokens),
        "generated_token_ids": direct_tokens[0].tolist(),
    }


def acceptance_report(report: dict) -> dict:
    """Apply the same explicit float32 tolerances as the CUDA test suite."""
    kernels = report["kernels"]
    router = kernels["router"]
    full_model = report["full_model"]
    lifecycle = report["storage_lifecycle"]
    checks = {
        "router_logits": router["logits_error"]["max_abs"] <= 2e-5,
        "router_top2_ids": router["top2_ids_equal"],
        "router_weights": router["weights_error"]["max_abs"] <= 2e-6,
        "all_experts_both_slots": kernels["all_route_ids_correct"],
        "expert_output_and_exact_gelu": (
            kernels["worst_selected_expert_output_max_abs"] <= 3e-4
        ),
        "one_token_logits": full_model["one_token_logit_error"]["max_abs"] <= 1e-3,
        "one_token_greedy_id": full_model["one_token_greedy_id_equal"],
        "greedy_64_tokens": (
            full_model["greedy_tokens_compared"] >= 64
            and full_model["all_greedy_tokens_equal"]
        ),
        "storage_lifecycle": all(
            stage["all_layers_valid"]
            and stage["state_dict_keys_unchanged"]
            and stage["no_layer_has_persistent_full_duplicate"]
            for stage in lifecycle.values()
        ),
        "real_checkpoint_cuda_storage": (
            report["real_checkpoint_cuda_storage"]["all_layers_valid"]
            and report["real_checkpoint_cuda_storage"]["state_dict_keys_unchanged"]
            and report["real_checkpoint_cuda_storage"][
                "no_layer_has_persistent_full_duplicate"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"fused MoE validation failed: {', '.join(failed)}")
    return {
        "all_passed": True,
        "checks": checks,
        "tolerances": {
            "router_logits_max_abs": 2e-5,
            "router_weights_max_abs": 2e-6,
            "selected_expert_output_max_abs": 3e-4,
            "full_model_one_token_logits_max_abs": 1e-3,
            "ids_and_greedy_tokens": "exact",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/minimoe_sft.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not fused_moe.HAS_TRITON:
        parser.error("fused MoE validation requires CUDA and Triton")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    lifecycle = lifecycle_report(checkpoint)
    model, _, metadata = load_engine_model(args.checkpoint, "cuda")
    expected_keys = set(checkpoint["model"])
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "gpu": torch.cuda.get_device_name(0),
            "driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_tracked_status": command_output(
                ["git", "status", "--short", "--untracked-files=no"]
            ).splitlines(),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256(args.checkpoint),
            "step": metadata["step"],
            "tokens_seen": metadata["tokens_seen"],
            "state_dict_key_count": len(expected_keys),
        },
        "kernels": kernel_report(model),
        "storage_lifecycle": lifecycle,
        "real_checkpoint_cuda_storage": model_storage_report(model, expected_keys),
        "full_model": full_model_report(model),
    }
    report["acceptance"] = acceptance_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
