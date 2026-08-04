"""CPU-safe tests for vLLM benchmark orchestration and validity gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from moe_engine.vllm_runtime import (
    DEFAULT_KV_CACHE_MEMORY_BYTES,
    MINIMUM_KV_CACHE_MEMORY_BYTES,
    STARTUP_MEMORY_GUARD_UTILIZATION,
    async_engine_kwargs,
    validate_kv_cache_memory_bytes,
)

_spec = importlib.util.spec_from_file_location(
    "vllm_compare",
    Path(__file__).parent.parent / "benchmarks" / "vllm_compare.py",
)
vllm_compare = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vllm_compare
_spec.loader.exec_module(vllm_compare)

_validation_spec = importlib.util.spec_from_file_location(
    "validate_hf_parity",
    Path(__file__).parent.parent / "scripts" / "validate_hf_parity.py",
)
validate_hf_parity = importlib.util.module_from_spec(_validation_spec)
sys.modules[_validation_spec.name] = validate_hf_parity
_validation_spec.loader.exec_module(validate_hf_parity)


def _system_result(system: str, tokens: list[int], value: float = 1.0) -> dict:
    return {
        "system": system,
        "reserved_kv_cache_bytes": (
            MINIMUM_KV_CACHE_MEMORY_BYTES
            if system == "engine"
            else DEFAULT_KV_CACHE_MEMORY_BYTES
        ),
        "peak_incremental_gpu_memory_mb": 10.0,
        "metrics": {
            "time_to_first_token_ms": value,
            "inter_token_latency_ms": [value] * 63,
            "mean_inter_token_latency_ms": value,
            "total_generation_time_ms": value * 64,
            "aggregate_tokens_per_second": 1000.0 / value,
            "peak_gpu_memory_mb": 100.0,
            "generated_token_ids": tokens,
        },
    }


def test_default_round_order_rotates_engine_and_eager_vllm():
    assert vllm_compare.rotated_system_order(0) == (
        "engine",
        "vllm-eager",
    )
    assert vllm_compare.rotated_system_order(1) == (
        "vllm-eager",
        "engine",
    )
    assert vllm_compare.rotated_system_order(2) == (
        "engine",
        "vllm-eager",
    )


def test_full_round_order_rotates_every_system_through_first_position():
    systems = vllm_compare.FULL_SYSTEMS
    assert vllm_compare.rotated_system_order(0, systems) == (
        "engine",
        "vllm",
        "vllm-eager",
    )
    assert vllm_compare.rotated_system_order(1, systems) == (
        "vllm",
        "vllm-eager",
        "engine",
    )
    assert vllm_compare.rotated_system_order(2, systems) == (
        "vllm-eager",
        "engine",
        "vllm",
    )


def test_paired_statistics_use_per_round_candidate_minus_baseline_deltas():
    rounds = []
    for index, engine_value in enumerate((10.0, 20.0, 30.0)):
        rounds.append(
            {
                "round_index": index,
                "systems": {
                    "engine": _system_result("engine", [1] * 64, engine_value),
                    "vllm": _system_result("vllm", [1] * 64, engine_value - 2.0),
                    "vllm-eager": _system_result(
                        "vllm-eager", [1] * 64, engine_value - 1.0
                    ),
                },
            }
        )
    paired = vllm_compare._paired_comparisons(rounds, (("engine", "vllm-eager"),))
    ttft = paired["vllm-eager_minus_engine"]["time_to_first_token_ms"]
    assert ttft["per_round_delta"] == [-1.0, -1.0, -1.0]
    assert ttft["mean_delta"] == -1.0
    assert ttft["ci95"] == [-1.0, -1.0]
    assert ttft["method"] == "paired two-sided Student-t interval"


def test_fixed_fp32_kv_cache_configuration_uses_utilization_only_as_guard():
    assert MINIMUM_KV_CACHE_MEMORY_BYTES == 37_748_736
    assert DEFAULT_KV_CACHE_MEMORY_BYTES == 75_497_472
    kwargs = async_engine_kwargs(
        "/model",
        enforce_eager=False,
        kv_cache_memory_bytes=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    assert kwargs["kv_cache_memory_bytes"] == DEFAULT_KV_CACHE_MEMORY_BYTES
    assert kwargs["dtype"] == "float32"
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["max_model_len"] == 1024
    assert kwargs["gpu_memory_utilization"] == STARTUP_MEMORY_GUARD_UTILIZATION == 0.5
    with pytest.raises(ValueError, match="too small"):
        validate_kv_cache_memory_bytes(MINIMUM_KV_CACHE_MEMORY_BYTES - 1)


def test_token_mismatch_aborts_before_summary_creation():
    record = {
        "round_index": 7,
        "systems": {
            "engine": _system_result("engine", [1] * 64),
            "vllm-eager": _system_result("vllm-eager", [1] * 63 + [2]),
        },
    }
    with pytest.raises(RuntimeError, match="greedy token mismatch in round 7"):
        vllm_compare._assert_round_token_equality(record)


def test_engine_round_warms_before_memory_monitor_and_measurement(monkeypatch):
    events = []

    class Model:
        def float(self):
            return self

        def set_moe_mode(self, mode):
            assert mode == "auto"

    class Config:
        max_seq_length = 1024

    class Monitor:
        def start(self):
            events.append("monitor-start")

        def stop(self):
            events.append("monitor-stop")
            return 100 * 2**20

    def fake_generate(model, prompt_ids):
        events.append("generate")
        return 0.0, [0.1 * (index + 1) for index in range(64)], [3] * 64

    monkeypatch.setattr(
        vllm_compare,
        "load_engine_model",
        lambda checkpoint, device: (Model(), Config(), {"step": 12}),
    )
    monkeypatch.setattr(vllm_compare, "_engine_generate", fake_generate)
    metrics, extra = vllm_compare.benchmark_engine(
        Path("checkpoint.pt"), [1] * 62, Monitor()
    )
    assert events == ["generate", "monitor-start", "generate", "monitor-stop"]
    assert metrics.generated_token_ids == [3] * 64
    assert metrics.stream_chunk_sizes == [1] * 64
    assert metrics.coalesced_stream_event_count == 0
    assert extra["reserved_kv_cache_bytes"] == MINIMUM_KV_CACHE_MEMORY_BYTES


def test_coalesced_vllm_delta_uses_one_user_visible_timestamp_per_chunk():
    token_ids = list(range(64))
    token_times = [1.1] + [1.2, 1.2] + [1.3] * 61
    metrics = vllm_compare._metrics(
        1.0,
        token_times,
        token_ids,
        100 * 2**20,
        stream_chunk_sizes=[1, 2, 61],
    )
    assert metrics.generated_token_ids == token_ids
    assert metrics.stream_chunk_sizes == [1, 2, 61]
    assert metrics.coalesced_stream_event_count == 2
    assert metrics.inter_token_latency_ms[1] == 0.0
    assert metrics.inter_token_latency_ms[-1] == 0.0
    assert metrics.mean_inter_token_latency_ms == pytest.approx(200.0 / 63)


def test_all_mode_records_rotating_raw_rounds(monkeypatch, tmp_path: Path):
    calls = []

    def fake_validation(args, output):
        return {"native_vllm_numerical_validation_passed": True}

    def fake_run(command, check):
        assert check is True
        system = command[command.index("--system") + 1]
        output = Path(command[command.index("--output") + 1])
        calls.append(system)
        output.write_text(json.dumps(_system_result(system, [9] * 64)))

    monkeypatch.setattr(vllm_compare, "_run_numerical_validation", fake_validation)
    monkeypatch.setattr(vllm_compare.subprocess, "run", fake_run)
    args = argparse.Namespace(
        output=tmp_path / "comparison.json",
        checkpoint=Path("checkpoint.pt"),
        model_dir=Path("model"),
        rounds=3,
        comparison="eager",
        gpu_index=0,
        kv_cache_memory_bytes=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    report = vllm_compare._run_all(args)
    assert calls == [
        "engine",
        "vllm-eager",
        "vllm-eager",
        "engine",
        "engine",
        "vllm-eager",
    ]
    assert [round_["execution_order"] for round_ in report["raw_rounds"]] == [
        ["engine", "vllm-eager"],
        ["vllm-eager", "engine"],
        ["engine", "vllm-eager"],
    ]
    assert report["all_64_generated_tokens_equal"] is True
    assert report["methodology"]["rounds"] == 3
    assert report["methodology"]["comparison"] == "eager"
    assert report["methodology"]["systems"] == ["engine", "vllm-eager"]


@pytest.mark.parametrize(
    ("comparison", "expected_mode"), (("eager", "eager"), ("full", "both"))
)
def test_numerical_gate_tracks_selected_comparison(
    monkeypatch, tmp_path: Path, comparison: str, expected_mode: str
):
    captured = {}

    def fake_run(command, check):
        assert check is True
        captured["command"] = command
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"native_vllm_numerical_validation_passed": True}))

    monkeypatch.setattr(vllm_compare.subprocess, "run", fake_run)
    args = argparse.Namespace(
        checkpoint=Path("checkpoint.pt"),
        model_dir=Path("model"),
        comparison=comparison,
        gpu_index=0,
        kv_cache_memory_bytes=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    report = vllm_compare._run_numerical_validation(args, tmp_path / "validation.json")
    command = captured["command"]
    assert command[command.index("--native-vllm-mode") + 1] == expected_mode
    assert report["native_vllm_numerical_validation_passed"] is True


def test_all_mode_refuses_summary_when_child_tokens_differ(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vllm_compare,
        "_run_numerical_validation",
        lambda args, output: {"native_vllm_numerical_validation_passed": True},
    )

    def fake_run(command, check):
        system = command[command.index("--system") + 1]
        output = Path(command[command.index("--output") + 1])
        tokens = [1] * 64 if system != "vllm-eager" else [1] * 63 + [2]
        output.write_text(json.dumps(_system_result(system, tokens)))

    monkeypatch.setattr(vllm_compare.subprocess, "run", fake_run)
    args = argparse.Namespace(
        output=tmp_path / "comparison.json",
        checkpoint=Path("checkpoint.pt"),
        model_dir=Path("model"),
        rounds=2,
        comparison="eager",
        gpu_index=0,
        kv_cache_memory_bytes=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    with pytest.raises(RuntimeError, match="greedy token mismatch in round 0"):
        vllm_compare._run_all(args)


def test_parser_defaults_to_21_rounds_and_fixed_cache():
    args = vllm_compare.build_parser().parse_args(
        ["--system", "all", "--device", "cuda"]
    )
    assert args.rounds == 21
    assert args.comparison == "eager"
    assert args.kv_cache_memory_bytes == DEFAULT_KV_CACHE_MEMORY_BYTES

    validator_args = validate_hf_parity.build_parser().parse_args([])
    assert validator_args.native_vllm_mode == "eager"


def test_native_validator_requests_full_vocabulary_logprobs_without_importing_vllm():
    source = (
        Path(__file__).parent.parent / "scripts" / "validate_hf_parity.py"
    ).read_text()
    assert "max_logprobs=-1" in source
    assert "logprobs=-1" in source
    assert "torch.log_softmax" in source
    assert "ATOL_VLLM_LOGPROBS" in source


def test_matching_tokens_cannot_hide_native_full_distribution_mismatch(monkeypatch):
    async def fake_oracle(*args, **kwargs):
        return torch.ones(4), [7] * 64

    monkeypatch.setattr(validate_hf_parity, "_native_vllm_oracle", fake_oracle)
    with pytest.raises(AssertionError):
        validate_hf_parity._validate_native_vllm(
            Path("model"),
            [1] * 62,
            torch.zeros(4),
            [7] * 64,
            DEFAULT_KV_CACHE_MEMORY_BYTES,
        )
