"""Paired, rotating-round single-request miniMoE comparison against vLLM.

Each system runs in a fresh process for every round. The process initializes and
warms its runtime before measuring exactly one request. The starting system
rotates by round so temporal GPU drift is paired across all three systems.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model
from moe_engine.vllm_runtime import (
    DEFAULT_KV_CACHE_MEMORY_BYTES,
    MAX_MODEL_LEN,
    MINIMUM_KV_CACHE_MEMORY_BYTES,
    STARTUP_MEMORY_GUARD_UTILIZATION,
    async_engine_kwargs,
    validate_kv_cache_memory_bytes,
)

PROMPT_TOKENS = 62
GENERATED_TOKENS = 64
DEFAULT_ROUNDS = 21
SYSTEMS = ("engine", "vllm", "vllm-eager")
COMPARISONS = (
    ("engine", "vllm"),
    ("engine", "vllm-eager"),
    ("vllm-eager", "vllm"),
)
PAIRED_FIELDS = {
    "time_to_first_token_ms": ("ms", "lower_is_better"),
    "mean_inter_token_latency_ms": ("ms", "lower_is_better"),
    "total_generation_time_ms": ("ms", "lower_is_better"),
    "aggregate_tokens_per_second": ("tokens_per_second", "higher_is_better"),
}


@dataclass
class RunMetrics:
    time_to_first_token_ms: float
    inter_token_latency_ms: list[float]
    mean_inter_token_latency_ms: float
    total_generation_time_ms: float
    aggregate_tokens_per_second: float
    peak_gpu_memory_mb: float
    generated_token_ids: list[int]
    stream_chunk_sizes: list[int]
    coalesced_stream_event_count: int


class GpuMemoryMonitor:
    """Sample whole-device usage, including vLLM worker subprocesses."""

    def __init__(self, gpu_index: int, interval_s: float = 0.002) -> None:
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError(
                "pynvml is required for GPU memory measurement"
            ) from error
        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def used_bytes(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def start(self) -> None:
        self.peak_bytes = self.used_bytes()
        self._stop.clear()

        def sample() -> None:
            while not self._stop.wait(self._interval_s):
                self.peak_bytes = max(self.peak_bytes, self.used_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.used_bytes())
        return self.peak_bytes


def rotated_system_order(round_index: int) -> tuple[str, ...]:
    shift = round_index % len(SYSTEMS)
    return SYSTEMS[shift:] + SYSTEMS[:shift]


def paired_t_summary(deltas: list[float], unit: str) -> dict:
    if not deltas:
        raise ValueError("paired confidence interval requires at least one delta")
    mean = statistics.mean(deltas)
    if len(deltas) < 2:
        interval = None
    else:
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
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }.get(len(deltas) - 1, 1.96)
        half_width = critical * statistics.stdev(deltas) / len(deltas) ** 0.5
        interval = [mean - half_width, mean + half_width]
    return {
        "unit": unit,
        "sample_count": len(deltas),
        "per_round_delta": deltas,
        "mean_delta": mean,
        "ci95": interval,
        "method": "paired two-sided Student-t interval",
    }


def _load_prompt_ids(model_dir: Path) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if len(prompt_ids) != PROMPT_TOKENS:
        raise RuntimeError(
            f"canonical prompt encoded to {len(prompt_ids)} tokens, expected {PROMPT_TOKENS}"
        )
    if tokenizer.model_max_length != MAX_MODEL_LEN:
        raise RuntimeError(
            f"tokenizer context limit is {tokenizer.model_max_length}, expected {MAX_MODEL_LEN}"
        )
    return prompt_ids


def _metrics(
    start: float,
    token_times: list[float],
    token_ids: list[int],
    peak_bytes: int,
    stream_chunk_sizes: list[int] | None = None,
) -> RunMetrics:
    if len(token_ids) != GENERATED_TOKENS or len(token_times) != GENERATED_TOKENS:
        raise RuntimeError(
            f"generation returned {len(token_ids)} tokens, expected {GENERATED_TOKENS}"
        )
    if stream_chunk_sizes is None:
        stream_chunk_sizes = [1] * len(token_ids)
    if not stream_chunk_sizes or sum(stream_chunk_sizes) != len(token_ids):
        raise RuntimeError("stream chunk sizes do not cover every generated token")
    intervals = [
        (current - previous) * 1000
        for previous, current in zip(token_times, token_times[1:])
    ]
    total_s = token_times[-1] - start
    return RunMetrics(
        time_to_first_token_ms=(token_times[0] - start) * 1000,
        inter_token_latency_ms=intervals,
        mean_inter_token_latency_ms=statistics.fmean(intervals),
        total_generation_time_ms=total_s * 1000,
        aggregate_tokens_per_second=GENERATED_TOKENS / total_s,
        peak_gpu_memory_mb=peak_bytes / 2**20,
        generated_token_ids=token_ids,
        stream_chunk_sizes=stream_chunk_sizes,
        coalesced_stream_event_count=sum(size > 1 for size in stream_chunk_sizes),
    )


def _engine_generate(
    model, prompt_ids: list[int]
) -> tuple[float, list[float], list[int]]:
    step_input = torch.tensor(prompt_ids, device="cuda", dtype=torch.long)[None, :]
    cache = model.new_cache(batch_size=1)
    token_times: list[float] = []
    token_ids: list[int] = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(GENERATED_TOKENS):
            logits = model(step_input, cache)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            token_times.append(time.perf_counter())
            token_ids.append(int(next_token.item()))
            step_input = next_token
    return start, token_times, token_ids


def benchmark_engine(
    checkpoint: Path,
    prompt_ids: list[int],
    monitor: GpuMemoryMonitor,
) -> tuple[RunMetrics, dict]:
    model, config, metadata = load_engine_model(checkpoint, "cuda")
    model.float().set_moe_mode("auto")
    if config.max_seq_length != MAX_MODEL_LEN:
        raise RuntimeError("engine checkpoint context limit is not 1,024")

    # Full warmup covers prefill, decode, and lazy Triton compilation.
    _engine_generate(model, prompt_ids)
    monitor.start()
    start, token_times, token_ids = _engine_generate(model, prompt_ids)
    metrics = _metrics(start, token_times, token_ids, monitor.stop())
    return metrics, {
        "checkpoint_step": metadata["step"],
        "moe_mode": "auto",
        "reserved_kv_cache_bytes": MINIMUM_KV_CACHE_MEMORY_BYTES,
    }


async def _vllm_generate(
    engine, prompt_ids: list[int]
) -> tuple[float, list[float], list[int], list[int]]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=GENERATED_TOKENS,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    token_times: list[float] = []
    token_ids: list[int] = []
    stream_chunk_sizes: list[int] = []
    start = time.perf_counter()
    async for output in engine.generate(
        {"prompt_token_ids": prompt_ids}, sampling, request_id=uuid.uuid4().hex
    ):
        now = time.perf_counter()
        delta_ids = list(output.outputs[0].token_ids)
        if not delta_ids:
            continue
        token_ids.extend(delta_ids)
        # DELTA outputs may be merged when the producer gets ahead. Tokens in
        # one chunk become visible together, so they share a delivery timestamp
        # and therefore have zero user-observed interval within that chunk.
        token_times.extend([now] * len(delta_ids))
        stream_chunk_sizes.append(len(delta_ids))
    return start, token_times, token_ids, stream_chunk_sizes


async def benchmark_vllm(
    model_dir: Path,
    prompt_ids: list[int],
    monitor: GpuMemoryMonitor,
    enforce_eager: bool,
    kv_cache_memory_bytes: int,
) -> tuple[RunMetrics, dict]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        **async_engine_kwargs(
            model_dir,
            enforce_eager=enforce_eager,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        # Full warmup covers prefill, decode, lazy compilation, and CUDA graphs.
        await _vllm_generate(engine, prompt_ids)
        monitor.start()
        start, token_times, token_ids, stream_chunk_sizes = await _vllm_generate(
            engine, prompt_ids
        )
        metrics = _metrics(
            start,
            token_times,
            token_ids,
            monitor.stop(),
            stream_chunk_sizes,
        )
        return metrics, {
            "enforce_eager": enforce_eager,
            "model_impl": "vllm",
            "uses_custom_engine_triton_moe": False,
            "reserved_kv_cache_bytes": kv_cache_memory_bytes,
        }
    finally:
        engine.shutdown()


def _run_isolated(args: argparse.Namespace) -> dict:
    if args.device != "cuda":
        raise RuntimeError("this benchmark requires the explicit flag --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    validate_kv_cache_memory_bytes(args.kv_cache_memory_bytes)
    torch.cuda.set_device(args.gpu_index)
    prompt_ids = _load_prompt_ids(args.model_dir)
    monitor = GpuMemoryMonitor(args.gpu_index)
    baseline_bytes = monitor.used_bytes()

    if args.system == "engine":
        metrics, extra = benchmark_engine(args.checkpoint, prompt_ids, monitor)
    elif args.system in {"vllm", "vllm-eager"}:
        metrics, extra = asyncio.run(
            benchmark_vllm(
                args.model_dir,
                prompt_ids,
                monitor,
                enforce_eager=args.system == "vllm-eager",
                kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            )
        )
    else:
        raise AssertionError("all mode must run in the parent process")
    result = {
        "system": args.system,
        "device": "cuda",
        "dtype": "float32",
        "active_requests": 1,
        "batch_size": 1,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": GENERATED_TOKENS,
        "context_limit": MAX_MODEL_LEN,
        "decoding": "greedy_ignore_eos",
        "warmup_requests": 1,
        "metrics": asdict(metrics),
        "baseline_gpu_memory_mb": baseline_bytes / 2**20,
        "peak_incremental_gpu_memory_mb": max(
            0.0, metrics.peak_gpu_memory_mb - baseline_bytes / 2**20
        ),
        **extra,
    }
    return result


def _assert_round_token_equality(round_record: dict) -> list[int]:
    systems = round_record["systems"]
    missing = set(SYSTEMS) - set(systems)
    if missing:
        raise RuntimeError(f"round is missing systems: {sorted(missing)}")
    reference = systems["engine"]["metrics"]["generated_token_ids"]
    mismatches = [
        system
        for system in SYSTEMS
        if systems[system]["metrics"]["generated_token_ids"] != reference
    ]
    if mismatches:
        raise RuntimeError(
            f"greedy token mismatch in round {round_record['round_index']}: {mismatches}"
        )
    return reference


def _assert_all_token_equality(rounds: list[dict]) -> list[int]:
    if not rounds:
        raise RuntimeError("benchmark produced no rounds")
    expected = _assert_round_token_equality(rounds[0])
    for round_record in rounds[1:]:
        actual = _assert_round_token_equality(round_record)
        if actual != expected:
            raise RuntimeError(
                f"engine greedy tokens changed in round {round_record['round_index']}"
            )
    return expected


def _system_summary(rounds: list[dict], system: str) -> dict:
    metrics = [round_record["systems"][system]["metrics"] for round_record in rounds]
    return {
        "rounds": len(rounds),
        "median": {
            field: statistics.median(measurement[field] for measurement in metrics)
            for field in (*PAIRED_FIELDS, "peak_gpu_memory_mb")
        },
        "median_peak_incremental_gpu_memory_mb": statistics.median(
            round_record["systems"][system]["peak_incremental_gpu_memory_mb"]
            for round_record in rounds
        ),
        "reserved_kv_cache_bytes": rounds[0]["systems"][system][
            "reserved_kv_cache_bytes"
        ],
    }


def _paired_comparisons(rounds: list[dict]) -> dict:
    comparisons = {}
    for baseline, candidate in COMPARISONS:
        name = f"{candidate}_minus_{baseline}"
        fields = {}
        for field, (unit, preference) in PAIRED_FIELDS.items():
            deltas = [
                round_record["systems"][candidate]["metrics"][field]
                - round_record["systems"][baseline]["metrics"][field]
                for round_record in rounds
            ]
            fields[field] = {
                **paired_t_summary(deltas, unit),
                "direction": f"{candidate} minus {baseline}",
                "preference": preference,
            }
        comparisons[name] = fields
    return comparisons


def _child_command(args: argparse.Namespace, system: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--system",
        system,
        "--device",
        "cuda",
        "--checkpoint",
        str(args.checkpoint),
        "--model-dir",
        str(args.model_dir),
        "--gpu-index",
        str(args.gpu_index),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--output",
        str(output),
    ]


def _run_numerical_validation(args: argparse.Namespace, output: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "validate_hf_parity.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--model-dir",
        str(args.model_dir),
        "--device",
        "cuda",
        "--gpu-index",
        str(args.gpu_index),
        "--validate-native-vllm",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    report = json.loads(output.read_text())
    if not report.get("native_vllm_numerical_validation_passed"):
        raise RuntimeError("native vLLM numerical validation did not pass")
    return report


def _run_all(args: argparse.Namespace) -> dict:
    if not args.output:
        raise ValueError("--system all requires --output")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output = output.with_name(f"{output.stem}-validation{output.suffix}")
    validation = _run_numerical_validation(args, validation_output)

    rounds = []
    for round_index in range(args.rounds):
        order = rotated_system_order(round_index)
        round_record = {
            "round_index": round_index,
            "execution_order": list(order),
            "systems": {},
        }
        for system in order:
            system_output = output.with_name(
                f"{output.stem}-round-{round_index:02d}-{system}{output.suffix}"
            )
            subprocess.run(_child_command(args, system, system_output), check=True)
            round_record["systems"][system] = json.loads(system_output.read_text())
        _assert_round_token_equality(round_record)
        rounds.append(round_record)

    generated_tokens = _assert_all_token_equality(rounds)
    return {
        "workload": {
            "prompt": PROMPT,
            "prompt_tokens": PROMPT_TOKENS,
            "generated_tokens": GENERATED_TOKENS,
            "context_limit": MAX_MODEL_LEN,
            "dtype": "float32",
            "active_requests": 1,
            "batch_size": 1,
        },
        "methodology": {
            "rounds": args.rounds,
            "rotation_period": len(SYSTEMS),
            "fresh_process_per_system_per_round": True,
            "warmup_requests_per_measured_process": 1,
            "initialization_and_compilation_timed": False,
            "paired_ci": "two-sided Student-t 95% confidence interval",
        },
        "kv_cache": {
            "fp32_minimum_bytes_for_one_1024_token_sequence": (
                MINIMUM_KV_CACHE_MEMORY_BYTES
            ),
            "vllm_reserved_bytes": args.kv_cache_memory_bytes,
            "vllm_safety_margin_multiplier": (
                args.kv_cache_memory_bytes / MINIMUM_KV_CACHE_MEMORY_BYTES
            ),
            "kv_cache_dtype": "float32 via model dtype",
            "reservation_policy": "fixed kv_cache_memory_bytes",
            "uses_gpu_memory_utilization_reservation": False,
            "startup_memory_guard_utilization": STARTUP_MEMORY_GUARD_UTILIZATION,
            "startup_memory_guard_controls_reservation": False,
        },
        "numerical_validation": validation,
        "all_64_generated_tokens_equal": True,
        "generated_token_ids": generated_tokens,
        "system_summary": {
            system: _system_summary(rounds, system) for system in SYSTEMS
        },
        "paired_comparisons": _paired_comparisons(rounds),
        "raw_rounds": rounds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=[*SYSTEMS, "all"])
    parser.add_argument("--device", required=True, choices=["cuda"])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/minimoe_sft.pt")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("checkpoints/minimoe-hf")
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least two")
    try:
        validate_kv_cache_memory_bytes(args.kv_cache_memory_bytes)
    except ValueError as error:
        parser.error(str(error))
    result = _run_all(args) if args.system == "all" else _run_isolated(args)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
