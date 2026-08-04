"""Isolated single-request miniMoE comparison against vLLM.

Every measured process uses the local converted tokenizer, a 62-token prompt,
64 greedy tokens, FP32, one active request, and a 1,024-token context limit.
The CLI requires an explicit CUDA device flag so it cannot select MPS or another
accelerator implicitly.
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

PROMPT_TOKENS = 62
GENERATED_TOKENS = 64
MAX_MODEL_LEN = 1024


@dataclass
class RunMetrics:
    time_to_first_token_ms: float
    inter_token_latency_ms: list[float]
    mean_inter_token_latency_ms: float
    total_generation_time_ms: float
    aggregate_tokens_per_second: float
    peak_gpu_memory_mb: float
    generated_token_ids: list[int]


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
) -> RunMetrics:
    if len(token_ids) != GENERATED_TOKENS or len(token_times) != GENERATED_TOKENS:
        raise RuntimeError(
            f"generation returned {len(token_ids)} tokens, expected {GENERATED_TOKENS}"
        )
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
    repeats: int,
    monitor: GpuMemoryMonitor,
) -> tuple[list[RunMetrics], dict]:
    model, config, metadata = load_engine_model(checkpoint, "cuda")
    model.float().set_moe_mode("auto")
    if config.max_seq_length != MAX_MODEL_LEN:
        raise RuntimeError("engine checkpoint context limit is not 1,024")
    _engine_generate(model, prompt_ids)
    runs = []
    for _ in range(repeats):
        monitor.start()
        start, token_times, token_ids = _engine_generate(model, prompt_ids)
        runs.append(_metrics(start, token_times, token_ids, monitor.stop()))
    return runs, {"checkpoint_step": metadata["step"], "moe_mode": "auto"}


async def _vllm_generate(
    engine, prompt_ids: list[int]
) -> tuple[float, list[float], list[int]]:
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
    start = time.perf_counter()
    async for output in engine.generate(
        {"prompt_token_ids": prompt_ids}, sampling, request_id=uuid.uuid4().hex
    ):
        now = time.perf_counter()
        delta_ids = list(output.outputs[0].token_ids)
        if len(delta_ids) != 1:
            raise RuntimeError(
                "vLLM did not stream exactly one token per decode iteration; "
                "per-token latency would be ambiguous"
            )
        token_ids.extend(delta_ids)
        token_times.append(now)
    return start, token_times, token_ids


async def benchmark_vllm(
    model_dir: Path,
    prompt_ids: list[int],
    repeats: int,
    monitor: GpuMemoryMonitor,
    enforce_eager: bool,
    gpu_memory_utilization: float,
) -> tuple[list[RunMetrics], dict]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tokenizer_mode="auto",
        trust_remote_code=True,
        model_impl="vllm",
        dtype="float32",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        await _vllm_generate(engine, prompt_ids)
        runs = []
        for _ in range(repeats):
            monitor.start()
            start, token_times, token_ids = await _vllm_generate(engine, prompt_ids)
            runs.append(_metrics(start, token_times, token_ids, monitor.stop()))
        return runs, {
            "enforce_eager": enforce_eager,
            "model_impl": "vllm",
            "uses_custom_engine_triton_moe": False,
        }
    finally:
        engine.shutdown()


def _summarize(system: str, runs: list[RunMetrics], extra: dict) -> dict:
    first_tokens = runs[0].generated_token_ids
    if any(run.generated_token_ids != first_tokens for run in runs[1:]):
        raise RuntimeError(f"{system} generated different tokens across repeats")

    def median(field: str) -> float:
        return statistics.median(getattr(run, field) for run in runs)

    return {
        "system": system,
        "device": "cuda",
        "dtype": "float32",
        "active_requests": 1,
        "batch_size": 1,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": GENERATED_TOKENS,
        "context_limit": MAX_MODEL_LEN,
        "decoding": "greedy_ignore_eos",
        "repeats": len(runs),
        "median": {
            "time_to_first_token_ms": median("time_to_first_token_ms"),
            "mean_inter_token_latency_ms": median("mean_inter_token_latency_ms"),
            "total_generation_time_ms": median("total_generation_time_ms"),
            "aggregate_tokens_per_second": median("aggregate_tokens_per_second"),
            "peak_gpu_memory_mb": median("peak_gpu_memory_mb"),
        },
        "generated_token_ids": first_tokens,
        "runs": [asdict(run) for run in runs],
        **extra,
    }


def _run_isolated(args: argparse.Namespace) -> dict:
    if args.device != "cuda":
        raise RuntimeError("this benchmark requires the explicit flag --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    torch.cuda.set_device(args.gpu_index)
    prompt_ids = _load_prompt_ids(args.model_dir)
    monitor = GpuMemoryMonitor(args.gpu_index)
    baseline_bytes = monitor.used_bytes()

    if args.system == "engine":
        runs, extra = benchmark_engine(
            args.checkpoint, prompt_ids, args.repeats, monitor
        )
    elif args.system in {"vllm", "vllm-eager"}:
        runs, extra = asyncio.run(
            benchmark_vllm(
                args.model_dir,
                prompt_ids,
                args.repeats,
                monitor,
                enforce_eager=args.system == "vllm-eager",
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
    else:
        raise AssertionError("all mode must run in the parent process")
    result = _summarize(args.system, runs, extra)
    result["baseline_gpu_memory_mb"] = baseline_bytes / 2**20
    result["peak_incremental_gpu_memory_mb"] = max(
        run.peak_gpu_memory_mb - result["baseline_gpu_memory_mb"] for run in runs
    )
    return result


def _run_all(args: argparse.Namespace) -> dict:
    if not args.output:
        raise ValueError("--system all requires --output")
    output = Path(args.output).resolve()
    per_system = []
    for system in ("engine", "vllm", "vllm-eager"):
        system_output = output.with_name(f"{output.stem}-{system}{output.suffix}")
        command = [
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
            "--repeats",
            str(args.repeats),
            "--gpu-index",
            str(args.gpu_index),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output",
            str(system_output),
        ]
        subprocess.run(command, check=True)
        per_system.append(json.loads(system_output.read_text()))
    reference_tokens = per_system[0]["generated_token_ids"]
    equality = {
        result["system"]: result["generated_token_ids"] == reference_tokens
        for result in per_system
    }
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
        "generated_token_equality_to_engine": equality,
        "all_generated_tokens_equal": all(equality.values()),
        "systems": per_system,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", required=True, choices=["engine", "vllm", "vllm-eager", "all"]
    )
    parser.add_argument("--device", required=True, choices=["cuda"])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/minimoe_sft.pt")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("checkpoints/minimoe-hf")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least one")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between zero and one")
    result = _run_all(args) if args.system == "all" else _run_isolated(args)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
