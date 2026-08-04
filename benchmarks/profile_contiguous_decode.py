"""Profile warmed one-token decode with the production contiguous KV cache.

The uninstrumented pass uses CUDA events (or synchronized wall time off CUDA).
A separate CPU-activity torch.profiler pass adds component scopes and exports a trace.
Instrumented component timings must not be treated as uninstrumented throughput.

Usage:
    uv run python benchmarks/profile_contiguous_decode.py
    uv run python benchmarks/profile_contiguous_decode.py \
        --output-dir results/contiguous_decode_profile_a6000
"""

import argparse
import gzip
import importlib.metadata
import json
import platform
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import tiktoken
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from moe_engine.benchmarking import PROMPT
from moe_engine.checkpoint import load_engine_model

COMPONENTS = (
    "qkv_attention_work",
    "moe_routing_top2",
    "expert_mlp_execution",
    "route_weight_combination",
    "fused_moe_decode",
    "layernorm_residual_output",
)
COMPONENT_LABELS = {
    "qkv_attention_work": "QKV and attention work",
    "moe_routing_top2": "MoE routing/top-2 selection",
    "expert_mlp_execution": "Expert MLP execution",
    "route_weight_combination": "Route-weight combination",
    "fused_moe_decode": "Fused MoE decode",
    "layernorm_residual_output": "LayerNorm/residual/output",
}


def sync(device: str) -> None:
    """Wait for asynchronous device work before reading a timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def profiled_attention_forward(self, x, cache, layer_idx, position):
    """Contiguous-cache attention with profiler-only subcomponent scopes."""
    with record_function("qkv_attention_work"):
        batch, seq_len, dim = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        cache.write(layer_idx, position, k, v)
        assert position == 0 or seq_len == 1, "prefill must start at position 0"
        keys, values = cache.read(layer_idx, position + seq_len)
        out = F.scaled_dot_product_attention(q, keys, values, is_causal=seq_len > 1)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out_proj(out)


def profiled_moe_forward(self, x, *, decode=False):
    """MoE oracle path split into routing, expert, and combination scopes."""
    if self.uses_direct(x, decode=decode):
        with record_function("fused_moe_decode"):
            return self._forward_direct(x)

    batch, seq_len, dim = x.shape
    flat_x = x.reshape(-1, dim)

    with (
        record_function("moe_routing_top2"),
        torch.autocast(device_type=x.device.type, enabled=False),
    ):
        router_logits = F.linear(flat_x.float(), self.router.weight.float())
        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

    with record_function("moe_expert_loop_total"):
        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            with record_function("moe_routing_top2"):
                token_ids, slot = torch.where(topk_indices == expert_id)
                if token_ids.numel() == 0:
                    continue
                weight = topk_weights[token_ids, slot].unsqueeze(-1).to(flat_x.dtype)
                expert_input = flat_x[token_ids]
            with record_function("expert_mlp_execution"):
                expert_output = expert(expert_input)
            with record_function("route_weight_combination"):
                out.index_add_(0, token_ids, weight * expert_output)
    return out.reshape(batch, seq_len, dim)


def profiled_block_forward(self, x, cache, layer_idx, position):
    with record_function("layernorm_residual_output"):
        normed = self.attn_norm(x)
    attention_out = self.attention(normed, cache, layer_idx, position)
    with record_function("layernorm_residual_output"):
        x = x + attention_out
        normed = self.moe_norm(x)
    moe_out = self.moe(normed, decode=position > 0 and normed.shape[1] == 1)
    with record_function("layernorm_residual_output"):
        return x + moe_out


def profiled_model_forward(self, token_ids, cache):
    """Model forward whose scopes partition the production decode work."""
    _, seq_len = token_ids.shape
    position = cache.position
    assert position + seq_len <= self.max_seq_length, "KV cache is full"

    with record_function("layernorm_residual_output"):
        pos = torch.arange(position, position + seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.positional_embedding(pos)
    for layer_idx, block in enumerate(self.MoEBlocks):
        x = block(x, cache, layer_idx, position)
    cache.position += seq_len
    with record_function("layernorm_residual_output"):
        return self.output_projection(self.final_norm(x))


@contextmanager
def instrument_model(model):
    """Temporarily add profiler scopes without changing the production path."""
    original_model_forward = model.forward
    original_blocks = []
    try:
        model.forward = MethodType(profiled_model_forward, model)
        for block in model.MoEBlocks:
            original_blocks.append(
                (block, block.forward, block.attention.forward, block.moe.forward)
            )
            block.forward = MethodType(profiled_block_forward, block)
            block.attention.forward = MethodType(
                profiled_attention_forward, block.attention
            )
            block.moe.forward = MethodType(profiled_moe_forward, block.moe)
        yield
    finally:
        model.forward = original_model_forward
        for block, block_forward, attention_forward, moe_forward in original_blocks:
            block.forward = block_forward
            block.attention.forward = attention_forward
            block.moe.forward = moe_forward


def prefill(model, prompt_ids):
    cache = model.new_cache()
    with torch.no_grad():
        logits = model(prompt_ids.unsqueeze(0), cache)
    return cache, logits[:, -1].argmax(dim=-1, keepdim=True)


def run_uninstrumented(model, prompt_ids, device: str, steps: int):
    """Return exact greedy tokens and per-step latency from the clean path."""
    cache, step_input = prefill(model, prompt_ids)
    tokens = []
    times_ms = []
    with torch.no_grad():
        if device == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            for start, end in zip(starts, ends):
                start.record()
                logits = model(step_input, cache)
                step_input = logits[:, -1].argmax(dim=-1, keepdim=True)
                end.record()
                tokens.append(step_input)
            sync(device)
            times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            for _ in range(steps):
                sync(device)
                start = time.perf_counter()
                logits = model(step_input, cache)
                step_input = logits[:, -1].argmax(dim=-1, keepdim=True)
                sync(device)
                times_ms.append((time.perf_counter() - start) * 1000)
                tokens.append(step_input)
    return torch.cat(tokens, dim=1).cpu(), times_ms


def run_profiled(model, prompt_ids, device: str, steps: int, trace_path: Path):
    """Run the same decode under component scopes and torch.profiler."""
    cache, step_input = prefill(model, prompt_ids)
    tokens = []
    # CPU activity captures Python/ATen dispatch, CUDA launches, and the
    # synchronous waits caused by dynamic nonzero/where routing. CUDA activity
    # is excluded because CUPTI drops buffers on some virtualized GPUs; clean
    # whole-step device latency is measured separately with CUDA events.
    activities = [ProfilerActivity.CPU]

    with (
        instrument_model(model),
        profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof,
        torch.no_grad(),
    ):
        for _ in range(steps):
            with record_function("decode_step_total"):
                logits = model(step_input, cache)
                with record_function("greedy_argmax"):
                    step_input = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(step_input)
    sync(device)
    prof.export_chrome_trace(str(trace_path))
    return torch.cat(tokens, dim=1).cpu(), prof


def profiler_report(prof, steps: int, device: str) -> dict:
    averages = {event.key: event for event in prof.key_averages()}
    total = averages["decode_step_total"]
    total_us = total.cpu_time_total
    components = []
    for name in COMPONENTS:
        event = averages.get(name)
        elapsed_us = event.cpu_time_total if event is not None else 0.0
        components.append(
            {
                "component": name,
                "instrumented_ms_per_step": elapsed_us / steps / 1000,
                "percent_of_instrumented_step": 100 * elapsed_us / total_us,
                "calls": event.count if event is not None else 0,
                "calls_per_step": event.count / steps if event is not None else 0,
            }
        )

    kernel_launch_events = [
        event for name, event in averages.items() if "LaunchKernel" in name
    ]
    scope_names = {
        *COMPONENTS,
        "decode_step_total",
        "greedy_argmax",
        "moe_expert_loop_total",
    }
    operator_events = [
        event
        for name, event in averages.items()
        if name not in scope_names and not name.startswith("ProfilerStep")
    ]
    top_cpu_operators = sorted(
        (
            {
                "operator": event.key,
                "cpu_self_ms_per_step": event.self_cpu_time_total / steps / 1000,
                "calls_per_step": event.count / steps,
            }
            for event in operator_events
            if event.self_cpu_time_total > 0
        ),
        key=lambda item: item["cpu_self_ms_per_step"],
        reverse=True,
    )[:20]
    expert_loop = averages.get("moe_expert_loop_total")
    cuda_launch_calls = sum(event.count for event in kernel_launch_events)
    return {
        "timing_basis": "CPU-scope wall time under CPU-activity torch.profiler",
        "instrumented_total_ms_per_step": total_us / steps / 1000,
        "instrumented_total_cpu_ms_per_step": total.cpu_time_total / steps / 1000,
        "components": components,
        "cuda_activity_collected": False,
        "cuda_launch_calls": (
            cuda_launch_calls if device == "cuda" and kernel_launch_events else None
        ),
        "cuda_launch_calls_per_step": (
            cuda_launch_calls / steps
            if device == "cuda" and kernel_launch_events
            else None
        ),
        "cuda_launch_cpu_ms_per_step": (
            sum(event.self_cpu_time_total for event in kernel_launch_events)
            / steps
            / 1000
            if device == "cuda" and kernel_launch_events
            else None
        ),
        "cpu_operator_self_ms_per_step": (
            sum(event.self_cpu_time_total for event in operator_events) / steps / 1000
        ),
        "cpu_operator_calls_per_step": sum(event.count for event in operator_events)
        / steps,
        "expert_loop": (
            {
                "cpu_scope_ms_per_step": expert_loop.cpu_time_total / steps / 1000,
                "unattributed_python_cpu_ms_per_step": expert_loop.self_cpu_time_total
                / steps
                / 1000,
                "calls_per_step": expert_loop.count / steps,
            }
            if expert_loop is not None
            else None
        ),
        "top_cpu_operators": top_cpu_operators,
    }


def hardware_report(device: str) -> dict:
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tiktoken": importlib.metadata.version("tiktoken"),
        "device": device,
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "gpu_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "triton": importlib.metadata.version("triton"),
                "driver": subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            }
        )
    return report


def markdown_summary(report: dict) -> str:
    clean = report["uninstrumented"]
    profiled = report["profiled"]
    components = sorted(
        profiled["components"],
        key=lambda item: item["instrumented_ms_per_step"],
        reverse=True,
    )
    dominant = components[0]
    component_map = {item["component"]: item for item in components}
    direct_active = component_map["fused_moe_decode"]["calls"] > 0
    moe_ms = sum(
        component_map[name]["instrumented_ms_per_step"]
        for name in (
            "moe_routing_top2",
            "expert_mlp_execution",
            "route_weight_combination",
            "fused_moe_decode",
        )
    )
    moe_percent = sum(
        component_map[name]["percent_of_instrumented_step"]
        for name in (
            "moe_routing_top2",
            "expert_mlp_execution",
            "route_weight_combination",
            "fused_moe_decode",
        )
    )
    expert_loop = profiled["expert_loop"]
    if expert_loop is None:
        python_loop_ms = 0.0
        python_loop_clean_percent = 0.0
        python_loop_detail = (
            "The PyTorch expert-loop scope was not active in this profile."
        )
        python_loop_conclusion = "The fused path contains no Python expert loop."
    else:
        python_loop_ms = expert_loop["unattributed_python_cpu_ms_per_step"]
        python_loop_clean_percent = 100 * python_loop_ms / clean["median_ms"]
        python_loop_detail = (
            f"The existing per-expert loop itself has {python_loop_ms:.3f} ms per step "
            "of profiler-unattributed Python CPU time "
            f"({python_loop_clean_percent:.1f}% of clean median latency); its nested "
            "tensor operations and kernel launches are reported in the component rows."
        )
        python_loop_conclusion = (
            "Pure Python loop overhead is not the dominant bottleneck."
            if python_loop_clean_percent < 50
            else "Pure Python loop overhead is the dominant bottleneck."
        )
    if direct_active:
        moe_conclusion = (
            "The combined value is the direct MoE critical-path attribution for "
            "comparison with the reference profile."
        )
        implementation_summary = [
            "The fixed-shape direct path uses three Triton launches per layer and keeps both selected expert ids and weights on device.",
            "Compare this report with an otherwise identical --moe reference run before drawing a performance conclusion.",
        ]
    elif moe_percent >= 50:
        moe_conclusion = "The combined MoE path dominates instrumented decode time, so it is the next kernel target."
        implementation_summary = [
            "The next MoE kernel should target batch-one decode directly: compute the float32 8-way router and normalized top-2 weights, run only the two selected 768x3072x768 GELU expert MLPs, and combine their weighted outputs without torch.where, per-expert Python dispatch, CPU reads of route counts, or index_add_ launches.",
            "A practical first design uses three fixed-shape Triton kernels per layer: router plus top-2 writes two device-resident ids and weights; selected-expert W1 plus GELU writes a [2, 3072] intermediate; selected-expert W2 plus weighted reduction writes one [768] output row.",
            "Keep the existing PyTorch expert loop as the correctness oracle and require logit and greedy-token parity before benchmarking.",
        ]
    else:
        moe_conclusion = (
            "The combined MoE path does not dominate instrumented decode time, so a fused MoE kernel "
            "should not precede optimization of the larger measured component."
        )
        implementation_summary = []
    nonzero = next(
        (
            item
            for item in profiled["top_cpu_operators"]
            if item["operator"] == "aten::nonzero"
        ),
        None,
    )
    nonzero_summary = (
        f"The largest individual CPU operator is aten::nonzero at {nonzero['cpu_self_ms_per_step']:.3f} ms per step across {nonzero['calls_per_step']:.1f} calls per step."
        if nonzero is not None
        else "The profile did not record an aten::nonzero operator."
    )
    launch_summary = (
        f"CUDA launches: {profiled['cuda_launch_calls_per_step']:.1f} per step; launch CPU time: {profiled['cuda_launch_cpu_ms_per_step']:.3f} ms per step."
        if profiled["cuda_launch_calls_per_step"] is not None
        else "CUDA launch events are unavailable in the CPU-activity-only trace; CPU operator dispatch time and call counts are reported instead."
    )
    lines = [
        "# One-token contiguous decode profile",
        "",
        f"Hardware: {report['hardware'].get('gpu', report['hardware']['device'])}.",
        f"Checkpoint step: {report['checkpoint_step']}.",
        f"Workload: {report['prompt_tokens']}-token benchmark prompt followed by {report['steps']} one-token greedy decode steps.",
        f"MoE mode: {report.get('moe', 'reference')}.",
        f"Clean median decode latency: {clean['median_ms']:.3f} ms ({clean['tokens_per_second']:.2f} tok/s).",
        f"Instrumented total: {profiled['instrumented_total_ms_per_step']:.3f} ms per step using {profiled['timing_basis']}.",
        "The component timings below come from the instrumented profiler run and are not uninstrumented throughput measurements.",
        "",
        "| component | instrumented ms/step | share | calls/step |",
        "|---|---:|---:|---:|",
    ]
    for item in components:
        lines.append(
            f"| {COMPONENT_LABELS[item['component']]} | {item['instrumented_ms_per_step']:.3f} | "
            f"{item['percent_of_instrumented_step']:.1f}% | {item['calls_per_step']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Combined MoE work: {moe_ms:.3f} ms per step ({moe_percent:.1f}% of instrumented decode time).",
            moe_conclusion,
            f"Largest component scope: {COMPONENT_LABELS[dominant['component']]} at {dominant['instrumented_ms_per_step']:.3f} ms per step ({dominant['percent_of_instrumented_step']:.1f}% of instrumented decode time).",
            python_loop_detail,
            python_loop_conclusion,
            nonzero_summary,
            f"CPU operator self time: {profiled['cpu_operator_self_ms_per_step']:.3f} ms per step across {profiled['cpu_operator_calls_per_step']:.1f} calls per step.",
            launch_summary,
            *implementation_summary,
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--moe",
        choices=["auto", "direct", "reference"],
        default="auto",
        help="fixed-shape Triton decode, PyTorch oracle, or automatic fallback",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/contiguous_decode_profile")
    )
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.steps < 1:
        parser.error("--warmup-steps and --steps must be positive")
    if 62 + max(args.warmup_steps, args.steps) > 1024:
        parser.error(
            "the prompt plus decode steps must fit the production context window"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")

    model, _, metadata = load_engine_model(args.checkpoint, args.device)
    model.set_moe_mode(args.moe)
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(PROMPT), device=args.device)
    if len(prompt_ids) != 62:
        raise RuntimeError(
            f"benchmark prompt changed: expected 62 tokens, found {len(prompt_ids)}"
        )

    # Warm up model kernels, allocator paths, and caches before either measurement.
    run_uninstrumented(model, prompt_ids, args.device, args.warmup_steps)
    sync(args.device)

    clean_tokens, clean_times = run_uninstrumented(
        model, prompt_ids, args.device, args.steps
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "trace.json"
    profiled_tokens, prof = run_profiled(
        model, prompt_ids, args.device, args.steps, trace_path
    )
    if not torch.equal(clean_tokens, profiled_tokens):
        raise RuntimeError("profiling instrumentation changed greedy decode outputs")
    trace_data = json.loads(trace_path.read_text())
    trace_event_count = len(trace_data.get("traceEvents", []))
    if trace_event_count == 0:
        raise RuntimeError("profiler trace contains no events")
    compressed_trace_path = trace_path.with_suffix(".json.gz")
    with (
        trace_path.open("rb") as source,
        gzip.open(compressed_trace_path, "wb", compresslevel=9) as destination,
    ):
        shutil.copyfileobj(source, destination)
    trace_path.unlink()

    clean_total_s = sum(clean_times) / 1000
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": metadata["step"],
        "hardware": hardware_report(args.device),
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "cache": "contiguous",
        "moe": args.moe,
        "decode": "one token per forward, greedy argmax",
        "outputs_match": True,
        "generated_token_ids": clean_tokens[0].tolist(),
        "uninstrumented": {
            "measurement": "CUDA events"
            if args.device == "cuda"
            else "synchronized wall time",
            "step_times_ms": clean_times,
            "median_ms": statistics.median(clean_times),
            "mean_ms": statistics.mean(clean_times),
            "p90_ms": percentile(clean_times, 0.9),
            "tokens_per_second": len(clean_times) / clean_total_s,
        },
        "profiled": profiler_report(prof, args.steps, args.device),
        "artifacts": {
            "trace": compressed_trace_path.name,
            "trace_event_count": trace_event_count,
        },
        "limitations": [
            "torch.profiler and record_function add CPU overhead, so component timings are reported separately from clean latency.",
            "The component profile collects CPU activity only because CUPTI can drop CUDA-activity buffers on virtualized GPUs; clean whole-step GPU latency still uses CUDA events.",
            "CPU scope time includes dispatch and synchronous GPU waits; asynchronous GPU work may be charged to a later synchronization scope, so values are critical-path attribution rather than isolated kernel durations.",
            "The repeated steps grow context from 62 tokens; results describe warmed early decode, not every context length.",
        ],
    }
    report_path = args.output_dir / "profile.json"
    summary_path = args.output_dir / "summary.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_path.write_text(markdown_summary(report))
    print(markdown_summary(report))
    print(f"Raw profile: {report_path}")
    print(f"Trace: {compressed_trace_path}")


if __name__ == "__main__":
    main()
