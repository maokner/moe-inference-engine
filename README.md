# moe-inference-engine

A from-scratch inference engine for [miniMoE](https://github.com/maokner/miniMoE), a 280M sparse Mixture-of-Experts model.
The roadmap includes custom Triton/CUDA kernels, paged KV caching, continuous batching, and quantization.

## The big picture

The main focus is the MoE forward path: gating, top-2 routing, and grouped expert GEMMs without padded per-expert batches.
The engine will be compared with HF `generate`, llama.cpp, and vLLM using throughput, time-to-first-token, and latency curves.

## Components

Planned build order:

### 1. Baseline server
Plain PyTorch + HF-style generate loop serving miniMoE over HTTP.
This provides the reference baseline.

### 2. Paged KV cache
Fixed-size KV blocks with a block table per sequence, so long and short requests share GPU memory without fragmentation.
Modeled on the PagedAttention design from vLLM.

### 3. Continuous batching
Iteration-level scheduling: new requests join the running batch at any decode step, finished requests leave immediately.

### 4. Fused MoE kernel
Triton first, CUDA if needed: fuse gating + top-2 dispatch + grouped expert GEMM into one path, avoiding padded per-expert batches.
Compared against the naive loop-over-experts implementation and against vLLM's `fused_moe`.

### 5. Quantization
Int8, then int4 groupwise weight quantization for the experts.
Report HellaSwag and perplexity changes with performance results.

### 6. CUDA graphs
Capture decode to reduce launch overhead at small batch sizes.

### 7. Speculative decoding (stretch)
A tiny dense draft model speculating for the MoE.
Only if milestones 1-6 land cleanly.

### 8. MLX port
Port the quantized model to MLX and publish on-device tokens/sec from an M-series MacBook.

## Benchmark methodology

- Same model, same weights, same prompts, same hardware for every system compared.
- Baselines: HF transformers `generate`, llama.cpp, and vLLM.
- Metrics: tokens/sec at batch 1, 8, 32; time-to-first-token; full throughput-latency curves, not single points.
- One rented 4090 or A100; exact GPU, driver, and library versions reported.
- The benchmark harness is published so every number regenerates with one command.
- Document remaining performance gaps against vLLM.

## Why this project

miniMoE provides a known model, tokenizer, checkpoint, and evaluation harness for testing inference work from model math through memory management.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and the miniMoE checkpoint (symlinked or copied into `checkpoints/`).

```bash
uv venv -p 3.12 && uv pip install -e ".[server]"

# smoke test
uv run python scripts/smoke_generate.py

# parity checks
uv run pytest tests/
uv run python scripts/check_parity.py

# benchmarks
uv run python benchmarks/bench.py --system reference
uv run python benchmarks/bench.py --system engine

# serve it over HTTP
uv run python scripts/serve.py
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

## Status

Milestone 1 done, milestone 2 underway: the engine now has its own forward pass (hand-written attention, inference-only MoE dispatch) with a contiguous KV cache.
Parity with the reference is reproducible from this repo: `pytest` covers the mechanism on tiny configs, and `scripts/check_parity.py` shows max logit diff 1.3e-4 and identical greedy generations on the real checkpoint ([results/parity_cpu.json](results/parity_cpu.json)).

Numbers on MacBook (MPS), 62-token prompt, 128 generated.
Reported figures use per-position median latencies across three runs.
Raw step times are stored in [results/](results/).

| system | decode tok/s | first half → second half | median / p90 latency | prefill |
|---|---|---|---|---|
| reference (no KV cache) | 2.7 | 3.3 → 2.3 (decays) | 258ms / 479ms | 139ms |
| engine (KV cache) | **21.4** | 21.2 → 21.7 (flat) | 47ms / 53ms | 118ms |

The reference decays because every step recomputes the whole sequence.
The cache reduces each decode step to one new transformer position; attention over cached history remains linear in context length, but at these context sizes the expert MLPs dominate, so the curve measures flat.
Next: paged KV blocks, then continuous batching.
See [moe-inference-engine.md](moe-inference-engine.md) for the full project plan, risks, and timeline.
