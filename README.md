# moe-inference-engine

A from-scratch inference engine for [miniMoE](https://github.com/maokner/miniMoE), a 280M sparse Mixture-of-Experts model.
The roadmap focuses on paged KV caching, custom Triton/CUDA kernels, and minimum single-request latency.

## The big picture

The main focus is the MoE forward path: gating, top-2 routing, and grouped expert GEMMs without padded per-expert batches.
The engine will be compared with HF `generate`, llama.cpp, and vLLM using single-request decode throughput, time-to-first-token, and latency curves.

## Components

Planned build order:

### 1. Baseline server
Plain PyTorch + HF-style generate loop serving miniMoE over HTTP.
This provides the reference baseline.

### 2. Paged KV cache
Fixed-size KV blocks with a block table per sequence, so long and short requests share GPU memory without fragmentation.
Modeled on the PagedAttention design from vLLM.

### 3. Direct paged attention
Read K/V values from non-contiguous cache blocks through the block table without rebuilding a contiguous K/V tensor at every layer and decode step.
The acceptance target is to match or beat the contiguous KV cache on the same single-request benchmark while preserving paged storage.

### 4. Fused MoE kernel
Triton first, CUDA if needed: fuse top-2 dispatch and grouped expert GEMMs into a low-latency path.
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
- Metrics: single-request tokens/sec, time-to-first-token, and per-token latency across context lengths.
- One rented 4090 or A100; exact GPU, driver, and library versions reported.
- The benchmark harness is published so every number regenerates with one command.
- Document remaining performance gaps against vLLM.

## Why this project

miniMoE provides a known model, tokenizer, checkpoint, and evaluation harness for testing inference work from model math through memory management.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and the miniMoE checkpoint (symlinked or copied into `checkpoints/`).

```bash
uv venv -p 3.12 && uv pip install -e ".[server,dev]"

# smoke test
uv run python scripts/smoke_generate.py

# parity checks
uv run pytest tests/
uv run python scripts/check_parity.py

# benchmarks
uv run python benchmarks/bench.py --system reference
uv run python benchmarks/bench.py --system engine
uv run python benchmarks/bench.py --system engine-paged

# serve it over HTTP
uv run python scripts/serve.py
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

## Status

Milestones 1 and 2 are complete.
The engine has a custom forward pass, a contiguous KV cache used by `generate()`, and a paged KV cache backed by a shared block pool.
Both cache implementations expose the same `write` and `read` interface to `CachedAttention`.
The HTTP server creates a `PagedKVCache` for each request and returns its blocks afterward.
Requests are serialized because the current pool holds one maximum-length sequence.
The server rejects over-length prompts and stops generation when the context window fills.
`pytest` checks both cache implementations on tiny configurations, including two sequences sharing one block pool.
On the real checkpoint, `scripts/check_parity.py` reports a maximum logit difference of 1.3e-4 and identical greedy generations ([results/parity_cpu.json](results/parity_cpu.json)).

Numbers on MacBook (MPS), 62-token prompt, 128 generated.
All three rows come from one back-to-back session at the same revision because MPS throughput varies between sessions.
Each row uses per-position median latencies across three runs.
Prefill times cover only the forward pass; cache allocation happens outside the timed region.
Raw step times are stored in [results/](results/).

| system | decode tok/s | first half → second half | median / p90 latency | prefill |
|---|---|---|---|---|
| reference (no KV cache) | 2.5 | 2.8 → 2.2 (decays) | 258ms / 431ms | 193ms |
| engine (KV cache) | **18.3** | 18.5 → 18.1 (flat) | 57ms / 63ms | 165ms |
| engine (paged KV cache) | 14.1 | 14.6 → 13.6 (flat) | 73ms / 82ms | 177ms |

Fresh single-request measurements on a Thunder Compute NVIDIA RTX A6000 use the same 62-token prompt and 128 greedy tokens, with each result taking per-position medians across three synchronized runs.
The contiguous and paged paths were repeated in reverse order because this virtualized GPU has noticeable run-to-run variance.
Raw step times are stored in [results/paged_only_cuda/](results/paged_only_cuda/).

| system | decode tok/s, first pass | decode tok/s, reverse-order pass | prefill |
|---|---:|---:|---:|
| reference (no KV cache) | 13.0 | - | 80ms |
| engine (contiguous KV cache) | **16.9** | **17.0** | 71ms / 70ms |
| engine (paged KV cache) | 15.3 | 16.4 | 102ms / 113ms |

The reference decays because every step recomputes the whole sequence.
The cache reduces each decode step to one new transformer position; attention over cached history remains linear in context length, but at these context sizes the expert MLPs dominate, so the curve measures flat.
The paged cache is about 23% slower than the contiguous cache on MPS and 4-10% slower across the two A6000 decode passes because `read()` gathers blocks into a contiguous tensor at every layer and step.
The shared pool reserves 36 MB of K/V storage for 1024 tokens.
The 190-token benchmark workload uses 12 of 64 blocks, or about 6.8 MB, while a contiguous cache reserves the full 36 MB for each sequence.
The serving path intentionally serializes generation requests; multi-request scheduling is outside the current scope.
Next: build a direct paged-attention kernel that matches or beats the contiguous cache's roughly 17 tok/s A6000 result, then rebuild the fused MoE kernel for additional single-request speed.
See [moe-inference-engine.md](moe-inference-engine.md) for the full project plan, risks, and timeline.
