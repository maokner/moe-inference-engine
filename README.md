# moe-inference-engine

A from-scratch LLM inference engine that serves [miniMoE](https://github.com/maokner/miniMoE) (a 280M sparse Mixture-of-Experts model trained from scratch) fast.
Custom Triton/CUDA kernels, paged KV cache, continuous batching, and quantization - benchmarked honestly against vLLM and llama.cpp.

## The big picture

Plenty of people have written mini-vLLM clones for dense models; that alone is table stakes.
The unclaimed part at portfolio scale is the **MoE forward path**: gating, top-2 routing, and grouped expert GEMMs without materializing padded per-expert batches.
This is a known performance-critical kernel in vLLM and SGLang (`fused_moe`), and it is genuinely hard to get right.

That fused MoE dispatch kernel is the headline of this project.
Everything else - the paged KV cache, the continuous batching scheduler, quantization, CUDA graphs - is the credible engine built around it.

The deliverable is numbers, not vibes: tokens/sec, time-to-first-token, and full throughput-latency curves against named baselines (HF `generate`, llama.cpp, vLLM) on named hardware.

## Components

Built in this order, so the engine is already useful before the hardest piece lands:

### 1. Baseline server
Plain PyTorch + HF-style generate loop serving miniMoE over HTTP.
This is the floor every later speedup is measured against.
Debugs fine on a MacBook with MPS - no GPU rental needed yet.

### 2. Paged KV cache
Fixed-size KV blocks with a block table per sequence, so long and short requests share GPU memory without fragmentation.
Modeled on the PagedAttention design from vLLM.

### 3. Continuous batching
Iteration-level scheduling: new requests join the running batch at any decode step, finished requests leave immediately.
This is where most of the throughput lives.

### 4. Fused MoE kernel (centerpiece)
Triton first, CUDA if needed: fuse gating + top-2 dispatch + grouped expert GEMM into one path, avoiding padded per-expert batches.
Compared against the naive loop-over-experts implementation and against vLLM's `fused_moe`.

### 5. Quantization
Int8, then int4 groupwise weight quantization for the experts.
Quality cost (HellaSwag delta, perplexity delta) is reported next to the speed gain, not instead of it.

### 6. CUDA graphs
Capture the decode step to kill kernel-launch overhead at small batch sizes.

### 7. Speculative decoding (stretch)
A tiny dense draft model speculating for the MoE.
Only if milestones 1-6 land cleanly.

### 8. MLX port (the Apple angle)
Port the quantized model to MLX and publish on-device tokens/sec from an M-series MacBook.

## Benchmark methodology

Honest measurement is half the credibility of this project:

- Same model, same weights, same prompts, same hardware for every system compared.
- Baselines: HF transformers `generate` (floor), llama.cpp, vLLM (ceiling).
- Metrics: tokens/sec at batch 1, 8, 32; time-to-first-token; full throughput-latency curves, not single points.
- One rented 4090 or A100; exact GPU, driver, and library versions reported.
- The benchmark harness is published so every number regenerates with one command.
- An explicit "where vLLM still beats this engine and why" section.

## Why this project

miniMoE means the model, tokenizer, and eval harness already exist, and I understand every layer of what the engine has to compute.
"Trained a 280M MoE from scratch, then wrote the engine that serves it" is a full-stack story an interviewer can drill into at any level - from routing math down to memory hierarchy.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and the miniMoE checkpoint (symlinked or copied into `checkpoints/`).

```bash
uv venv -p 3.12 && uv pip install -e ".[server]"

# prove the model generates on this machine
uv run python scripts/smoke_generate.py

# measure the baseline: prefill latency + decode tokens/sec
uv run python benchmarks/bench.py

# serve it over HTTP
uv run python scripts/serve.py
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

## Status

Milestone 1 in progress: baseline server and benchmark harness running on MacBook (MPS).
First recorded floor: **2.2 tok/s decode, 105ms prefill** for the reference no-KV-cache loop ([results/baseline_mps.json](results/baseline_mps.json)).
See [moe-inference-engine.md](moe-inference-engine.md) for the full project plan, risks, and timeline.
