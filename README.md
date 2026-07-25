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
uv run python benchmarks/bench_batch.py --batch-sizes 1,8,32

# serve it over HTTP
uv run python scripts/serve.py
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

## Status

Milestones 1 through 3 are complete and milestone 4 is underway.
The fused MoE Triton kernel has landed: (token, expert) pairs are sorted by expert and padded to GEMM-block boundaries (vLLM's `moe_align_block_size` scheme), then two grouped-GEMM launches replace the Python loop over experts, with bias, exact-erf GELU, and the routing weight fused into the epilogues.
The expert loop remains as the parity oracle; `tests/test_fused_moe.py` checks the kernel against it from single-token decode shapes up to prefill shapes, including token-exact greedy generation.
Still open in milestone 4: the head-to-head against vLLM's `fused_moe` on a dedicated GPU.
The engine has a custom forward pass, a contiguous KV cache used by `generate()`, and a paged KV cache backed by a shared block pool.
Both cache implementations expose the same `write` and `read` interface to `CachedAttention`.
A scheduling engine (`moe_engine/engine.py`) runs iteration-level batching: requests join the running batch between decode steps and leave the moment they finish, returning their KV blocks to the pool.
Admission reserves each request's worst-case block count up front, so decoding never runs out of blocks mid-flight; vLLM instead admits optimistically and preempts, which serves more requests at the cost of swap/recompute machinery.
`Model.forward_decode` advances a batch of sequences at different positions in one forward pass; embeddings, norms, and the expert MLPs run batched, while attention loops over sequences until a future paged-attention kernel replaces the loop.
The HTTP server submits each request to the shared engine and waits on its completion event, so concurrent requests genuinely decode together.
The server rejects over-length prompts and stops generation when the context window fills.
`pytest` covers cache parity, batched-vs-sequential decode parity, mid-flight batch joins and leaves, admission under block pressure, and the server endpoint.
On the real checkpoint, `scripts/check_parity.py` reports a maximum logit difference of 1.3e-4 and identical greedy generations ([results/parity_cpu.json](results/parity_cpu.json)).

Each table below comes from one back-to-back session on a rented GPU (Thunder Compute virtualized RTX A6000, driver 610.43.02, torch 2.13.0+cu130); the single-sequence table and the milestone-4 tables are from different sessions, so compare within a table only.
Single-sequence rows: 62-token prompt, 128 greedy tokens, per-position medians across three runs; prefill covers only the forward pass, with cache allocation outside the timed region.
Raw step times are stored in [results/](results/).

| system | decode tok/s | median / p90 latency | prefill |
|---|---|---|---|
| reference (no KV cache) | 15.2 | 66ms / 73ms | 75ms |
| engine (KV cache) | 16.6 | 61ms / 65ms | 86ms |
| engine (paged KV cache) | 22.8 | 44ms / 52ms | 97ms |

One MoE layer from the real checkpoint, fused kernel against the expert loop ([results/moe_kernel_cuda.json](results/moe_kernel_cuda.json)):

| tokens | expert loop | fused kernel | speedup |
|---|---|---|---|
| 1 | 3761µs | 1696µs | 2.2x |
| 8 | 6042µs | 1934µs | 3.1x |
| 32 | 8016µs | 2883µs | 2.8x |
| 256 | 6167µs | 1866µs | 3.3x |
| 1024 | 6183µs | 1559µs | 4.0x |

An earlier version of the alignment step hid a `torch.bincount` device sync in every call, which serialized the launch pipeline and capped the fused path near 6% of fp32 peak; removing it (`scatter_add_` into a fixed-size tensor) let the ~20 launches pipeline, and the fused path now sustains about 30% of the A6000's fp32 peak at 1024 tokens.
The large-shape fused rows therefore measure real arithmetic; the small-token rows and the whole loop column remain launch- and dispatch-bound (the loop issues roughly 60 kernels per layer against the fused path's ~20) and swing tens of percent between repeated runs on this virtualized GPU - the loop at 1 token measured 7.5ms and 3.8ms in back-to-back runs - so read the speedup column as "several times", not as a precise ratio.
The definitive arithmetic comparison is the dedicated-GPU vLLM head-to-head.
That comparison must also account for precision: this kernel runs exact fp32 (`input_precision="ieee"`) to stay directly comparable with the reference model, while vLLM's `fused_moe` runs fp16/bf16 through tensor cores, several times fp32 peak on this hardware.

Continuous batching end to end, 62-token prompt, 128 greedy tokens per request, loop and fused measured back to back ([results/engine_batch_cuda.json](results/engine_batch_cuda.json)):

| batch size | aggregate tok/s (loop) | aggregate tok/s (fused) |
|---|---|---|
| 1 | 20.3 | 30.4 |
| 8 | 91.8 | 109.6 |
| 32 | 136.9 | **155.4** |

The kernel buys 50% end to end at batch 1 and 13% at batch 32, where the per-sequence attention loop and Python scheduling take a growing share of each step - those are the next targets.

Read the single-sequence table with skepticism: a 280M model decodes in sub-millisecond kernels, so on a virtualized network-attached GPU those rows are dominated by per-launch latency, which swings heavily between sessions (the paged system measured 17.6 and then 22.8 tok/s in two sessions at the same revision, and the contiguous/paged ordering flipped).
In particular, the KV cache's benefit is real but unmeasurable here: it removes recompute this GPU was never short of, while the wall-clock cost is kernel-launch overhead that caching does not reduce.
The MPS runs show the compute-bound case instead: the same code measures 2.5 tok/s uncached (decaying as context grows) versus 18.3 tok/s cached ([results/engine_kvcache_mps.json](results/engine_kvcache_mps.json)).
The robust result is the batching trend: aggregate throughput scales 5.1x from batch 1 to 32 on the fused path (6.7x on the loop path), even though attention still loops over sequences in Python - a future paged-attention kernel targets that loop.
The two harnesses also time differently (`bench.py` synchronizes every token, `bench_batch.py` once per run), so compare within a table, not across tables.
`bench_batch.py` submits all requests up front and measures steady-state batching; mid-flight joins and leaves are covered by tests rather than benchmarks.
The shared block pool reserves 36 MB of K/V storage per 1024 tokens; the single-sequence workload pins 12 of 64 blocks, about 6.8 MB, while a contiguous cache reserves the full 36 MB per sequence.
Development numbers on the MacBook (MPS) from milestones 2 and 3 remain in [results/](results/) as `*_mps.json`.
Next: the vLLM `fused_moe` head-to-head on a dedicated (non-virtualized) GPU, then quantization.
See [moe-inference-engine.md](moe-inference-engine.md) for the full project plan, risks, and timeline.
