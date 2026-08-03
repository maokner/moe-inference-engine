# moe-inference-engine

A from-scratch inference engine for [miniMoE](https://github.com/maokner/miniMoE), a 280M sparse Mixture-of-Experts model.
The production serving path optimizes latency for exactly one active request with a contiguous KV cache.
Paged KV caching and direct Triton paged attention remain available as learning and benchmark implementations.

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
uv run python scripts/check_parity.py --device cuda --engine-cache paged-direct

# benchmarks (per-system harness)
uv run python benchmarks/bench.py --system reference
uv run python benchmarks/bench.py --system engine
uv run python benchmarks/bench.py --system engine-paged --paged-attention gather
uv run python benchmarks/bench.py --system engine-paged --paged-attention direct

# interleaved comparison, context scaling, and attention microbenchmark
uv run python benchmarks/ab_compare.py
uv run python benchmarks/context_sweep.py
uv run python benchmarks/attention_micro.py

# warmed one-token contiguous decode profile (CUDA by default)
uv run python benchmarks/profile_contiguous_decode.py \
  --output-dir results/contiguous_decode_profile

# serve one request at a time with the default contiguous KV cache
uv run python scripts/serve.py
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'

# explicitly select paged serving for study or benchmarking
uv run python scripts/serve.py --cache paged --paged-attention auto
uv run python scripts/serve.py --cache paged --paged-attention gather
uv run python scripts/serve.py --cache paged --paged-attention direct
```

## Status

Milestones 1, 2, and 3 are complete.
The engine has a custom forward pass, a contiguous KV cache, a paged KV cache backed by a shared block pool, and a direct paged-attention Triton kernel that removes most of the old gather path's overhead.
The synchronous HTTP server creates an independent contiguous `KVCache` for every request by default because that path has the lowest measured single-request latency.
`--cache paged` explicitly enables the shared block pool, and `--paged-attention auto|gather|direct` selects its attention implementation.
Paged requests return their blocks after success or failure, while contiguous caches require no explicit `free()` call.
Requests remain serialized, and continuous batching, scheduling, and multi-request admission are deliberately out of scope.

### Direct paged attention (milestone 3)

The old paged path called `PagedKVCache.read()`, which gathered every physical block into freshly allocated contiguous K and V tensors at every layer of every decode step.
The decode kernel in [src/moe_engine/paged_attention.py](src/moe_engine/paged_attention.py) removes that gather: for logical token position `p` it looks up `physical_block = block_table[p // block_size]` on device and reads the K/V rows at offset `p % block_size` inside that block, straight out of the shared pool.
Each of the 8 attention heads is one Triton program.
A head loops over the context in 128-token tiles with an online (running-max) softmax, and pads head dimension 96 to a 128-wide tile under a mask so no invalid columns are read.
The block table lives twice: the Python list handles allocation, freeing, and tests, and a small int32 device mirror is appended to only when a new block is reserved (once per 16 tokens), never rebuilt per layer or step.
Everything else stays in PyTorch: QKV and output projections, cache writes, routing, and the expert MLPs.

Prefill never uses the kernel.
During an initial prefill the entire attention history is exactly the K/V the layer just projected, so the engine writes the blocks for later decode but computes prompt attention directly from those contiguous projection tensors with causal SDPA.
Multi-token cache writes are also one indexed scatter per pool now instead of a Python loop over tokens.
Together these removed the paged prefill regression (roughly 102-113 ms before, at parity with the contiguous cache now).

Each block allocator picks one of three attention modes.
`gather` is the original read-and-reconstruct path, kept as the correctness oracle and an explicit benchmark target.
`direct` forces the kernel and raises where it cannot run.
`auto` (the default) uses the kernel for CUDA decode and falls back to `gather` decode on CPU/MPS.

### Correctness

`pytest` compares the kernel against the gather-plus-SDPA oracle at contexts from 1 to 1024, across block boundaries, with deliberately non-contiguous physical block ids, partially filled final blocks, and production head dimensions; CUDA tests skip cleanly off-GPU.
Kernel-level checks pass at atol 2e-5 / rtol 1e-5 in float32; model-level checks use the suite-wide 1e-4.
On the real checkpoint on CUDA, all three engine paths (contiguous, paged-gather, paged-direct) report the same 6.0e-5 maximum absolute logit difference against the reference model and identical 32-token greedy generations ([results/direct_paged_cuda/](results/direct_paged_cuda/)).
Forcing `direct` on an unsupported device or dtype is validated before the cache write, so the rejected request allocates no blocks and leaves the cache untouched.

### A6000 results

Measured on a Thunder Compute NVIDIA RTX A6000 (virtualized; torch 2.13.0+cu130, triton 3.7.1) with the 62-token prompt workload.
Sequential per-system runs on this GPU drift by tens of percent over minutes ([results/direct_paged_cuda/](results/direct_paged_cuda/) keeps two ordered passes of `bench.py` as evidence), so the headline numbers come from [benchmarks/ab_compare.py](benchmarks/ab_compare.py), which interleaves short rounds of all three paths with a rotating start order so drift hits every path equally.
The final run used 21 rounds of 64 greedy tokens each.
Decode throughput is total completed steps divided by total decode time, while median and p90 remain latency statistics.
Raw step times, per-round totals and throughput, and paired confidence intervals are in [results/direct_paged_cuda/ab_compare.json](results/direct_paged_cuda/ab_compare.json):

| system | prefill | decode median / p90 | decode tok/s |
|---|---:|---:|---:|
| engine (contiguous KV cache) | 92.4ms | 62.0ms / 82.0ms | **15.34** |
| engine (paged, gather) | 95.1ms | 66.1ms / 84.9ms | 14.58 |
| engine (paged, direct kernel) | 88.6ms | 63.0ms / 85.7ms | 14.97 |

Direct paged attention is 2.7% faster than gather, but 2.4% slower than contiguous in aggregate throughput on this run, so the original match-or-beat acceptance target was not met.
The paired mean step-latency delta for direct minus contiguous is +1.595ms with a 95% t-interval of +0.234ms to +2.956ms, which supports a small but statistically detectable slowdown rather than an exact tie.
Direct minus gather is -1.802ms with a 95% interval of -3.072ms to -0.532ms, supporting the direct kernel's improvement over reconstruction.
The cache-path margins are small relative to each roughly 65ms decode step.
The attention microbenchmark ([results/direct_paged_cuda/attention_micro.json](results/direct_paged_cuda/attention_micro.json)) isolates what the direct kernel replaced per layer call, but it does not attribute the rest of whole-model decode time:

| context | contiguous read+SDPA | paged gather+SDPA | direct kernel |
|---:|---:|---:|---:|
| 16 | 197us | 406us | 147us |
| 64 | 108us | 502us | 140us |
| 256 | 106us | 493us | 146us |
| 1024 | 98us | 489us | 119us |

The gather path pays for index/copy/reshape kernel chains on every call; the direct path is one kernel launch and stays nearly flat in context length.
An interleaved context sweep at 16/64/256/1024 cached tokens shows the same picture end to end: direct tracks contiguous within about 1 ms per step at every context while gather is slowest, worst at 1024 ([results/direct_paged_cuda/context_sweep.json](results/direct_paged_cuda/context_sweep.json)).

### One-token contiguous decode profile

The reproducible profiler uses the real checkpoint and the canonical 62-token benchmark prompt, warms 32 one-token greedy steps, then records 64 clean and 64 instrumented decode steps.
On a Thunder Compute NVIDIA RTX A6000 with driver 610.43.02, torch 2.13.0+cu130, and Triton 3.7.1, clean CUDA-event timing measured a 65.460ms median, 64.801ms mean, 68.065ms p90, and 15.43 tok/s.
The separate CPU-activity `torch.profiler` pass measured 78.840ms per instrumented step.
These component times include profiler overhead and synchronous GPU waits, so they are not presented as clean throughput:

| component | instrumented ms/step | share |
|---|---:|---:|
| MoE routing/top-2 selection | 55.022 | 69.8% |
| QKV and attention work | 6.325 | 8.0% |
| Expert MLP execution | 5.400 | 6.8% |
| LayerNorm/residual/output | 4.537 | 5.8% |
| Route-weight combination | 2.409 | 3.1% |

The combined MoE path accounts for 62.832ms, or 79.7% of the instrumented step, but the Python interpreter loop itself accounts for only 2.494ms, or 3.8% of clean median latency.
The actual dominant operation is route discovery: the one-token path executes `torch.where` for every expert in every layer, producing 48 `aten::nonzero` calls and 38.196ms of CPU operator time per step.
The evidence therefore rejects the claim that pure Python loop overhead or expert GEMM execution is the dominant bottleneck.
The next kernel must eliminate dynamic route discovery, CPU reads, and per-expert dispatch in addition to fusing expert math.

The concrete batch-one design is three fixed-shape Triton kernels per layer: an fp32 8-way router/top-2 kernel that writes two device-resident ids and normalized weights, a selected-expert 768x3072 W1 plus GELU kernel that writes `[2, 3072]`, and a selected-expert 3072x768 W2 plus weighted-reduction kernel that writes one `[768]` row.
The existing PyTorch path remains the oracle, with logit and greedy-token parity required before performance comparison.

Raw CUDA-event step times, operator tables, hardware metadata, generated tokens, and limitations are in [profile.json](results/contiguous_decode_profile_a6000/profile.json).
The [human summary](results/contiguous_decode_profile_a6000/summary.md), [compressed CPU profiler trace](results/contiguous_decode_profile_a6000/trace.json.gz), and [run log](results/contiguous_decode_profile_a6000/run.log) reproduce the report.
Kineto/CUPTI dropped CUDA-activity buffers on this virtualized GPU, so the authoritative component pass intentionally collected CPU activity only; the failed CUDA-activity validation is preserved in [cupti_activity_validation.log](results/contiguous_decode_profile_a6000/cupti_activity_validation.log) rather than being reported as valid kernel timing.
Real-checkpoint parity for contiguous, paged-gather, and paged-direct serving is also preserved in [results/contiguous_decode_profile_a6000/](results/contiguous_decode_profile_a6000/); all three paths have the same 6.01e-5 maximum logit difference and no greedy-token mismatch.

Earlier milestone-2 measurements (MacBook MPS and the first A6000 session, gather path with the old per-token writes) are preserved in [results/](results/) and [results/paged_only_cuda/](results/paged_only_cuda/) for history.
The shared pool reserves 36 MB of K/V storage for 1024 tokens; the 190-token benchmark workload holds 12 of 64 blocks, about 6.8 MB, while a contiguous cache reserves the full 36 MB per sequence.
Known limitation: absolute tok/s on Thunder's virtualized GPUs is depressed by high launch latency and varies between sessions, so cross-session comparisons are invalid; the dedicated-GPU vLLM comparison planned for later milestones will produce the citable absolute numbers.
The remaining direct-versus-contiguous gap is small enough that contiguous caching is now the production default and CUDA graphs remain the most relevant later cache-path optimization.
The dedicated one-token profile in `benchmarks/profile_contiguous_decode.py` confirms that synchronization-heavy route discovery, not pure Python loop overhead or expert GEMMs, is the next kernel target.
See [moe-inference-engine.md](moe-inference-engine.md) for the full project plan, risks, and timeline.
