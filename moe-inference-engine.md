# From-Scratch MoE Inference Engine

## Goal

Build an inference engine for miniMoE with custom kernels, paged KV caching, continuous batching, and quantization.
Compare it with vLLM and llama.cpp on fixed workloads.

## Motivation

The project covers the path from model operations to device-level execution.
Results include throughput, time-to-first-token, and latency curves on specified hardware.

## Fused MoE dispatch

The main kernel covers gating, top-2 routing, and grouped expert GEMMs without padded per-expert batches.
KV caching, batching, and quantization provide the surrounding inference path.

## Scope, in build order

1. **Baseline server.** Plain PyTorch generation over HTTP.
2. **Paged KV cache.** Fixed-size blocks and a block table per sequence.
3. **Continuous batching.** Add and remove requests between decode iterations.
4. **Fused MoE kernel.** Fuse gating, top-2 dispatch, and grouped expert GEMM in Triton or CUDA.
5. **Quantization.** Int8 and int4 groupwise expert weights, with HellaSwag and perplexity measurements.
6. **CUDA graphs.** Capture the decode step to kill launch overhead at small batch sizes.
7. **Speculative decoding.** Use a small dense draft model.
8. **MLX port.** Run the quantized model on Apple Silicon.

## Benchmark methodology

- Same model, same weights, same prompts, same hardware for every system compared.
- Baselines: HF transformers generate, llama.cpp, and vLLM.
- Metrics: tokens/sec at batch 1, 8, 32; time-to-first-token; full throughput-latency curves, not single points.
- One rented 4090 or A100; report exact GPU, driver, and versions.
- Publish the benchmark harness and raw results.
- Document remaining performance gaps against vLLM.

## Target results

- Built an LLM inference engine from scratch in Python/Triton serving a 280M sparse MoE: paged KV cache, continuous batching, CUDA graphs, and int4/int8 quantization.
- Wrote a fused MoE kernel (gating + top-k dispatch + grouped expert GEMM) reaching X% of vLLM's fused_moe throughput; engine end-to-end reaches Nx HF generate and Y% of vLLM at batch 1-32 on one A100.
- Ported the quantized model to MLX, serving Z tokens/sec on-device on an M-series MacBook.

The values remain placeholders until the relevant milestones are complete.

## Existing foundation

miniMoE provides the model, tokenizer, checkpoint, and evaluation harness.
The routing ablation pipeline provides a base for controlled measurements.

## Hardware and cost

GPU work runs on one rented 4090 or A100.
MLX work runs on the development MacBook.

## Risks

- **Triton learning curve.** Build cache and batching support before the fused kernel.
- **Performance gap against vLLM.** Report the gap and profile its causes.
- **Scope creep toward a general-purpose engine.** Serve miniMoE and at most one dense reference model for baseline comparisons.
  Do not build a model zoo.

## Site/resume timing

Do not replace Warden on the site until the first benchmark chart exists.
A "Building" card with no results is weaker than Warden's finished card.
Swap when milestone 3 or 4 has numbers.

## Next steps

- Read the vLLM paged-attention and fused_moe source and the PagedAttention paper before writing any code; know the reference designs cold.
- Milestone 1: baseline server + benchmark harness first, so every later change lands as a measured delta.
- Rent the GPU only from milestone 2 onward; milestone 1 debugs fine on the MacBook with MPS.
