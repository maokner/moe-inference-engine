# From-Scratch MoE Inference Engine

## Goal

Build a low-latency, single-request inference engine for miniMoE with a contiguous production KV cache, custom kernels, and quantization.
Keep paged KV caching and direct paged attention as explicit learning and benchmark paths.
Compare it with vLLM and llama.cpp on fixed single-request workloads.

## Motivation

The project covers the path from model operations to device-level execution.
Results include throughput, time-to-first-token, and latency curves on specified hardware.

## Custom kernel path

The direct paged-attention kernel is implemented: decode attention reads non-contiguous K/V blocks through the block table in one Triton kernel, and prefill attends over the contiguous projection outputs instead of gathering blocks back.
On the final 21-round interleaved A6000 comparison it reached 14.97 tok/s, ahead of the old gather path at 14.58 tok/s but behind the contiguous cache at 15.34 tok/s.
The paired direct-minus-contiguous latency delta was +1.595ms with a 95% interval of +0.234ms to +2.956ms, so the data supports a small slowdown and the original match-or-beat target remains unmet.
The synchronous server therefore uses contiguous caching by default.
Paged serving remains available through `--cache paged`, with `--paged-attention auto|gather|direct` selecting the fallback or Triton implementation.
The one-token contiguous-decode profile attributes 69.8% of instrumented time to routing/top-2 selection and 79.7% to the combined MoE path.
Pure Python loop overhead is only 3.8% of clean median latency; 48 `aten::nonzero` calls from per-expert `torch.where` route discovery are the dominant operation at 38.196ms per token.
Quantization provides the surrounding inference path.

## Scope, in build order

1. **Baseline server.** Plain PyTorch generation over HTTP. Done.
2. **Paged KV cache.** Fixed-size blocks and a block table per sequence, retained as an explicit learning and benchmark path. Done.
3. **Direct paged attention.** Read non-contiguous K/V blocks directly instead of gathering them into contiguous tensors on every decode step, with the acceptance target of matching or beating the contiguous KV cache on the same single-request benchmark. Implemented: beat gather by 2.7% but trailed contiguous by 2.4% on the final interleaved A6000 run.
4. **Fused MoE kernel.** Fuse top-2 dispatch and grouped expert GEMMs in Triton or CUDA.
5. **Quantization.** Int8 and int4 groupwise expert weights, with HellaSwag and perplexity measurements.
6. **CUDA graphs.** Capture the decode step to kill launch overhead at small batch sizes.
7. **Speculative decoding.** Use a small dense draft model.
8. **MLX port.** Run the quantized model on Apple Silicon.

## Benchmark methodology

- Same model, same weights, same prompts, same hardware for every system compared.
- Baselines: HF transformers generate, llama.cpp, and vLLM.
- Metrics: single-request tokens/sec, time-to-first-token, and per-token latency across context lengths.
- One rented 4090 or A100; report exact GPU, driver, and versions.
- Publish the benchmark harness and raw results.
- Document remaining performance gaps against vLLM.

## Target results

- Built an LLM inference engine from scratch in Python/Triton serving a 280M sparse MoE with a contiguous production KV cache, explicit paged and direct-attention benchmark paths, CUDA graphs, and int4/int8 quantization.
- Wrote a fused MoE kernel reaching X% of vLLM's `fused_moe` throughput; single-request generation reaches Nx HF `generate` and Y% of vLLM on one dedicated GPU.
- Ported the quantized model to MLX, serving Z tokens/sec on-device on an M-series MacBook.

The values remain placeholders until the relevant milestones are complete.

## Existing foundation

miniMoE provides the model, tokenizer, checkpoint, and evaluation harness.
The routing ablation pipeline provides a base for controlled measurements.

## Hardware and cost

GPU work runs on one rented NVIDIA GPU with the exact model, driver, and software versions reported for every result.
MLX work runs on the development MacBook.

## Risks

- **Triton learning curve.** Keep the PyTorch cache and expert-loop implementations as correctness oracles for each custom kernel.
- **Performance gap against vLLM.** Report the gap and profile its causes.
- **Scope creep toward a general-purpose engine.** Serve miniMoE and at most one dense reference model for baseline comparisons.
  Do not build a model zoo.
- **Serving architecture drift.** Keep exactly one active request and the contiguous cache as the production default.
  Do not add continuous batching, scheduling, admission control, or multi-request abstractions.

## Site/resume timing

Do not replace Warden on the site until the first benchmark chart exists.
A "Building" card with no results is weaker than Warden's finished card.
Swap when the direct paged-attention or fused-MoE milestone has dedicated-GPU numbers.

## Next steps

- Build a batch-one fused MoE path that removes dynamic route discovery and CPU synchronization, not only the 2.494ms of pure Python loop overhead.
- Use three fixed-shape Triton kernels per layer: fp32 router/top-2 to two device-resident ids and weights, selected-expert W1 plus GELU to `[2, 3072]`, then selected-expert W2 plus weighted reduction to `[768]`.
- Keep the expert-loop implementation as the correctness oracle for any future fused path.
- CUDA graphs are the most relevant later optimization for the remaining direct-attention launch-overhead gap against contiguous SDPA.
- Benchmark discipline from milestone 3 carries forward: interleave systems within one session on virtualized GPUs; sequential passes drift too much to resolve small differences.
