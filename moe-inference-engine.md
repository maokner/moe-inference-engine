# From-Scratch MoE Inference Engine: Serve miniMoE Fast

## One-sentence thesis

Write the inference engine that serves miniMoE from scratch - custom Triton/CUDA kernels, paged KV cache, continuous batching, quantization - and benchmark it honestly against vLLM and llama.cpp.

## Why this could impress the target companies

Annapurna Labs, NVIDIA AI, Apple AI, and xAI infra all hire the same core profile: people who understand what happens between the model math and the silicon.
This project is that job, performed in public.
It also completes a story no other student has: "trained a 280M MoE from scratch" is rare, and "trained it, then wrote the engine that serves it" is a full-stack narrative an interviewer can drill into for an hour at any level of the stack.
The deliverable is numbers, not vibes: tokens/sec, time-to-first-token, and throughput-latency curves against named baselines on named hardware.

## The differentiator: the fused MoE dispatch path

Plenty of people have written mini-vLLM clones for dense models; that alone is table stakes.
The unclaimed part at portfolio scale is the MoE forward path: gate, top-2 routing, and grouped expert GEMMs without materializing padded per-expert batches.
This is a known performance-critical kernel in vLLM and SGLang (fused_moe), and it is genuinely hard to get right.
That kernel is the headline of the project.
Everything else (KV cache, batching, quantization) is the credible engine around it.

## Scope, in build order

1. **Baseline server.** Plain PyTorch + HF-style generate loop serving miniMoE over HTTP. This is the floor every speedup is measured against.
2. **KV cache done right.** Paged KV cache (fixed-size blocks, block table per sequence) so long and short requests share memory without fragmentation.
3. **Continuous batching.** Iteration-level scheduling: new requests join the running batch at any decode step, finished ones leave. This is where most of the throughput lives.
4. **The fused MoE kernel (centerpiece).** Triton first, CUDA if needed: fuse gating + top-2 dispatch + grouped expert GEMM. Compare against the naive loop-over-experts implementation and against vLLM's fused_moe.
5. **Quantization.** Int8 then int4 groupwise weight quantization for the experts. Report the quality cost (HellaSwag delta, perplexity delta) next to the speed gain, not instead of it.
6. **CUDA graphs.** Capture the decode step to kill launch overhead at small batch sizes.
7. **Stretch: speculative decoding.** A tiny dense draft model speculating for the MoE. Only if milestones 1-6 land cleanly.
8. **The Apple angle.** Port the quantized model to MLX and publish on-device numbers from the MacBook. About a week of work, and it turns one project into a specific talking point for Apple AI.

## Benchmark methodology (this is half the credibility)

- Same model, same weights, same prompts, same hardware for every system compared.
- Baselines: HF transformers generate (floor), llama.cpp, vLLM (ceiling).
- Metrics: tokens/sec at batch 1, 8, 32; time-to-first-token; full throughput-latency curves, not single points.
- One rented 4090 or A100; report exact GPU, driver, and versions.
- Publish the harness so every number regenerates with one command, same standard as the miniMoE eval pipeline.
- An honest "where vLLM still beats me and why" section reads as more credible, not less.

## Target resume bullets (write toward these)

- Built an LLM inference engine from scratch in Python/Triton serving a 280M sparse MoE: paged KV cache, continuous batching, CUDA graphs, and int4/int8 quantization.
- Wrote a fused MoE kernel (gating + top-k dispatch + grouped expert GEMM) reaching X% of vLLM's fused_moe throughput; engine end-to-end reaches Nx HF generate and Y% of vLLM at batch 1-32 on one A100.
- Ported the quantized model to MLX, serving Z tokens/sec on-device on an M-series MacBook.

The numbers are placeholders; the shape of the claim is the point.
If X ends up at 60%, that is still a strong bullet with the honest comparison attached.

## Why me

miniMoE means the model, tokenizer, and eval harness already exist and I understand every layer of what the engine has to compute.
The routing ablation pipeline already proved I can run controlled, reproducible measurements.
CS 61C covered the memory-hierarchy intuitions the kernel work builds on; Fall 2026 CS 162 lands right as the scheduling/batching layer gets built.

## Hardware and cost

Everything runs on one rented 4090 or A100 (Lambda/RunPod-class pricing, tens of dollars total, nothing like the miniMoE pretraining bill).
MLX work runs free on the MacBook.

## Risks

- **Triton learning curve.** Mitigation: milestone order is chosen so the engine is already useful (2-3) before the hard kernel (4). If the fused kernel stalls, an engine with paged KV + continuous batching + quantization is still a real project.
- **"Within Y% of vLLM" comes out embarrassing.** Mitigation: the honest-methodology framing absorbs this; report the gap and explain the causes. The floor comparison (HF generate) will be a large multiple regardless.
- **Scope creep toward a general-purpose engine.** Serve miniMoE and at most one dense reference model (for baseline comparability). Do not build a model zoo.

## Site/resume timing

Do not replace Warden on the site until the first benchmark chart exists.
A "Building" card with no results is weaker than Warden's finished card.
Swap when milestone 3 or 4 has numbers.

## Next steps

- Read the vLLM paged-attention and fused_moe source and the PagedAttention paper before writing any code; know the reference designs cold.
- Milestone 1: baseline server + benchmark harness first, so every later change lands as a measured delta.
- Rent the GPU only from milestone 2 onward; milestone 1 debugs fine on the MacBook with MPS.
