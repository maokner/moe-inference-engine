# miniMoE inference engine

A low-latency, single-request inference engine for [miniMoE](https://github.com/maokner/miniMoE), built in PyTorch and Triton.

**[miniMoE weights](https://huggingface.co/mokner123/miniMoE)**

miniMoE is a 280M-parameter sparse Mixture-of-Experts model with 110M active parameters per token I build earlier.
Here, I rebuild its inference path to study which serving optimizations matter when exactly one request is active.

## Optimizations

The engine supports both contiguous and paged KV caches.
Contiguous caching is the serving default because it delivers the lowest measured latency at batch size one.
Paged caching adds allocation and lookup overhead that is useful for managing many concurrent sequences, but provides no memory-management advantage for this workload.

The paged path includes a custom Triton attention kernel that reads K/V directly through the block table instead of gathering blocks into contiguous tensors before every decode step.
Prefill still uses PyTorch SDPA because its newly projected K/V tensors are already contiguous.
The direct kernel improves paged decode throughput by 2.7%, but contiguous caching remains 2.4% faster end to end.

The server deliberately handles one request at a time.
Continuous batching and scheduling are omitted because they optimize aggregate throughput rather than the single-user latency this project measures.

## Results

Measured with the real checkpoint on an NVIDIA RTX A6000 using a 62-token prompt and interleaved 64-token greedy decode runs.

| KV cache | Median decode latency | Throughput |
|---|---:|---:|
| Contiguous | **62.0 ms** | **15.34 tok/s** |
| Paged, gather | 66.1 ms | 14.58 tok/s |
| Paged, direct Triton | 63.0 ms | 14.97 tok/s |

All three paths match the reference model within `6.01e-5` maximum logit error and produce identical greedy tokens.
A one-token profile attributes 69.8% of instrumented latency to MoE routing, driven by synchronization-heavy per-expert route discovery rather than expert matrix multiplication.

Raw measurements are in [`results/`](results/).

## Run

Place `minimoe_sft.pt` in `checkpoints/`, then:

```bash
uv sync --extra server
uv run python scripts/serve.py
```

Use `--cache paged --paged-attention direct` to run the Triton paged-attention path.
