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

A fixed-shape Triton MoE decode path handles the production batch-one float32 CUDA shape.
It keeps routing ids and weights on device, evaluates only the selected two experts, and retains the PyTorch MoE as the prefill and correctness-oracle path.
The three-kernel path has been validated with the real checkpoint on an RTX A6000.

## Results

Measured with the real checkpoint on an NVIDIA RTX A6000 using a 62-token prompt and interleaved 64-token greedy decode runs.

| KV cache | Median decode latency | Throughput |
|---|---:|---:|
| Contiguous | **62.0 ms** | **15.34 tok/s** |
| Paged, gather | 66.1 ms | 14.58 tok/s |
| Paged, direct Triton | 63.0 ms | 14.97 tok/s |

All three paths match the reference model within `6.01e-5` maximum logit error and produce identical greedy tokens.

The fused-MoE comparison uses 21 interleaved rounds and 1,344 measured decode tokens per mode.
Throughput is total completed steps divided by total decode time.

| MoE decode | Total decode time | Median | p90 | Aggregate throughput |
|---|---:|---:|---:|---:|
| PyTorch reference | 70.064978 s | 53.110 ms | 60.028 ms | 19.182 tok/s |
| Direct Triton | 10.270138 s | 7.555 ms | 8.193 ms | **130.865 tok/s** |

Direct throughput is 582.23% higher, and the paired direct-minus-reference latency delta is -44.490 ms per step with a 95% confidence interval of [-46.612, -42.368] ms.
Real-model one-token logits have `1.62e-5` maximum and `2.81e-6` mean absolute error, and all 64 greedy tokens match exactly.
The profile shows that combined MoE work fell from 45.869 ms to 1.960 ms per instrumented step, `aten::nonzero` fell from 48 calls to zero, and QKV plus attention is now the largest component.

Raw measurements are in [`results/`](results/).

## Run

Place `minimoe_sft.pt` in `checkpoints/`, then:

```bash
uv sync --extra server
uv run python scripts/serve.py --device cuda
```

The general server and benchmark entry points default to CPU so accelerators are never selected implicitly.
Pass `--device cuda` or `--device mps` explicitly when that accelerator is intended.
Use `--cache paged --paged-attention direct` to run the Triton paged-attention path.
Use `--moe reference|direct|auto` to select the PyTorch MoE oracle, force the Triton decode path, or use automatic fallback.
