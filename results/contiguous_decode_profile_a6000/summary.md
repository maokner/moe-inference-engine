# One-token contiguous decode profile

Hardware: NVIDIA RTX A6000.
Checkpoint step: 6358.
Workload: 62-token benchmark prompt followed by 64 one-token greedy decode steps.
Clean median decode latency: 65.460 ms (15.43 tok/s).
Instrumented total: 78.840 ms per step using CPU-scope wall time under CPU-activity torch.profiler.
The component timings below come from the instrumented profiler run and are not uninstrumented throughput measurements.

| component | instrumented ms/step | share | calls/step |
|---|---:|---:|---:|
| MoE routing/top-2 selection | 55.022 | 69.8% | 54.0 |
| QKV and attention work | 6.325 | 8.0% | 6.0 |
| Expert MLP execution | 5.400 | 6.8% | 12.0 |
| LayerNorm/residual/output | 4.537 | 5.8% | 20.0 |
| Route-weight combination | 2.409 | 3.1% | 12.0 |

Combined MoE routing, expert, and route-combination work: 62.832 ms per step (79.7% of instrumented decode time).
The combined MoE path dominates instrumented decode time, so it is the next kernel target.
Largest component scope: MoE routing/top-2 selection at 55.022 ms per step (69.8% of instrumented decode time).
The existing per-expert loop itself has 2.494 ms per step of profiler-unattributed Python CPU time (3.8% of clean median latency); its nested tensor operations and kernel launches are reported in the component rows.
Pure Python loop overhead is not the dominant bottleneck.
The largest individual CPU operator is aten::nonzero at 38.196 ms per step across 48.0 calls per step.
CPU operator self time: 61.665 ms per step across 1659.0 calls per step.
CUDA launch events are unavailable in the CPU-activity-only trace; CPU operator dispatch time and call counts are reported instead.
The next MoE kernel should target batch-one decode directly: compute the float32 8-way router and normalized top-2 weights, run only the two selected 768x3072x768 GELU expert MLPs, and combine their weighted outputs without torch.where, per-expert Python dispatch, CPU reads of route counts, or index_add_ launches.
A practical first design uses three fixed-shape Triton kernels per layer: router plus top-2 writes two device-resident ids and weights; selected-expert W1 plus GELU writes a [2, 3072] intermediate; selected-expert W2 plus weighted reduction writes one [768] output row.
Keep the existing PyTorch expert loop as the correctness oracle and require logit and greedy-token parity before benchmarking.

## Limitations

- torch.profiler and record_function add CPU overhead, so component timings are reported separately from clean latency.
- The component profile collects CPU activity only because CUPTI can drop CUDA-activity buffers on virtualized GPUs; clean whole-step GPU latency still uses CUDA events.
- CPU scope time includes dispatch and synchronous GPU waits; asynchronous GPU work may be charged to a later synchronization scope, so values are critical-path attribution rather than isolated kernel durations.
- The repeated steps grow context from 62 tokens; results describe warmed early decode, not every context length.
