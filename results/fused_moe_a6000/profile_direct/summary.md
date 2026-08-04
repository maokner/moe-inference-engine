# One-token contiguous decode profile

Hardware: NVIDIA RTX A6000.
Checkpoint step: 6358.
Workload: 62-token benchmark prompt followed by 32 one-token greedy decode steps.
MoE mode: direct.
Clean median decode latency: 6.841 ms (142.97 tok/s).
Instrumented total: 9.994 ms per step using CPU-scope wall time under CPU-activity torch.profiler.
The component timings below come from the instrumented profiler run and are not uninstrumented throughput measurements.

| component | instrumented ms/step | share | calls/step |
|---|---:|---:|---:|
| QKV and attention work | 3.917 | 39.2% | 6.0 |
| LayerNorm/residual/output | 2.847 | 28.5% | 20.0 |
| Fused MoE decode | 1.960 | 19.6% | 6.0 |
| MoE routing/top-2 selection | 0.000 | 0.0% | 0.0 |
| Expert MLP execution | 0.000 | 0.0% | 0.0 |
| Route-weight combination | 0.000 | 0.0% | 0.0 |

Combined MoE work: 1.960 ms per step (19.6% of instrumented decode time).
The combined value is the direct MoE critical-path attribution for comparison with the reference profile.
Largest component scope: QKV and attention work at 3.917 ms per step (39.2% of instrumented decode time).
The PyTorch expert-loop scope was not active in this profile.
The fused path contains no Python expert loop.
The profile did not record an aten::nonzero operator.
CPU operator self time: 4.794 ms per step across 573.0 calls per step.
CUDA launch events are unavailable in the CPU-activity-only trace; CPU operator dispatch time and call counts are reported instead.
The fixed-shape direct path uses three Triton launches per layer and keeps both selected expert ids and weights on device.
Compare this report with an otherwise identical --moe reference run before drawing a performance conclusion.

## Limitations

- torch.profiler and record_function add CPU overhead, so component timings are reported separately from clean latency.
- The component profile collects CPU activity only because CUPTI can drop CUDA-activity buffers on virtualized GPUs; clean whole-step GPU latency still uses CUDA events.
- CPU scope time includes dispatch and synchronous GPU waits; asynchronous GPU work may be charged to a later synchronization scope, so values are critical-path attribution rather than isolated kernel durations.
- The repeated steps grow context from 62 tokens; results describe warmed early decode, not every context length.
