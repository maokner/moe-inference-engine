# miniMoE versus vLLM on one RTX A6000

## Conclusion

Faithful optimized vLLM is faster than the custom miniMoE engine for this single-request workload.
Its median aggregate throughput is 305.46 tokens/s versus 96.63 tokens/s, a 3.16x result or 216.10% higher throughput.
Its median total generation time is 209.52 ms versus 662.30 ms, a 68.36% reduction.
The paired 95% confidence intervals exclude zero for every reported optimized-vLLM comparison, so the data support this conclusion despite temporal GPU drift.

Eager vLLM is slower than the custom engine after the first token.
It has a faster median TTFT but 39.71% lower median aggregate throughput and 65.86% higher median total generation time.

## Workload and environment

- Source commit: `f9eccd636f354436bebfedc9fa2fec715f58afc3`
- GPU: NVIDIA RTX A6000 with 49,140 MiB and driver 610.43.02
- Python: 3.12.13
- PyTorch: 2.9.1+cu128 with CUDA 12.8
- Triton: 3.5.1
- Transformers: 4.57.6
- vLLM: 0.14.1
- Checkpoint SHA-256: `f8f9e91d05f00fe1d1579dc2d5ce0f2f728d0183f9424733fd93542e979d9529`
- One active request, batch size one, FP32, 62 prompt tokens, 64 greedy tokens with EOS ignored, and a 1,024-token context limit
- Twenty-one paired outer rounds with rotating first-system order and a fresh fully warmed process for every system-round

## Median performance

| System | TTFT | Mean ITL | Total time | Throughput | Peak GPU memory | Incremental GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Custom engine | 118.37 ms | 8.721 ms | 662.30 ms | 96.63 tok/s | 2,330.38 MiB | 1,458.00 MiB |
| Optimized vLLM | 23.44 ms | 2.987 ms | 209.52 ms | 305.46 tok/s | 2,467.69 MiB | 1,595.31 MiB |
| Eager vLLM | 35.61 ms | 16.751 ms | 1,098.49 ms | 58.26 tok/s | 2,589.69 MiB | 1,717.31 MiB |

## Paired candidate-minus-baseline results

All intervals are two-sided paired Student-t 95% confidence intervals over 21 per-round deltas.

| Comparison | Metric | Mean delta | 95% CI |
|---|---|---:|---:|
| Optimized vLLM minus custom engine | TTFT | -90.665 ms | [-99.635, -81.694] ms |
| Optimized vLLM minus custom engine | Mean ITL | -5.734 ms | [-6.370, -5.099] ms |
| Optimized vLLM minus custom engine | Total time | -451.932 ms | [-497.919, -405.945] ms |
| Optimized vLLM minus custom engine | Throughput | +202.694 tok/s | [+195.502, +209.885] tok/s |
| Eager vLLM minus custom engine | TTFT | -78.140 ms | [-87.215, -69.064] ms |
| Eager vLLM minus custom engine | Mean ITL | +7.850 ms | [+7.114, +8.585] ms |
| Eager vLLM minus custom engine | Total time | +416.382 ms | [+364.966, +467.799] ms |
| Eager vLLM minus custom engine | Throughput | -38.659 tok/s | [-44.575, -32.742] tok/s |
| Optimized vLLM minus eager vLLM | TTFT | -12.525 ms | [-14.496, -10.554] ms |
| Optimized vLLM minus eager vLLM | Mean ITL | -13.584 ms | [-14.131, -13.037] ms |
| Optimized vLLM minus eager vLLM | Total time | -868.314 ms | [-903.241, -833.387] ms |
| Optimized vLLM minus eager vLLM | Throughput | +241.352 tok/s | [+235.194, +247.511] tok/s |

## Correctness gate

The converted checkpoint values match the source checkpoint exactly.
Hugging Face and the custom engine have zero direct logit difference on the validation input.
Against the original `nn.MultiheadAttention` reference on CUDA, both have maximum absolute logit error `3.44038e-4` and mean absolute error `2.23860e-5`.

The native optimized-vLLM full 50,304-entry normalized next-token distribution has maximum absolute error `1.82190e-2` and mean absolute error `6.48891e-3` against Hugging Face.
The eager-vLLM distribution has maximum absolute error `1.73550e-2` and mean absolute error `6.30710e-3`.
Both pass the evidence-based `atol=2e-2, rtol=2e-4` gate.

All 64 greedy token IDs match exactly across the original model, Hugging Face model, custom engine, optimized vLLM, and eager vLLM.
Every one of the 63 timed system-round records also contains the same exact 64-token sequence.

## Memory policy and run health

vLLM reserves exactly 75,497,472 KV-cache bytes, twice the 37,748,736-byte FP32 minimum for one 1,024-token sequence.
The fixed reservation is reported separately from whole-device peak and incremental memory.
The 0.5 utilization value is only a vLLM 0.14.1 startup free-memory guard and does not size or reserve the cache.
The previous 90% utilization-based reservation policy is not used.

The initial smoke exposed normal vLLM `DELTA` output coalescing and was retained as failed-smoke evidence.
Commit `f9eccd6` records co-delivered tokens at one user-visible timestamp without disabling asynchronous scheduling.
The repeated smoke and complete 21-round benchmark passed.
The full benchmark had zero failed or retried rounds.
No stream events happened to coalesce in the final 21-round sample, but raw chunk sizes and event counts are present in every record.

The full A6000 test suite passed with 78 tests passed and 1 skipped.
An independent local audit verified round order, child-record identity, finite metrics, stream coverage, token equality, numerical-validation status, and 21-sample paired statistics.
All copied remote artifacts passed the recorded SHA-256 manifest.

## Remaining runtime caveat

vLLM 0.14.1 reports that it has no tuned `E=8,N=3072` FP32 FusedMoE configuration file for the RTX A6000 and therefore uses its default MoE configuration.
That is the actual default optimized-vLLM runtime requested for this comparison, but a future vLLM release or an upstream tuning profile could change the result.
