# Fused MoE A6000 validation

## Environment

- GPU: NVIDIA RTX A6000 with 49,140 MiB.
- Driver: 610.43.02.
- Runtime: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.
- Checkpoint: step 6358, SHA-256 `f8f9e91d05f00fe1d1579dc2d5ce0f2f728d0183f9424733fd93542e979d9529`.
- Benchmark source: tracked-clean commit `f04346ace6962cfd8070dbe31b9da27879d8d9dd`.
- Workload: one active request, contiguous KV cache, fixed 62-token prompt, and 64 repeated one-token greedy decode steps per round.

## Correctness

The full CUDA suite completed with 61 passed and one expected capability-gated skip in 30.40 seconds.
The skipped paged-attention negative test is intentionally skipped when the current machine supports the direct kernel.

The router kernel matched PyTorch top-2 ids exactly.
Router-logit maximum and mean absolute errors were `1.7881e-7` and `6.7055e-8`.
Normalized routing-weight maximum and mean absolute errors were `8.9407e-8` and `4.4703e-8`, and the direct weights summed to `0.99999988`.

Every expert was exercised as both the first and second route with distinct logits, weights, W1 biases, and W2 biases.
All requested route ids were correct.
The worst complete selected-expert output had maximum absolute error `5.9605e-8` and mean absolute error `9.5849e-9` against the exact PyTorch GELU oracle.

With the real checkpoint and fixed 62-token prompt, one-token direct-versus-reference logits had maximum absolute error `1.6212e-5` and mean absolute error `2.8136e-6`.
The next greedy token matched, and all 64 generated greedy tokens matched exactly.
Independent contiguous and direct-paged parity runs also matched all 64 greedy tokens.

The real checkpoint retained every original state-dict key.
Ordinary `load_state_dict`, `load_state_dict(assign=True)`, CPU-to-CUDA conversion, and the loaded CUDA model all retained contiguous canonical storage and correct per-expert pointer strides.
Each layer has exactly four unique expert storages whose total bytes equal the logical expert parameter bytes, so the model retains no persistent full duplicate packed expert weights.

Paged and contiguous cache regression tests confirmed that a rejected forced-direct decode leaves cache position and K/V state unchanged.
The paged test starts on a block boundary and also confirms the Python block table, free-block set, device mirror, and allocator K/V pools remain unchanged.

The reference and direct CUDA HTTP servers returned identical deterministic text, token count, and finish reason for the fixed prompt and 64-token request.

## Interleaved benchmark

The primary throughput metric is total completed decode steps divided by total decode time.
Median and p90 latency are reported separately and are not inverted to estimate throughput.

| MoE mode | Steps | Total decode seconds | Median ms | p90 ms | Aggregate tok/s |
|---|---:|---:|---:|---:|---:|
| Reference | 1,344 | 70.064978 | 53.110 | 60.028 | 19.182 |
| Direct Triton | 1,344 | 10.270138 | 7.555 | 8.193 | 130.865 |

Direct aggregate throughput was 582.23% higher than reference throughput.
Mean latency derived from the aggregate totals fell from 52.132 ms to 7.641 ms, an 85.34% reduction.
Across the 21 paired rounds, the direct-minus-reference mean latency delta was -44.490 ms per step with a paired 95% confidence interval of [-46.612, -42.368] ms.
Expressed per round as a relative latency change, the paired mean was -85.219% with a 95% confidence interval of [-86.009%, -84.429%].
Every paired round retained identical greedy token ids.

## Profile comparison

The clean profile is separate from the instrumented component profile.

| Metric | Reference | Direct Triton |
|---|---:|---:|
| Clean median decode latency | 55.147 ms | 6.841 ms |
| Clean throughput | 18.12 tok/s | 142.97 tok/s |
| Instrumented total | 58.001 ms/step | 9.994 ms/step |
| Combined MoE | 45.869 ms/step, 79.1% | 1.960 ms/step, 19.6% |
| `aten::nonzero` | 48 calls/step, 27.295 ms/step | 0 calls/step |
| CPU operator calls | 1,659/step | 573/step |

The direct MoE path issues exactly three Triton launches per layer, or 18 direct-MoE launches across six layers.
CUDA-activity launch totals are not reported because CUPTI dropped buffers on this virtualized GPU; the preserved profiler therefore uses CPU activity and reports operator calls consistently for both modes.
The dominant component moved from MoE routing at 39.969 ms/step and 68.9% to QKV and attention work at 3.917 ms/step and 39.2%.
No focused MoE microbenchmark was needed because the end-to-end result and component attribution agree and expose no remaining MoE regression.

## Limitations

- The virtualized A6000 showed round-to-round drift, which is why systems were interleaved and starting order rotated.
- Profile component timings include profiler overhead and are not throughput measurements.
- Decode context grows from the 62-token prompt during each run, so results describe warmed early decode rather than every context length.
