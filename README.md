# miniMoE inference engine

A low-latency, single-request inference engine for [miniMoE](https://github.com/maokner/miniMoE), built with PyTorch and Triton.

miniMoE is a 280M-parameter sparse MoE model with 110M active parameters per token.
The weights are available on [Hugging Face](https://huggingface.co/mokner123/miniMoE).

## What is implemented

- Contiguous KV caching for the default single-request path.
- Paged KV caching with direct Triton attention.
- Fused Triton MoE decode for batch-one CUDA inference.
- A synchronous HTTP server with no continuous batching.

## A6000 results

All benchmarks use the real checkpoint and produce identical greedy tokens.

| KV cache | Median latency | Throughput |
|---|---:|---:|
| Contiguous | **62.0 ms** | **15.34 tok/s** |
| Paged, gather | 66.1 ms | 14.58 tok/s |
| Paged, direct Triton | 63.0 ms | 14.97 tok/s |

| MoE decode | Median latency | Throughput |
|---|---:|---:|
| PyTorch | 53.11 ms | 19.18 tok/s |
| Triton | **7.56 ms** | **130.87 tok/s** |

The Triton MoE path improves throughput by **582%**.

Full benchmark output is in [`results/`](results/).

## Run

Place `minimoe_sft.pt` in `checkpoints/`, then run:

```bash
uv sync --extra server
uv run python scripts/serve.py --device cuda
```

Optional modes:

```bash
--cache paged --paged-attention direct
--moe reference|direct|auto
```

## Hugging Face and vLLM comparison

The repository now includes a deterministic Hugging Face checkpoint converter, an exact local tiktoken tokenizer, a Transformers oracle, and a vLLM 0.14.x out-of-tree model plugin.
The plugin uses vLLM's own attention and fused MoE runtime and does not reuse this engine's Triton kernels.

See [`docs/vllm_compatibility.md`](docs/vllm_compatibility.md) for the compatibility inspection, native-plugin numerical gate, and the 21-round paired benchmark with rotating system order and fixed KV-cache memory.
