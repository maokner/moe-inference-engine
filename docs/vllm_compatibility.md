# miniMoE Hugging Face and vLLM compatibility

## Architecture contract

The conversion preserves miniMoE as its own `MiniMoEForCausalLM` architecture.
It does not identify the model as Mixtral, Grok, or another architecture.

The model uses learned absolute positional embeddings, six pre-normalized decoder blocks, eight full attention heads, and a 1,024-token context limit.
Every sparse block uses eight biased `Linear(768, 3072) -> GELU -> Linear(3072, 768)` experts and a learned top-2 router.
The router computes top-k over FP32 logits, normalizes only the selected two logits, and takes their weighted sum.
The exact 50,257-token GPT-2 tiktoken vocabulary is preserved while the model's 50,304-row padded token embedding is tied to a biased output projection.

The converted state dict uses one deterministic prefix mapping.

| Source prefix | Hugging Face prefix |
|---|---|
| `token_embedding.` | `model.token_embedding.` |
| `positional_embedding.` | `model.positional_embedding.` |
| `MoEBlocks.` | `model.MoEBlocks.` |
| `final_norm.` | `model.final_norm.` |
| `output_projection.` | `lm_head.` |

Keys are processed in lexical order, tensors remain FP32, and no tensor is transposed, concatenated, split, or otherwise changed.
The converter rejects a checkpoint unless its token and output weights are bitwise equal.
The generated `conversion_manifest.json` records the source SHA-256 and every key mapping.

## Transformers backend inspection

The initial Hugging Face implementation follows vLLM's Transformers modeling backend contract.
`MiniMoEModel` advertises `_supports_attention_backend = True`, every attention block has a stable `layer_idx`, and attention calls `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`.
Keyword arguments flow from the base model to every attention block.
The sparse block exposes an `experts` `nn.ModuleList` subclass whose forward signature is `(hidden_states, top_k_index, top_k_weights)`.

Inspection of vLLM 0.14.1 confirms that its Transformers wrapper sets `_attn_implementation = "vllm"`, registers a `vllm_flash_attention_forward` function, and routes the projected Q, K, and V tensors through vLLM `Attention` instances.
The attention path is therefore replaced by vLLM's paged KV cache and selected attention backend.

The same inspection found that the generic Transformers MoE path is not faithful for miniMoE.

- vLLM 0.14.1 guards generic MoE support behind Transformers 5.0.0.dev0, while the vLLM 0.14.1 package resolves against Transformers 4.x.
- The generic replacement chooses `silu` for an unfamiliar architecture and constructs `FusedMoE` with its default gated activation layout.
- Its checkpoint loader recognizes three-projection expert conventions such as `gate_proj/down_proj/up_proj`, `w1/w2/w3`, and Grok's three linear layers.
- miniMoE has exactly two expert projections with an ordinary non-gated GELU between them.
- The generic causal wrapper creates `ParallelLMHead` without a bias and skips the complete `lm_head` prefix when embeddings are tied, but miniMoE has a learned output bias.

Changing the architecture string to `Grok1` would still select a gated GELU operation and would be an architecture substitution, so it is explicitly rejected.

The inspected upstream sources are:

- [Transformers backend base, v0.14.1](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/models/transformers/base.py)
- [Transformers backend MoE replacement, v0.14.1](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/models/transformers/moe.py)
- [Transformers causal wrapper, v0.14.1](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/models/transformers/causal.py)
- [vLLM FusedMoE layer, v0.14.1](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/layers/fused_moe/layer.py)

## Native plugin fallback

The `vllm.general_plugins` entry point registers `MiniMoEForCausalLM` lazily.
The native implementation uses vLLM `Attention` for all attention work and vLLM `FusedMoE` with the required `gelu_no_mul` activation, `is_act_and_mul=False`, and `has_bias=True`.
In vLLM 0.14.1 the non-gated flag controls projection packing, while the separate no-multiply activation name controls the fused kernel's output width.
The plugin also loads expert biases explicitly because the release's generic fused-MoE weight loader silently declines bias tensors.
It uses vLLM linear, embedding, output-head, and logits components for the remaining runtime-sensitive operations.
It never imports or invokes `moe_engine.fused_moe` or `moe_engine.paged_attention`.

The plugin targets vLLM 0.14.x because native model internals are not covered by vLLM's plugin compatibility guarantee.
The benchmark requests `model_impl="vllm"` so a silent fallback to the incompatible generic Transformers MoE path cannot occur.

## Conversion and validation

Run conversion on any CPU-only host:

```bash
uv run python scripts/convert_to_hf.py \
  --checkpoint checkpoints/minimoe_sft.pt \
  --output checkpoints/minimoe-hf
```

Run full real-checkpoint parity on an authorized CUDA host:

```bash
uv run python scripts/validate_hf_parity.py \
  --checkpoint checkpoints/minimoe_sft.pt \
  --model-dir checkpoints/minimoe-hf \
  --device cuda \
  --validate-native-vllm \
  --output results/vllm_comparison/parity.json
```

The validator requires exact converted checkpoint values and keeps the original CPU tolerances of `atol=1e-5, rtol=1e-5` for Hugging Face and `atol=1e-4, rtol=1e-4` for the engine.
On CUDA, both the Hugging Face and engine SDPA paths use `atol=5e-4, rtol=1e-4` against the vendored `nn.MultiheadAttention` reference because the real-checkpoint A6000 diagnostic measured a shared maximum error of `3.44038e-4`, mean error of `2.23860e-5`, and zero argmax differences.
Hugging Face and engine logits are still compared directly with `atol=1e-5, rtol=1e-5`.
For both native vLLM modes it requests all 50,304 normalized next-token log-probabilities and compares them with the Hugging Face FP32 distribution using `atol=2e-2, rtol=2e-4`.
An untimed A6000 tolerance probe measured maximum and mean absolute normalized log-probability errors of `1.82190e-2` and `6.48891e-3` for optimized vLLM, and `1.73550e-2` and `6.30710e-3` for eager vLLM.
Both complete 64-token greedy sequences matched during that probe.
The absolute tolerance retains a narrow margin over the observed fused FP32 kernel error while still checking the complete attention, router, biased GELU expert, tied output-weight, and learned output-bias path.
All five 64-token greedy sequences must match exactly.

Run all three benchmarks in isolated processes:

```bash
uv sync --extra vllm
uv run python benchmarks/vllm_compare.py \
  --system all \
  --device cuda \
  --checkpoint checkpoints/minimoe_sft.pt \
  --model-dir checkpoints/minimoe-hf \
  --rounds 21 \
  --output results/vllm_comparison/comparison.json
```

The harness fixes one active request, batch size one, FP32, the same local tokenizer, the canonical 62-token prompt, 64 greedy tokens with EOS ignored, and a 1,024-token context limit.
It runs the untimed native-vLLM numerical validator before benchmarking and refuses to publish a performance summary if any round has a token mismatch.
Each of the default 21 rounds rotates the first system, and each system-round runs in a fresh process with a complete warmup before its one measured request.
vLLM may merge adjacent `DELTA` outputs when its producer gets ahead of the consumer.
The harness accepts every non-empty chunk, assigns one observed wall-clock delivery timestamp to all tokens in that chunk, and records chunk sizes plus the number of coalesced events in the raw metrics.
Co-delivered tokens therefore have zero user-visible interval without disabling vLLM's default asynchronous scheduling.
The report retains raw per-round results and provides paired per-round deltas with two-sided Student-t 95% confidence intervals for time to first token, mean inter-token latency, total generation time, and throughput.

The minimum FP32 KV capacity is `1024 positions * 6 layers * 8 heads * 96 head dimensions * 2 for K and V * 4 bytes = 37,748,736 bytes`.
vLLM receives a fixed default reservation of 75,497,472 bytes, which is a two-times safety margin and can be changed with `--kv-cache-memory-bytes`.
vLLM 0.14.1 still checks `gpu_memory_utilization` against free memory during early startup, even though its fixed-byte cache branch later ignores that value.
The harness sets the startup-only guard to 0.5 so a preceding isolated validation process cannot trip the default 0.9 free-memory check.
The report marks that guard as non-reserving and records the fixed reserved KV bytes separately from sampled whole-device peak and incremental memory.
The comparison does not use the previous 90-percent utilization reservation policy.

No local MPS, local vLLM, or CUDA model workload is part of the implementation test suite.
Runtime validation remains pending until the parent task authorizes the A6000 run.
