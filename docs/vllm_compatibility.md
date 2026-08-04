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
The native implementation uses vLLM `Attention` for all attention work and vLLM `FusedMoE` with `activation="gelu"`, `is_act_and_mul=False`, and `has_bias=True`.
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
  --output results/vllm_comparison/parity.json
```

The validator requires exact converted checkpoint values, Hugging Face versus original logits within `atol=1e-5, rtol=1e-5`, engine versus original logits within `atol=1e-4, rtol=1e-4`, and exact equality of all 64 greedy tokens.

Run all three benchmarks in isolated processes:

```bash
uv sync --extra vllm
uv run python benchmarks/vllm_compare.py \
  --system all \
  --device cuda \
  --checkpoint checkpoints/minimoe_sft.pt \
  --model-dir checkpoints/minimoe-hf \
  --output results/vllm_comparison/comparison.json
```

The harness fixes one active request, batch size one, FP32, the same local tokenizer, the canonical 62-token prompt, 64 greedy tokens with EOS ignored, and a 1,024-token context limit.
It records time to first token, every inter-token latency, mean inter-token latency, total generation time, aggregate generated tokens per second, whole-device peak and incremental GPU memory, and generated-token equality.

No local MPS, local vLLM, or CUDA model workload is part of the implementation test suite.
Runtime validation remains pending until the parent task authorizes the A6000 run.
