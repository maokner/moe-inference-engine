"""Shared, import-safe vLLM runtime configuration for miniMoE."""

from __future__ import annotations

from pathlib import Path

MAX_MODEL_LEN = 1024
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 8
HIDDEN_SIZE = 768
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
FP32_BYTES = 4

# One sequence stores K and V for every layer, head, and context position.
MINIMUM_KV_CACHE_MEMORY_BYTES = (
    MAX_MODEL_LEN * NUM_HIDDEN_LAYERS * NUM_ATTENTION_HEADS * HEAD_DIM * 2 * FP32_BYTES
)
KV_CACHE_SAFETY_FACTOR = 2
DEFAULT_KV_CACHE_MEMORY_BYTES = MINIMUM_KV_CACHE_MEMORY_BYTES * KV_CACHE_SAFETY_FACTOR
# vLLM 0.14.1 checks this legacy utilization value at startup even when the
# fixed-byte KV-cache path later ignores it. Keep the check below the available
# memory without allowing it to size or reserve the cache.
STARTUP_MEMORY_GUARD_UTILIZATION = 0.5


def validate_kv_cache_memory_bytes(value: int) -> int:
    if value < MINIMUM_KV_CACHE_MEMORY_BYTES:
        raise ValueError(
            "KV-cache capacity is too small for one 1,024-token FP32 miniMoE "
            f"sequence: {value} < {MINIMUM_KV_CACHE_MEMORY_BYTES} bytes"
        )
    return value


def async_engine_kwargs(
    model_dir: str | Path,
    *,
    enforce_eager: bool,
    kv_cache_memory_bytes: int,
    max_logprobs: int = 20,
) -> dict:
    """Return the exact vLLM 0.14.x arguments without importing vLLM."""
    validate_kv_cache_memory_bytes(kv_cache_memory_bytes)
    return {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "tokenizer_mode": "auto",
        "trust_remote_code": True,
        "model_impl": "vllm",
        "dtype": "float32",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 1,
        "max_num_batched_tokens": MAX_MODEL_LEN,
        "enable_prefix_caching": False,
        "enforce_eager": enforce_eager,
        "disable_log_stats": True,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "gpu_memory_utilization": STARTUP_MEMORY_GUARD_UTILIZATION,
        "swap_space": 0,
        "max_logprobs": max_logprobs,
    }
