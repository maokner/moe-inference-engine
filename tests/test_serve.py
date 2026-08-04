"""Integration tests for the HTTP generation endpoint."""

import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from moe_engine.model import KVCache, PagedKVCache
from moe_engine.model import Model as EngineModel
from moe_engine.model import ModelConfig as EngineConfig

# The vocab must cover real gpt2 token ids, since the server tokenizes with gpt2.
TINY_SERVE = {
    "max_seq_length": 32,
    "vocab_size": 50257,
    "num_layers": 2,
    "hidden_dim": 32,
    "num_experts": 4,
    "top_k": 2,
}

_spec = importlib.util.spec_from_file_location(
    "serve", Path(__file__).parent.parent / "scripts" / "serve.py"
)
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)


@pytest.fixture()
def client():
    torch.manual_seed(0)
    serve.model = EngineModel(EngineConfig(**TINY_SERVE)).eval()
    serve.cache_mode = "contiguous"
    serve.allocator = None
    serve.device = "cpu"
    return TestClient(serve.app, raise_server_exceptions=False)


@pytest.fixture()
def paged_client(client):
    serve.cache_mode = "paged"
    serve.allocator = serve.model.new_block_allocator(
        num_blocks=2, block_size=16, attention_mode="gather"
    )
    return client


def test_cli_defaults_to_contiguous_and_accepts_explicit_paged_mode():
    defaults = serve.build_parser().parse_args([])
    assert defaults.cache == "contiguous"
    assert defaults.paged_attention == "auto"
    assert defaults.moe == "auto"
    assert defaults.device == "cpu"

    paged = serve.build_parser().parse_args(
        ["--cache", "paged", "--paged-attention", "gather"]
    )
    assert paged.cache == "paged"
    assert paged.paged_attention == "gather"


def test_generate_uses_contiguous_cache_by_default(client):
    seen_caches = []
    real_generate = serve.model.generate

    def spying_generate(*args, **kwargs):
        seen_caches.append(kwargs["cache"])
        return real_generate(*args, **kwargs)

    serve.model.generate = spying_generate
    resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < body["new_tokens"] <= 4 or body["finish_reason"] == "eos"
    assert isinstance(seen_caches[0], KVCache)
    assert not hasattr(seen_caches[0], "free")


def test_explicit_paged_cache_frees_blocks_after_success(paged_client):
    seen_caches = []
    real_generate = serve.model.generate

    def spying_generate(*args, **kwargs):
        seen_caches.append(kwargs["cache"])
        return real_generate(*args, **kwargs)

    serve.model.generate = spying_generate
    resp = paged_client.post("/generate", json={"prompt": "hello", "max_new_tokens": 4})
    assert resp.status_code == 200
    assert isinstance(seen_caches[0], PagedKVCache)
    assert len(serve.allocator.free_blocks) == 2


def test_finish_reason_length_and_context_limit(client):
    resp = client.post(
        "/generate",
        json={"prompt": "hello world", "max_new_tokens": 2, "temperature": 0},
    )
    assert resp.status_code == 200
    assert resp.json()["finish_reason"] == "length"

    resp = client.post(
        "/generate",
        json={"prompt": "hello world", "max_new_tokens": 64, "temperature": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["finish_reason"] == "context_limit"
    # A 2-token prompt in a 32-token context fits 32 - 2 + 1 generated tokens.
    assert body["new_tokens"] == TINY_SERVE["max_seq_length"] - 2 + 1


def test_over_length_prompt_returns_400(client):
    resp = client.post("/generate", json={"prompt": "word " * 40, "max_new_tokens": 4})
    assert resp.status_code == 400
    assert "context" in resp.json()["detail"]


def test_invalid_fields_return_422(client):
    bad_requests = [
        {"prompt": "", "max_new_tokens": 4},
        {"prompt": "hi", "max_new_tokens": 0},
        {"prompt": "hi", "max_new_tokens": -3},
        {"prompt": "hi", "temperature": -1},
    ]
    for payload in bad_requests:
        assert client.post("/generate", json=payload).status_code == 422, payload


def test_paged_validation_failure_does_not_allocate_blocks(paged_client):
    resp = paged_client.post("/generate", json={"prompt": "", "max_new_tokens": 4})
    assert resp.status_code == 422
    assert len(serve.allocator.free_blocks) == 2


def test_contiguous_generation_failure_needs_no_cleanup(client):
    def broken_generate(*args, **kwargs):
        raise RuntimeError("boom")

    serve.model.generate = broken_generate
    with pytest.raises(RuntimeError, match="boom"):
        serve.generate(serve.GenerateRequest(prompt="hello", max_new_tokens=4))


def test_paged_blocks_freed_when_generation_raises(paged_client):
    def broken_generate(*args, **kwargs):
        raise RuntimeError("boom")

    serve.model.generate = broken_generate
    resp = paged_client.post("/generate", json={"prompt": "hello", "max_new_tokens": 4})
    assert resp.status_code == 500
    assert len(serve.allocator.free_blocks) == 2


def test_concurrent_requests_are_serialized(client):
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_generate(token_ids, *args, **kwargs):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with counter_lock:
            active -= 1
        return token_ids.unsqueeze(0)

    serve.model.generate = slow_generate
    payload = {"prompt": "hello", "max_new_tokens": 4}
    with ThreadPoolExecutor(2) as pool:
        second_client = TestClient(serve.app, raise_server_exceptions=False)
        responses = list(
            pool.map(
                lambda c: c.post("/generate", json=payload), [client, second_client]
            )
        )

    assert all(r.status_code == 200 for r in responses)
    assert max_active == 1  # generate_lock keeps decodes one at a time.
