"""Tests for the batched decode path and the continuous batching engine.

The tiny model's vocabulary (96) cannot contain the gpt2 EOS id (50256),
so greedy runs here always stop on max_new_tokens or the context limit.
Tests drive `engine.step()` directly for deterministic scheduling.
"""

import pytest
import torch

import moe_engine.engine
from moe_engine.engine import Engine, Request
from moe_engine.model import Model, ModelConfig

TINY = dict(max_seq_length=32, vocab_size=96, num_layers=2, hidden_dim=32, num_experts=4, top_k=2)


def make_model():
    torch.manual_seed(0)
    return Model(ModelConfig(**TINY)).eval()


def make_engine(model, num_blocks=32, block_size=4, max_batch_size=4):
    allocator = model.new_block_allocator(num_blocks=num_blocks, block_size=block_size)
    return Engine(model, allocator, max_batch_size=max_batch_size)


def greedy_request(prompt, max_new_tokens):
    return Request(prompt_ids=prompt.tolist(), max_new_tokens=max_new_tokens, temperature=0)


def step_until_done(engine, requests, limit=64):
    for _ in range(limit):
        if all(r.done.is_set() for r in requests):
            return
        engine.step()
    raise AssertionError("engine did not finish within the step limit")


def expected_new_ids(model, prompt, max_new_tokens):
    out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
    return out[0, len(prompt) :].tolist()


def test_forward_decode_matches_single_sequence_forward():
    """A batched decode step must equal each sequence decoding alone."""
    model = make_model()
    allocator = model.new_block_allocator(num_blocks=32, block_size=4)
    lengths = [5, 9, 3]
    prompts = [torch.randint(0, TINY["vocab_size"], (1, n)) for n in lengths]

    with torch.no_grad():
        single_caches = [allocator.new_cache() for _ in prompts]
        batch_caches = [allocator.new_cache() for _ in prompts]
        for prompt, single, batched in zip(prompts, single_caches, batch_caches):
            model(prompt, single)
            model(prompt, batched)

        for _ in range(3):
            tokens = torch.randint(0, TINY["vocab_size"], (len(prompts), 1))
            expected = torch.cat(
                [model(tokens[i : i + 1], single_caches[i]) for i in range(len(prompts))]
            )
            got = model.forward_decode(tokens, batch_caches)
            torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_engine_matches_sequential_generate():
    """Requests decoded together must produce the same tokens as one-at-a-time generate()."""
    model = make_model()
    engine = make_engine(model)
    prompts = [torch.randint(0, TINY["vocab_size"], (n,)) for n in [4, 7, 5]]
    expected = [expected_new_ids(model, p, 8) for p in prompts]

    requests = [engine.submit(greedy_request(p, 8)) for p in prompts]
    step_until_done(engine, requests)

    for request, want in zip(requests, expected):
        assert request.error is None
        assert request.new_ids == want
        assert request.finish_reason == "length"
    assert len(engine.allocator.free_blocks) == engine.allocator.num_blocks
    assert engine.reserved_blocks == 0


def test_requests_join_and_leave_mid_flight():
    """The batch must change composition between steps without corrupting outputs."""
    model = make_model()
    engine = make_engine(model)
    long_prompt = torch.randint(0, TINY["vocab_size"], (6,))
    short_prompt = torch.randint(0, TINY["vocab_size"], (4,))
    want_long = expected_new_ids(model, long_prompt, 12)
    want_short = expected_new_ids(model, short_prompt, 2)

    long_req = engine.submit(greedy_request(long_prompt, 12))
    engine.step()
    engine.step()  # The long request is now mid-decode.
    short_req = engine.submit(greedy_request(short_prompt, 2))
    step_until_done(engine, [short_req])

    assert short_req.new_ids == want_short  # Joined a busy batch, still exact.
    assert not long_req.done.is_set()  # And left it early without disturbing it.
    step_until_done(engine, [long_req])
    assert long_req.new_ids == want_long


def test_admission_waits_for_free_blocks():
    """A request that does not fit must wait until a running one releases blocks."""
    model = make_model()
    # Worst case per request: 4 + 8 - 1 = 11 tokens -> 3 blocks of 4.
    engine = make_engine(model, num_blocks=4, block_size=4)
    prompts = [torch.randint(0, TINY["vocab_size"], (4,)) for _ in range(2)]
    expected = [expected_new_ids(model, p, 8) for p in prompts]

    first = engine.submit(greedy_request(prompts[0], 8))
    second = engine.submit(greedy_request(prompts[1], 8))
    engine.step()
    assert len(engine.running) == 1  # Only 1 free block remains: second waits.
    assert list(engine.waiting) == [second]

    step_until_done(engine, [first, second])
    assert first.new_ids == expected[0]
    assert second.new_ids == expected[1]
    assert len(engine.allocator.free_blocks) == 4


def test_batch_size_cap_limits_admission():
    model = make_model()
    engine = make_engine(model, max_batch_size=2)
    requests = [
        engine.submit(greedy_request(torch.randint(0, TINY["vocab_size"], (4,)), 8))
        for _ in range(3)
    ]
    engine.step()
    assert len(engine.running) == 2
    assert len(engine.waiting) == 1
    step_until_done(engine, requests)
    assert all(r.error is None for r in requests)


def test_eos_finishes_request_without_appending(monkeypatch):
    """An EOS sample must end the request, excluded from the output."""
    model = make_model()
    engine = make_engine(model)
    eos = torch.tensor([[moe_engine.engine.EOS_TOKEN_ID]])
    monkeypatch.setattr(
        moe_engine.engine, "sample_next_token", lambda logits, temperature, top_k: eos
    )

    request = engine.submit(greedy_request(torch.randint(0, TINY["vocab_size"], (4,)), 8))
    engine.step()
    assert request.done.is_set()
    assert request.finish_reason == "eos"
    assert request.new_ids == []
    assert len(engine.allocator.free_blocks) == engine.allocator.num_blocks


def test_failed_decode_reports_error_and_frees_blocks(monkeypatch):
    model = make_model()
    engine = make_engine(model)

    def broken_forward_decode(token_ids, caches):
        raise RuntimeError("boom")

    monkeypatch.setattr(model, "forward_decode", broken_forward_decode)
    request = engine.submit(greedy_request(torch.randint(0, TINY["vocab_size"], (4,)), 8))
    engine.step()

    assert request.done.is_set()
    assert isinstance(request.error, RuntimeError)
    assert len(engine.allocator.free_blocks) == engine.allocator.num_blocks
    assert engine.reserved_blocks == 0


def test_failed_sampling_reports_error_and_engine_survives(monkeypatch):
    """A sampling crash must fail that request cleanly, not kill the engine."""
    model = make_model()
    engine = make_engine(model)

    def broken_sample(logits, temperature, top_k):
        raise RuntimeError("probability tensor contains nan")

    monkeypatch.setattr(moe_engine.engine, "sample_next_token", broken_sample)
    doomed = engine.submit(greedy_request(torch.randint(0, TINY["vocab_size"], (4,)), 8))
    engine.step()
    assert doomed.done.is_set()
    assert isinstance(doomed.error, RuntimeError)
    assert len(engine.allocator.free_blocks) == engine.allocator.num_blocks

    monkeypatch.undo()  # The next request must run normally on the same engine.
    prompt = torch.randint(0, TINY["vocab_size"], (4,))
    request = engine.submit(greedy_request(prompt, 4))
    step_until_done(engine, [request])
    assert request.error is None
    assert request.new_ids == expected_new_ids(model, prompt, 4)


def test_scheduler_crash_fails_all_requests_and_keeps_accounting(monkeypatch):
    """run_forever's crash handler must fail queued requests without
    corrupting reserved_blocks (they never reserved anything)."""
    model = make_model()
    engine = make_engine(model)
    requests = [
        engine.submit(greedy_request(torch.randint(0, TINY["vocab_size"], (4,)), 8))
        for _ in range(3)
    ]

    def broken_worst_case(request):
        raise RuntimeError("scheduler bug")

    monkeypatch.setattr(engine, "_worst_case_blocks", broken_worst_case)
    engine.start()
    for request in requests:
        assert request.done.wait(timeout=5)
        assert isinstance(request.error, RuntimeError)
    engine.stop()
    assert engine.reserved_blocks == 0
    assert len(engine.allocator.free_blocks) == engine.allocator.num_blocks

    # Accounting intact: a fresh request must still be admittable.
    monkeypatch.undo()
    prompt = torch.randint(0, TINY["vocab_size"], (4,))
    revived = Request(prompt_ids=prompt.tolist(), max_new_tokens=4, temperature=0)
    engine.stopped = False
    engine.submit(revived)
    step_until_done(engine, [revived])
    assert revived.error is None
    assert revived.new_ids == expected_new_ids(model, prompt, 4)


def test_submit_rejects_invalid_requests():
    model = make_model()
    engine = make_engine(model, num_blocks=2, block_size=4)  # Pool holds 8 tokens.
    good = torch.randint(0, TINY["vocab_size"], (4,))
    with pytest.raises(ValueError, match="at least one token"):
        engine.submit(Request(prompt_ids=[], max_new_tokens=4))
    with pytest.raises(ValueError, match="max_seq_length"):
        engine.submit(Request(prompt_ids=[0] * 33, max_new_tokens=4))
    with pytest.raises(ValueError, match="temperature"):
        engine.submit(Request(prompt_ids=good.tolist(), max_new_tokens=4, temperature=-1))
    with pytest.raises(ValueError, match="more KV blocks than the pool"):
        # 4 + 32 - 1 tokens worst case can never fit in an 8-token pool.
        engine.submit(Request(prompt_ids=good.tolist(), max_new_tokens=32))
    assert not engine.waiting  # Nothing invalid was queued.


def test_context_limit_finishes_cleanly():
    """A request that outgrows the context must stop with the generate() rules."""
    model = make_model()
    engine = make_engine(model, num_blocks=8, block_size=4)
    prompt = torch.randint(0, TINY["vocab_size"], (TINY["max_seq_length"] - 2,))
    want = expected_new_ids(model, prompt, 10)

    request = engine.submit(greedy_request(prompt, 10))
    step_until_done(engine, [request])
    assert request.finish_reason == "context_limit"
    assert request.new_ids == want
    # The last emitted token never enters the cache, so max - prompt + 1 fit.
    assert len(request.new_ids) == 3
