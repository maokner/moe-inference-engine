"""Continuous batching engine: one decode loop shared by all requests.

Requests join the running batch between decode steps and leave the moment
they finish, returning their KV blocks to the pool for queued requests.
This is iteration-level scheduling as introduced by Orca and used in vLLM.

Admission is conservative: a request is only admitted once the pool can
cover its worst-case block count, so decoding can never run out of blocks
mid-flight. vLLM instead admits optimistically and preempts sequences when
the pool fills; that serves more concurrent requests but needs swapping or
recomputation machinery we do not want yet.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass, field

import torch

from moe_engine.model import EOS_TOKEN_ID, PagedKVCache, sample_next_token


@dataclass
class Request:
    """One generation request; `done` fires when it finishes or fails."""

    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    new_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # "eos", "length", or "context_limit".
    error: Exception | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cache: PagedKVCache | None = None  # Owned by the engine while running.
    reserved: int = 0  # Blocks reserved at admission; 0 until admitted.


class Engine:
    """Owns the model, the block pool, and the scheduling loop.

    `submit()` is called from any thread; `step()` runs one scheduling
    iteration. The server runs `start()` for a background loop, while tests
    drive `step()` directly for deterministic control.
    """

    def __init__(self, model, allocator, max_batch_size: int = 8):
        self.model = model
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.reserved_blocks = 0  # Worst-case blocks promised to running requests.
        self.has_work = threading.Condition()
        self.stopped = False

    def _worst_case_blocks(self, request: Request) -> int:
        # The final sampled token is returned without entering the cache,
        # so a finished request caches at most prompt + max_new - 1 tokens.
        tokens = min(
            len(request.prompt_ids) + request.max_new_tokens - 1,
            self.model.max_seq_length,
        )
        return -(-tokens // self.allocator.block_size)

    def submit(self, request: Request) -> Request:
        """Validate and queue a request. Safe to call from any thread."""
        if len(request.prompt_ids) == 0:
            raise ValueError("prompt must contain at least one token")
        if len(request.prompt_ids) > self.model.max_seq_length:
            raise ValueError(
                f"prompt is {len(request.prompt_ids)} tokens; "
                f"max_seq_length is {self.model.max_seq_length}"
            )
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not math.isfinite(request.temperature) or request.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if request.top_k is not None and request.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._worst_case_blocks(request) > self.allocator.num_blocks:
            raise ValueError("request needs more KV blocks than the pool holds")
        with self.has_work:
            self.waiting.append(request)
            self.has_work.notify()
        return request

    @torch.no_grad()
    def step(self) -> None:
        """One scheduling iteration: admit, prefill, then decode one token."""
        admitted = []
        with self.has_work:
            # First come, first served: never skip the head of the queue,
            # so a large request cannot starve behind smaller ones.
            while self.waiting:
                request = self.waiting[0]
                if len(self.running) + len(admitted) >= self.max_batch_size:
                    break
                needed = self._worst_case_blocks(request)
                if self.reserved_blocks + needed > self.allocator.num_blocks:
                    break
                request.reserved = needed
                self.reserved_blocks += needed
                admitted.append(self.waiting.popleft())

        for request in admitted:
            self._prefill(request)
        if self.running:
            self._decode_step()

    def _device(self):
        return self.model.token_embedding.weight.device

    def _prefill(self, request: Request) -> None:
        """Run the whole prompt through the model and sample its first token."""
        request.cache = self.allocator.new_cache()
        try:
            ids = torch.tensor([request.prompt_ids], device=self._device())
            logits = self.model(ids, request.cache)
            self._append_sampled(request, logits)
        except Exception as error:  # noqa: BLE001 - the request must not hang.
            self._finish(request, error=error)
            return
        if request.finish_reason is None:
            self.running.append(request)
        else:
            self._finish(request)

    def _decode_step(self) -> None:
        """Advance every running sequence by one token in a single forward."""
        batch = self.running
        token_ids = torch.tensor(
            [[request.new_ids[-1]] for request in batch], device=self._device()
        )
        try:
            logits = self.model.forward_decode(token_ids, [r.cache for r in batch])
        except Exception as error:  # noqa: BLE001
            # A failed batched forward leaves every cache suspect; fail them all.
            self.running = []
            for request in batch:
                self._finish(request, error=error)
            return
        for i, request in enumerate(batch):
            try:
                self._append_sampled(request, logits[i : i + 1])
            except Exception as error:  # noqa: BLE001
                request.error = error
        self.running = [r for r in batch if r.finish_reason is None and r.error is None]
        for request in batch:
            if request.finish_reason is not None or request.error is not None:
                self._finish(request, error=request.error)

    def _append_sampled(self, request: Request, logits) -> None:
        """Sample from the last position and apply the stopping rules."""
        token = sample_next_token(
            logits[:, -1, :], request.temperature, request.top_k
        ).item()
        if token == EOS_TOKEN_ID:
            request.finish_reason = "eos"
            return
        request.new_ids.append(token)
        if len(request.new_ids) == request.max_new_tokens:
            request.finish_reason = "length"
        elif request.cache.position >= self.model.max_seq_length:
            # The token just sampled has no cache slot for another step.
            request.finish_reason = "context_limit"

    def _finish(self, request: Request, error: Exception | None = None) -> None:
        """Release the request's blocks and reservation, then wake the caller.

        Subtract the reservation recorded at admission, not a recomputed
        value: requests failed straight from the waiting queue never
        reserved anything, and subtracting for them corrupts admission.
        """
        request.error = error
        if request.cache is not None:
            request.cache.free()
            request.cache = None
        self.reserved_blocks -= request.reserved
        request.reserved = 0
        request.done.set()

    def start(self) -> threading.Thread:
        """Run the scheduling loop in a daemon thread (used by the server)."""
        thread = threading.Thread(target=self.run_forever, daemon=True)
        thread.start()
        return thread

    def run_forever(self) -> None:
        while True:
            with self.has_work:
                while not self.waiting and not self.running and not self.stopped:
                    self.has_work.wait()
                if self.stopped:
                    return
            try:
                self.step()
            except Exception as error:  # noqa: BLE001
                # A scheduling bug must not kill this thread: callers wait on
                # their events forever. Fail everything in flight and keep going.
                with self.has_work:
                    in_flight = list(self.waiting) + self.running
                    self.waiting.clear()
                    self.running = []
                for request in in_flight:
                    self._finish(request, error=error)

    def stop(self) -> None:
        with self.has_work:
            self.stopped = True
            self.has_work.notify()
