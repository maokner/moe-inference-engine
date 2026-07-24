"""Serve the engine model through a synchronous HTTP endpoint.

Usage:
    uv run python scripts/serve.py
    curl -s localhost:8000/generate -d '{"prompt": "The capital of France is"}'
"""

import argparse
import threading
import time

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from moe_engine.checkpoint import load_engine_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(64, ge=1)
    temperature: float = Field(0.8, ge=0.0, allow_inf_nan=False)


app = FastAPI(title="moe-engine")
enc = tiktoken.get_encoding("gpt2")
model = None  # Initialized in main().
allocator = None  # Shared KV block pool, initialized in main().
device = pick_device()

# The pool holds one max-length sequence, so serialize generation requests.
generate_lock = threading.Lock()


@app.post("/generate")
def generate(req: GenerateRequest):
    prompt_ids = torch.tensor(enc.encode(req.prompt), device=device)
    if len(prompt_ids) == 0:
        raise HTTPException(status_code=400, detail="prompt tokenizes to zero tokens")
    if len(prompt_ids) > model.max_seq_length:
        raise HTTPException(
            status_code=400,
            detail=f"prompt is {len(prompt_ids)} tokens; the model's context is {model.max_seq_length}",
        )

    with generate_lock:
        cache = allocator.new_cache()
        start = time.perf_counter()
        try:
            out = model.generate(
                prompt_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=50,
                cache=cache,
            )
        finally:
            cache.free()
        elapsed = time.perf_counter() - start

    new_ids = out[0, len(prompt_ids):].tolist()
    # generate() returns only tokens, so infer the stopping condition.
    if len(new_ids) == req.max_new_tokens:
        finish_reason = "length"
    elif len(prompt_ids) + len(new_ids) > model.max_seq_length:
        finish_reason = "context_limit"
    else:
        finish_reason = "eos"
    return {
        "text": enc.decode(new_ids),
        "new_tokens": len(new_ids),
        "finish_reason": finish_reason,
        "tokens_per_sec": round(len(new_ids) / elapsed, 1),
    }


def main() -> None:
    global model, allocator
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, _, metadata = load_engine_model(args.checkpoint, device)
    # Size the pool for one maximum-length request.
    block_size = 16
    num_blocks = -(-model.max_seq_length // block_size)  # Ceiling division.
    allocator = model.new_block_allocator(num_blocks=num_blocks, block_size=block_size)
    print(f"Serving checkpoint step {metadata['step']} on {device}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
