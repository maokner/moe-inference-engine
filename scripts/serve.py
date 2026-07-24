"""Serve the engine model with continuous batching.

Each HTTP request becomes an engine Request and waits on its completion
event; a single background engine thread batches all active requests into
shared decode steps, so concurrent requests genuinely run together.

Usage:
    uv run python scripts/serve.py
    curl -s localhost:8000/generate -d '{"prompt": "The capital of France is"}'
"""

import argparse
import time

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from moe_engine.checkpoint import load_engine_model
from moe_engine.engine import Engine, Request


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
engine = None  # Initialized in main().
device = pick_device()


@app.post("/generate")
def generate(req: GenerateRequest):
    prompt_ids = enc.encode(req.prompt)
    if len(prompt_ids) == 0:
        raise HTTPException(status_code=400, detail="prompt tokenizes to zero tokens")
    if len(prompt_ids) > model.max_seq_length:
        raise HTTPException(
            status_code=400,
            detail=f"prompt is {len(prompt_ids)} tokens; the model's context is {model.max_seq_length}",
        )

    request = Request(
        prompt_ids=prompt_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=50,
    )
    start = time.perf_counter()
    engine.submit(request)
    request.done.wait()
    elapsed = time.perf_counter() - start

    if request.error is not None:
        raise HTTPException(status_code=500, detail="generation failed")
    return {
        "text": enc.decode(request.new_ids),
        "new_tokens": len(request.new_ids),
        "finish_reason": request.finish_reason,
        # Wall time includes any queueing, so this is end-to-end throughput.
        "tokens_per_sec": round(len(request.new_ids) / elapsed, 1),
    }


def main() -> None:
    global model, engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    model, _, metadata = load_engine_model(args.checkpoint, device)
    # Size the pool so a full batch of maximum-length sequences always fits;
    # admission then only ever waits on the batch-size cap.
    block_size = 16
    blocks_per_seq = -(-model.max_seq_length // block_size)  # Ceiling division.
    allocator = model.new_block_allocator(
        num_blocks=args.max_batch_size * blocks_per_seq, block_size=block_size
    )
    engine = Engine(model, allocator, max_batch_size=args.max_batch_size)
    engine.start()
    print(f"Serving checkpoint step {metadata['step']} on {device}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
