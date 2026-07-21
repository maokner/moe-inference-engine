"""Serve the reference model through a synchronous HTTP endpoint.

Usage:
    uv run python scripts/serve.py
    curl -s localhost:8000/generate -d '{"prompt": "The capital of France is"}'
"""

import argparse
import time

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from moe_engine.checkpoint import load_reference_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8


app = FastAPI(title="moe-engine baseline")
enc = tiktoken.get_encoding("gpt2")
model = None  # Initialized in main().
device = pick_device()


@app.post("/generate")
def generate(req: GenerateRequest):
    prompt_ids = torch.tensor(enc.encode(req.prompt), device=device)

    start = time.perf_counter()
    out = model.generate(
        prompt_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=50,
    )
    elapsed = time.perf_counter() - start

    new_ids = out[0, len(prompt_ids):].tolist()
    return {
        "text": enc.decode(new_ids),
        "new_tokens": len(new_ids),
        "tokens_per_sec": round(len(new_ids) / elapsed, 1),
    }


def main() -> None:
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, _, metadata = load_reference_model(args.checkpoint, device)
    print(f"Serving checkpoint step {metadata['step']} on {device}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
