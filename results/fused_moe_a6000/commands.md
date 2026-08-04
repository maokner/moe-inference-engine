# A6000 validation commands

The snapshot environment supplied the Python runtime while `PYTHONPATH=src` selected the validation checkout's exact source.

```bash
export SNAP_PY=/home/ubuntu/moe-inference-engine/.venv/bin/python
cd /home/ubuntu/moe-validation
sha256sum checkpoints/minimoe_sft.pt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short --branch
```

```bash
uvx ruff check benchmarks/bench.py benchmarks/moe_ab_compare.py benchmarks/profile_contiguous_decode.py scripts/check_parity.py scripts/serve.py scripts/validate_fused_moe.py src/moe_engine/fused_moe.py src/moe_engine/model.py tests/test_benchmark_cli.py tests/test_fused_moe.py tests/test_serve.py
uvx ruff format --check benchmarks/bench.py benchmarks/moe_ab_compare.py benchmarks/profile_contiguous_decode.py scripts/check_parity.py scripts/serve.py scripts/validate_fused_moe.py src/moe_engine/fused_moe.py src/moe_engine/model.py tests/test_benchmark_cli.py tests/test_fused_moe.py tests/test_serve.py
PYTHONPATH=src "$SNAP_PY" -m compileall -q src benchmarks scripts tests
git diff --check
MINIMOE_CHECKPOINT=checkpoints/minimoe_sft.pt PYTHONPATH=src "$SNAP_PY" -m pytest -q -rs
```

```bash
PYTHONPATH=src "$SNAP_PY" scripts/validate_fused_moe.py --checkpoint checkpoints/minimoe_sft.pt --output results/fused_moe_a6000/fused_moe_validation.json
PYTHONPATH=src "$SNAP_PY" scripts/check_parity.py --device cuda --engine-cache contiguous --moe direct --new-tokens 64 --output results/fused_moe_a6000/parity_contiguous_direct.json
PYTHONPATH=src "$SNAP_PY" scripts/check_parity.py --device cuda --engine-cache paged-direct --moe direct --new-tokens 64 --output results/fused_moe_a6000/parity_paged_direct.json
```

The HTTP request body used the canonical 62-token prompt with `max_new_tokens` set to 64 and `temperature` set to zero.

```bash
PYTHONPATH=src "$SNAP_PY" scripts/serve.py --device cuda --cache contiguous --moe reference --port 8011
curl -sf -X POST http://127.0.0.1:8011/generate -H 'content-type: application/json' --data-binary "$PROMPT_JSON"
PYTHONPATH=src "$SNAP_PY" scripts/serve.py --device cuda --cache contiguous --moe direct --port 8011
curl -sf -X POST http://127.0.0.1:8011/generate -H 'content-type: application/json' --data-binary "$PROMPT_JSON"
```

```bash
PYTHONPATH=src "$SNAP_PY" benchmarks/moe_ab_compare.py --checkpoint checkpoints/minimoe_sft.pt --rounds 21 --new-tokens 64 --output results/fused_moe_a6000/moe_ab_compare.json
PYTHONPATH=src "$SNAP_PY" benchmarks/profile_contiguous_decode.py --checkpoint checkpoints/minimoe_sft.pt --device cuda --moe reference --warmup-steps 16 --steps 32 --output-dir results/fused_moe_a6000/profile_reference
PYTHONPATH=src "$SNAP_PY" benchmarks/profile_contiguous_decode.py --checkpoint checkpoints/minimoe_sft.pt --device cuda --moe direct --warmup-steps 16 --steps 32 --output-dir results/fused_moe_a6000/profile_direct
```
