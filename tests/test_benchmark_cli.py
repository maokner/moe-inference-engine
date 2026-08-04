"""Parser coverage for safe benchmark device selection."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).parent.parent / "benchmarks" / "bench.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_benchmark_defaults_to_cpu_and_requires_explicit_accelerator():
    defaults = bench.build_parser().parse_args([])
    assert defaults.device == "cpu"

    assert bench.build_parser().parse_args(["--device", "cuda"]).device == "cuda"
    assert bench.build_parser().parse_args(["--device", "mps"]).device == "mps"
