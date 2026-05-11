"""metis cross-engine benchmark suite.

Runs reproducible benchmarks across all engines and produces numbers you
can point at: which engine wins where, by how much, and at what quality.

Run:
    python -m benchmarks.run --quick      # ~1 min smoke benchmark
    python -m benchmarks.run --full       # ~10 min full sweep
    python -m benchmarks.run --suite qubo # specific suite

Output goes to benchmarks/results/<timestamp>.json and a markdown summary.
"""

from .harness import Benchmark, BenchmarkResult, run_suite
from .quantum_sweep import clifford_vs_general, quantum_size_sweep
from .qubo_sweep import constrained_vs_unconstrained, qubo_size_sweep

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "run_suite",
    "qubo_size_sweep",
    "constrained_vs_unconstrained",
    "quantum_size_sweep",
    "clifford_vs_general",
]
