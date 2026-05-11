"""CLI for running metis benchmarks.

Usage:
    python -m benchmarks.run --quick           # 60s smoke test
    python -m benchmarks.run --full            # full sweep, several minutes
    python -m benchmarks.run --suite qubo      # one suite only
    python -m benchmarks.run --suite quantum
    python -m benchmarks.run --output results/myrun.json
"""

from __future__ import annotations

import argparse
import os
import time

from metis import default_router

from .harness import render_markdown_summary, run_suite, write_results_json
from .quantum_sweep import (
    clifford_vs_general,
    quantum_size_sweep,
    random_clifford_circuit,
)
from .qubo_sweep import (
    constrained_vs_unconstrained,
    quality_at_large_n,
    qubo_size_sweep,
)


def build_quick_suite():
    """Smoke benchmark: a handful of problems, one trial each, ~60s total."""
    return (
        qubo_size_sweep(sizes=[8, 16, 50, 500], n_trials=2)
        + constrained_vs_unconstrained(n=12, cardinality=4, n_trials=2)
        + quantum_size_sweep(sizes=[4, 20, 50, 200], n_trials=1)
        + clifford_vs_general(n=4, n_trials=2)
    )


def build_full_suite():
    """Full sweep: comprehensive coverage, several minutes."""
    return (
        qubo_size_sweep(sizes=[8, 12, 16, 20, 50, 100, 500, 2000], n_trials=3)
        + constrained_vs_unconstrained(n=20, cardinality=5, n_trials=3)
        + quality_at_large_n(n=100, n_trials=1)
        + quantum_size_sweep(sizes=[4, 10, 20, 28, 50, 200, 1000], n_trials=2)
        + clifford_vs_general(n=8, n_trials=3)
        + random_clifford_circuit(n=50, depth=100, n_trials=2)
    )


def build_suite_by_name(name: str):
    if name == "qubo":
        return (
            qubo_size_sweep(sizes=[8, 12, 16, 20, 50, 100, 500, 2000], n_trials=3)
            + constrained_vs_unconstrained(n=20, cardinality=5, n_trials=3)
            + quality_at_large_n(n=100, n_trials=1)
        )
    if name == "quantum":
        return (
            quantum_size_sweep(sizes=[4, 10, 20, 28, 50, 200, 1000], n_trials=2)
            + clifford_vs_general(n=8, n_trials=3)
            + random_clifford_circuit(n=50, depth=100, n_trials=2)
        )
    raise ValueError(f"unknown suite: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run metis benchmarks")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", action="store_true", help="Quick smoke benchmark (~60s)"
    )
    group.add_argument(
        "--full", action="store_true", help="Full sweep (several minutes)"
    )
    group.add_argument(
        "--suite", choices=("qubo", "quantum"), help="Run a specific suite"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="JSON output path (default: results/<timestamp>.json)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-measurement progress lines",
    )
    args = parser.parse_args()

    if args.quick:
        benchmarks = build_quick_suite()
        suite_label = "quick"
    elif args.suite:
        benchmarks = build_suite_by_name(args.suite)
        suite_label = args.suite
    else:
        benchmarks = build_full_suite()
        suite_label = "full"

    print(f"metis benchmark suite: {suite_label}")
    print(f"  {len(benchmarks)} problems")
    router = default_router()
    print(
        f"  {len(router.engines())} engines: "
        f"{', '.join(e.name for e in router.engines())}"
    )
    print()

    t0 = time.perf_counter()
    results = run_suite(router, benchmarks, quiet=args.quiet)
    elapsed = time.perf_counter() - t0

    print()
    print(f"Done in {elapsed:.1f}s")
    print()

    md = render_markdown_summary(results)
    print(md)

    # Write JSON
    if args.output is None:
        os.makedirs("benchmarks/results", exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = f"benchmarks/results/{suite_label}_{ts}.json"
    else:
        json_path = args.output
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    write_results_json(results, json_path)

    # Also write markdown alongside
    md_path = json_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(md)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
