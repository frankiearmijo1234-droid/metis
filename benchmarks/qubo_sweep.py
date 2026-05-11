"""QUBO benchmarks: where does each engine win?

Three suites here:

1. qubo_size_sweep: increasing n with a fixed random seed. Shows the
   crossover from classical (exact, fastest at small n) to OR-Tools
   (exact, fastest at moderate n) to SA (heuristic, only choice past ~1000).

2. constrained_vs_unconstrained: same QUBO with and without a cardinality
   constraint. Demonstrates that adding constraints flips the routing
   decision since classical and SA refuse constrained problems.

3. quality_at_large_n: how close does SA get to OR-Tools' optimum when
   both are eligible? Establishes the speed-vs-quality tradeoff.
"""

from __future__ import annotations

import numpy as np

from metis import Problem, ProblemKind

from .harness import Benchmark


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n, n))
    return (Q + Q.T) / 2


def _qubo_obj(sol):
    """Pull the function value out of a Solution."""
    if sol.value.get("x") is None:
        return float("inf")
    return float(sol.value["fun"])


def qubo_size_sweep(
    sizes: list[int] | None = None, n_trials: int = 3, seed: int = 0
) -> list[Benchmark]:
    """Random unconstrained QUBO at increasing n."""
    if sizes is None:
        sizes = [8, 12, 16, 20, 50, 100, 500, 2000]
    benchmarks = []
    for n in sizes:
        Q = _random_qubo(n, seed=seed + n)
        # Match SA hints to the size; otherwise default heuristics may make
        # SA estimate or run too slowly to be fair.
        n_sweeps = min(max(100, n), 1000)
        bench = Benchmark(
            problem_id=f"qubo_n{n}",
            problem=Problem(
                kind=ProblemKind.OPTIMIZATION,
                payload={"qubo_Q": Q, "qubo_solve": True},
                hints={
                    "size": n,
                    "n_sweeps": n_sweeps,
                    "n_restarts": 2,
                    "seed": 0,
                    "time_budget_s": 5.0,
                },
            ),
            extract_objective=_qubo_obj,
            n_trials=n_trials,
            timeout_sec=30.0,
        )
        benchmarks.append(bench)
    return benchmarks


def constrained_vs_unconstrained(
    n: int = 20, cardinality: int = 5, n_trials: int = 3, seed: int = 0
) -> list[Benchmark]:
    """Same QUBO, with and without a cardinality cap. Constrained version
    must route to OR-Tools because classical and SA refuse constrained
    problems."""
    Q = _random_qubo(n, seed)
    return [
        Benchmark(
            problem_id=f"qubo_n{n}_unconstrained",
            problem=Problem(
                kind=ProblemKind.OPTIMIZATION,
                payload={"qubo_Q": Q, "qubo_solve": True},
                hints={
                    "size": n,
                    "n_sweeps": 500,
                    "n_restarts": 4,
                    "seed": 0,
                    "time_budget_s": 5.0,
                },
            ),
            extract_objective=_qubo_obj,
            n_trials=n_trials,
        ),
        Benchmark(
            problem_id=f"qubo_n{n}_card{cardinality}",
            problem=Problem(
                kind=ProblemKind.OPTIMIZATION,
                payload={
                    "qubo_Q": Q,
                    "qubo_solve": True,
                    "linear_constraints": [
                        {"coeffs": [1.0] * n, "lo": None, "hi": cardinality},
                    ],
                },
                hints={"size": n, "time_budget_s": 5.0},
            ),
            extract_objective=_qubo_obj,
            n_trials=n_trials,
        ),
    ]


def quality_at_large_n(
    n: int = 100, n_trials: int = 1, seed: int = 0
) -> list[Benchmark]:
    """At n=100 both OR-Tools and SA are eligible. OR-Tools tries to prove
    optimality (slow); SA gives a fast heuristic answer. We measure both
    speed and the quality gap."""
    Q = _random_qubo(n, seed)
    # Two versions: short SA budget (heuristic) and long SA budget
    # (more thorough). Both compared against OR-Tools' best.
    return [
        Benchmark(
            problem_id=f"qubo_n{n}_quality",
            problem=Problem(
                kind=ProblemKind.OPTIMIZATION,
                payload={"qubo_Q": Q, "qubo_solve": True},
                hints={
                    "size": n,
                    "n_sweeps": 500,
                    "n_restarts": 4,
                    "seed": 0,
                    "time_budget_s": 10.0,
                },
            ),
            extract_objective=_qubo_obj,
            n_trials=n_trials,
            timeout_sec=15.0,
        ),
    ]
