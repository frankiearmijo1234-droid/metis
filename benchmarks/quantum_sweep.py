"""Quantum circuit benchmarks: state-vector vs stabilizer.

The point of these is to make the cliff visible. State-vector simulation
hits a hard memory wall around n=28 (this engine's cap; in absolute terms,
n=33 needs 64 GB and n=50 is 16 PB). Stabilizer keeps going because it
stores O(n^2) bits, not 2^n complex amplitudes — but only for Clifford
circuits.

Suites:

1. quantum_size_sweep_clifford: GHZ at n in [4..1000]. State-vector wins
   small, dies past 28. Stabilizer wins past ~25 and goes to 1000+.

2. clifford_vs_general: same circuit shape with and without a T gate.
   Adding T forces routing to state-vector (T is non-Clifford).

3. random_clifford_circuit: more realistic Clifford workload (random gates
   instead of GHZ structure). Same routing crossover, different constants.
"""

from __future__ import annotations

import numpy as np

from metis import Problem, ProblemKind

from .harness import Benchmark


def _ghz_circuit(n: int) -> list[dict]:
    """n-qubit GHZ: H on qubit 0, then CNOT cascade."""
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    return ops


def _ghz_check(sol):
    """Quality metric for a GHZ sample: fraction of shots that landed in
    {|00..0>, |11..1>}. Should be 1.0."""
    counts = sol.value.get("counts")
    if counts is None:
        return 0.0
    n_shots = sum(counts.values())
    if n_shots == 0:
        return 0.0
    correct = 0
    for outcome, c in counts.items():
        if set(outcome) == {"0"} or set(outcome) == {"1"}:
            correct += c
    return correct / n_shots


def quantum_size_sweep(
    sizes: list[int] | None = None, n_shots: int = 5, n_trials: int = 2
) -> list[Benchmark]:
    """GHZ at increasing n. State-vector caps at 28; stabilizer goes to 1000+."""
    if sizes is None:
        sizes = [4, 10, 20, 28, 50, 200, 1000]
    benchmarks = []
    for n in sizes:
        ops = _ghz_circuit(n)
        bench = Benchmark(
            problem_id=f"ghz_n{n}",
            problem=Problem(
                kind=ProblemKind.QUANTUM_CIRCUIT,
                payload={
                    "n_qubits": n,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": n_shots, "seed": 0},
                },
            ),
            extract_objective=_ghz_check,
            n_trials=n_trials,
            timeout_sec=120.0,  # 1000-qubit case can take ~30s/run
        )
        benchmarks.append(bench)
    return benchmarks


def clifford_vs_general(n: int = 8, n_trials: int = 3) -> list[Benchmark]:
    """Same circuit, with and without a non-Clifford T gate.

    Without T: stabilizer engine eligible.
    With T: only state-vector engine eligible.
    """
    base_ops = _ghz_circuit(n)
    with_t = base_ops + [{"gate": "T", "qubits": [0]}]
    return [
        Benchmark(
            problem_id=f"clifford_only_n{n}",
            problem=Problem(
                kind=ProblemKind.QUANTUM_CIRCUIT,
                payload={
                    "n_qubits": n,
                    "ops": base_ops,
                    "task": "sample",
                    "task_args": {"n_shots": 100, "seed": 0},
                },
            ),
            extract_objective=_ghz_check,
            n_trials=n_trials,
        ),
        Benchmark(
            problem_id=f"with_t_gate_n{n}",
            problem=Problem(
                kind=ProblemKind.QUANTUM_CIRCUIT,
                payload={
                    "n_qubits": n,
                    "ops": with_t,
                    "task": "sample",
                    "task_args": {"n_shots": 100, "seed": 0},
                },
            ),
            n_trials=n_trials,
        ),
    ]


def random_clifford_circuit(
    n: int = 50, depth: int = 100, n_shots: int = 5, n_trials: int = 2, seed: int = 0
) -> list[Benchmark]:
    """A random Clifford circuit at moderate size.

    More realistic workload than pure GHZ because it has substantive
    entanglement throughout. We expect both engines to give the same answer
    but stabilizer to win on speed past n~25.
    """
    rng = np.random.default_rng(seed)
    single_clifford_gates = ["H", "S", "X", "Y", "Z"]
    ops = []
    for _ in range(depth):
        if rng.random() < 0.5 or n < 2:
            # single-qubit gate
            gate = rng.choice(single_clifford_gates)
            q = int(rng.integers(0, n))
            ops.append({"gate": gate, "qubits": [q]})
        else:
            # CNOT
            q1, q2 = rng.choice(n, size=2, replace=False)
            ops.append({"gate": "CNOT", "qubits": [int(q1), int(q2)]})

    return [
        Benchmark(
            problem_id=f"rand_clifford_n{n}_d{depth}",
            problem=Problem(
                kind=ProblemKind.QUANTUM_CIRCUIT,
                payload={
                    "n_qubits": n,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": n_shots, "seed": 0},
                },
            ),
            n_trials=n_trials,
            timeout_sec=60.0,
        ),
    ]
