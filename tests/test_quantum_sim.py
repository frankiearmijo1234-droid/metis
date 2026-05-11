"""Tests for the quantum simulation engine (qmlx wrapper)."""

import numpy as np
import pytest

from metis import Problem, ProblemKind, QuantumStateVector


@pytest.fixture
def engine():
    return QuantumStateVector()


def test_handles_quantum_circuit(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 2, "ops": []},
    )
    assert engine.can_handle(p)


def test_rejects_optimization(engine):
    p = Problem(kind=ProblemKind.OPTIMIZATION, payload={})
    assert not engine.can_handle(p)


def test_rejects_too_many_qubits(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 100, "ops": []},
    )
    assert not engine.can_handle(p)


def test_bell_state(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
            ],
            "task": "probabilities",
        },
    )
    sol = engine.solve(p)
    probs = sol.value["probabilities"]
    np.testing.assert_allclose(probs, [0.5, 0, 0, 0.5], atol=1e-5)


def test_ghz_5_qubit(engine):
    n = 5
    ops = [{"gate": "H", "qubits": [0]}]
    for i in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [i, i + 1]})
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": n, "ops": ops, "task": "probabilities"},
    )
    sol = engine.solve(p)
    probs = sol.value["probabilities"]
    expected = np.zeros(2**n)
    expected[0] = 0.5
    expected[-1] = 0.5
    np.testing.assert_allclose(probs, expected, atol=1e-5)


def test_sample_returns_counts(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
            ],
            "task": "sample",
            "task_args": {"n_shots": 1000, "seed": 0},
        },
    )
    sol = engine.solve(p)
    counts = sol.value["counts"]
    # Bell state samples should only contain 00 and 11
    assert set(counts.keys()).issubset({"00", "11"})
    # Both outcomes should appear (with seed=0 and 1000 shots, vanishingly
    # unlikely to miss one entirely)
    assert "00" in counts and "11" in counts
    assert counts["00"] + counts["11"] == 1000


def test_rotation_gate_with_param(engine):
    """RY(pi) on |0> gives |1> (up to global phase)."""
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 1,
            "ops": [
                {"gate": "RY", "qubits": [0], "params": [np.pi]},
            ],
            "task": "probabilities",
        },
    )
    sol = engine.solve(p)
    probs = sol.value["probabilities"]
    np.testing.assert_allclose(probs, [0, 1], atol=1e-5)


def test_rejects_unknown_gate(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [{"gate": "NOTAREALGATE", "qubits": [0]}],
            "task": "probabilities",
        },
    )
    with pytest.raises(ValueError, match="unknown gate"):
        engine.solve(p)


def test_cost_grows_exponentially_with_qubits(engine):
    """Cost estimate should reflect 2^n scaling."""
    costs = []
    for n in [4, 8, 12, 16]:
        p = Problem(
            kind=ProblemKind.QUANTUM_CIRCUIT,
            payload={"n_qubits": n, "ops": [{"gate": "H", "qubits": [0]}]},
        )
        costs.append(engine.estimate_cost(p))
    # Each step of +4 qubits should ~16x the cost
    for i in range(len(costs) - 1):
        ratio = costs[i + 1] / costs[i]
        assert 8 < ratio < 32, f"ratio {i}: {ratio}"
