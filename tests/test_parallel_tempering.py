"""Tests for the parallel tempering engine."""

import numpy as np
import pytest

from metis import (
    ClassicalOptimizer,
    ParallelTempering,
    Problem,
    ProblemKind,
)


@pytest.fixture
def engine():
    return ParallelTempering()


# ---------- can_handle / estimate_cost ----------


def test_handles_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(20), "qubo_solve": True},
        hints={"size": 20},
    )
    assert engine.can_handle(p)


def test_rejects_continuous(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: 0, "x0": np.zeros(3)},
    )
    assert not engine.can_handle(p)


def test_rejects_constrained(engine):
    """PT doesn't handle linear constraints; OR-Tools should."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(10),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1.0] * 10, "hi": 5}],
        },
    )
    assert not engine.can_handle(p)


def test_rejects_other_kinds(engine):
    p = Problem(kind=ProblemKind.QUANTUM_CIRCUIT, payload={})
    assert not engine.can_handle(p)


# ---------- correctness ----------


def test_solves_diagonal_qubo_exactly(engine):
    """Q = -I has unique optimum at all-ones with f = -n."""
    n = 8
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 100, "n_replicas": 4, "seed": 0},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(-n)
    assert all(v == 1.0 for v in sol.value["x"])


def test_finds_optimum_on_random_qubos():
    """PT should match brute force on every random small QUBO."""
    pt = ParallelTempering()
    cls = ClassicalOptimizer()
    for seed in range(8):
        n = 12
        rng = np.random.default_rng(seed * 17)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2

        truth = cls.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n},
            )
        ).value["fun"]
        pt_val = pt.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n, "n_sweeps": 200, "n_replicas": 4, "seed": 0},
            )
        ).value["fun"]

        assert pt_val == pytest.approx(
            truth, abs=1e-6
        ), f"seed={seed}: PT found {pt_val}, truth is {truth}"


def test_solution_x_consistent_with_fun(engine):
    n = 30
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 100, "n_replicas": 4, "seed": 0},
    )
    sol = engine.solve(p)
    x = np.asarray(sol.value["x"])
    Qs = (Q + Q.T) / 2
    recomputed = float(x @ Qs @ x)
    assert sol.value["fun"] == pytest.approx(recomputed, abs=1e-9)


def test_metadata_records_swap_acceptance(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(20), "qubo_solve": True},
        hints={"size": 20, "n_sweeps": 50, "n_replicas": 8, "seed": 0},
    )
    sol = engine.solve(p)
    assert "swap_acceptance_rate" in sol.value
    assert 0 <= sol.value["swap_acceptance_rate"] <= 1


def test_seed_makes_deterministic(engine):
    n = 20
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    payload = {"qubo_Q": Q, "qubo_solve": True}
    hints = {"size": n, "n_sweeps": 100, "n_replicas": 4, "seed": 12345}
    s1 = engine.solve(Problem(ProblemKind.OPTIMIZATION, payload, hints))
    s2 = engine.solve(Problem(ProblemKind.OPTIMIZATION, payload, hints))
    assert s1.value["fun"] == s2.value["fun"]
    np.testing.assert_array_equal(s1.value["x"], s2.value["x"])


# ---------- validation / caps ----------


def test_rejects_nan_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.full((5, 5), np.nan), "qubo_solve": True},
        hints={"size": 5},
    )
    with pytest.raises(ValueError, match="NaN"):
        engine.solve(p)


def test_rejects_complex_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5, dtype=complex), "qubo_solve": True},
        hints={"size": 5},
    )
    with pytest.raises(ValueError, match="complex"):
        engine.solve(p)


def test_rejects_huge_n_replicas(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5, "n_replicas": 1_000_000, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_replicas"):
        engine.solve(p)


def test_rejects_huge_n_sweeps(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5, "n_sweeps": 10**9, "n_replicas": 4, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_sweeps"):
        engine.solve(p)


def test_rejects_invalid_T_min(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5, "T_min": -1.0, "n_replicas": 4, "seed": 0},
    )
    with pytest.raises(ValueError, match="T_min"):
        engine.solve(p)


def test_rejects_T_max_below_T_min(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5, "T_min": 1.0, "T_max": 0.5, "n_replicas": 4, "seed": 0},
    )
    with pytest.raises(ValueError, match="T_max"):
        engine.solve(p)
