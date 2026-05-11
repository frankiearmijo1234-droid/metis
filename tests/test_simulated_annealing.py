"""Tests for the simulated annealing engine."""

import numpy as np
import pytest

from metis import ClassicalOptimizer, Problem, ProblemKind, SimulatedAnnealing


@pytest.fixture
def engine():
    return SimulatedAnnealing()


def test_handles_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
    )
    assert engine.can_handle(p)


def test_rejects_continuous(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: 0, "x0": np.array([0.0])},
    )
    assert not engine.can_handle(p)


def test_finds_known_optimum_small(engine):
    """For Q = -I with n=5, optimum is all ones with f = -5."""
    n = 5
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "seed": 0, "n_sweeps": 200, "n_restarts": 4},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(-n)


def test_matches_brute_force_on_random_qubos():
    """SA must find the same optimum as classical brute force for n that
    classical can handle. We try several random seeds."""
    sa = SimulatedAnnealing()
    classical = ClassicalOptimizer()
    n = 8
    for seed in [0, 1, 2, 7, 42]:
        rng = np.random.default_rng(seed)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        p = Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n, "seed": 0, "n_sweeps": 800, "n_restarts": 8},
        )
        sa_sol = sa.solve(p)
        classical_sol = classical.solve(p)
        # Ground truth from brute force:
        truth = classical_sol.value["fun"]
        # SA should find it (or be very close) at this size
        assert sa_sol.value["fun"] == pytest.approx(truth, abs=1e-6), (
            f"seed {seed}: SA found {sa_sol.value['fun']}, " f"truth is {truth}"
        )


def test_handles_large_qubo_without_crashing(engine):
    """Just make sure SA scales to sizes brute force can't touch."""
    n = 40
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "seed": 0, "n_sweeps": 200, "n_restarts": 2},
    )
    sol = engine.solve(p)
    # We can't verify optimality (no ground truth), but the function value
    # should be finite and the solution should be a binary vector.
    assert np.isfinite(sol.value["fun"])
    assert set(np.unique(sol.value["x"]).tolist()).issubset({0.0, 1.0})


def test_seed_makes_deterministic(engine):
    n = 12
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    payload = {"qubo_Q": Q, "qubo_solve": True}
    hints = {"size": n, "seed": 12345, "n_sweeps": 300, "n_restarts": 3}
    p1 = Problem(kind=ProblemKind.OPTIMIZATION, payload=payload, hints=hints)
    p2 = Problem(kind=ProblemKind.OPTIMIZATION, payload=payload, hints=hints)
    s1 = engine.solve(p1)
    s2 = engine.solve(p2)
    assert s1.value["fun"] == s2.value["fun"]
    np.testing.assert_array_equal(s1.value["x"], s2.value["x"])
