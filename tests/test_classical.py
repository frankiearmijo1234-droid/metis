"""Tests for the classical optimizer engine."""

import numpy as np
import pytest

from metis import ClassicalOptimizer, Problem, ProblemKind


@pytest.fixture
def engine():
    return ClassicalOptimizer()


def test_can_handle_continuous(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: float(np.sum(x**2)), "x0": np.zeros(3)},
    )
    assert engine.can_handle(p)


def test_can_handle_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
    )
    assert engine.can_handle(p)


def test_rejects_other_kinds(engine):
    p = Problem(kind=ProblemKind.QUANTUM_CIRCUIT, payload={})
    assert not engine.can_handle(p)


def test_rejects_oversized_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(30), "qubo_solve": True},
        hints={"size": 30},
    )
    # can_handle is True, but cost is inf so router will skip it
    assert engine.estimate_cost(p) == float("inf")


def test_solves_quadratic_continuous(engine):
    """f(x) = (x-3)^2 has min at x=3."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: float((x[0] - 3) ** 2), "x0": np.array([0.0])},
    )
    sol = engine.solve(p)
    assert abs(sol.value["x"][0] - 3.0) < 1e-4
    assert sol.value["fun"] < 1e-6


def test_solves_qubo_brute_force_finds_optimum(engine):
    """For Q = -I, x^T Q x = -sum(x_i). Optimum is x = [1,1,...,1] with value = -n."""
    n = 6
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n},
    )
    sol = engine.solve(p)
    np.testing.assert_array_equal(sol.value["x"], np.ones(n))
    assert sol.value["fun"] == -n


def test_solves_known_qubo_optimum(engine):
    """Hand-constructed QUBO with known optimum.
    Q = [[2, -3], [-3, 2]]. Try all 4 assignments:
      [0,0]: 0; [0,1]: 2; [1,0]: 2; [1,1]: 2 + 2 + 2*(-3) = -2.
    """
    Q = np.array([[2.0, -3.0], [-3.0, 2.0]])
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": 2},
    )
    sol = engine.solve(p)
    np.testing.assert_array_equal(sol.value["x"], np.array([1.0, 1.0]))
    assert sol.value["fun"] == pytest.approx(-2.0)


def test_solves_rosenbrock(engine):
    """Classic test function. Minimum at (1, 1) with value 0."""

    def rosen(x):
        return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": rosen, "x0": np.array([-1.2, 1.0])},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] < 1e-6
    np.testing.assert_allclose(sol.value["x"], [1.0, 1.0], atol=1e-3)
