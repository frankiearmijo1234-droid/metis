"""Tests for the OR-Tools (CP-SAT) engine."""

import numpy as np
import pytest

from metis import Problem, ProblemKind

ortools = pytest.importorskip("ortools")
from metis.engines.ortools_engine import MAX_QUBO_N, ORTools


@pytest.fixture
def engine():
    return ORTools()


# ---------- can_handle / estimate_cost ----------


def test_handles_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
    )
    assert engine.can_handle(p)


def test_handles_ilp(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"ilp_solve": True, "objective_coeffs": [1, 2, 3]},
    )
    assert engine.can_handle(p)


def test_rejects_continuous(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: 0, "x0": np.zeros(3)},
    )
    assert not engine.can_handle(p)


def test_rejects_other_kinds(engine):
    p = Problem(kind=ProblemKind.QUANTUM_CIRCUIT, payload={})
    assert not engine.can_handle(p)


def test_rejects_oversized_qubo(engine):
    """Beyond cap, can_handle returns False."""
    n = MAX_QUBO_N + 1
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(n), "qubo_solve": True},
    )
    assert not engine.can_handle(p)


# ---------- QUBO correctness ----------


def test_solves_diagonal_qubo_exactly(engine):
    """Q = -I has unique optimum at all-ones with f = -n."""
    n = 8
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "time_budget_s": 5},
    )
    sol = engine.solve(p)
    np.testing.assert_array_equal(sol.value["x"], np.ones(n))
    assert sol.value["fun"] == pytest.approx(-n)
    assert sol.value["is_optimal"]


def test_solves_known_2var_qubo(engine):
    """Q = [[2, -3], [-3, 2]]: optimum at [1,1] with fun = -2."""
    Q = np.array([[2.0, -3.0], [-3.0, 2.0]])
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": 2, "time_budget_s": 5},
    )
    sol = engine.solve(p)
    np.testing.assert_array_equal(sol.value["x"], np.ones(2))
    assert sol.value["fun"] == pytest.approx(-2)


def test_qubo_with_cardinality_constraint(engine):
    """Pick exactly k of n: Q = -I, sum(x) <= k. Optimum is k ones."""
    n, k = 8, 3
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": Q,
            "qubo_solve": True,
            "linear_constraints": [
                {"coeffs": [1.0] * n, "lo": None, "hi": k},
            ],
        },
        hints={"size": n, "time_budget_s": 5},
    )
    sol = engine.solve(p)
    assert int(sol.value["x"].sum()) == k
    assert sol.value["fun"] == pytest.approx(-k)


def test_qubo_with_lower_bound_constraint(engine):
    """Pick at least k of n with non-trivial costs."""
    n = 6
    Q = np.eye(n)  # each x_i adds 1 to objective; min wants all zeros
    constraint = [{"coeffs": [1.0] * n, "lo": 3, "hi": None}]
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": Q,
            "qubo_solve": True,
            "linear_constraints": constraint,
        },
        hints={"size": n, "time_budget_s": 5},
    )
    sol = engine.solve(p)
    # Must include at least 3 to satisfy constraint; minimum is exactly 3
    assert int(sol.value["x"].sum()) == 3
    assert sol.value["fun"] == pytest.approx(3)


def test_infeasible_constraint_returns_no_solution(engine):
    """Constraint that no x can satisfy."""
    n = 3
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(n),
            "qubo_solve": True,
            "linear_constraints": [
                # sum(x) >= 100, but x has only 3 binary vars (max sum = 3)
                {"coeffs": [1.0] * n, "lo": 100, "hi": None},
            ],
        },
        hints={"size": n, "time_budget_s": 2},
    )
    sol = engine.solve(p)
    assert sol.value["x"] is None
    assert sol.value["status"] == "INFEASIBLE"
    assert "warning" in sol.value


# ---------- ILP correctness ----------


def test_solves_classic_lp_textbook_problem(engine):
    """max 5x + 4y s.t. 6x + 4y <= 24, x + 2y <= 6, x,y in [0, 10] integer.
    Integer optimum: x=4, y=0 -> 20."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "ilp_solve": True,
            "objective_coeffs": [5, 4],
            "var_lo": [0, 0],
            "var_hi": [10, 10],
            "linear_constraints": [
                {"coeffs": [6, 4], "lo": None, "hi": 24},
                {"coeffs": [1, 2], "lo": None, "hi": 6},
            ],
            "minimize": False,
        },
        hints={"time_budget_s": 5},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(20)


def test_minimize_ilp(engine):
    """minimize x + y s.t. x + y >= 5, x,y in [0, 10] -> 5."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "ilp_solve": True,
            "objective_coeffs": [1, 1],
            "var_lo": [0, 0],
            "var_hi": [10, 10],
            "linear_constraints": [
                {"coeffs": [1, 1], "lo": 5, "hi": None},
            ],
            "minimize": True,
        },
        hints={"time_budget_s": 5},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(5)


# ---------- Validation / security ----------


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


def test_rejects_huge_time_budget(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5, "time_budget_s": 100_000},
    )
    with pytest.raises(ValueError, match="time_budget"):
        engine.solve(p)


def test_rejects_malformed_constraint(engine):
    """Coeffs vector wrong length."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(5),
            "qubo_solve": True,
            "linear_constraints": [
                {"coeffs": [1, 1], "hi": 3}
            ],  # only 2 coeffs for n=5
        },
        hints={"size": 5},
    )
    with pytest.raises(ValueError, match="length"):
        engine.solve(p)


def test_rejects_nan_coefficients(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(3),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1, float("nan"), 1], "hi": 1}],
        },
        hints={"size": 3},
    )
    with pytest.raises(ValueError, match="invalid value"):
        engine.solve(p)
