"""Integration tests: did the router pick the right engine?

These verify routing-decision logic across all four engines including
OR-Tools.
"""

import numpy as np
import pytest

from metis import _ORTOOLS_AVAILABLE, Problem, ProblemKind, default_router

ortools_required = pytest.mark.skipif(
    not _ORTOOLS_AVAILABLE,
    reason="OR-Tools not installed",
)


@ortools_required
def test_constrained_qubo_routes_to_ortools():
    """Only OR-Tools handles constraints; classical and SA must refuse."""
    router = default_router()
    n = 10
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": Q,
            "qubo_solve": True,
            "linear_constraints": [
                {"coeffs": [1.0] * n, "lo": None, "hi": 5},
            ],
        },
        hints={"size": n, "time_budget_s": 5},
    )
    sol = router.solve(p)
    assert sol.engine_name == "ortools_cpsat"
    # Constraint must actually be enforced
    assert int(sol.value["x"].sum()) <= 5


@ortools_required
def test_ilp_routes_to_ortools():
    """ILP problems are OR-Tools-only."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "ilp_solve": True,
            "objective_coeffs": [3, 2, 1],
            "var_lo": [0, 0, 0],
            "var_hi": [10, 10, 10],
            "linear_constraints": [
                {"coeffs": [1, 1, 1], "lo": None, "hi": 5},
            ],
            "minimize": False,
        },
        hints={"time_budget_s": 5},
    )
    sol = router.solve(p)
    assert sol.engine_name == "ortools_cpsat"


@ortools_required
def test_classical_refuses_constrained_problem():
    """Direct check: classical.can_handle returns False for constrained."""
    from metis import ClassicalOptimizer

    c = ClassicalOptimizer()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(5),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1] * 5, "hi": 2}],
        },
    )
    assert not c.can_handle(p)


@ortools_required
def test_sa_refuses_constrained_problem():
    """Direct check: SA.can_handle returns False for constrained."""
    from metis import SimulatedAnnealing

    sa = SimulatedAnnealing()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(5),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1] * 5, "hi": 2}],
        },
    )
    assert not sa.can_handle(p)


@ortools_required
def test_unconstrained_small_qubo_still_classical():
    """Adding OR-Tools shouldn't change small-QUBO routing."""
    router = default_router()
    np.random.seed(0)
    Q = np.random.randn(8, 8)
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": 8},
    )
    sol = router.solve(p)
    assert sol.engine_name == "classical"


@ortools_required
def test_quantum_circuits_unaffected_by_ortools():
    """Adding OR-Tools shouldn't change quantum-circuit routing."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "probabilities",
        },
    )
    sol = router.solve(p)
    assert sol.engine_name == "qmlx_statevector"
