"""Integration tests: did the router pick the *right* engine for the problem?

These are the tests that verify our cost estimates are calibrated correctly,
not just that everything runs without crashing.
"""

import numpy as np
import pytest

from metis import Problem, ProblemKind, default_router


def test_small_qubo_routes_to_classical():
    """Brute force should win for n=8 because SA's overhead is wasteful."""
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


def test_large_qubo_routes_to_non_classical():
    """At n=30, classical brute force is infeasible. With OR-Tools, PT, and SA
    all available, one of the heuristic/exact non-classical engines wins.
    Either way, classical should NOT be picked."""
    router = default_router()
    np.random.seed(0)
    Q = np.random.randn(30, 30)
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": 30, "n_sweeps": 100, "seed": 0, "time_budget_s": 5},
    )
    sol = router.solve(p)
    assert sol.engine_name in (
        "simulated_annealing",
        "parallel_tempering",
        "ortools_cpsat",
    )
    assert sol.engine_name != "classical"


def test_very_large_qubo_routes_to_simulated_annealing():
    """At very large n with tight SA budget, SA's cost estimate undercuts
    OR-Tools (which clamps to its full time budget). The router prefers
    fast-with-uncertainty over exact-but-time-budget-eaten."""
    router = default_router()
    np.random.seed(0)
    n = 2000  # past OR-Tools' size cap (1000), so it's ineligible
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 20, "n_restarts": 1, "seed": 0},
    )
    sol = router.solve(p)
    assert sol.engine_name == "simulated_annealing"


def test_continuous_optimization_routes_to_classical():
    """No quantum or SA engine handles continuous; classical is the only fit."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "objective": lambda x: float(np.sum(x**2)),
            "x0": np.array([1.0, 2.0, 3.0]),
        },
    )
    sol = router.solve(p)
    assert sol.engine_name == "classical"
    np.testing.assert_allclose(sol.value["x"], [0, 0, 0], atol=1e-4)


def test_quantum_circuit_routes_to_qmlx():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "CNOT", "qubits": [1, 2]},
            ],
            "task": "probabilities",
        },
    )
    sol = router.solve(p)
    assert sol.engine_name == "qmlx_statevector"


def test_routing_decision_is_recorded():
    """Every solution should carry an audit trail of the routing decision."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
        hints={"size": 5},
    )
    sol = router.solve(p)
    decision = sol.metadata["routing_decision"]
    assert decision.chosen == sol.engine_name
    assert len(decision.candidates) >= 1
    assert decision.reason  # non-empty


def test_unknown_problem_kind_at_optimization_with_no_payload_fields():
    """A malformed optimization request should raise NoEngineAvailableError."""
    from metis import NoEngineAvailableError

    router = default_router()
    p = Problem(kind=ProblemKind.OPTIMIZATION, payload={"nonsense": True})
    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_search_problem_kind_has_no_engine_yet():
    """ProblemKind.SEARCH isn't implemented; should fail cleanly."""
    from metis import NoEngineAvailableError

    router = default_router()
    p = Problem(kind=ProblemKind.SEARCH, payload={})
    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_cross_engine_correctness_qubo():
    """For a QUBO that both classical and SA can handle (small n=10),
    they should give the same optimal value."""
    router = default_router()
    np.random.seed(0)
    n = 10
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2

    # Force classical
    p_classical = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n},
    )
    classical_sol = router.solve(p_classical)
    assert classical_sol.engine_name == "classical"

    # Run SA directly with thorough settings
    from metis import SimulatedAnnealing

    sa = SimulatedAnnealing()
    sa_sol = sa.solve(
        Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n, "seed": 0, "n_sweeps": 1000, "n_restarts": 10},
        )
    )

    # Both should find the same global minimum
    assert sa_sol.value["fun"] == pytest.approx(classical_sol.value["fun"], abs=1e-6)
