"""Adversarial input tests.

Every test here corresponds to a hostile input that previously slipped past
validation. They prevent regression — if any of these starts passing without
raising, we've reopened a known security hole.
"""

import numpy as np
import pytest

from metis import Problem, ProblemKind, default_router

# ---------- QUBO with non-finite values ----------


def test_nan_qubo_rejected():
    """Pre-fix: SA returned {'x': None, 'fun': inf} silently."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.full((5, 5), np.nan), "qubo_solve": True},
        hints={"size": 5, "n_sweeps": 5, "n_restarts": 1},
    )
    with pytest.raises(ValueError, match="non-finite"):
        router.solve(p)


def test_inf_qubo_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.full((5, 5), np.inf), "qubo_solve": True},
        hints={"size": 5, "n_sweeps": 5, "n_restarts": 1},
    )
    with pytest.raises(ValueError, match="non-finite"):
        router.solve(p)


def test_partially_nan_qubo_rejected():
    """Even a single NaN must be rejected."""
    Q = np.eye(5)
    Q[2, 3] = np.nan
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": 5, "n_sweeps": 5, "n_restarts": 1},
    )
    with pytest.raises(ValueError, match="non-finite"):
        router.solve(p)


def test_non_square_qubo_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.zeros((3, 5)), "qubo_solve": True},
        hints={"size": 3},
    )
    with pytest.raises(ValueError, match="square"):
        router.solve(p)


# ---------- Quantum circuit hostile inputs ----------


def test_oversized_n_qubits_rejected():
    """Pre-fix: passes can_handle if you bypass MCP layer."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 1000, "ops": [], "task": "probabilities"},
    )
    from metis import NoEngineAvailableError

    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_negative_n_qubits_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": -5, "ops": [], "task": "probabilities"},
    )
    from metis import NoEngineAvailableError

    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_boolean_n_qubits_rejected():
    """Python booleans are technically ints; we explicitly reject them."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": True, "ops": [], "task": "probabilities"},
    )
    from metis import NoEngineAvailableError

    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_ops_not_a_list_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 2, "ops": "not-a-list", "task": "probabilities"},
    )
    from metis import NoEngineAvailableError

    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_million_op_circuit_rejected():
    """Resource cap: 2M ops should be refused at engine.can_handle()."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [{"gate": "H", "qubits": [0]}] * 2_000_000,
            "task": "probabilities",
        },
    )
    from metis import NoEngineAvailableError

    with pytest.raises(NoEngineAvailableError):
        router.solve(p)


def test_qubit_index_out_of_range_rejected():
    """Engine raises ValueError when qubit index >= n_qubits."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [{"gate": "H", "qubits": [99]}],
            "task": "probabilities",
        },
    )
    with pytest.raises((ValueError, IndexError), match="out of range"):
        router.solve(p)


def test_negative_qubit_index_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [{"gate": "H", "qubits": [-1]}],
            "task": "probabilities",
        },
    )
    with pytest.raises((ValueError, IndexError), match="out of range"):
        router.solve(p)


def test_unknown_gate_rejected():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 2,
            "ops": [{"gate": "HACKGATE", "qubits": [0]}],
            "task": "probabilities",
        },
    )
    with pytest.raises(ValueError, match="unknown gate"):
        router.solve(p)


# ---------- Misleading 'success' on NaN objective ----------


def test_nan_objective_reports_unreliable():
    """When the user's objective always returns NaN, the result must clearly
    say so rather than reporting success=True with garbage."""
    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": lambda x: float("nan"), "x0": np.zeros(5)},
        hints={"size": 5},
    )
    sol = router.solve(p)
    assert sol.value["success"] is False
    assert "warning" in sol.value
    assert sol.value["nonfinite_evaluations"] > 0


def test_partially_nan_objective_succeeds_with_real_value():
    """If objective returns NaN sometimes but a valid minimum is found, the
    real success flag stays True. We test with f(x) = NaN if any x[i] > 100,
    else x[0]^2. The minimum at x=[0,0,...] should still be found."""

    def patchy(x):
        if np.any(x > 100):
            return float("nan")
        return float(x[0] ** 2)

    router = default_router()
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": patchy, "x0": np.array([5.0, 5.0])},
        hints={"size": 2},
    )
    sol = router.solve(p)
    assert sol.value["success"] is True
    assert sol.value["fun"] < 1e-6


# ---------- MCP eval-bypass cannot regress ----------


def test_mcp_server_no_minimize_function():
    """The eval-based minimize_function was removed for security. The
    replacement minimize_quadratic uses structured input."""
    # Importing the mcp_server requires the mcp package; we just check the
    # source for the dangerous symbol.
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mcp_path = os.path.join(repo_root, "claude_skill", "mcp_server.py")
    with open(mcp_path) as f:
        src = f.read()
    # No raw eval() call should exist
    assert "eval(code" not in src, "MCP server must not eval user expressions"
    # The replacement should be present
    assert "minimize_quadratic" in src
