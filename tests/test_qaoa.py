"""Tests for the QAOA engine.

QAOA is a hybrid quantum/classical algorithm. We test:
1. It finds known optima at small n.
2. It refuses problems that exceed its qubit cap.
3. It's opt-in: the router only picks it when prefer_qaoa is set.
4. Validation rejects bad inputs.
5. The optimizer reports useful diagnostics.
"""

import numpy as np
import pytest

from metis import (
    QAOA,
    Problem,
    ProblemKind,
    default_router,
)


@pytest.fixture
def engine():
    return QAOA()


# ---------- can_handle / opt-in routing ----------


def test_handles_small_qubo_with_prefer_hint(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4, "prefer_qaoa": True},
    )
    assert engine.can_handle(p)


def test_handles_when_method_is_qaoa(engine):
    """method='qaoa' is the alternative opt-in signal."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4, "method": "qaoa"},
    )
    assert engine.can_handle(p)


def test_refuses_without_opt_in(engine):
    """QAOA refuses by default -- it's slower than SA for typical QUBO."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4},
    )
    assert not engine.can_handle(p)


def test_refuses_oversized_problem(engine):
    """QAOA caps at 18 qubits."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(20), "qubo_solve": True},
        hints={"size": 20, "prefer_qaoa": True},
    )
    assert not engine.can_handle(p)


def test_refuses_constrained(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(4),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1.0] * 4, "hi": 2}],
        },
        hints={"size": 4, "prefer_qaoa": True},
    )
    assert not engine.can_handle(p)


def test_router_does_not_pick_qaoa_by_default():
    """In the default router, a normal QUBO routes to classical/SA, not QAOA."""
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
    assert sol.engine_name != "qaoa"


def test_router_picks_qaoa_when_requested():
    """With prefer_qaoa=True, QAOA becomes eligible (and ought to be picked
    if no cheaper opt-in alternative is registered)."""
    router = default_router()
    np.random.seed(0)
    Q = np.random.randn(6, 6)
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={
            "size": 6,
            "prefer_qaoa": True,
            "p": 2,
            "max_iter": 20,
            "n_shots": 256,
            "seed": 0,
        },
    )
    sol = router.solve(p)
    # QAOA is eligible. Whether it wins depends on cost estimate.
    decision = sol.metadata["routing_decision"]
    qaoa_eligible = any(name == "qaoa" for name, _ in decision.candidates)
    assert qaoa_eligible


# ---------- correctness ----------


def test_finds_diagonal_optimum(engine):
    """Q = -I has unique optimum at all-ones."""
    n = 4
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={
            "size": n,
            "prefer_qaoa": True,
            "p": 2,
            "max_iter": 30,
            "n_shots": 256,
            "seed": 0,
        },
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(-n)
    assert all(v == 1.0 for v in sol.value["x"])


def test_finds_maxcut_ring_optimum(engine):
    """Anti-ferromagnetic 6-cycle: known optimum is -6 (6 cuts)."""
    n = 6
    Q = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        Q[i, i] -= 1.0
        Q[j, j] -= 1.0
        Q[i, j] += 1.0
        Q[j, i] += 1.0
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={
            "size": n,
            "prefer_qaoa": True,
            "p": 3,
            "max_iter": 100,
            "n_shots": 1024,
            "seed": 0,
        },
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(-6)


def test_metadata_records_qaoa_diagnostics(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={
            "size": 4,
            "prefer_qaoa": True,
            "p": 2,
            "max_iter": 20,
            "n_shots": 128,
            "seed": 0,
        },
    )
    sol = engine.solve(p)
    assert "qaoa_expected_energy" in sol.value
    assert "optimizer_iters" in sol.value
    assert sol.value["p"] == 2


# ---------- validation ----------


def test_rejects_nan_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.full((4, 4), np.nan), "qubo_solve": True},
        hints={"size": 4, "prefer_qaoa": True},
    )
    with pytest.raises(ValueError, match="NaN"):
        engine.solve(p)


def test_rejects_huge_p(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4, "prefer_qaoa": True, "p": 1000, "seed": 0},
    )
    with pytest.raises(ValueError, match="p must be"):
        engine.solve(p)


def test_rejects_huge_max_iter(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4, "prefer_qaoa": True, "max_iter": 10**9, "seed": 0},
    )
    with pytest.raises(ValueError, match="max_iter"):
        engine.solve(p)


def test_rejects_huge_n_shots(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(4), "qubo_solve": True},
        hints={"size": 4, "prefer_qaoa": True, "n_shots": 10**9, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_shots"):
        engine.solve(p)
