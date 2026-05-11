"""Tests for the MLX-accelerated SA engine.

These tests run the engine via its NumPy fallback when MLX is unavailable
(e.g., on CI that doesn't have Apple hardware). They verify:
- Correctness parity with the CPU SA engine.
- The engine declines small problems (where MLX overhead would hurt).
- Validation of inputs and resource caps.

MLX-specific behavior (actual GPU dispatch) is verified on Mac in dev.
"""

import numpy as np
import pytest

from metis import (
    Problem,
    ProblemKind,
    SimulatedAnnealing,
    SimulatedAnnealingMLX,
    is_mlx_available,
)


@pytest.fixture
def engine():
    return SimulatedAnnealingMLX()


def test_handles_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(500), "qubo_solve": True},
        hints={"size": 500},
    )
    assert engine.can_handle(p)


def test_declines_small_qubo_without_prefer_hint(engine):
    """Below MLX_MIN_PROFITABLE_N, engine should refuse."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(50), "qubo_solve": True},
        hints={"size": 50},
    )
    assert not engine.can_handle(p)


def test_accepts_small_qubo_with_prefer_hint(engine):
    """User can force-prefer MLX even for small problems."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(50), "qubo_solve": True},
        hints={"size": 50, "prefer_mlx": True},
    )
    assert engine.can_handle(p)


def test_rejects_constrained_qubo(engine):
    """MLX SA does not handle constraints; OR-Tools should win."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={
            "qubo_Q": np.eye(500),
            "qubo_solve": True,
            "linear_constraints": [{"coeffs": [1.0] * 500, "hi": 100}],
        },
        hints={"size": 500},
    )
    assert not engine.can_handle(p)


def test_finds_known_optimum_at_diagonal_qubo(engine):
    """Q = -I has unique optimum at all-ones with f = -n."""
    n = 256
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 100, "n_restarts": 4, "seed": 0},
    )
    sol = engine.solve(p)
    assert sol.value["fun"] == pytest.approx(-n)
    assert all(v == 1.0 for v in sol.value["x"])


def test_matches_cpu_sa_within_quality_tolerance():
    """At n=300 both engines should find solutions of comparable quality."""
    cpu_sa = SimulatedAnnealing()
    mlx_sa = SimulatedAnnealingMLX()

    n = 300
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    payload = {"qubo_Q": Q, "qubo_solve": True}
    hints = {"size": n, "n_sweeps": 100, "n_restarts": 4, "seed": 0}

    cpu_sol = cpu_sa.solve(Problem(ProblemKind.OPTIMIZATION, payload, hints))
    mlx_sol = mlx_sa.solve(Problem(ProblemKind.OPTIMIZATION, payload, hints))

    # Both should find an answer of similar quality.
    # Don't expect identical solutions because the algorithms differ
    # slightly (vectorized batch vs serial), but quality should be close.
    rel_gap = abs(mlx_sol.value["fun"] - cpu_sol.value["fun"]) / abs(
        cpu_sol.value["fun"]
    )
    assert rel_gap < 0.10, (
        f"Quality gap too large: mlx={mlx_sol.value['fun']:.2f}, "
        f"cpu={cpu_sol.value['fun']:.2f}, gap={rel_gap*100:.1f}%"
    )


def test_solution_x_is_consistent_with_fun(engine):
    """sol.value['fun'] should equal x^T Q x."""
    n = 300
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 50, "n_restarts": 2, "seed": 0},
    )
    sol = engine.solve(p)
    x = np.asarray(sol.value["x"])
    Qs = (Q + Q.T) / 2
    recomputed = float(x @ Qs @ x)
    assert sol.value["fun"] == pytest.approx(recomputed, rel=1e-3)


def test_backend_field_recorded(engine):
    """Solution should declare which backend was actually used."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(300), "qubo_solve": True},
        hints={"size": 300, "n_sweeps": 20, "n_restarts": 1, "seed": 0},
    )
    sol = engine.solve(p)
    assert sol.value["backend"] in ("mlx", "numpy_fallback")
    if is_mlx_available():
        assert sol.value["backend"] == "mlx"
    else:
        assert sol.value["backend"] == "numpy_fallback"


# ---------- Validation / caps ----------


def test_rejects_nan_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.full((500, 500), np.nan), "qubo_solve": True},
        hints={"size": 500},
    )
    with pytest.raises(ValueError, match="NaN"):
        engine.solve(p)


def test_rejects_complex_qubo(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(500, dtype=complex), "qubo_solve": True},
        hints={"size": 500},
    )
    with pytest.raises(ValueError, match="complex"):
        engine.solve(p)


def test_rejects_huge_n_sweeps(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(500), "qubo_solve": True},
        hints={"size": 500, "n_sweeps": 10**9, "n_restarts": 2, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_sweeps"):
        engine.solve(p)


def test_rejects_huge_n_restarts(engine):
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(500), "qubo_solve": True},
        hints={"size": 500, "n_sweeps": 50, "n_restarts": 10**6, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_restarts"):
        engine.solve(p)


def test_rejects_oversized_qubo(engine):
    """MLX engine has its own cap independent of CPU SA."""
    n = 50_000
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.zeros((1, 1)), "qubo_solve": True},  # placeholder
        hints={"size": n},
    )

    # Construct an intentionally oversized Q: we do this without actually
    # allocating the 50K x 50K matrix to keep the test fast.
    class FakeBigArray:
        shape = (n, n)

    p.payload["qubo_Q"] = FakeBigArray()
    assert not engine.can_handle(p)
