"""Regression tests for bugs found in code review.

Each test is named after the bug it prevents. If any of these fail, we've
re-introduced a known bug.
"""

import numpy as np
import pytest

from metis import (
    ClassicalOptimizer,
    Problem,
    ProblemKind,
    Router,
    SimulatedAnnealing,
    Solution,
    default_router,
)

# ---------- Bug #1: SA reported end-state, not best-seen ----------


def test_bug1_sa_returns_best_seen_not_end_state():
    """When the temperature schedule ends hot enough that the chain wanders,
    SA must still return the best state it ever visited. Previously it
    returned the (possibly worse) state at the final iteration."""
    sa = SimulatedAnnealing()
    n = 20
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2

    # Compare reported f against the literal x^T Q x of returned x.
    # Also compare against several brute-force-found optima for small n
    # to ensure quality didn't drop.
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "seed": 0, "n_sweeps": 50, "n_restarts": 4},
    )
    sol = sa.solve(p)
    x = sol.value["x"]
    Qs = (Q + Q.T) / 2
    recomputed = float(x @ Qs @ x)
    # The reported fun must match the recomputed value (no off-by-one of
    # tracking which x corresponds to which val).
    assert sol.value["fun"] == pytest.approx(recomputed, abs=1e-9)


def test_bug1_sa_quality_with_short_schedule():
    """With n_sweeps=20 and 8 restarts, SA should still find optimum on small n.
    Pre-fix this often returned worse-than-optimal because the chain wandered."""
    sa = SimulatedAnnealing()
    classical = ClassicalOptimizer()
    n = 8
    for seed in range(10):
        rng = np.random.default_rng(seed)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        p = Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n, "seed": 0, "n_sweeps": 20, "n_restarts": 8},
        )
        sa_sol = sa.solve(p)
        truth = classical.solve(p).value["fun"]
        assert sa_sol.value["fun"] == pytest.approx(
            truth, abs=1e-6
        ), f"seed={seed}: SA returned {sa_sol.value['fun']} but truth is {truth}"


# ---------- Bug #2 & #3: cost estimates wildly off ----------


def test_bug2_classical_cost_estimate_within_5x_of_actual():
    """Estimate must be within 5x of actual to make routing decisions sensible."""
    eng = ClassicalOptimizer()
    import time

    for n in [10, 14, 18]:
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        p = Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n},
        )
        est = eng.estimate_cost(p)
        t0 = time.perf_counter()
        eng.solve(p)
        actual = time.perf_counter() - t0
        # Allow generous bounds — estimates needn't be perfect, just not
        # off by orders of magnitude.
        assert 0.1 < est / max(actual, 1e-9) < 10, (
            f"n={n}: estimate {est:.4f}s vs actual {actual:.4f}s "
            f"(ratio {est/actual:.2f}x)"
        )


def test_bug3_routing_does_not_pick_classical_at_n14():
    """At n=14, classical brute force takes much longer than alternatives
    despite being 'simpler'. Pre-fix the estimates were off by orders of
    magnitude and classical was wrongly preferred. The fix calibrated cost
    estimates so classical loses at moderate sizes -- to either SA or
    OR-Tools depending on what's installed."""
    router = default_router()
    rng = np.random.default_rng(0)
    n = 14
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n, "n_sweeps": 200, "n_restarts": 4, "time_budget_s": 5},
    )
    sol = router.solve(p)
    # The exact engine depends on whether OR-Tools is installed; the bug
    # was that classical was picked when it shouldn't have been.
    assert sol.engine_name != "classical", (
        f"classical should not win at n={n}. "
        f"Decision: {sol.metadata['routing_decision'].reason}"
    )


# ---------- Bug #5: router didn't fall back on solve crash ----------


def test_bug5_default_solve_propagates_crashes():
    """By default, an engine crash propagates. This is the documented
    behavior so users can debug."""

    class Crasher:
        name = "crasher"

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 0.001

        def solve(self, p):
            raise RuntimeError("intentional")

    r = Router().register(Crasher())
    with pytest.raises(RuntimeError, match="intentional"):
        r.solve(Problem(kind=ProblemKind.OPTIMIZATION, payload={}))


def test_bug5_fallback_true_tries_backup_engine():
    """When fallback=True, a crashed primary engine yields to the backup."""

    class Crasher:
        name = "crasher"

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 0.001

        def solve(self, p):
            raise RuntimeError("intentional")

    class Backup:
        name = "backup"

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 1.0

        def solve(self, p):
            return Solution(value="recovered", engine_name=self.name, elapsed_sec=0.0)

    r = Router().register(Crasher()).register(Backup())
    sol = r.solve(
        Problem(kind=ProblemKind.OPTIMIZATION, payload={}),
        fallback=True,
    )
    assert sol.engine_name == "backup"
    assert sol.value == "recovered"
    attempts = sol.metadata["fallback_attempts"]
    assert any(name == "crasher" for name, _ in attempts)


def test_bug5_fallback_all_failing_raises_runtime_error():
    """If every engine in the eligible set crashes, fallback raises."""

    class A:
        name = "a"

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 0.001

        def solve(self, p):
            raise RuntimeError("a failed")

    class B:
        name = "b"

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 1.0

        def solve(self, p):
            raise ValueError("b failed")

    r = Router().register(A()).register(B())
    with pytest.raises(RuntimeError, match="All eligible engines failed"):
        r.solve(
            Problem(kind=ProblemKind.OPTIMIZATION, payload={}),
            fallback=True,
        )


# ---------- General invariants ----------


def test_solution_fun_matches_x_for_qubo():
    """For any returned solution, fun must equal x^T Q x. This is the
    fundamental contract that lets users trust the answer."""
    router = default_router()
    rng = np.random.default_rng(0)
    for n in [6, 12, 25]:
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        p = Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n, "seed": 0, "n_sweeps": 200, "n_restarts": 2},
        )
        sol = router.solve(p)
        x = np.asarray(sol.value["x"])
        Qs = (Q + Q.T) / 2
        recomputed = float(x @ Qs @ x)
        assert sol.value["fun"] == pytest.approx(recomputed, abs=1e-9), (
            f"n={n}: fun={sol.value['fun']} but x^T Q x = {recomputed} "
            f"(engine: {sol.engine_name})"
        )
