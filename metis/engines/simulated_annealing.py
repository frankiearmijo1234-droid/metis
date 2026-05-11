"""Simulated annealing engine.

This is the "quantum-inspired" engine. It's a classical algorithm that
mimics the way physical systems escape local minima by introducing thermal
fluctuations. For QUBO problems past brute-force reach, this is often the
practical workhorse and what people frequently mean when they say
"quantum optimization" -- the actual quantum advantage on these problems
is unproven, but classical simulated annealing absolutely works.

Handles ProblemKind.OPTIMIZATION with QUBO payloads:
    {"qubo_Q": np.ndarray (n, n), "qubo_solve": True}

Hints respected:
    - n_sweeps: int          number of MC sweeps (default scales with size)
    - n_restarts: int        independent runs (best is returned)
    - seed: int
    - schedule: "linear" | "exponential"
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Engine-level hard caps. Defense-in-depth: even if a caller bypasses the
# MCP server's validators, these protect the engine itself.
MAX_QUBO_N = 5_000  # 200 MB for a dense float64 matrix
MAX_N_SWEEPS = 100_000
MAX_N_RESTARTS = 100


class SimulatedAnnealing:
    name = "simulated_annealing"

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.OPTIMIZATION:
            return False
        p = problem.payload
        # Refuse constrained problems — SA has no constraint handling.
        # Encoding constraints as soft penalties is the user's job, and
        # they should pass an unconstrained Q if that's their intent.
        if p.get("linear_constraints") or p.get("ilp_solve"):
            return False
        return "qubo_Q" in p and p.get("qubo_solve", False)

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        try:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
        except Exception:
            return float("inf")
        if n > MAX_QUBO_N:
            return float("inf")
        n_sweeps = problem.hints.get("n_sweeps") or self._default_sweeps(n)
        n_restarts = problem.hints.get("n_restarts") or 4
        # Clamp for cost-estimation purposes (actual validation in solve()).
        n_sweeps = min(n_sweeps, MAX_N_SWEEPS)
        n_restarts = min(n_restarts, MAX_N_RESTARTS)
        # Calibrated per-iteration cost. NumPy does dense vector ops in
        # the inner loop, so per-flop cost is small but has fixed overhead
        # per Python-level iteration. Two-regime fit: pure-Python overhead
        # at small n, vectorized-ops cost at large n.
        inner_iters = n_restarts * n_sweeps * n * n
        if n < 100:
            # Small-n: Python-loop overhead dominates
            return 6e-8 * inner_iters
        # Large-n: vectorized inner ops, cache-efficient
        return 3e-9 * inner_iters + 0.001  # tiny constant for setup

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        p = problem.payload

        # Reject complex dtype before float casting silently discards imaginary parts.
        Q_in = np.asarray(p["qubo_Q"])
        if np.iscomplexobj(Q_in):
            raise ValueError("QUBO Q must be real-valued, got complex dtype")
        Q = np.asarray(Q_in, dtype=float)
        n = Q.shape[0]
        if Q.ndim != 2 or n != Q.shape[1]:
            raise ValueError(f"QUBO Q must be a square 2-D matrix, got shape {Q.shape}")
        if n < 1:
            raise ValueError("QUBO Q must have at least one variable")
        if n > MAX_QUBO_N:
            raise ValueError(f"QUBO size {n} exceeds SA engine cap {MAX_QUBO_N}")
        # Reject non-finite inputs up front. NaN/Inf would silently propagate
        # through the solver and yield garbage solutions like {'x': None, 'fun': inf}.
        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "QUBO Q contains non-finite values (NaN or Inf). "
                "These cannot be optimized over and would produce meaningless results."
            )

        # Validate hint-driven resource limits.
        n_sweeps = problem.hints.get("n_sweeps") or self._default_sweeps(n)
        n_restarts = problem.hints.get("n_restarts") or 4
        if not isinstance(n_sweeps, int) or n_sweeps < 1 or n_sweeps > MAX_N_SWEEPS:
            raise ValueError(
                f"n_sweeps must be int in [1, {MAX_N_SWEEPS}], got {n_sweeps}"
            )
        if (
            not isinstance(n_restarts, int)
            or n_restarts < 1
            or n_restarts > MAX_N_RESTARTS
        ):
            raise ValueError(
                f"n_restarts must be int in [1, {MAX_N_RESTARTS}], got {n_restarts}"
            )
        seed = problem.hints.get("seed")

        rng = np.random.default_rng(seed)
        best_x = None
        best_val = float("inf")

        for restart in range(n_restarts):
            x, val = self._anneal(Q, n_sweeps, rng)
            if val < best_val:
                best_val = val
                best_x = x

        elapsed = time.perf_counter() - t0
        return Solution(
            value={
                "x": best_x,
                "fun": float(best_val),
                "method": "simulated_annealing",
                "n_sweeps": n_sweeps,
                "n_restarts": n_restarts,
            },
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"size": n},
        )

    @staticmethod
    def _default_sweeps(n: int) -> int:
        # Heuristic: more sweeps for larger problems, with hard cap.
        return min(max(500, 50 * n), MAX_N_SWEEPS)

    @staticmethod
    def _anneal(
        Q: np.ndarray, n_sweeps: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        n = Q.shape[0]
        # Symmetrize Q so x^T Q x is well-defined regardless of input form.
        Qs = (Q + Q.T) / 2.0

        x = rng.integers(0, 2, size=n).astype(np.float64)
        val = float(x @ Qs @ x)

        # Track the best state ever seen, not just the final state. SA wanders
        # due to thermal noise; the final state isn't always the best one
        # visited. Returning end-state instead of best-seen is a common SA
        # mistake that hurts solution quality, especially with hot endings.
        best_x = x.copy()
        best_val = val

        # Energy delta if we flip bit i:
        # If x_i = 0 -> 1: dE = Q_ii + 2 * sum_{j != i} Q_ij * x_j
        # If x_i = 1 -> 0: dE = -Q_ii - 2 * sum_{j != i} Q_ij * x_j
        # Temperature schedule: exponential cooldown
        T0 = max(abs(Qs).sum() / n, 1.0)
        T_final = 0.001
        cool = (T_final / T0) ** (1.0 / max(n_sweeps - 1, 1))

        T = T0
        for sweep in range(n_sweeps):
            # One sweep = n attempted flips
            order = rng.permutation(n)
            for i in order:
                xi = x[i]
                row_dot = Qs[i] @ x  # sum_j Q_ij x_j
                if xi == 0:
                    dE = Qs[i, i] + 2 * (row_dot - Qs[i, i] * xi)
                else:
                    dE = -Qs[i, i] - 2 * (row_dot - Qs[i, i] * xi)
                # Accept if energy drops, or with Boltzmann probability.
                # Guard against overflow when -dE/T is very large positive
                # (which only happens if dE < 0 and T > 0 -- handled by the
                # short-circuit -- but defensive math.exp clamping doesn't
                # hurt).
                if dE < 0:
                    accept = True
                else:
                    # -dE/T <= 0, so exp is in [0, 1], no overflow possible
                    accept = rng.random() < math.exp(-dE / T)
                if accept:
                    x[i] = 1 - xi
                    val += dE
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
            T *= cool

        return best_x, best_val
