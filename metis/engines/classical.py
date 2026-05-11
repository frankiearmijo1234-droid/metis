"""Classical optimization engine.

Uses scipy.optimize for continuous problems and a simple branch-and-bound
fallback for small QUBO. This is the boring, reliable baseline. Often beats
the fancier engines at moderate sizes because decades of numerical
optimization research live behind scipy.

Handles ProblemKind.OPTIMIZATION with payload of the form:
    {
        "objective": callable(x: np.ndarray) -> float,
        "x0": np.ndarray of shape (n,),    # initial guess
        "bounds": [(lo, hi), ...] | None,
        "method": "BFGS" | "L-BFGS-B" | "Nelder-Mead" | None,
    }

Or for QUBO/Ising problems:
    {
        "qubo_Q": np.ndarray of shape (n, n),  # symmetric
        "qubo_solve": True,
    }

The hint `size` is used for cost estimation. Up to size=20 we'll happily
brute-force a QUBO; past that we route elsewhere.

Defensive validation on every input. The Python API is a public attack
surface in addition to the MCP layer.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Engine-level hard caps. The MCP layer has its own caps; these are
# defense-in-depth for callers using the Python API directly.
MAX_QUBO_N = 22  # 2^22 = 4M iterations; past this -> infeasible
MAX_CONTINUOUS_DIM = 10_000  # absurd ceiling for safety


class ClassicalOptimizer:
    name = "classical"

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.OPTIMIZATION:
            return False
        p = problem.payload
        # Refuse constrained problems — this engine has no constraint
        # handling. Routing one here would silently ignore the constraints
        # and return a wrong answer.
        if p.get("linear_constraints") or p.get("ilp_solve"):
            return False
        # Continuous problem: needs an objective callable
        if callable(p.get("objective")) and "x0" in p:
            return True
        # QUBO problem (unconstrained)
        if "qubo_Q" in p and p.get("qubo_solve"):
            return True
        return False

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        size = problem.hints.get("size") or self._infer_size(p)
        if "qubo_Q" in p:
            # Brute-force QUBO: 2^n iterations * O(n^2) work each.
            # Measured constant ~5e-6 s per iteration in pure Python on a
            # mid-range machine (calibrated against actual runs at n=8..18).
            # Refuse beyond MAX_QUBO_N to avoid multi-minute runtimes.
            if size > MAX_QUBO_N:
                return float("inf")
            return 5e-6 * (2**size)
        # Continuous: scipy is great up to ~thousands of dims for unconstrained
        if size > MAX_CONTINUOUS_DIM:
            return float("inf")
        if size <= 2000:
            return 0.001 + 0.0001 * size
        return 0.5 + 0.001 * size  # still feasible, just slower

    def solve(self, problem: Problem) -> Solution:
        from scipy.optimize import minimize  # local import keeps cold start fast

        t0 = time.perf_counter()
        p = problem.payload

        if "qubo_Q" in p:
            Q = self._validate_qubo(p["qubo_Q"])
            value = self._solve_qubo_bruteforce(Q)
        else:
            x0 = self._validate_x0(p["x0"])
            objective = p["objective"]
            method = p.get("method") or "L-BFGS-B"
            # Track whether the objective ever returned non-finite values.
            # If it did, the final `success` flag from scipy is misleading
            # because we substituted sentinels.
            nonfinite_count = [0]
            SENTINEL = 1e18

            def wrapped_obj(x):
                # Pass a copy so the user's function can't mutate scipy's
                # internal state.
                result = objective(np.array(x, copy=True))
                fval = float(result)
                if not np.isfinite(fval):
                    nonfinite_count[0] += 1
                    return SENTINEL
                return fval

            res = minimize(
                wrapped_obj,
                x0,
                method=method,
                bounds=p.get("bounds"),
            )
            # If the reported best value equals the sentinel, the objective
            # returned non-finite for every evaluation; we have no real answer.
            real_success = bool(res.success) and float(res.fun) < SENTINEL / 2
            value = {
                "x": res.x,
                "fun": float(res.fun),
                "success": real_success,
                "n_iter": int(res.get("nit", 0)),
                "message": str(res.message),
                "nonfinite_evaluations": nonfinite_count[0],
            }
            if not real_success and nonfinite_count[0] > 0:
                value["warning"] = (
                    f"Objective returned non-finite values "
                    f"{nonfinite_count[0]} times; result is unreliable."
                )

        elapsed = time.perf_counter() - t0
        return Solution(
            value=value, engine_name=self.name, elapsed_sec=elapsed, metadata={}
        )

    @staticmethod
    def _validate_qubo(Q_input) -> np.ndarray:
        # Reject complex dtype explicitly rather than silently casting.
        arr = np.asarray(Q_input)
        if np.iscomplexobj(arr):
            raise ValueError("QUBO Q must be real-valued, got complex dtype")
        Q = np.asarray(arr, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"QUBO Q must be square, got shape {Q.shape}")
        n = Q.shape[0]
        if n < 1:
            raise ValueError("QUBO Q must have at least one variable")
        if n > MAX_QUBO_N:
            raise ValueError(f"QUBO size {n} exceeds classical engine cap {MAX_QUBO_N}")
        if not np.all(np.isfinite(Q)):
            raise ValueError("QUBO Q contains NaN or inf values")
        return Q

    @staticmethod
    def _validate_x0(x0_input) -> np.ndarray:
        x0 = np.asarray(x0_input, dtype=float)
        if x0.ndim != 1:
            raise ValueError(f"x0 must be 1-D, got shape {x0.shape}")
        if x0.size < 1 or x0.size > MAX_CONTINUOUS_DIM:
            raise ValueError(f"x0 dim {x0.size} outside [1, {MAX_CONTINUOUS_DIM}]")
        if not np.all(np.isfinite(x0)):
            raise ValueError("x0 contains NaN or inf values")
        return x0

    @staticmethod
    def _infer_size(payload: dict) -> int:
        if "qubo_Q" in payload:
            return int(np.asarray(payload["qubo_Q"]).shape[0])
        if "x0" in payload:
            return int(np.asarray(payload["x0"]).size)
        return 1

    @staticmethod
    def _solve_qubo_bruteforce(Q: np.ndarray) -> dict:
        """Find x in {0,1}^n minimizing x^T Q x. O(2^n * n^2)."""
        n = Q.shape[0]
        best_x = None
        best_val = float("inf")
        for i in range(2**n):
            x = np.array([(i >> b) & 1 for b in range(n)], dtype=float)
            val = float(x @ Q @ x)
            if val < best_val:
                best_val = val
                best_x = x
        return {"x": best_x, "fun": best_val, "method": "bruteforce"}
