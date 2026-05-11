"""OR-Tools engine: constraint programming and integer linear programming.

Wraps Google's OR-Tools CP-SAT solver, which is the state-of-the-art free
solver for combinatorial optimization. CP-SAT solves QUBO problems exactly
(unlike SA which is heuristic) and also handles a much wider class:
problems with linear constraints, integer variables, scheduling structure,
etc.

Why this is in metis:
- Many real problems aren't pure QUBO. They have constraints like "exactly
  k items selected" or "x_i + x_j <= 1 for some pairs". SA can encode these
  as soft penalties, but a real CP solver finds exact answers faster.
- For pure unconstrained QUBO at moderate size (n ~ 50-200), CP-SAT is
  often faster than SA and returns a proven optimum. The router prefers it
  in that range.
- For very large problems or when "good enough fast" beats "exact slow",
  the router falls back to SA.

Handles ProblemKind.OPTIMIZATION with two payload forms:

1. QUBO with optional constraints:
    {
        "qubo_Q": np.ndarray (n, n),
        "qubo_solve": True,
        "linear_constraints": [             # optional
            {"coeffs": [...], "lo": int, "hi": int},
            ...
        ],
    }

2. Integer linear program:
    {
        "ilp_solve": True,
        "objective_coeffs": [...],          # length n
        "var_lo": [...] | None,             # length n; default 0
        "var_hi": [...] | None,             # length n; default 1 (binary)
        "linear_constraints": [
            {"coeffs": [...], "lo": int, "hi": int},
            ...
        ],
        "minimize": True,                   # default True
    }

Hints respected:
    - time_budget_s: float, default 30.0
    - n_workers: int, default 4
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Engine-level caps. CP-SAT is fast but problems can still get unwieldy.
MAX_QUBO_N = 1_000  # state-of-the-art CP-SAT handles 1000s of binary vars
MAX_ILP_N = 10_000
MAX_CONSTRAINTS = 100_000
MAX_TIME_BUDGET_S = 600  # 10 minutes ceiling
DEFAULT_TIME_BUDGET_S = 30.0


# CP-SAT optimizes integer objectives. To handle real-valued QUBO matrices,
# we scale up by this factor and round, then divide back. 1e6 gives ~6 digits
# of precision which is plenty for routing decisions.
COEFF_SCALE = 1_000_000


class ORTools:
    name = "ortools_cpsat"

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.OPTIMIZATION:
            return False
        p = problem.payload
        if "qubo_Q" in p and p.get("qubo_solve"):
            try:
                n = int(np.asarray(p["qubo_Q"]).shape[0])
            except Exception:
                return False
            return 1 <= n <= MAX_QUBO_N
        if p.get("ilp_solve"):
            n = len(p.get("objective_coeffs", []))
            return 1 <= n <= MAX_ILP_N
        return False

    def estimate_cost(self, problem: Problem) -> float:
        """CP-SAT is dramatically faster than SA on constrained problems but
        can be slower than SA on pure unconstrained QUBO at large n.

        Heuristic:
        - QUBO with constraints: CP-SAT wins; cost = small constant + linear in n
        - QUBO without constraints, n <= 100: CP-SAT often wins
        - QUBO without constraints, n > 200: SA wins on average
        - ILP: CP-SAT is the only honest answer
        """
        p = problem.payload
        time_budget = float(problem.hints.get("time_budget_s", DEFAULT_TIME_BUDGET_S))
        if "qubo_Q" in p:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
            n_constraints = len(p.get("linear_constraints", []))
            if n_constraints > 0:
                # Constraint problems: CP-SAT scales well
                return 0.05 + 0.001 * n + 0.0001 * n_constraints
            # Pure QUBO: cost grows roughly as n^2 in CP-SAT for binary
            # quadratic. Capped by time_budget.
            est = 0.01 + 5e-5 * n * n
            return min(est, time_budget)
        if p.get("ilp_solve"):
            n = len(p["objective_coeffs"])
            n_constraints = len(p.get("linear_constraints", []))
            est = 0.05 + 0.001 * n + 0.0001 * n_constraints
            return min(est, time_budget)
        return float("inf")

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        p = problem.payload
        if "qubo_Q" in p and p.get("qubo_solve"):
            value = self._solve_qubo(p, problem.hints)
        elif p.get("ilp_solve"):
            value = self._solve_ilp(p, problem.hints)
        else:
            raise ValueError("OR-Tools engine called with unrecognized payload")
        elapsed = time.perf_counter() - t0
        return Solution(
            value=value, engine_name=self.name, elapsed_sec=elapsed, metadata={}
        )

    # ---------- QUBO via CP-SAT ----------

    def _solve_qubo(self, payload: dict, hints: dict) -> dict:
        from ortools.sat.python import cp_model

        Q = self._validate_qubo(payload["qubo_Q"])
        n = Q.shape[0]
        constraints = self._validate_constraints(
            payload.get("linear_constraints", []),
            n,
        )
        time_budget = self._validate_time_budget(hints.get("time_budget_s"))
        n_workers = self._validate_workers(hints.get("n_workers"))

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(n)]

        # Symmetrize Q: x^T Q x = sum_i Q_ii x_i + sum_{i<j} (Q_ij + Q_ji) x_i x_j.
        # CP-SAT needs integer coefficients; scale and round.
        Qs = (Q + Q.T) / 2.0

        # Build objective. Quadratic terms are encoded by introducing a
        # product variable z_ij = x_i AND x_j (since both binary). For
        # diagonal terms it's just x_i (since x_i^2 = x_i for booleans).
        obj_terms = []
        for i in range(n):
            coeff = int(round(Qs[i, i] * COEFF_SCALE))
            if coeff != 0:
                obj_terms.append(coeff * x[i])

        for i in range(n):
            for j in range(i + 1, n):
                # Coefficient on x_i x_j in x^T Qs x is 2*Qs[i,j].
                coeff = int(round(2 * Qs[i, j] * COEFF_SCALE))
                if coeff == 0:
                    continue
                z = model.NewBoolVar(f"z_{i}_{j}")
                # z = x_i AND x_j
                model.AddBoolAnd([x[i], x[j]]).OnlyEnforceIf(z)
                model.AddBoolOr([x[i].Not(), x[j].Not()]).OnlyEnforceIf(z.Not())
                obj_terms.append(coeff * z)

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # Linear constraints
        for c in constraints:
            terms = [
                int(round(c["coeffs"][i] * COEFF_SCALE)) * x[i]
                for i in range(n)
                if c["coeffs"][i] != 0
            ]
            lo = int(round(c["lo"] * COEFF_SCALE)) if c["lo"] is not None else None
            hi = int(round(c["hi"] * COEFF_SCALE)) if c["hi"] is not None else None
            if terms:
                expr = sum(terms)
                if lo is not None and hi is not None:
                    model.AddLinearConstraint(expr, lo, hi)
                elif lo is not None:
                    model.Add(expr >= lo)
                elif hi is not None:
                    model.Add(expr <= hi)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget
        solver.parameters.num_search_workers = n_workers
        status = solver.Solve(model)

        status_name = solver.StatusName(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            x_vals = np.array([solver.Value(x[i]) for i in range(n)], dtype=float)
            # Recompute fun in original (real) units for transparency.
            fun = float(x_vals @ Qs @ x_vals)
            return {
                "x": x_vals,
                "fun": fun,
                "method": "ortools_cpsat",
                "status": status_name,
                "is_optimal": status == cp_model.OPTIMAL,
                "n_constraints": len(constraints),
            }

        # Infeasible or unknown
        return {
            "x": None,
            "fun": float("inf"),
            "method": "ortools_cpsat",
            "status": status_name,
            "is_optimal": False,
            "n_constraints": len(constraints),
            "warning": f"solver returned {status_name}; no solution found",
        }

    # ---------- ILP via CP-SAT ----------

    def _solve_ilp(self, payload: dict, hints: dict) -> dict:
        from ortools.sat.python import cp_model

        coeffs = np.asarray(payload["objective_coeffs"], dtype=float)
        n = coeffs.size
        if n < 1 or n > MAX_ILP_N:
            raise ValueError(f"ILP size {n} outside [1, {MAX_ILP_N}]")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("objective_coeffs contains non-finite values")

        var_lo = payload.get("var_lo")
        var_hi = payload.get("var_hi")
        if var_lo is None:
            var_lo = [0] * n
        if var_hi is None:
            var_hi = [1] * n
        if len(var_lo) != n or len(var_hi) != n:
            raise ValueError("var_lo and var_hi must have length n")
        for lo, hi in zip(var_lo, var_hi):
            if not isinstance(lo, int) or not isinstance(hi, int):
                raise ValueError("var bounds must be integers")
            if lo > hi:
                raise ValueError(f"var bound {lo} > {hi}")

        constraints = self._validate_constraints(
            payload.get("linear_constraints", []),
            n,
        )
        minimize = payload.get("minimize", True)
        time_budget = self._validate_time_budget(hints.get("time_budget_s"))
        n_workers = self._validate_workers(hints.get("n_workers"))

        model = cp_model.CpModel()
        x = [model.NewIntVar(int(var_lo[i]), int(var_hi[i]), f"x{i}") for i in range(n)]

        obj_terms = [
            int(round(coeffs[i] * COEFF_SCALE)) * x[i]
            for i in range(n)
            if coeffs[i] != 0
        ]
        if obj_terms:
            if minimize:
                model.Minimize(sum(obj_terms))
            else:
                model.Maximize(sum(obj_terms))

        for c in constraints:
            terms = [
                int(round(c["coeffs"][i] * COEFF_SCALE)) * x[i]
                for i in range(n)
                if c["coeffs"][i] != 0
            ]
            lo = int(round(c["lo"] * COEFF_SCALE)) if c["lo"] is not None else None
            hi = int(round(c["hi"] * COEFF_SCALE)) if c["hi"] is not None else None
            if terms:
                expr = sum(terms)
                if lo is not None and hi is not None:
                    model.AddLinearConstraint(expr, lo, hi)
                elif lo is not None:
                    model.Add(expr >= lo)
                elif hi is not None:
                    model.Add(expr <= hi)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget
        solver.parameters.num_search_workers = n_workers
        status = solver.Solve(model)

        status_name = solver.StatusName(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            x_vals = np.array([solver.Value(x[i]) for i in range(n)], dtype=float)
            fun = float(coeffs @ x_vals)
            return {
                "x": x_vals,
                "fun": fun,
                "method": "ortools_cpsat",
                "status": status_name,
                "is_optimal": status == cp_model.OPTIMAL,
                "n_constraints": len(constraints),
            }
        return {
            "x": None,
            "fun": float("inf") if minimize else float("-inf"),
            "method": "ortools_cpsat",
            "status": status_name,
            "is_optimal": False,
            "n_constraints": len(constraints),
            "warning": f"solver returned {status_name}; no solution found",
        }

    # ---------- validation helpers ----------

    @staticmethod
    def _validate_qubo(Q_input) -> np.ndarray:
        arr = np.asarray(Q_input)
        if np.iscomplexobj(arr):
            raise ValueError("QUBO Q must be real-valued, got complex dtype")
        Q = np.asarray(arr, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"QUBO Q must be square, got shape {Q.shape}")
        n = Q.shape[0]
        if n < 1 or n > MAX_QUBO_N:
            raise ValueError(f"QUBO size {n} outside [1, {MAX_QUBO_N}]")
        if not np.all(np.isfinite(Q)):
            raise ValueError("QUBO Q contains NaN or inf values")
        return Q

    @staticmethod
    def _validate_constraints(constraints, n):
        if not isinstance(constraints, list):
            raise ValueError("linear_constraints must be a list")
        if len(constraints) > MAX_CONSTRAINTS:
            raise ValueError(
                f"too many constraints: {len(constraints)} > {MAX_CONSTRAINTS}"
            )
        out = []
        for i, c in enumerate(constraints):
            if not isinstance(c, dict):
                raise ValueError(f"constraint {i} must be a dict")
            coeffs = c.get("coeffs", [])
            if len(coeffs) != n:
                raise ValueError(
                    f"constraint {i}: coeffs has length {len(coeffs)}, " f"expected {n}"
                )
            for v in coeffs:
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    raise ValueError(f"constraint {i}: coeffs has invalid value")
            lo = c.get("lo")
            hi = c.get("hi")
            if lo is None and hi is None:
                raise ValueError(f"constraint {i}: must have lo or hi")
            for bound, label in [(lo, "lo"), (hi, "hi")]:
                if bound is not None and (
                    not isinstance(bound, (int, float)) or not np.isfinite(bound)
                ):
                    raise ValueError(f"constraint {i}: {label} invalid")
            out.append({"coeffs": list(coeffs), "lo": lo, "hi": hi})
        return out

    @staticmethod
    def _validate_time_budget(t):
        if t is None:
            return DEFAULT_TIME_BUDGET_S
        if not isinstance(t, (int, float)) or t <= 0 or t > MAX_TIME_BUDGET_S:
            raise ValueError(f"time_budget_s must be in (0, {MAX_TIME_BUDGET_S}]")
        return float(t)

    @staticmethod
    def _validate_workers(w):
        if w is None:
            return 4
        if not isinstance(w, int) or w < 1 or w > 32:
            raise ValueError("n_workers must be int in [1, 32]")
        return w
