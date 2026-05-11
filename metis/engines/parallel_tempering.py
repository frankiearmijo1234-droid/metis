"""Parallel tempering (replica-exchange MCMC) engine.

Runs N replicas of SA at a geometric ladder of temperatures. After each
sweep, adjacent replicas attempt a state swap with the Metropolis criterion

    P(swap) = min(1, exp((1/T_i - 1/T_{i+1}) * (E_i - E_{i+1})))

This makes the high-temperature replicas wander widely while the low-T ones
refine, and the swaps feed promising states down the temperature ladder.
For rugged energy landscapes (frustrated QUBO, spin glasses) PT typically
finds better optima than SA at the same compute budget.

Compared to SA:
- More replicas at once -> more total work for a given n_sweeps, but better
  exploration -> often fewer total sweeps needed for the same quality.
- The cost estimate accounts for n_replicas of work.

Handles ProblemKind.OPTIMIZATION with QUBO payload (same shape as SA).
Hints respected:
    - n_sweeps: int                      MC sweeps per replica (default scales with n)
    - n_replicas: int                    number of temperature replicas (default 8)
    - T_min: float                       coldest temperature (default 0.001)
    - T_max: float | None                hottest temperature (default: from |Q|)
    - swap_interval: int                 sweeps between swap attempts (default 1)
    - seed: int
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Engine caps (defense-in-depth).
MAX_QUBO_N = 5_000
MAX_N_SWEEPS = 100_000
MAX_N_REPLICAS = 64
MIN_N_REPLICAS = 2


class ParallelTempering:
    name = "parallel_tempering"

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.OPTIMIZATION:
            return False
        p = problem.payload
        if p.get("linear_constraints") or p.get("ilp_solve"):
            return False
        if "qubo_Q" not in p or not p.get("qubo_solve", False):
            return False
        try:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
        except Exception:
            return False
        return 1 <= n <= MAX_QUBO_N

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        try:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
        except Exception:
            return float("inf")
        if n > MAX_QUBO_N:
            return float("inf")
        n_sweeps = problem.hints.get("n_sweeps") or self._default_sweeps(n)
        n_replicas = problem.hints.get("n_replicas") or 8
        n_sweeps = min(n_sweeps, MAX_N_SWEEPS)
        n_replicas = min(max(n_replicas, MIN_N_REPLICAS), MAX_N_REPLICAS)
        # Same vectorized batched algorithm as MLX-SA NumPy fallback,
        # but with n_replicas chains instead of n_restarts. Per-iteration
        # cost calibrated on this CPU.
        return 0.005 + 1e-8 * n_sweeps * n_replicas * n * n

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        p = problem.payload

        # Validate Q
        Q_in = np.asarray(p["qubo_Q"])
        if np.iscomplexobj(Q_in):
            raise ValueError("QUBO Q must be real-valued, got complex dtype")
        Q = np.asarray(Q_in, dtype=np.float64)
        n = Q.shape[0]
        if Q.ndim != 2 or n != Q.shape[1]:
            raise ValueError(f"QUBO Q must be square, got {Q.shape}")
        if n < 1 or n > MAX_QUBO_N:
            raise ValueError(f"n={n} outside [1, {MAX_QUBO_N}]")
        if not np.all(np.isfinite(Q)):
            raise ValueError("QUBO Q contains NaN or inf values")
        Qs = (Q + Q.T) / 2.0

        # Hints
        n_sweeps = problem.hints.get("n_sweeps") or self._default_sweeps(n)
        n_replicas = problem.hints.get("n_replicas") or 8
        T_min = float(problem.hints.get("T_min", 0.001))
        T_max = problem.hints.get("T_max")
        swap_interval = int(problem.hints.get("swap_interval", 1))
        seed = problem.hints.get("seed")

        if not isinstance(n_sweeps, int) or n_sweeps < 1 or n_sweeps > MAX_N_SWEEPS:
            raise ValueError(f"n_sweeps must be int in [1, {MAX_N_SWEEPS}]")
        if (
            not isinstance(n_replicas, int)
            or n_replicas < MIN_N_REPLICAS
            or n_replicas > MAX_N_REPLICAS
        ):
            raise ValueError(
                f"n_replicas must be int in [{MIN_N_REPLICAS}, {MAX_N_REPLICAS}]"
            )
        if T_min <= 0:
            raise ValueError("T_min must be > 0")
        if T_max is not None and (T_max <= T_min):
            raise ValueError("T_max must be > T_min")
        if swap_interval < 1 or swap_interval > 10_000:
            raise ValueError("swap_interval must be in [1, 10000]")

        if T_max is None:
            T_max = max(np.abs(Qs).sum() / n, 1.0)

        # Geometric temperature ladder: T_i = T_min * (T_max/T_min)^(i/(R-1))
        ratio = T_max / T_min
        if n_replicas == 1:
            temperatures = np.array([T_min])
        else:
            temperatures = T_min * (ratio ** (np.arange(n_replicas) / (n_replicas - 1)))

        rng = np.random.default_rng(seed)

        # Run PT
        best_x, best_val, swap_stats = self._run_pt(
            Qs,
            n_sweeps,
            n_replicas,
            temperatures,
            swap_interval,
            rng,
        )

        elapsed = time.perf_counter() - t0
        return Solution(
            value={
                "x": best_x,
                "fun": float(best_val),
                "method": "parallel_tempering",
                "n_sweeps": n_sweeps,
                "n_replicas": n_replicas,
                "T_min": float(T_min),
                "T_max": float(T_max),
                "swap_acceptance_rate": float(
                    swap_stats["accepted"] / max(swap_stats["attempted"], 1)
                ),
            },
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"size": n, "n_replicas": n_replicas},
        )

    @staticmethod
    def _default_sweeps(n: int) -> int:
        return min(max(200, 20 * n), MAX_N_SWEEPS)

    @staticmethod
    def _run_pt(
        Qs: np.ndarray,
        n_sweeps: int,
        n_replicas: int,
        temperatures: np.ndarray,
        swap_interval: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, dict]:
        """Run PT and return (best_x, best_val, swap_stats).

        Vectorizes the single-bit-flip move across all replicas, the same
        way the MLX-SA engine does for restarts. The crucial difference
        from SA: replicas run at *different* temperatures, and we attempt
        swaps between adjacent ladder rungs.
        """
        n = Qs.shape[0]
        diag = np.diag(Qs)

        # Initialize each replica at a random bitstring
        x = rng.integers(0, 2, size=(n_replicas, n)).astype(np.float64)
        Qx = x @ Qs  # (R, n) -- precomputed inner product
        val = np.sum(Qx * x, axis=1)  # (R,)
        best_x = x[0].copy()
        best_val = val.min()
        best_idx = int(val.argmin())
        best_x = x[best_idx].copy()

        swap_attempted = 0
        swap_accepted = 0

        # Inverse temperatures, used for swap criterion
        beta = 1.0 / temperatures  # (R,)

        for sweep in range(n_sweeps):
            # ---- Bit-flip sweep on every replica ----
            order = rng.permutation(n)
            for i in order:
                xi = x[:, i]  # (R,)
                row_dot = Qx[:, i]  # (R,)
                # dE for flipping bit i on each replica:
                sign = 1 - 2 * xi
                dE = sign * (diag[i] + 2 * (row_dot - diag[i] * xi))
                # Acceptance per replica with that replica's temperature
                with np.errstate(over="ignore"):
                    accept_p = np.exp(-dE * beta)
                rand = rng.random(size=n_replicas)
                accept = (dE < 0) | (rand < accept_p)
                # Apply flips
                delta_x = np.where(accept, 1 - 2 * xi, 0.0)
                x[:, i] = x[:, i] + delta_x
                val = val + accept.astype(np.float64) * dE
                # Update Qx incrementally (rank-1 update on column i)
                Qx = Qx + np.outer(delta_x, Qs[i, :])

            # ---- Replica swap step ----
            if (sweep + 1) % swap_interval == 0 and n_replicas >= 2:
                # Alternate parity to avoid bias: even sweeps swap (0,1),(2,3),...
                # odd sweeps swap (1,2),(3,4),...
                start = sweep % 2
                for r in range(start, n_replicas - 1, 2):
                    # Metropolis swap criterion:
                    # P = min(1, exp((beta_r - beta_{r+1}) * (E_r - E_{r+1})))
                    delta_beta = beta[r] - beta[r + 1]
                    delta_E = val[r] - val[r + 1]
                    log_p = delta_beta * delta_E
                    swap_attempted += 1
                    if log_p >= 0 or rng.random() < math.exp(log_p):
                        # Swap states r and r+1
                        x[[r, r + 1]] = x[[r + 1, r]]
                        Qx[[r, r + 1]] = Qx[[r + 1, r]]
                        val[r], val[r + 1] = val[r + 1], val[r]
                        swap_accepted += 1

            # Track global best
            cur_min = val.min()
            if cur_min < best_val:
                best_val = float(cur_min)
                best_x = x[int(val.argmin())].copy()

        return (
            best_x,
            best_val,
            {
                "attempted": swap_attempted,
                "accepted": swap_accepted,
            },
        )
