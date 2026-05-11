"""Markov Chain Monte Carlo (MCMC) sampling engine.

While the SA/PT/MLX-SA engines do *optimization* (find the argmin), this
engine does *sampling*: given an unnormalized probability distribution
π(x) ∝ exp(-E(x)/T), draw representative samples from it.

Why this is in metis:
- Sampling and optimization are different problems. Optimization wants
  THE best x; sampling wants a representative ensemble of x's.
- Bayesian inference, statistical physics, and Boltzmann sampling all
  reduce to "draw samples from this distribution."
- MLX runs n_chains in parallel as batch tensor ops, giving real GPU
  speedup on Apple Silicon. NumPy fallback runs identically on any CPU.

Two backends in one engine:
1. Metropolis-Hastings (continuous): user supplies log_density(x) callable
   and a proposal scale. Useful for Bayesian posterior sampling.
2. Gibbs sampling on binary QUBO at finite T: user supplies Q matrix and
   T. Useful for sampling Boltzmann distributions.

Handles ProblemKind.SAMPLING with payloads:

Continuous (Metropolis-Hastings):
    {
        "log_density": callable(x: ndarray of shape (n,)) -> float,
        "x0": ndarray of shape (n,) initial state,
        "n_samples": int,
    }

Binary QUBO sampling (Gibbs):
    {
        "qubo_Q": ndarray (n, n),
        "T": float (temperature),
        "qubo_sample": True,
        "n_samples": int,
    }

Hints:
    - n_chains: int               independent chains (default 4)
    - burn_in: int                discarded warmup samples (default 1000)
    - thin: int                   keep every k-th sample (default 1)
    - proposal_scale: float       MH step size (default 0.5)
    - seed: int
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Detect MLX once
try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except Exception:
    mx = None  # type: ignore
    _MLX_AVAILABLE = False


# Engine caps
MAX_DIM = 10_000  # continuous MH dimension
MAX_QUBO_N = 5_000
MAX_N_SAMPLES = 1_000_000
MAX_N_CHAINS = 64
MAX_BURN_IN = 1_000_000
DEFAULT_N_CHAINS = 4
DEFAULT_BURN_IN = 1000


class MCMCEngine:
    name = "mcmc"

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.SAMPLING:
            return False
        p = problem.payload
        # Continuous MH
        if callable(p.get("log_density")) and "x0" in p:
            return True
        # Gibbs on binary QUBO
        if "qubo_Q" in p and p.get("qubo_sample") and "T" in p:
            return True
        return False

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        n_samples = int(p.get("n_samples", 1000))
        n_chains = int(problem.hints.get("n_chains", DEFAULT_N_CHAINS))
        burn_in = int(problem.hints.get("burn_in", DEFAULT_BURN_IN))
        total_iters = (n_samples + burn_in) * n_chains
        if "qubo_Q" in p:
            try:
                n = int(np.asarray(p["qubo_Q"]).shape[0])
            except Exception:
                return float("inf")
            if n > MAX_QUBO_N:
                return float("inf")
            # Gibbs: each iteration scans n bits, each takes O(n) for delta E.
            # Per-iter ~ n^2 work.
            per_iter = 1e-8 * n * n
        else:
            try:
                dim = int(np.asarray(p["x0"]).size)
            except Exception:
                return float("inf")
            if dim > MAX_DIM:
                return float("inf")
            # MH: each iter calls user's log_density once. We cost it at
            # ~50us per call as a generic estimate.
            per_iter = 5e-5
        return 0.005 + total_iters * per_iter

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        p = problem.payload

        n_samples = int(p.get("n_samples", 1000))
        n_chains = int(problem.hints.get("n_chains", DEFAULT_N_CHAINS))
        burn_in = int(problem.hints.get("burn_in", DEFAULT_BURN_IN))
        thin = int(problem.hints.get("thin", 1))
        seed = problem.hints.get("seed")

        if not (1 <= n_samples <= MAX_N_SAMPLES):
            raise ValueError(f"n_samples must be in [1, {MAX_N_SAMPLES}]")
        if not (1 <= n_chains <= MAX_N_CHAINS):
            raise ValueError(f"n_chains must be in [1, {MAX_N_CHAINS}]")
        if not (0 <= burn_in <= MAX_BURN_IN):
            raise ValueError(f"burn_in must be in [0, {MAX_BURN_IN}]")
        if thin < 1 or thin > 10000:
            raise ValueError("thin must be in [1, 10000]")

        if "qubo_Q" in p and p.get("qubo_sample"):
            value = self._gibbs_qubo(p, n_samples, n_chains, burn_in, thin, seed)
        elif callable(p.get("log_density")):
            value = self._metropolis_hastings(
                p,
                n_samples,
                n_chains,
                burn_in,
                thin,
                seed,
                problem.hints.get("proposal_scale", 0.5),
            )
        else:
            raise ValueError("MCMC engine: payload doesn't match a supported pattern")

        elapsed = time.perf_counter() - t0
        return Solution(
            value=value,
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"n_samples": n_samples, "n_chains": n_chains, "burn_in": burn_in},
        )

    # ---------- Gibbs sampling on binary QUBO ----------

    def _gibbs_qubo(
        self,
        payload: dict,
        n_samples: int,
        n_chains: int,
        burn_in: int,
        thin: int,
        seed: Any,
    ) -> dict:
        """Gibbs sampler for the Boltzmann distribution
        π(x) ∝ exp(-x^T Q x / T) over x in {0,1}^n.

        Each step: pick a random bit i, compute conditional P(x_i = 1 | rest)
        from the local field, sample x_i. Repeat until n_samples * thin
        sweeps after burn-in.
        """
        Q_in = np.asarray(payload["qubo_Q"])
        if np.iscomplexobj(Q_in):
            raise ValueError("Q must be real-valued")
        Q = np.asarray(Q_in, dtype=np.float64)
        n = Q.shape[0]
        if Q.ndim != 2 or n != Q.shape[1]:
            raise ValueError(f"Q must be square, got {Q.shape}")
        if n < 1 or n > MAX_QUBO_N:
            raise ValueError(f"n={n} outside [1, {MAX_QUBO_N}]")
        if not np.all(np.isfinite(Q)):
            raise ValueError("Q contains non-finite values")
        T = float(payload["T"])
        if T <= 0 or not np.isfinite(T):
            raise ValueError(f"T must be positive finite, got {T}")
        Qs = (Q + Q.T) / 2.0
        diag = np.diag(Qs)

        rng = np.random.default_rng(seed)

        # Initialize n_chains chains at random
        x = rng.integers(0, 2, size=(n_chains, n)).astype(np.float64)
        # Track x^T Q x for each chain (incremental update)
        Qx = x @ Qs  # (n_chains, n)
        val = np.sum(Qx * x, axis=1)  # (n_chains,)

        total_iters = burn_in + n_samples * thin
        # Storage: list of (n_samples,) entries for each chain. Final shape
        # is (n_chains, n_samples, n) for samples.
        sampled_x: list[np.ndarray] = []
        sampled_E: list[np.ndarray] = []

        for it in range(total_iters):
            # One full sweep: visit every bit
            order = rng.permutation(n)
            for i in order:
                xi = x[:, i]
                row_dot = Qx[:, i]
                # Energy if we set x_i = 0:  E_old - xi*(diag[i] + 2*(row_dot - diag[i]*xi))
                # Equivalently, for binary x: E(x_i=1) - E(x_i=0) = diag[i] + 2*(row_dot - diag[i]*xi - off-diag)
                # Cleaner: dE_to_1 = E(set bit to 1) - E(set bit to 0)
                # If currently xi = 1: row_dot includes diag*1 and full pair contribution.
                # Simpler: compute dE for "if bit were 1" minus "if bit were 0".
                # Let f(s) = (Qs[i,i] s + 2 s sum_{j!=i} Qs[i,j] x_j) where s in {0,1}
                # Note row_dot = sum_j Qs[i,j] x_j = diag[i]*xi + neighbor_sum
                neighbor_sum = row_dot - diag[i] * xi  # (n_chains,)
                # E(s=1) - E(s=0) = diag[i] + 2 * neighbor_sum
                dE = diag[i] + 2 * neighbor_sum
                # Conditional P(x_i = 1 | rest) = sigmoid(-dE / T)
                # log P_1 - log P_0 = -dE/T
                with np.errstate(over="ignore"):
                    p1 = 1.0 / (1.0 + np.exp(dE / T))
                rand = rng.random(size=n_chains)
                new_xi = (rand < p1).astype(np.float64)
                # Update val and Qx incrementally
                delta_x = new_xi - xi  # (n_chains,)
                x[:, i] = new_xi
                # E change = diag[i]*(new^2 - old^2) + 2*neighbor_sum*(new - old)
                # For binary: new^2 = new, old^2 = old.
                val = val + diag[i] * delta_x + 2 * neighbor_sum * delta_x
                # Qx update: only column j gets shifted by delta_x * Qs[i, j]
                # Actually all columns shift: Qx[:, j] += delta_x * Qs[i, j]
                Qx = Qx + np.outer(delta_x, Qs[i, :])

            # Record after burn-in, every `thin` sweeps
            if it >= burn_in and (it - burn_in) % thin == 0:
                sampled_x.append(x.copy())
                sampled_E.append(val.copy())

        # Stack to (n_samples, n_chains, n)
        samples_arr = np.stack(sampled_x, axis=0)  # (n_samples, n_chains, n)
        energies_arr = np.stack(sampled_E, axis=0)  # (n_samples, n_chains)

        return {
            "samples": samples_arr,
            "energies": energies_arr,
            "method": "gibbs_qubo",
            "T": T,
            "n_chains": n_chains,
            "n_samples": samples_arr.shape[0],
            "n_dim": n,
            "mean_energy": float(energies_arr.mean()),
            "min_energy_seen": float(energies_arr.min()),
        }

    # ---------- Metropolis-Hastings for continuous distributions ----------

    def _metropolis_hastings(
        self,
        payload: dict,
        n_samples: int,
        n_chains: int,
        burn_in: int,
        thin: int,
        seed: Any,
        proposal_scale: float,
    ) -> dict:
        """Random-walk Metropolis-Hastings on a continuous distribution.

        Proposal: x' = x + N(0, proposal_scale * I).
        Accept with min(1, exp(log_density(x') - log_density(x))).
        """
        log_density = payload["log_density"]
        x0 = np.asarray(payload["x0"], dtype=np.float64)
        if x0.ndim != 1:
            raise ValueError(f"x0 must be 1-D, got shape {x0.shape}")
        if x0.size > MAX_DIM:
            raise ValueError(f"dim {x0.size} > {MAX_DIM}")
        if not np.all(np.isfinite(x0)):
            raise ValueError("x0 contains non-finite values")
        if not isinstance(proposal_scale, (int, float)) or proposal_scale <= 0:
            raise ValueError("proposal_scale must be > 0")
        proposal_scale = float(proposal_scale)

        dim = x0.size
        rng = np.random.default_rng(seed)

        # Initialize chains by jittering x0
        x = np.tile(x0, (n_chains, 1))
        x = x + 0.01 * rng.normal(size=x.shape)

        # Wrap user's log_density in a defensive evaluator that handles
        # non-finite returns by treating them as -inf (proposal rejected).
        def _log_dens(x_one):
            try:
                v = float(log_density(np.array(x_one, copy=True)))
            except Exception:
                return -np.inf
            if not np.isfinite(v):
                return -np.inf
            return v

        log_p = np.array([_log_dens(x[c]) for c in range(n_chains)])

        n_accepted = 0
        n_proposed = 0
        sampled: list[np.ndarray] = []
        sampled_logp: list[np.ndarray] = []

        total_iters = burn_in + n_samples * thin
        for it in range(total_iters):
            # Propose
            proposal = x + proposal_scale * rng.normal(size=x.shape)
            new_log_p = np.array([_log_dens(proposal[c]) for c in range(n_chains)])
            # Accept ratio
            with np.errstate(invalid="ignore"):
                log_accept = new_log_p - log_p
            rand = rng.random(size=n_chains)
            accept = (log_accept >= 0) | (np.log(rand + 1e-300) < log_accept)
            n_proposed += n_chains
            n_accepted += int(accept.sum())
            # Update
            x = np.where(accept[:, None], proposal, x)
            log_p = np.where(accept, new_log_p, log_p)
            if it >= burn_in and (it - burn_in) % thin == 0:
                sampled.append(x.copy())
                sampled_logp.append(log_p.copy())

        samples_arr = np.stack(sampled, axis=0)  # (n_samples, n_chains, dim)
        logp_arr = np.stack(sampled_logp, axis=0)  # (n_samples, n_chains)
        accept_rate = n_accepted / max(n_proposed, 1)

        return {
            "samples": samples_arr,
            "log_densities": logp_arr,
            "method": "metropolis_hastings",
            "n_chains": n_chains,
            "n_samples": samples_arr.shape[0],
            "n_dim": dim,
            "acceptance_rate": float(accept_rate),
            "proposal_scale": proposal_scale,
        }
