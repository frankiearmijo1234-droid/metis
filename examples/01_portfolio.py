"""Portfolio optimization showcase.

Demonstrates metis picking the right engine for problems of different sizes.

Same problem class -- a portfolio QUBO with risk and budget constraints --
sized differently. metis routes each one to the engine that handles it best.

Run:
    python examples/01_portfolio.py
"""

import numpy as np

from metis import Problem, ProblemKind, default_router


def build_portfolio_qubo(
    n_assets: int, seed: int = 0, risk_weight: float = 0.5, budget: int | None = None
) -> np.ndarray:
    """Build a QUBO encoding a portfolio selection problem.

    Variables x_i ∈ {0, 1}: include asset i in the portfolio.
    Objective: maximize expected return - risk_weight * variance,
                subject to (soft) budget constraint.

    Returns a Q matrix such that minimizing x^T Q x corresponds to the
    desired portfolio.
    """
    rng = np.random.default_rng(seed)
    # Synthetic returns and covariances
    returns = rng.uniform(0.05, 0.20, size=n_assets)
    A = rng.normal(size=(n_assets, n_assets))
    cov = (A @ A.T) / n_assets * 0.05  # positive semi-definite

    # We want to MAXIMIZE returns - risk_weight * variance,
    # i.e. MINIMIZE -returns + risk_weight * variance.
    Q = risk_weight * cov - np.diag(returns)

    # Soft budget constraint: penalty * (sum(x) - budget)^2
    if budget is not None:
        penalty = 1.0
        # (sum x - b)^2 = sum x_i x_j - 2b sum x_i + b^2
        # As a QUBO contribution: penalty * (1 on off-diagonal + (1 - 2b) on diagonal)
        Q += penalty * (
            np.ones((n_assets, n_assets))
            - np.eye(n_assets)
            + np.diag(np.full(n_assets, 1 - 2 * budget))
        )

    return Q


def solve_portfolio(n_assets: int, budget: int, router):
    Q = build_portfolio_qubo(n_assets, seed=0, budget=budget)
    problem = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q, "qubo_solve": True},
        hints={"size": n_assets, "n_sweeps": 500, "n_restarts": 5, "seed": 0},
    )
    return router.solve(problem)


def main():
    router = default_router()

    print("=== metis portfolio optimization demo ===\n")
    print("Same problem class, three sizes. Watch which engine metis picks.\n")

    sizes = [(8, 4), (16, 8), (40, 20)]
    print(f"{'n_assets':>8}  {'engine':>22}  {'time':>8}  {'objective':>12}")
    print("-" * 60)
    for n, budget in sizes:
        sol = solve_portfolio(n, budget, router)
        x = sol.value["x"]
        n_selected = int(x.sum())
        print(
            f"{n:>8}  {sol.engine_name:>22}  "
            f"{sol.elapsed_sec*1000:>6.1f}ms  {sol.value['fun']:>12.4f}  "
            f"({n_selected} assets selected)"
        )
        decision = sol.metadata["routing_decision"]
        print(f"{'':>10}  why: {decision.reason}")
        print()

    print("Key takeaway: metis picked classical (brute force) for n=8,")
    print("parallel_tempering for n=16 and n=40. Same code path, same API.")


if __name__ == "__main__":
    main()
