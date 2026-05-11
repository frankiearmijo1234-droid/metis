"""All five engines in one demo.

Shows metis routing the same kind of problem to different engines based
on size, structure, and task. Each section ends with a one-liner showing
which engine the router picked and why.

Run:
    python examples/02_all_engines.py
"""

import numpy as np

from metis import Problem, ProblemKind, default_router


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show(sol, label=""):
    decision = sol.metadata["routing_decision"]
    print(f"  Engine:    {sol.engine_name}")
    print(f"  Time:      {sol.elapsed_sec*1000:.1f} ms")
    print(f"  Decided:   {decision.reason}")
    if label:
        print(f"  Result:    {label}")


def main():
    router = default_router()

    print("metis: 10 engines, one router")
    print("-" * 70)
    print("Engines registered:")
    for e in router.engines():
        print(f"  - {e.name}")

    # ---------- 1. Small QUBO -> classical ----------
    section("1. Small QUBO (n=8): brute force is exact and fast")
    np.random.seed(0)
    Q = np.random.randn(8, 8)
    Q = (Q + Q.T) / 2
    sol = router.solve(
        Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": 8},
        )
    )
    show(sol, f"optimal value = {sol.value['fun']:.4f}")

    # ---------- 2. Constrained QUBO -> OR-Tools ----------
    section("2. Constrained QUBO: only OR-Tools handles linear constraints")
    n = 12
    Q = -np.eye(n)  # max sum(x), so unconstrained would be all-ones
    constraint = [{"coeffs": [1.0] * n, "lo": None, "hi": 5}]  # at most 5 ones
    sol = router.solve(
        Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True, "linear_constraints": constraint},
            hints={"size": n, "time_budget_s": 5},
        )
    )
    show(sol, f"selected {int(sol.value['x'].sum())}/{n}, fun={sol.value['fun']}")

    # ---------- 3. Big unconstrained QUBO -> SA ----------
    section("3. Big unconstrained QUBO (n=2000): SA scales beyond exact solvers")
    n = 2000
    np.random.seed(0)
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2
    sol = router.solve(
        Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={"qubo_Q": Q, "qubo_solve": True},
            hints={"size": n, "n_sweeps": 30, "n_restarts": 1, "seed": 0},
        )
    )
    show(sol, f"fun={sol.value['fun']:.4f}")

    # ---------- 4. ILP -> OR-Tools ----------
    section("4. Integer linear program: textbook resource allocation")
    sol = router.solve(
        Problem(
            kind=ProblemKind.OPTIMIZATION,
            payload={
                "ilp_solve": True,
                "objective_coeffs": [5, 4],
                "var_lo": [0, 0],
                "var_hi": [10, 10],
                "linear_constraints": [
                    {"coeffs": [6, 4], "lo": None, "hi": 24},
                    {"coeffs": [1, 2], "lo": None, "hi": 6},
                ],
                "minimize": False,
            },
            hints={"time_budget_s": 5},
        )
    )
    show(sol, f"max 5x+4y = {sol.value['fun']}, x={sol.value['x']}")

    # ---------- 5. Small quantum circuit -> qmlx ----------
    section("5. Small quantum circuit (n=4 GHZ with T gate): qmlx state-vector")
    n = 4
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    ops.append({"gate": "T", "qubits": [0]})  # non-Clifford -> stabilizer can't
    sol = router.solve(
        Problem(
            kind=ProblemKind.QUANTUM_CIRCUIT,
            payload={"n_qubits": n, "ops": ops, "task": "probabilities"},
        )
    )
    probs = sol.value["probabilities"]
    show(sol, f"non-zero probs: {[f'{p:.3f}' for p in probs if p > 0.01]}")

    # ---------- 6. Big quantum circuit -> stabilizer ----------
    section("6. 1000-qubit GHZ: stabilizer makes the impossible routine")
    print("  (state-vector simulation would need 16 EXABYTES of memory)")
    n = 1000
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    sol = router.solve(
        Problem(
            kind=ProblemKind.QUANTUM_CIRCUIT,
            payload={
                "n_qubits": n,
                "ops": ops,
                "task": "sample",
                "task_args": {"n_shots": 3, "seed": 0},
            },
        )
    )
    counts = sol.value["counts"]
    # Show sample bits succinctly
    for outcome in counts:
        first_bits = outcome[:30]
        last_bits = outcome[-30:]
        print(f"  Sample: {first_bits}...{last_bits} ({counts[outcome]} times)")
    show(sol)

    print()
    print("-" * 70)
    print("That's all 10 engines. Same router, same API, same audit trail.")


if __name__ == "__main__":
    main()
