"""End-to-end system test for metis.

Runs the whole system through realistic workloads and reports honestly:
- Did each engine actually work?
- Did the router make sensible decisions?
- Did engines agree on overlapping problems?
- Where are the pain points?

Designed for two audiences:
1. The author: pre-ship sanity check that the whole thing holds together.
2. The reviewer: a hiring manager or eng team who pulls the repo and runs
   one command to evaluate it.

Run:
    python -m benchmarks.system_test           # full test, ~3-5 min
    python -m benchmarks.system_test --quick   # ~30s smoke version

Output is structured: every check has a PASS/FAIL/PAIN_POINT marker plus
a one-line summary. Final report counts checks and lists every pain point
we found.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from metis import (
    _ORTOOLS_AVAILABLE,
    QAOA,
    ClassicalOptimizer,
    MCMCEngine,
    MPSSimulator,
    ParallelTempering,
    Problem,
    ProblemKind,
    QuantumStateVector,
    SimulatedAnnealing,
    SimulatedAnnealingMLX,
    StabilizerSimulator,
    default_router,
    is_mlx_available,
)

# ---------- Result types ----------

PASS = "PASS"
FAIL = "FAIL"
PAIN = "PAIN"


@dataclass
class CheckResult:
    phase: str
    name: str
    status: str
    elapsed_sec: float
    detail: str = ""
    pain_point: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == PASS


@dataclass
class TestRun:
    results: list[CheckResult] = field(default_factory=list)
    started_at: float = 0.0

    def add(self, r: CheckResult) -> None:
        self.results.append(r)
        marker = {"PASS": "✓", "FAIL": "✗", "PAIN": "⚠"}[r.status]
        print(f"  [{marker} {r.elapsed_sec:>6.2f}s] {r.name}", flush=True)
        if r.detail:
            print(f"      {r.detail}", flush=True)
        if r.pain_point:
            print(f"      PAIN POINT: {r.pain_point}", flush=True)

    def summarize(self) -> int:
        passed = sum(1 for r in self.results if r.status == PASS)
        failed = sum(1 for r in self.results if r.status == FAIL)
        pain = sum(1 for r in self.results if r.status == PAIN)
        elapsed = time.time() - self.started_at
        print()
        print("=" * 72)
        print(f"  metis system test: {len(self.results)} checks in {elapsed:.1f}s")
        print(f"    {passed} passed, {failed} failed, {pain} pain points")
        print("=" * 72)
        if pain:
            print()
            print("Pain points (things that work but signal friction or design gaps):")
            for r in self.results:
                if r.status == PAIN:
                    print(f"  - [{r.phase}] {r.name}: {r.pain_point}")
        if failed:
            print()
            print("Failures (things that should have worked):")
            for r in self.results:
                if r.status == FAIL:
                    print(f"  - [{r.phase}] {r.name}: {r.detail}")
        return 0 if failed == 0 else 1


def run_check(
    run: TestRun,
    phase: str,
    name: str,
    fn: Callable[[], tuple[str, str, str]],
) -> None:
    """Helper: run fn(), capture timing/exceptions, append to results.

    fn returns (status, detail, pain_point). status in {PASS, FAIL, PAIN}.
    Any uncaught exception is recorded as FAIL.
    """
    t0 = time.time()
    try:
        status, detail, pain = fn()
    except Exception as e:
        status = FAIL
        detail = f"uncaught: {type(e).__name__}: {e}"
        pain = ""
        traceback.print_exc(file=sys.stderr)
    run.add(CheckResult(phase, name, status, time.time() - t0, detail, pain))


# ---------- Phase 1: Smoke ----------


def phase_smoke(run: TestRun, quick: bool) -> None:
    print()
    print("Phase 1: Smoke -- every engine handles a problem it should")
    print("-" * 60)

    def smoke_classical():
        eng = ClassicalOptimizer()
        Q = -np.eye(5)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": 5},
            )
        )
        if sol.value["fun"] != -5:
            return FAIL, f"expected -5, got {sol.value['fun']}", ""
        return PASS, f"found optimum -5 in {sol.elapsed_sec*1000:.1f}ms", ""

    def smoke_sa():
        eng = SimulatedAnnealing()
        Q = -np.eye(20)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": 20, "n_sweeps": 100, "seed": 0},
            )
        )
        if sol.value["fun"] != -20:
            return (
                PAIN,
                f"expected -20, got {sol.value['fun']}",
                "SA didn't find trivial optimum -- may need more sweeps",
            )
        return PASS, "found optimum -20", ""

    def smoke_mlx_sa():
        eng = SimulatedAnnealingMLX()
        Q = -np.eye(300)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": 300, "n_sweeps": 50, "seed": 0},
            )
        )
        backend = sol.value.get("backend")
        # Quality check: should find -300 or very close
        if sol.value["fun"] > -290:
            return (
                PAIN,
                f"fun={sol.value['fun']:.1f} (expected -300), backend={backend}",
                "MLX-SA quality on diagonal QUBO is below expected",
            )
        return PASS, f"backend={backend}, fun={sol.value['fun']:.1f}", ""

    def smoke_pt():
        eng = ParallelTempering()
        Q = -np.eye(20)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": 20, "n_sweeps": 100, "n_replicas": 4, "seed": 0},
            )
        )
        if sol.value["fun"] != -20:
            return (
                PAIN,
                f"expected -20, got {sol.value['fun']}",
                "PT failed trivial QUBO",
            )
        rate = sol.value["swap_acceptance_rate"]
        if rate < 0.05 or rate > 0.95:
            return (
                PAIN,
                f"swap rate {rate:.2f} out of healthy band [0.05, 0.95]",
                "PT swap acceptance rate suggests temperature ladder is poorly tuned",
            )
        return PASS, f"swap_rate={rate:.2f}", ""

    def smoke_qaoa():
        eng = QAOA()
        Q = -np.eye(4)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {
                    "size": 4,
                    "prefer_qaoa": True,
                    "p": 2,
                    "max_iter": 30,
                    "n_shots": 256,
                    "seed": 0,
                },
            )
        )
        if sol.value["fun"] != -4:
            return (
                PAIN,
                f"expected -4, got {sol.value['fun']}",
                "QAOA didn't find trivial optimum -- could need higher p or more iters",
            )
        return PASS, f"p={sol.value['p']}, iters={sol.value['optimizer_iters']}", ""

    def smoke_mcmc():
        eng = MCMCEngine()
        sol = eng.solve(
            Problem(
                ProblemKind.SAMPLING,
                {
                    "qubo_Q": -np.eye(4),
                    "qubo_sample": True,
                    "T": 0.01,
                    "n_samples": 100,
                },
                {"n_chains": 2, "burn_in": 200, "seed": 0},
            )
        )
        min_E = sol.value["min_energy_seen"]
        if min_E != -4:
            return (
                PAIN,
                f"low-T Gibbs didn't reach ground -4, got {min_E}",
                "MCMC may need more burn-in",
            )
        return PASS, f"min_energy={min_E}", ""

    def smoke_qmlx():
        eng = QuantumStateVector()
        sol = eng.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": 2,
                    "ops": [
                        {"gate": "H", "qubits": [0]},
                        {"gate": "CNOT", "qubits": [0, 1]},
                    ],
                    "task": "probabilities",
                },
            )
        )
        probs = sol.value["probabilities"]
        if abs(probs[0] - 0.5) > 1e-5 or abs(probs[3] - 0.5) > 1e-5:
            return FAIL, f"Bell state probs wrong: {probs}", ""
        return PASS, f"Bell state probs = {probs}", ""

    def smoke_stabilizer():
        eng = StabilizerSimulator()
        ops = [{"gate": "H", "qubits": [0]}]
        for q in range(99):
            ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
        sol = eng.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": 100,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": 5, "seed": 0},
                },
            )
        )
        all_zero, all_one = "0" * 100, "1" * 100
        for k in sol.value["counts"]:
            if k not in (all_zero, all_one):
                return FAIL, f"non-GHZ outcome {k}", ""
        return PASS, f"100-qubit GHZ in {sol.elapsed_sec*1000:.0f}ms", ""

    def smoke_mps():
        eng = MPSSimulator()
        ops = [{"gate": "H", "qubits": [0]}]
        for q in range(49):
            ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
        sol = eng.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": 50,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": 5, "seed": 0},
                },
                {"prefer_mps": True, "bond_dim": 8},
            )
        )
        all_zero, all_one = "0" * 50, "1" * 50
        for k in sol.value["counts"]:
            if k not in (all_zero, all_one):
                return FAIL, f"non-GHZ outcome {k}", ""
        bd = sol.value["bond_dim_used"]
        return PASS, f"50-qubit GHZ at bond_dim_used={bd}", ""

    def smoke_ortools():
        if not _ORTOOLS_AVAILABLE:
            return (
                PAIN,
                "OR-Tools not installed",
                "OR-Tools is an optional dep; install with [ortools] extra",
            )
        from metis import ORTools

        eng = ORTools()
        Q = -np.eye(8)
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {
                    "qubo_Q": Q,
                    "qubo_solve": True,
                    "linear_constraints": [
                        {"coeffs": [1.0] * 8, "lo": None, "hi": 3},
                    ],
                },
                {"size": 8, "time_budget_s": 5},
            )
        )
        if int(sum(sol.value["x"])) != 3:
            return FAIL, f"constraint violated: sum={sum(sol.value['x'])}", ""
        if sol.value["fun"] != -3:
            return FAIL, f"expected -3, got {sol.value['fun']}", ""
        return PASS, "cardinality-3 cut found, fun=-3", ""

    run_check(run, "smoke", "classical engine", smoke_classical)
    run_check(run, "smoke", "simulated_annealing engine", smoke_sa)
    run_check(run, "smoke", "simulated_annealing_mlx engine", smoke_mlx_sa)
    run_check(run, "smoke", "parallel_tempering engine", smoke_pt)
    run_check(run, "smoke", "qaoa engine", smoke_qaoa)
    run_check(run, "smoke", "mcmc engine", smoke_mcmc)
    run_check(run, "smoke", "qmlx_statevector engine", smoke_qmlx)
    run_check(run, "smoke", "stabilizer engine", smoke_stabilizer)
    run_check(run, "smoke", "mps engine", smoke_mps)
    run_check(run, "smoke", "ortools_cpsat engine", smoke_ortools)


# ---------- Phase 2: Routing ----------


def phase_routing(run: TestRun, quick: bool) -> None:
    print()
    print("Phase 2: Routing -- the right engine wins for each region")
    print("-" * 60)
    router = default_router()

    def route_check(
        label: str,
        problem: Problem,
        expected_set: set,
        not_expected_set: set = frozenset(),
    ):
        def fn():
            sol = router.solve(problem)
            engine = sol.engine_name
            if expected_set and engine not in expected_set:
                return FAIL, f"expected one of {sorted(expected_set)}, got {engine}", ""
            if engine in not_expected_set:
                return FAIL, f"engine {engine} should not have been picked", ""
            return PASS, f"-> {engine}", ""

        return fn

    rng = np.random.default_rng(0)

    # Tiny QUBO: classical
    Q8 = rng.normal(size=(8, 8))
    Q8 = (Q8 + Q8.T) / 2
    run_check(
        run,
        "routing",
        "n=8 QUBO routes correctly",
        route_check(
            "n=8",
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q8, "qubo_solve": True},
                {"size": 8},
            ),
            {"classical", "simulated_annealing", "ortools_cpsat", "parallel_tempering"},
        ),
    )

    # Big unconstrained QUBO: SA family wins
    Q500 = rng.normal(size=(500, 500))
    Q500 = (Q500 + Q500.T) / 2
    run_check(
        run,
        "routing",
        "n=500 unconstrained -> SA family",
        route_check(
            "n=500",
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q500, "qubo_solve": True},
                {
                    "size": 500,
                    "n_sweeps": 50,
                    "n_restarts": 2,
                    "seed": 0,
                    "time_budget_s": 5,
                },
            ),
            {
                "simulated_annealing",
                "simulated_annealing_mlx",
                "parallel_tempering",
                "ortools_cpsat",
            },
            not_expected_set={"classical"},
        ),
    )

    # Constrained QUBO: must go to OR-Tools
    if _ORTOOLS_AVAILABLE:
        Q12 = rng.normal(size=(12, 12))
        Q12 = (Q12 + Q12.T) / 2
        run_check(
            run,
            "routing",
            "constrained QUBO -> ortools_cpsat",
            route_check(
                "constrained",
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {
                        "qubo_Q": Q12,
                        "qubo_solve": True,
                        "linear_constraints": [
                            {"coeffs": [1.0] * 12, "lo": None, "hi": 5},
                        ],
                    },
                    {"size": 12, "time_budget_s": 5},
                ),
                {"ortools_cpsat"},
            ),
        )

    # Quantum circuit with arbitrary gates: qmlx
    run_check(
        run,
        "routing",
        "small quantum circuit -> qmlx",
        route_check(
            "qcirc",
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": 4,
                    "ops": [
                        {"gate": "H", "qubits": [0]},
                        {"gate": "T", "qubits": [0]},
                        {"gate": "CNOT", "qubits": [0, 1]},
                    ],
                    "task": "probabilities",
                },
            ),
            {"qmlx_statevector"},
        ),
    )

    # Big Clifford circuit: stabilizer
    big_clifford_ops = [{"gate": "H", "qubits": [0]}]
    for q in range(99):
        big_clifford_ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    run_check(
        run,
        "routing",
        "100q Clifford circuit -> stabilizer",
        route_check(
            "100qC",
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": 100,
                    "ops": big_clifford_ops,
                    "task": "sample",
                    "task_args": {"n_shots": 3, "seed": 0},
                },
            ),
            {"stabilizer"},
        ),
    )

    # Sampling problem: mcmc is the only one
    run_check(
        run,
        "routing",
        "sampling problem -> mcmc",
        route_check(
            "sample",
            Problem(
                ProblemKind.SAMPLING,
                {"qubo_Q": np.eye(4), "qubo_sample": True, "T": 1.0, "n_samples": 50},
                {"n_chains": 2, "burn_in": 20, "seed": 0},
            ),
            {"mcmc"},
        ),
    )

    # Routing audit trail is present and informative
    def routing_decision_audit():
        sol = router.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": -np.eye(10), "qubo_solve": True},
                {"size": 10, "n_sweeps": 50, "seed": 0},
            )
        )
        decision = sol.metadata.get("routing_decision")
        if decision is None:
            return FAIL, "routing_decision absent from solution metadata", ""
        if not decision.candidates:
            return FAIL, "decision.candidates empty", ""
        if decision.chosen != sol.engine_name:
            return FAIL, "decision.chosen != engine_name", ""
        if not decision.reason:
            return (
                PAIN,
                "decision.reason empty",
                "audit trail lacks human-readable explanation",
            )
        return (
            PASS,
            f"{len(decision.candidates)} candidates, {len(decision.rejected)} rejected",
            "",
        )

    run_check(run, "routing", "audit trail is present", routing_decision_audit)


# ---------- Phase 3: Stress ----------


def phase_stress(run: TestRun, quick: bool) -> None:
    print()
    print("Phase 3: Stress -- engines at non-trivial scale")
    print("-" * 60)

    n_qubits_stress = 50 if quick else 200

    def stress_stabilizer_big():
        eng = StabilizerSimulator()
        n = 500 if quick else 1000
        ops = [{"gate": "H", "qubits": [0]}]
        for q in range(n - 1):
            ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
        t0 = time.time()
        sol = eng.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": n,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": 2, "seed": 0},
                },
            )
        )
        elapsed = time.time() - t0
        if elapsed > 120:
            return (
                PAIN,
                f"stabilizer at n={n} took {elapsed:.0f}s",
                f"stabilizer too slow at n={n}; consider Aaronson/Gottesman bit-packed impl",
            )
        return PASS, f"n={n} GHZ in {elapsed:.1f}s", ""

    def stress_mps_chain():
        eng = MPSSimulator()
        n = n_qubits_stress
        # Random Clifford-like circuit at moderate depth
        rng = np.random.default_rng(0)
        ops = []
        for layer in range(5):
            for q in range(n):
                gate = rng.choice(["H", "S", "X", "Y", "Z"])
                ops.append({"gate": gate, "qubits": [int(q)]})
            for q in range(0, n - 1, 2):
                ops.append({"gate": "CNOT", "qubits": [int(q), int(q + 1)]})
        sol = eng.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {
                    "n_qubits": n,
                    "ops": ops,
                    "task": "sample",
                    "task_args": {"n_shots": 3, "seed": 0},
                },
                {"prefer_mps": True, "bond_dim": 16},
            )
        )
        bd = sol.value["bond_dim_used"]
        trunc = sol.value["truncation_error"]
        if trunc > 0.1:
            return (
                PAIN,
                f"truncation_error={trunc:.3f} at bond_dim_used={bd}",
                "MPS at bond_dim=16 unable to represent state; bump bond_dim",
            )
        return PASS, f"n={n} circuit, bond_dim_used={bd}, trunc={trunc:.2e}", ""

    def stress_pt_quality_vs_sa():
        """At a frustrated QUBO, PT should match or beat SA quality."""
        n = 30
        rng = np.random.default_rng(7)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        sa = SimulatedAnnealing()
        pt = ParallelTempering()
        sa_sol = sa.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n, "n_sweeps": 200, "n_restarts": 4, "seed": 0},
            )
        )
        pt_sol = pt.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n, "n_sweeps": 200, "n_replicas": 8, "seed": 0},
            )
        )
        gap = pt_sol.value["fun"] - sa_sol.value["fun"]
        if gap > 0.5:
            return (
                PAIN,
                f"PT worse than SA by {gap:.3f}",
                "PT has heavier compute budget but worse quality here -- temperature ladder may need tuning",
            )
        return PASS, f"SA={sa_sol.value['fun']:.3f}, PT={pt_sol.value['fun']:.3f}", ""

    def stress_sa_n2000():
        eng = SimulatedAnnealing()
        n = 1000 if quick else 2000
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        t0 = time.time()
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n, "n_sweeps": 30, "n_restarts": 2, "seed": 0},
            )
        )
        elapsed = time.time() - t0
        if elapsed > 30:
            return (
                PAIN,
                f"SA at n={n} took {elapsed:.1f}s",
                "SA scaling at large n could use MLX backend",
            )
        return PASS, f"n={n} in {elapsed:.1f}s, fun={sol.value['fun']:.0f}", ""

    def stress_qaoa_p_scaling():
        """QAOA at p=4 on a moderate problem -- meaningful workload."""
        eng = QAOA()
        n = 6
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        cls_sol = ClassicalOptimizer().solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n},
            )
        )
        truth = cls_sol.value["fun"]
        sol = eng.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {
                    "size": n,
                    "prefer_qaoa": True,
                    "p": 4,
                    "max_iter": 80,
                    "n_shots": 1024,
                    "seed": 0,
                },
            )
        )
        gap = sol.value["fun"] - truth
        if gap > 0.5:
            return (
                PAIN,
                f"QAOA gap to optimum = {gap:.3f}",
                "QAOA finds suboptimal answer at p=4 -- typical limitation, deeper p helps",
            )
        return (
            PASS,
            f"truth={truth:.3f}, QAOA={sol.value['fun']:.3f} (gap {gap:.3f})",
            "",
        )

    run_check(run, "stress", "stabilizer at large n (GHZ)", stress_stabilizer_big)
    run_check(run, "stress", "MPS on random circuit", stress_mps_chain)
    run_check(run, "stress", "PT vs SA quality on rugged QUBO", stress_pt_quality_vs_sa)
    run_check(run, "stress", "SA at large n", stress_sa_n2000)
    run_check(run, "stress", "QAOA at depth p=4", stress_qaoa_p_scaling)


# ---------- Phase 4: Cross-validation ----------


def phase_cross_validation(run: TestRun, quick: bool) -> None:
    print()
    print("Phase 4: Cross-validation -- engines must agree where they overlap")
    print("-" * 60)

    def cross_sa_vs_classical():
        """SA must find the same optimum as brute force on small QUBOs."""
        sa = SimulatedAnnealing()
        cls = ClassicalOptimizer()
        misses = 0
        for seed in range(8):
            n = 8
            rng = np.random.default_rng(seed)
            Q = rng.normal(size=(n, n))
            Q = (Q + Q.T) / 2
            truth = cls.solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {"size": n},
                )
            ).value["fun"]
            found = sa.solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {"size": n, "n_sweeps": 200, "n_restarts": 4, "seed": 0},
                )
            ).value["fun"]
            if abs(found - truth) > 1e-6:
                misses += 1
        if misses > 0:
            return (
                PAIN,
                f"{misses}/8 cases SA missed brute-force optimum",
                "SA quality at small n could be improved with more restarts",
            )
        return PASS, "8/8 cases SA matched brute force", ""

    def cross_pt_vs_classical():
        pt = ParallelTempering()
        cls = ClassicalOptimizer()
        misses = 0
        for seed in range(8):
            n = 8
            rng = np.random.default_rng(seed)
            Q = rng.normal(size=(n, n))
            Q = (Q + Q.T) / 2
            truth = cls.solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {"size": n},
                )
            ).value["fun"]
            found = pt.solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {"size": n, "n_sweeps": 100, "n_replicas": 4, "seed": 0},
                )
            ).value["fun"]
            if abs(found - truth) > 1e-6:
                misses += 1
        if misses > 0:
            return (
                PAIN,
                f"{misses}/8 cases PT missed brute-force optimum",
                "PT quality at small n could need more sweeps",
            )
        return PASS, "8/8 cases PT matched brute force", ""

    def cross_mps_vs_qmlx():
        """MPS at full bond dim should match state-vector at small n."""
        qmlx = QuantumStateVector()
        mps_eng = MPSSimulator()
        n = 6
        rng = np.random.default_rng(0)
        ops = []
        for layer in range(3):
            for q in range(n):
                gate = rng.choice(["RX", "RY", "RZ"])
                theta = float(rng.uniform(0, 2 * np.pi))
                ops.append({"gate": gate, "qubits": [int(q)], "params": [theta]})
            for q in range(0, n - 1, 2):
                ops.append({"gate": "CNOT", "qubits": [int(q), int(q + 1)]})
        p_qmlx = Problem(
            ProblemKind.QUANTUM_CIRCUIT,
            {"n_qubits": n, "ops": ops, "task": "probabilities"},
        )
        p_mps = Problem(
            ProblemKind.QUANTUM_CIRCUIT,
            {"n_qubits": n, "ops": ops, "task": "probabilities"},
            {"prefer_mps": True, "bond_dim": 64},
        )
        probs_qmlx = np.array(qmlx.solve(p_qmlx).value["probabilities"])
        probs_mps = np.array(mps_eng.solve(p_mps).value["probabilities"])
        max_diff = float(np.max(np.abs(probs_qmlx - probs_mps)))
        if max_diff > 1e-4:
            return FAIL, f"max abs diff = {max_diff:.2e}", ""
        return PASS, f"max abs diff = {max_diff:.2e}", ""

    def cross_stabilizer_vs_qmlx():
        """Stabilizer and qmlx must agree on Bell-state sample distributions."""
        qmlx = QuantumStateVector()
        stab = StabilizerSimulator()
        ops = [{"gate": "H", "qubits": [0]}, {"gate": "CNOT", "qubits": [0, 1]}]
        p = Problem(
            ProblemKind.QUANTUM_CIRCUIT,
            {
                "n_qubits": 2,
                "ops": ops,
                "task": "sample",
                "task_args": {"n_shots": 5000, "seed": 0},
            },
        )
        c_qmlx = qmlx.solve(p).value["counts"]
        c_stab = stab.solve(p).value["counts"]
        # Both should have only "00" and "11"
        if set(c_qmlx.keys()) != set(c_stab.keys()):
            return (
                FAIL,
                f"different outcome sets: qmlx={set(c_qmlx)}, stab={set(c_stab)}",
                "",
            )
        # Ratios should both be ~50/50
        for outcome in ("00", "11"):
            r_qmlx = c_qmlx.get(outcome, 0) / 5000
            r_stab = c_stab.get(outcome, 0) / 5000
            if abs(r_qmlx - r_stab) > 0.05:
                return (
                    PAIN,
                    f"distribution mismatch on {outcome}: qmlx={r_qmlx:.3f} stab={r_stab:.3f}",
                    "stabilizer & state-vector differ by >5% -- statistical or correctness?",
                )
        return PASS, "Bell distributions agree to <5%", ""

    def cross_mcmc_finds_qubo_ground():
        """MCMC at low T should reach the QUBO ground state, agreeing with
        a direct optimizer."""
        mcmc = MCMCEngine()
        cls = ClassicalOptimizer()
        n = 8
        rng = np.random.default_rng(2)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        truth = cls.solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {"qubo_Q": Q, "qubo_solve": True},
                {"size": n},
            )
        ).value["fun"]
        sol = mcmc.solve(
            Problem(
                ProblemKind.SAMPLING,
                {"qubo_Q": Q, "qubo_sample": True, "T": 0.05, "n_samples": 200},
                {"n_chains": 4, "burn_in": 1000, "seed": 0},
            )
        )
        min_seen = sol.value["min_energy_seen"]
        if abs(min_seen - truth) > 1e-3:
            return (
                PAIN,
                f"MCMC min={min_seen:.4f}, truth={truth:.4f}",
                "MCMC at low T should hit ground -- might need more burn-in or lower T",
            )
        return PASS, f"MCMC min={min_seen:.4f} matches truth", ""

    run_check(run, "cross", "SA matches brute force on 8 QUBOs", cross_sa_vs_classical)
    run_check(run, "cross", "PT matches brute force on 8 QUBOs", cross_pt_vs_classical)
    run_check(run, "cross", "MPS matches state-vector at n=6", cross_mps_vs_qmlx)
    run_check(
        run,
        "cross",
        "stabilizer matches state-vector on Bell",
        cross_stabilizer_vs_qmlx,
    )
    run_check(
        run,
        "cross",
        "MCMC reaches QUBO ground state at low T",
        cross_mcmc_finds_qubo_ground,
    )


# ---------- Phase 5: Hard probes ----------


def phase_hard_probes(run: TestRun, quick: bool) -> None:
    print()
    print("Phase 5: Hard probes -- where does the system genuinely struggle?")
    print("-" * 60)

    def probe_mps_high_entanglement():
        """MPS at fixed bond dim should fail on highly-entangled circuits.
        This isn't a bug -- it's a fundamental limitation. The pain point is
        whether we EXPOSE the failure clearly."""
        n = 10
        rng = np.random.default_rng(0)
        ops = []
        # Volume-law entangling circuit
        for layer in range(20):
            for q in range(n):
                ops.append(
                    {
                        "gate": "RX",
                        "qubits": [q],
                        "params": [float(rng.uniform(0, np.pi))],
                    }
                )
                ops.append(
                    {
                        "gate": "RY",
                        "qubits": [q],
                        "params": [float(rng.uniform(0, np.pi))],
                    }
                )
            for i in range(n):
                for j in range(i + 1, n):
                    ops.append({"gate": "CNOT", "qubits": [i, j]})
        mps = MPSSimulator()
        qmlx = QuantumStateVector()
        sol_mps = mps.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {"n_qubits": n, "ops": ops, "task": "probabilities"},
                {"prefer_mps": True, "bond_dim": 16},
            )
        )
        sol_qmlx = qmlx.solve(
            Problem(
                ProblemKind.QUANTUM_CIRCUIT,
                {"n_qubits": n, "ops": ops, "task": "probabilities"},
            )
        )
        p_mps = np.array(sol_mps.value["probabilities"])
        p_qmlx = np.array(sol_qmlx.value["probabilities"])
        total_var = float(np.sum(np.abs(p_mps - p_qmlx)) / 2)
        trunc = sol_mps.value.get("truncation_error", 0)
        # Truncation error is reported, so the user CAN diagnose the failure.
        # That's the right outcome -- engine returns garbage but flags it.
        if trunc <= 0.01:
            return (
                PAIN,
                f"truncation reported as {trunc:.4f} but TV={total_var:.2%}",
                "high-entanglement state but truncation_error stayed near zero -- needs better detection",
            )
        return (
            PASS,
            f"truncation_error={trunc:.2f} correctly signals MPS failure (TV={total_var:.2%})",
            "",
        )

    def probe_qaoa_increases_quality_with_p():
        """As p increases, QAOA should not get worse. Otherwise the optimizer
        is stuck in local minima."""
        n = 6
        rng = np.random.default_rng(11)
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        truth = (
            ClassicalOptimizer()
            .solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {"size": n},
                )
            )
            .value["fun"]
        )
        gaps = {}
        for p_depth in [1, 2, 4]:
            sol = QAOA().solve(
                Problem(
                    ProblemKind.OPTIMIZATION,
                    {"qubo_Q": Q, "qubo_solve": True},
                    {
                        "size": n,
                        "prefer_qaoa": True,
                        "p": p_depth,
                        "max_iter": 80,
                        "n_shots": 1024,
                        "seed": 0,
                    },
                )
            )
            gaps[p_depth] = sol.value["fun"] - truth
        # Gap at p=4 should not be much worse than p=1
        if gaps[4] > gaps[1] + 0.5:
            return (
                PAIN,
                f"gaps p=1:{gaps[1]:.3f} p=2:{gaps[2]:.3f} p=4:{gaps[4]:.3f}",
                "QAOA quality degrades with deeper p -- classical optimizer struggles in higher-D parameter space",
            )
        return PASS, f"gaps p=1:{gaps[1]:.3f} p=2:{gaps[2]:.3f} p=4:{gaps[4]:.3f}", ""

    def probe_mcmc_bimodal_mode_trap():
        """Random-walk MH cannot easily escape between separated modes.
        The pain is that the engine doesn't warn the user."""

        def log_pdf_bimodal(x):
            return float(
                np.log(
                    np.exp(-0.5 * np.sum((x - 3) ** 2))
                    + np.exp(-0.5 * np.sum((x + 3) ** 2))
                    + 1e-300
                )
            )

        sol = MCMCEngine().solve(
            Problem(
                ProblemKind.SAMPLING,
                {
                    "log_density": log_pdf_bimodal,
                    "x0": np.array([0.0, 0.0]),
                    "n_samples": 2000,
                },
                {"n_chains": 4, "burn_in": 1000, "proposal_scale": 1.0, "seed": 0},
            )
        )
        samples = sol.value["samples"]  # (n_samples, n_chains, dim)
        # Count how many chains stay one-sided
        stuck_chains = 0
        for c in range(samples.shape[1]):
            n_pos = int((samples[:, c, 0] > 0).sum())
            total = samples.shape[0]
            ratio = n_pos / total
            if ratio < 0.05 or ratio > 0.95:
                stuck_chains += 1
        if stuck_chains > 0:
            return (
                PAIN,
                f"{stuck_chains}/4 chains trapped in one mode at proposal_scale=1",
                "MCMC engine has no convergence diagnostic (R-hat, ESS) to flag mode trapping",
            )
        return PASS, "no chain trapping detected", ""

    def probe_routing_diagnostic_value():
        """When routing leads to a suboptimal solve (e.g., OR-Tools timing
        out before optimum), can the user diagnose it from the audit trail?"""
        rng = np.random.default_rng(0)
        n = 100
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        sol = default_router().solve(
            Problem(
                ProblemKind.OPTIMIZATION,
                {
                    "qubo_Q": Q,
                    "qubo_solve": True,
                    "linear_constraints": [{"coeffs": [1.0] * n, "lo": None, "hi": 50}],
                },
                {"size": n, "time_budget_s": 1},
            )
        )
        # The user should be able to see (a) which engine, (b) is_optimal,
        # (c) why routing chose this engine.
        decision = sol.metadata["routing_decision"]
        is_optimal = sol.value.get("is_optimal")
        status = sol.value.get("status")
        # Pain point: if the time budget caused FEASIBLE-but-not-OPTIMAL,
        # the user might miss it without explicitly checking is_optimal.
        if status == "FEASIBLE" and not is_optimal:
            return (
                PAIN,
                "OR-Tools returned FEASIBLE with is_optimal=False but routing reason makes no mention of time-budget impact",
                "routing audit trail doesn't surface engine-internal status (FEASIBLE vs OPTIMAL)",
            )
        return PASS, f"status={status}, is_optimal={is_optimal}", ""

    run_check(
        run,
        "probe",
        "MPS high-entanglement failure is detectable",
        probe_mps_high_entanglement,
    )
    run_check(run, "probe", "QAOA depth scaling", probe_qaoa_increases_quality_with_p)
    run_check(run, "probe", "MCMC bimodal mode trap", probe_mcmc_bimodal_mode_trap)
    run_check(
        run,
        "probe",
        "routing diagnostics for suboptimal solves",
        probe_routing_diagnostic_value,
    )


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(description="End-to-end system test for metis")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: smaller problem sizes (~30s)"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  metis end-to-end system test")
    print("=" * 72)
    print(f"  mode: {'quick' if args.quick else 'full'}")
    print(f"  MLX available: {is_mlx_available()}")
    print(f"  OR-Tools available: {_ORTOOLS_AVAILABLE}")
    router = default_router()
    print(f"  engines registered: {[e.name for e in router.engines()]}")

    run = TestRun(started_at=time.time())

    phase_smoke(run, args.quick)
    phase_routing(run, args.quick)
    phase_stress(run, args.quick)
    phase_cross_validation(run, args.quick)
    phase_hard_probes(run, args.quick)

    return run.summarize()


if __name__ == "__main__":
    sys.exit(main())
