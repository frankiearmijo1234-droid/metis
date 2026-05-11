# metis

**Run quantum simulation, optimization, and sampling on your Mac. The right algorithm picks itself. No CUDA. No cloud. No vendor lock-in.**

[![CI](https://github.com/frankiearmijo1234-droid/metis/actions/workflows/ci.yml/badge.svg)](https://github.com/frankiearmijo1234-droid/metis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen)](https://github.com/frankiearmijo1234-droid/metis/actions)
[![Engines](https://img.shields.io/badge/engines-10-purple)](https://github.com/frankiearmijo1234-droid/metis#engines)
[![System test](https://img.shields.io/badge/system_test-29%2F31_with_2_documented_pain_points-yellow)](https://github.com/frankiearmijo1234-droid/metis#known-limitations)

---

## If you're on Apple Silicon and you've ever:

- Hit `pip install qsim` and discovered [CUDA support is not available on macOS](https://quantumai.google/qsim/install_qsimcirq) — leaving you on the slow CPU path.
- Spent an afternoon [pinning BLAS to Accelerate](https://schuckert.medium.com/apple-silicon-m1-for-the-quantum-physicist-4f101ea99cf4) just to get TeNPy or quspin to run at half-speed.
- Manually downloaded an old Microsoft.Quantum.Sdk nupkg, [extracted it as a ZIP](https://www.strathweb.com/2022/07/running-q-and-qdk-on-arm64-mac/), and copied native binaries into `.nuget/packages` to get QDK working on M1.
- Stared at an unsigned alpha binary, run `xattr -d com.apple.quarantine /Applications/...` to bypass Gatekeeper, just to use the *only* MLX-native quantum simulator.
- Stood up Qiskit *and* Cirq *and* D-Wave Ocean *and* OR-Tools *and* PennyLane in one project because [no single library handles all of them](https://medium.com/@adnanmasood/quantum-sundays-27-qubo-inc-selecting-the-right-solver-in-a-nisq-world-0ee797d3f2c3).
- Wondered which solver to pick — tabu? simulated annealing? Gurobi? CPLEX? D-Wave? QAOA? — and had to learn five APIs to find out.
- Watched your laptop fans scream as a 30-qubit state vector ate 16 GB of RAM, then realized you could have used a *stabilizer simulator* and run 1,000 qubits on the same machine.

…then you've felt the pain that metis fixes.

---

## What it is

metis is a Python library for hard computational problems. You hand it a problem. It picks the best algorithm for that specific problem from a roster of ten engines, runs it on your local hardware (with MLX acceleration where Metal helps), and returns the answer with a full audit trail of why it picked what it picked.

Ten engines. One API. One install. No CUDA. No cloud account. No license server. MIT licensed, signed by no one but Apple's compiler when you run it.

## Install

metis depends on [`qmlx`](https://github.com/frankiearmijo1234-droid/qmlx), the state-vector simulator behind the quantum engine. Until both are published to PyPI, install from source:

```bash
# Install qmlx first (the state-vector simulator metis wraps)
git clone https://github.com/frankiearmijo1234-droid/qmlx.git
cd qmlx
pip install -e .
cd ..

# Then install metis itself
git clone https://github.com/frankiearmijo1234-droid/metis.git
cd metis
pip install -e ".[all,dev]"   # all = MLX + OR-Tools + Claude Code MCP
```

Verify the install:

```bash
pytest                                    # should be 200/200
python -m benchmarks.system_test --quick  # should be 29 passed, 0 failed, 2 documented pain points
```

```python
from metis import default_router, Problem, ProblemKind

router = default_router()

# A 1000-qubit GHZ state. State-vector simulators can't touch this — they'd
# need 16 EXABYTES of RAM. metis routes to the stabilizer engine.
n = 1000
ops = [{"gate": "H", "qubits": [0]}]
for q in range(n - 1):
    ops.append({"gate": "CNOT", "qubits": [q, q + 1]})

sol = router.solve(Problem(
    kind=ProblemKind.QUANTUM_CIRCUIT,
    payload={"n_qubits": n, "ops": ops, "task": "sample",
             "task_args": {"n_shots": 5}},
))

print(sol.engine_name)                              # 'stabilizer'
print(sol.metadata["routing_decision"].reason)
print(list(sol.value["counts"].keys())[0][:50])     # all 0s or all 1s
```

That's a 1,000-qubit circuit running locally in 25 seconds on a CPU. With MLX, less.

---

## Why this exists

There's no single algorithm that's best for every hard problem.

- A 10-variable optimization wants brute force.
- A 100-variable *constrained* optimization wants CP-SAT.
- A 5,000-variable unconstrained optimization wants simulated annealing.
- A 20-qubit circuit with arbitrary rotations wants a state-vector simulator.
- A 1,000-qubit Clifford circuit wants a stabilizer simulator.
- A low-entanglement state on 200 qubits wants an MPS tensor network.
- A Bayesian posterior wants MCMC.
- A frustrated spin glass wants parallel tempering.

Most "AI compute" tools force you to pick the engine yourself, then learn its API, then juggle it with three other libraries when one engine doesn't fit.

metis picks for you. You describe the problem. metis inspects size, structure, and constraints, then routes to the engine that handles it best. Engines are interchangeable behind a uniform interface — adding a new one (your own custom solver, a connector to D-Wave, whatever) doesn't change anything for callers.

---

## What you get out of the box

| Engine | Handles | Wins when |
|---|---|---|
| `classical` | continuous opt, small QUBO | n ≤ 22 (exact answers, sub-millisecond) |
| `simulated_annealing` | unconstrained QUBO, any size | large n, fast heuristic |
| `simulated_annealing_mlx` | unconstrained QUBO, n ≥ 256 | large n on Apple Silicon |
| `parallel_tempering` | unconstrained QUBO | rugged landscapes (frustrated systems) |
| `qaoa` | small QUBO (≤ 18 qubits) | hybrid quantum/classical, opt-in |
| `mcmc` | continuous & binary sampling | the only sampling-capable engine |
| `ortools_cpsat` | QUBO + linear constraints, ILP | constrained problems, exact answers |
| `qmlx_statevector` | quantum circuits, any gate | up to 28 qubits, exact |
| `mps` | quantum circuits | low entanglement, n up to 200, opt-in |
| `stabilizer` | Clifford circuits | up to 10,000+ qubits |

Ten engines. One router. One audit trail. Same `Problem(kind, payload)` everywhere.

---

## Sample run — the all-engines demo

```bash
python examples/02_all_engines.py
```

Six different problems, six different engines picked automatically:

```
1. Small QUBO (n=8)               → classical            (1.0ms,  optimal)
2. Constrained QUBO (n=12, hi=5)  → ortools_cpsat        (4.2ms,  optimal)
3. Big QUBO (n=2000)              → simulated_annealing  (0.5s,   heuristic)
4. ILP (resource alloc)           → ortools_cpsat        (3.2ms,  optimal)
5. Quantum + T gate (n=4)         → qmlx_statevector     (3ms,    exact)
6. 1000-qubit GHZ                 → stabilizer           (25.6s,  exact)
```

Same router. Same API. Six different engines. Each pick has a routing reason you can inspect.

---

## How routing works

Every engine answers two questions about every problem:

1. `can_handle(problem)` — am I eligible?
2. `estimate_cost(problem)` — how long will I take?

The router asks every engine, picks the lowest cost from the eligible ones, and records every consideration in a `RoutingDecision`:

```python
sol.metadata["routing_decision"].chosen      # 'ortools_cpsat'
sol.metadata["routing_decision"].candidates  # [('ortools_cpsat', 0.05)]
sol.metadata["routing_decision"].rejected    # [('classical', 'can_handle=False'), ...]
sol.metadata["routing_decision"].reason      # human-readable explanation
```

When something is slower or different than expected, you see exactly why. No magic. No guessing.

---

## Drive it from Claude Code

```json
{
  "mcpServers": {
    "metis": {
      "command": "python3",
      "args": ["-m", "metis.mcp_server"]
    }
  }
}
```

Then in Claude Code:

> "I have 30 stocks with these expected returns and this covariance matrix. Pick a portfolio that maximizes Sharpe under a 12-position cardinality constraint."

Claude calls `solve_qubo` with the constraint. metis routes to OR-Tools (the only engine that handles linear constraints). You get the optimal answer with a routing explanation.

> "Run a 100-qubit GHZ state and sample 1,000 measurements."

Claude calls `run_quantum_circuit`. metis routes to stabilizer (qmlx caps at 28 qubits). You get the counts in seconds.

The MCP server exposes `list_engines`, `solve_qubo`, `run_quantum_circuit`, and `minimize_quadratic`. Defense-in-depth input validation at every layer.

---

## What metis is honest about

This is a working, tested portfolio piece — not a research artifact and not a polished commercial product. Things you should know:

- **Not a quantum computer.** Quantum simulation runs on classical hardware. For real quantum execution, plug in IBM/IonQ/AWS Braket via their SDKs. metis doesn't try to replace those.
- **Not a supercomputer.** metis makes your existing Apple Silicon work efficiently — no magical capability boost.
- **Not unbounded.** Stabilizer only handles Clifford circuits. State-vector caps at 28 qubits. QUBO solvers cap at 5,000 variables. These are honest physical limits, not bugs.
- **Two known pain points** are documented in the system test. MCMC has no convergence diagnostics (R-hat, ESS) — multimodal posteriors can mode-trap silently. Routing audit trail doesn't surface engine-internal status (FEASIBLE vs OPTIMAL when OR-Tools terminates on time budget). Both are documented limitations rather than hidden bugs. Run `python -m benchmarks.system_test` and you'll see them.

What it *is*: a clean, small, extensible library that lets you and Claude Code use the right algorithm without thinking about it.

---

## Tests & benchmarks

```bash
pytest                                    # 200 unit tests, ~15s
python -m benchmarks.system_test          # end-to-end system test, ~33s
python -m benchmarks.run --quick          # cross-engine performance, ~60s
python -m benchmarks.run --full           # full sweep, several minutes
```

Unit tests cover per-engine correctness, cross-engine validation, adversarial inputs, and routing decisions. The system test deliberately hunts for friction across five phases: smoke, routing, stress, cross-validation, and hard probes. The benchmark suite produces JSON output you can compare across hardware.

---

## Architecture

```
metis/
├── metis/
│   ├── types.py                 # Problem, Solution, Engine protocol
│   ├── router.py                # the dispatcher with optional fallback
│   ├── engines/
│   │   ├── classical.py
│   │   ├── simulated_annealing.py
│   │   ├── simulated_annealing_mlx.py    # MLX-accelerated SA
│   │   ├── parallel_tempering.py
│   │   ├── qaoa.py                       # hybrid quantum/classical
│   │   ├── mcmc.py                       # MH + Gibbs sampling
│   │   ├── ortools_engine.py             # CP-SAT
│   │   ├── stabilizer.py                 # 1,000+ qubit Clifford
│   │   ├── quantum_sim.py                # qmlx state-vector
│   │   └── mps.py                        # tensor-network simulator
├── benchmarks/                  # cross-engine perf + system test
├── tests/                       # 200 unit tests
├── examples/
│   ├── 01_portfolio.py          # portfolio optimization showcase
│   └── 02_all_engines.py        # all engines, one router
└── claude_skill/
    └── mcp_server.py            # Claude Code interface
```

---

## Security

Every input crosses two validation layers (MCP server + engine). Resource caps prevent DoS through huge circuits, oversized matrices, billion-iteration sweeps, or huge time budgets. No `eval`, `exec`, `pickle`, `subprocess`, or `shell=True` in production code. 21+ adversarial regression tests for every known security bug.

---

## License

MIT. Yours to use, fork, and ship.
