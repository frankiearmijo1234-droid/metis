# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. The next release will appear here.

## [0.6.0] — 2026-05-10

Initial public release.

### Features

**Ten compute engines, one router, one API:**

- `classical` — scipy continuous optimization and brute-force QUBO for n ≤ 22
- `simulated_annealing` — CPU simulated annealing for unconstrained QUBO
- `simulated_annealing_mlx` — MLX-accelerated batched SA for Apple Silicon (NumPy fallback elsewhere)
- `parallel_tempering` — replica-exchange MCMC for rugged QUBO landscapes
- `qaoa` — hybrid quantum/classical optimizer, opt-in via `prefer_qaoa=True`
- `mcmc` — Metropolis-Hastings and Gibbs sampling for `ProblemKind.SAMPLING`
- `ortools_cpsat` — Google CP-SAT for QUBO with linear constraints and ILP
- `qmlx_statevector` — qmlx-backed state-vector simulator up to 28 qubits
- `mps` — matrix product state tensor-network simulator up to 200 qubits, opt-in
- `stabilizer` — Aaronson-Gottesman tableau for 10,000+ qubit Clifford circuits

**Router infrastructure:**

- `Problem(kind, payload, hints)` / `Solution(value, engine_name, ...)` core types
- Cost-based dispatch: every engine answers `can_handle()` and `estimate_cost()`
- `RoutingDecision` audit trail records candidates, rejected engines, and reason
- Optional fallback to the next-best engine on solve failure

**Tooling:**

- MCP server in `claude_skill/mcp_server.py` for Claude Code integration
- Cross-engine benchmark suite at `python -m benchmarks.run`
- End-to-end system test at `python -m benchmarks.system_test`
- Two demos: `examples/01_portfolio.py`, `examples/02_all_engines.py`

**Quality bar:**

- 200 unit tests covering correctness, cross-engine validation, adversarial inputs
- 31 system-test checks across smoke, routing, stress, cross-validation, and hard probes
- Honest documentation of known pain points (MCMC convergence diagnostics, routing audit transparency)

### Security

- No `eval`, `exec`, `pickle`, `subprocess`, or `shell=True` in production code
- Two-layer input validation (MCP server + engine)
- Per-engine resource caps prevent DoS via oversized work budgets
- 21+ adversarial regression tests guard against re-introducing known attacks

### Known limitations

Documented honestly in the system test:

- MCMC has no convergence diagnostics (R-hat, ESS) — mode trapping can go unflagged on multimodal posteriors
- Routing audit trail doesn't surface engine-internal status (FEASIBLE vs OPTIMAL when OR-Tools terminates on time budget)
- MPS at fixed bond dimension silently produces approximate results when entanglement exceeds capacity (`truncation_error` is reported but defaulting users might miss it)
- MLX-SA cost estimates are calibrated against NumPy fallback; real Apple Silicon performance needs first-run recalibration

[Unreleased]: https://github.com/frankiearmijo1234-droid/metis/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/frankiearmijo1234-droid/metis/releases/tag/v0.6.0
