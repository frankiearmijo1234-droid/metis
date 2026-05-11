# Security Policy

## Threat model

metis runs on a user's local machine and may be reachable through an MCP
server from a coding assistant (like Claude Code). The realistic threats are:

1. **Untrusted input.** An LLM driver or user passes adversarial data to
   the MCP server — huge matrices, NaN/inf, hostile callables, oversized
   work budgets, malformed gate sequences.
2. **Resource exhaustion.** Requests that would consume unbounded memory
   or CPU time (10⁹ iterations, 50,000-qubit circuits, 10⁹ shots).
3. **Code execution via deserialization.** Inputs that, if naively
   eval'd or unpickled, could run arbitrary code.

Out of scope:
- Attacks that require write access to the file system or modification
  of the metis source itself.
- Side-channel attacks against the host hardware.
- Vulnerabilities in transitive dependencies (these go to the upstream
  project; we'll bump versions when fixes are available).

## Mitigations in place

- **No `eval`, `exec`, `pickle`, `subprocess`, `shell=True`** in
  production code. Audited via grep at every release.
- **Two-layer validation.** Every MCP-accepted input is validated again
  inside the engine. No engine trusts that its caller validated.
- **Resource caps per engine.** Each engine declares its own caps on
  size, time budget, iteration counts, and qubit counts. Examples:
  classical caps QUBO at n=22, SA caps at 5,000 variables and 100,000
  sweeps, stabilizer caps at 10,000 qubits, MPS caps at 200 qubits and
  bond_dim 256.
- **NaN/inf rejection.** All numeric inputs checked with `np.isfinite`
  before any computation begins.
- **Safe MCP entry points.** `minimize_quadratic(A, b)` takes only an
  array and vector — no callables. The previous `minimize_function`
  endpoint, which took a callable, was removed in v0.2.0 because the
  `eval()`-based dispatch was exploitable via `__subclasses__`.
- **21+ adversarial regression tests** in `tests/test_adversarial.py`
  covering every known input attack. Any new exploit pattern becomes a
  permanent test.

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Email: Frankiearmijo1234@gmail.com

We aim to acknowledge within 72 hours and provide a fix or mitigation
plan within 14 days. Coordinated disclosure preferred — we'll work with
you on a public-disclosure timeline.

If you've found a working exploit, please include:
- A minimal reproducer
- The version of metis you're testing against
- Your assessment of severity

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.6.x   | ✅ Yes    |
| < 0.6   | ❌ No     |

Security fixes are backported to the latest minor version only. If
you're on an older version, the fix is to upgrade.
