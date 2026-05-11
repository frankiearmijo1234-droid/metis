"""MCP server exposing metis to Claude Code.

Provides natural-language access to metis's compute orchestrator. The LLM
sees the available tools and picks among them; metis itself picks among
engines. Two layers of routing: tool selection and engine selection.

Setup:
    pip install "metis[mcp]"

Run standalone:
    python -m metis.mcp_server
"""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise ImportError("mcp package not installed. Run: pip install 'metis[mcp]'") from e

import numpy as np

from metis import Problem, ProblemKind, default_router

# Hard limits matching qmlx + metis security model
MAX_N_VARS = 100
MAX_N_QUBITS = 28
MAX_N_OPS = 100_000
MAX_N_SWEEPS = 100_000
MAX_N_RESTARTS = 100
MAX_N_SHOTS = 1_000_000
EXEC_TIMEOUT_SEC = 60

ALLOWED_GATES = {
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "SDG",
    "T",
    "TDG",
    "RX",
    "RY",
    "RZ",
    "PHASE",
    "CNOT",
    "CX",
    "CZ",
    "SWAP",
}

mcp = FastMCP("metis")
_router = default_router()


# ---------- helpers ----------


def _validate_qubo_matrix(Q: Any, n: int) -> np.ndarray:
    arr = np.asarray(Q, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"Q must be {n}x{n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Q contains non-finite values")
    return arr


def _validate_circuit_ops(ops: list, n_qubits: int) -> list:
    if not isinstance(ops, list):
        raise ValueError("ops must be a list")
    if len(ops) > MAX_N_OPS:
        raise ValueError(f"too many ops (max {MAX_N_OPS})")
    out = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"ops[{i}] must be a dict")
        gate = str(op.get("gate", "")).upper()
        if gate not in ALLOWED_GATES:
            raise ValueError(f"ops[{i}].gate '{gate}' not in allowed set")
        qubits = op.get("qubits", [])
        if not isinstance(qubits, list) or not all(
            isinstance(q, int) and 0 <= q < n_qubits for q in qubits
        ):
            raise ValueError(f"ops[{i}].qubits invalid")
        params = op.get("params", [])
        if not isinstance(params, list):
            raise ValueError(f"ops[{i}].params must be a list")
        for p in params:
            if not isinstance(p, (int, float)) or not (-1e9 < float(p) < 1e9):
                raise ValueError(f"ops[{i}].params has invalid value")
        out.append({"gate": gate, "qubits": qubits, "params": params})
    return out


# ---------- tools ----------


@mcp.tool()
def list_engines() -> dict:
    """List available compute engines and what they do."""
    return {
        "engines": [
            {
                "name": e.name,
                "doc": (
                    (e.__class__.__doc__ or "").splitlines()[0]
                    if e.__class__.__doc__
                    else ""
                ),
            }
            for e in _router.engines()
        ]
    }


@mcp.tool()
def solve_qubo(
    Q: list[list[float]],
    n_sweeps: int = 500,
    n_restarts: int = 4,
    seed: int | None = None,
) -> dict:
    """Minimize x^T Q x where x is a binary vector of length n.

    QUBO (Quadratic Unconstrained Binary Optimization) encodes a huge class
    of combinatorial problems: portfolio selection, max-cut, scheduling,
    set cover. metis picks brute force for small n (exact) or simulated
    annealing for large n (heuristic but scalable).

    Args:
        Q: n x n matrix as a list of lists. Symmetric is preferred.
        n_sweeps: SA hint, larger = more thorough (default 500).
        n_restarts: SA hint, more restarts = more reliable (default 4).
        seed: RNG seed for reproducibility.

    Returns: {"x": [...], "fun": float, "engine": str, "elapsed_sec": float}
    """
    arr = np.asarray(Q, dtype=float)
    n = arr.shape[0]
    if n < 1 or n > MAX_N_VARS:
        raise ValueError(f"n must be in [1, {MAX_N_VARS}], got {n}")
    arr = _validate_qubo_matrix(arr, n)
    # Cap resource hints to prevent DoS via huge sweep/restart counts.
    if not isinstance(n_sweeps, int) or n_sweeps < 1 or n_sweeps > MAX_N_SWEEPS:
        raise ValueError(f"n_sweeps must be int in [1, {MAX_N_SWEEPS}]")
    if not isinstance(n_restarts, int) or n_restarts < 1 or n_restarts > MAX_N_RESTARTS:
        raise ValueError(f"n_restarts must be int in [1, {MAX_N_RESTARTS}]")

    problem = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": arr, "qubo_solve": True},
        hints={"size": n, "n_sweeps": n_sweeps, "n_restarts": n_restarts, "seed": seed},
    )
    sol = _router.solve(problem)
    return {
        "x": [int(v) for v in sol.value["x"]],
        "fun": float(sol.value["fun"]),
        "engine": sol.engine_name,
        "elapsed_sec": sol.elapsed_sec,
        "routing_reason": sol.metadata["routing_decision"].reason,
    }


@mcp.tool()
def run_quantum_circuit(
    n_qubits: int,
    ops: list[dict],
    task: str = "probabilities",
    n_shots: int = 1000,
    seed: int | None = None,
) -> dict:
    """Run a quantum circuit and return measurement results.

    Args:
        n_qubits: Number of qubits, 1..28.
        ops: List of gate operations. Each op:
             {"gate": str, "qubits": [int], "params": [float] (optional)}
             Allowed gates: H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP, ...
        task: "probabilities" returns full distribution; "sample" returns
              outcome counts.
        n_shots: Number of samples (only for task="sample").
        seed: RNG seed.
    """
    if n_qubits < 1 or n_qubits > MAX_N_QUBITS:
        raise ValueError(f"n_qubits must be in [1, {MAX_N_QUBITS}]")
    ops = _validate_circuit_ops(ops, n_qubits)
    if task not in ("probabilities", "sample", "expectation_z"):
        raise ValueError("task must be 'probabilities', 'sample', or 'expectation_z'")
    if task == "sample":
        if not isinstance(n_shots, int) or n_shots < 1 or n_shots > MAX_N_SHOTS:
            raise ValueError(f"n_shots must be int in [1, {MAX_N_SHOTS}]")

    payload = {"n_qubits": n_qubits, "ops": ops, "task": task}
    if task == "sample":
        payload["task_args"] = {"n_shots": int(n_shots), "seed": seed}

    problem = Problem(kind=ProblemKind.QUANTUM_CIRCUIT, payload=payload)
    sol = _router.solve(problem)
    return {
        "result": sol.value,
        "engine": sol.engine_name,
        "elapsed_sec": sol.elapsed_sec,
    }


@mcp.tool()
def minimize_quadratic(
    A: list[list[float]], b: list[float], x0: list[float] | None = None
) -> dict:
    """Minimize x^T A x + b^T x for a continuous vector x.

    This is a safer, structured alternative to passing arbitrary expressions.
    Most real continuous-optimization needs that LLMs encounter (least-squares,
    portfolio mean-variance, regularized regression) reduce to this form.

    For non-quadratic objectives, use the Python API directly with a real
    callable rather than going through the MCP server.

    Args:
        A: n x n matrix. If positive semi-definite, the minimum is unique;
           otherwise scipy will find a local one.
        b: length-n linear coefficient vector.
        x0: optional starting point (defaults to zeros).
    """
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n = b_arr.size
    if A_arr.shape != (n, n):
        raise ValueError(f"A must be {n}x{n}, got {A_arr.shape}")
    if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(b_arr)):
        raise ValueError("A or b contains non-finite values")
    if n > MAX_N_VARS:
        raise ValueError(f"n={n} exceeds limit {MAX_N_VARS}")

    if x0 is None:
        x0_arr = np.zeros(n)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.size != n:
            raise ValueError(f"x0 size {x0_arr.size} != n={n}")

    def objective(x):
        return float(x @ A_arr @ x + b_arr @ x)

    problem = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"objective": objective, "x0": x0_arr},
        hints={"size": n},
    )
    sol = _router.solve(problem)
    return {
        "x": [float(v) for v in sol.value["x"]],
        "fun": float(sol.value["fun"]),
        "success": bool(sol.value.get("success", True)),
        "engine": sol.engine_name,
        "elapsed_sec": sol.elapsed_sec,
    }


if __name__ == "__main__":
    mcp.run()
