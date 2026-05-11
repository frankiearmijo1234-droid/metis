"""Quantum circuit simulation engine, backed by qmlx.

Handles ProblemKind.QUANTUM_CIRCUIT with payload:
    {
        "n_qubits": int,
        "ops": [{"gate": "H", "qubits": [0], "params": []}, ...],
        "task": "probabilities" | "sample" | "expectation_z",
        "task_args": {...},
    }

Cost estimate scales with 2^n * len(ops), since that's the work the
state-vector simulator does. Refuses problems above 28 qubits.
"""

from __future__ import annotations

import time

from ..types import Problem, ProblemKind, Solution

# Map gate names to Circuit method names (case-insensitive)
GATE_METHODS = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "SDG": "sdg",
    "T": "t",
    "TDG": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PHASE": "phase",
    "CNOT": "cnot",
    "CX": "cnot",
    "CZ": "cz",
    "SWAP": "swap",
}
ROTATION_GATES = {"RX", "RY", "RZ", "PHASE"}
TWO_QUBIT_GATES = {"CNOT", "CX", "CZ", "SWAP"}


class QuantumStateVector:
    name = "qmlx_statevector"
    MAX_QUBITS = 28
    MAX_OPS = 100_000  # past this -> seconds to minutes; reject
    MAX_N_SHOTS = 1_000_000

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.QUANTUM_CIRCUIT:
            return False
        p = problem.payload
        n = p.get("n_qubits")
        if not isinstance(n, int) or isinstance(n, bool):
            return False
        if n < 1 or n > self.MAX_QUBITS:
            return False
        ops = p.get("ops")
        if not isinstance(ops, list):
            return False
        if len(ops) > self.MAX_OPS:
            return False
        return True

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        n = int(p["n_qubits"])
        n_ops = len(p["ops"])
        # State vector has 2^n complex amplitudes; each op touches all of them.
        # On Apple Silicon with MLX, ~1e9 amplitude-ops/sec is realistic.
        # On CPU-NumPy ~1e8.
        ops_work = n_ops * (2**n)
        return ops_work / 5e8

    def solve(self, problem: Problem) -> Solution:
        from qmlx import Circuit  # imported here to keep startup fast

        t0 = time.perf_counter()
        p = problem.payload
        n = int(p["n_qubits"])
        if n < 1 or n > self.MAX_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {self.MAX_QUBITS}], got {n}")
        ops = p["ops"]
        if not isinstance(ops, list) or len(ops) > self.MAX_OPS:
            raise ValueError(f"ops must be a list of length <= {self.MAX_OPS}")
        task = p.get("task", "probabilities")

        circuit = Circuit(n)
        for op in ops:
            self._apply_op(circuit, op, n)
        sv = circuit.run()

        if task == "probabilities":
            value = {"probabilities": sv.probabilities().tolist()}
        elif task == "sample":
            args = p.get("task_args", {})
            n_shots = args.get("n_shots", 1000)
            if (
                not isinstance(n_shots, int)
                or n_shots < 1
                or n_shots > self.MAX_N_SHOTS
            ):
                raise ValueError(
                    f"n_shots must be int in [1, {self.MAX_N_SHOTS}], got {n_shots}"
                )
            seed = args.get("seed")
            samples = sv.sample(n_shots, seed=seed)
            counts: dict[str, int] = {}
            for s in samples:
                bits = format(int(s), f"0{n}b")
                counts[bits] = counts.get(bits, 0) + 1
            value = {"counts": counts, "n_shots": n_shots}
        elif task == "expectation_z":
            args = p.get("task_args", {})
            qubit = int(args.get("qubit", 0))
            if qubit < 0 or qubit >= n:
                raise ValueError(f"qubit index {qubit} out of range [0, {n})")
            value = {"expectation_z": sv.expectation_z(qubit), "qubit": qubit}
        else:
            raise ValueError(f"unknown task: {task}")

        elapsed = time.perf_counter() - t0
        return Solution(
            value=value,
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"n_qubits": n, "n_ops": len(ops), "task": task},
        )

    @staticmethod
    def _apply_op(circuit, op: dict, n_qubits: int) -> None:
        if not isinstance(op, dict):
            raise ValueError(f"op must be a dict, got {type(op).__name__}")
        gate_name = str(op.get("gate", "")).upper()
        method_name = GATE_METHODS.get(gate_name)
        if method_name is None:
            raise ValueError(f"unknown gate: {gate_name}")
        method = getattr(circuit, method_name)
        qubits = op.get("qubits", [])
        if not isinstance(qubits, list):
            raise ValueError("qubits must be a list")
        for q in qubits:
            if not isinstance(q, int) or isinstance(q, bool):
                raise ValueError(f"qubit indices must be ints, got {q!r}")
            if q < 0 or q >= n_qubits:
                raise ValueError(f"qubit index {q} out of range [0, {n_qubits})")
        params = op.get("params", [])
        if not isinstance(params, list):
            raise ValueError("params must be a list")
        for prm in params:
            if not isinstance(prm, (int, float)) or isinstance(prm, bool):
                raise ValueError(f"params must be numbers, got {type(prm).__name__}")
            f = float(prm)
            if not (-1e9 < f < 1e9) or f != f:  # rejects nan and out-of-range
                raise ValueError(f"param value {prm} out of safe range or non-finite")
        if gate_name in ROTATION_GATES:
            if len(params) != 1 or len(qubits) != 1:
                raise ValueError(f"{gate_name} requires 1 param and 1 qubit")
            method(params[0], qubits[0])
        elif gate_name in TWO_QUBIT_GATES:
            if len(qubits) != 2 or qubits[0] == qubits[1]:
                raise ValueError(f"{gate_name} requires 2 distinct qubits")
            method(qubits[0], qubits[1])
        else:
            if len(qubits) != 1:
                raise ValueError(f"{gate_name} requires exactly 1 qubit")
            method(qubits[0])
