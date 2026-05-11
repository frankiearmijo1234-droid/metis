"""Stabilizer simulator engine: 1000+ qubit Clifford circuits.

State-vector simulation needs 2^n complex amplitudes -- 30 qubits is 16 GB.
For Clifford circuits (the gate set H, S, CNOT and their products) there's
a much better representation: the stabilizer tableau, which uses O(n^2)
bits and supports any Clifford gate in O(n) time.

This is a real classical-simulation capability boost. Clifford circuits are
limited (they cannot create universal quantum computation alone -- you need
a non-Clifford gate like T to escape) but they cover a huge slice of useful
quantum work: error-correction circuits, Bell pair generation at scale,
GHZ state preparation, graph states, randomized benchmarking, and the entire
analysis of stabilizer codes. They're also exactly the regime where claims
of "we simulated N qubits on a laptop" hold.

Reference: Aaronson & Gottesman 2004 (quant-ph/0406196) "Improved Simulation
of Stabilizer Circuits".

Tableau representation
----------------------
For an n-qubit state we keep 2n+1 stabilizer/destabilizer rows. Each row is
2n bits (X part, Z part) plus a sign bit. Rows 0..n-1 are destabilizers
(used for measurement); rows n..2n-1 are stabilizers (the actual stabilizer
group); row 2n is a scratch row.

Tableau layout (ours): a single (2n+1) x (2n+1) numpy uint8 matrix M where:
  - M[r, 0:n]   = X part of row r
  - M[r, n:2n]  = Z part of row r
  - M[r, 2n]    = sign bit of row r (0 = +, 1 = -)

Initial |0...0> state has destabilizers = X_i, stabilizers = Z_i, all signs +.

Supported gates
---------------
- H, S, S^dagger, X, Y, Z (single-qubit Clifford)
- CNOT (CX), CZ, SWAP (two-qubit Clifford)
- Z-basis measurement of a qubit (returns a bit)

NOT supported: T, T-dagger, RX/RY/RZ, arbitrary unitaries. Those are
non-Clifford. Use the qmlx_statevector engine for those.

Handles ProblemKind.QUANTUM_CIRCUIT with payload form same as quantum_sim
but only Clifford gates and Z-basis measurement are accepted.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

# Engine caps. Stabilizer storage is O(n^2) bits, so even 5000 qubits is
# only ~3 MB. The bottleneck is gate-application speed (each two-qubit
# gate touches ~2n entries).
MAX_QUBITS = 10_000
MAX_OPS = 1_000_000

CLIFFORD_GATES = {"H", "S", "SDG", "X", "Y", "Z", "CNOT", "CX", "CZ", "SWAP"}
SINGLE_QUBIT = {"H", "S", "SDG", "X", "Y", "Z"}
TWO_QUBIT = {"CNOT", "CX", "CZ", "SWAP"}


class Tableau:
    """Stabilizer tableau over n qubits.

    Stored as one (2n+1) x (2n+1) uint8 matrix:
      - rows 0..n-1: destabilizers
      - rows n..2n-1: stabilizers
      - row 2n: scratch (used during measurement)
      - cols 0..n-1: X part, cols n..2n-1: Z part, col 2n: sign bit
    """

    def __init__(self, n: int):
        if n < 1 or n > MAX_QUBITS:
            raise ValueError(f"n must be in [1, {MAX_QUBITS}], got {n}")
        self.n = n
        self.M = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.uint8)
        # Destabilizer i: X on qubit i  -> M[i, i] = 1
        for i in range(n):
            self.M[i, i] = 1
        # Stabilizer i: Z on qubit i  -> M[n+i, n+i] = 1
        for i in range(n):
            self.M[n + i, n + i] = 1
        # All signs initially 0 (+).

    # ---- Single-qubit gates ----

    def h(self, q: int) -> None:
        """Hadamard on qubit q. Swaps X and Z parts; sign gets ^= x*z."""
        self._check_q(q)
        n = self.n
        x = self.M[:, q].copy()
        z = self.M[:, n + q].copy()
        # sign ^= x*z
        self.M[:, 2 * n] ^= x & z
        # swap X and Z columns
        self.M[:, q] = z
        self.M[:, n + q] = x

    def s(self, q: int) -> None:
        """Phase gate (S). Z stays, X becomes Y; sign ^= x*z."""
        self._check_q(q)
        n = self.n
        x = self.M[:, q]
        z = self.M[:, n + q]
        self.M[:, 2 * n] ^= x & z
        # Z part XORs in X part:  z' = z XOR x
        self.M[:, n + q] = z ^ x

    def sdg(self, q: int) -> None:
        """S-dagger: equivalent to S^3 = Z * S. We can do it as S then Z."""
        self.s(q)
        self.z(q)

    def x(self, q: int) -> None:
        """Pauli X. Sign flips on rows where the Z part has bit q set."""
        self._check_q(q)
        n = self.n
        self.M[:, 2 * n] ^= self.M[:, n + q]

    def z(self, q: int) -> None:
        """Pauli Z. Sign flips on rows where the X part has bit q set."""
        self._check_q(q)
        n = self.n
        self.M[:, 2 * n] ^= self.M[:, q]

    def y(self, q: int) -> None:
        """Pauli Y = iXZ. Sign flips where X^Z bit q set."""
        self._check_q(q)
        n = self.n
        self.M[:, 2 * n] ^= self.M[:, q] ^ self.M[:, n + q]

    # ---- Two-qubit gates ----

    def cnot(self, control: int, target: int) -> None:
        """CNOT with given control and target."""
        self._check_q(control)
        self._check_q(target)
        if control == target:
            raise ValueError("CNOT requires distinct qubits")
        n = self.n
        c = control
        t = target
        # sign ^= x_c * z_t * (x_t XOR z_c XOR 1)
        xc = self.M[:, c]
        zt = self.M[:, n + t]
        xt = self.M[:, t]
        zc = self.M[:, n + c]
        self.M[:, 2 * n] ^= xc & zt & (xt ^ zc ^ 1)
        # x_t ^= x_c ; z_c ^= z_t
        self.M[:, t] ^= xc
        self.M[:, n + c] ^= zt

    def cz(self, q1: int, q2: int) -> None:
        """CZ = H_2 * CNOT * H_2."""
        self.h(q2)
        self.cnot(q1, q2)
        self.h(q2)

    def swap(self, q1: int, q2: int) -> None:
        """SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)."""
        self.cnot(q1, q2)
        self.cnot(q2, q1)
        self.cnot(q1, q2)

    # ---- Measurement ----

    def measure_z(self, q: int, rng: np.random.Generator) -> int:
        """Measure qubit q in Z basis. Returns 0 or 1.

        Following Aaronson-Gottesman algorithm (quant-ph/0406196).
        """
        self._check_q(q)
        n = self.n
        # Find the FIRST stabilizer row with X bit on qubit q. If one
        # exists, the outcome is random; otherwise it's deterministic.
        # np.argmax over a boolean array returns the index of the first
        # True (or 0 if all False -- we check separately).
        stab_xq = self.M[n : 2 * n, q]
        if stab_xq.any():
            p = n + int(np.argmax(stab_xq))
            # Random outcome case. Vectorize the rowsum loop:
            # for every row i with X bit on q (except p), do rowsum(i, p).
            mask = self.M[:, q].astype(bool).copy()
            mask[p] = False
            rows_to_update = np.where(mask)[0]
            for i in rows_to_update:
                self._rowsum(int(i), p)
            outcome = int(rng.integers(0, 2))
            self.M[p - n, :] = self.M[p, :].copy()
            self.M[p, :] = 0
            self.M[p, n + q] = 1
            self.M[p, 2 * n] = outcome
            return outcome

        # Deterministic case.
        self.M[2 * n, :] = 0
        dest_xq = self.M[0:n, q]
        rows_to_apply = np.where(dest_xq == 1)[0]
        for i in rows_to_apply:
            self._rowsum(2 * n, n + int(i))
        return int(self.M[2 * n, 2 * n])

    # ---- helpers ----

    def _check_q(self, q: int) -> None:
        if not isinstance(q, (int, np.integer)) or q < 0 or q >= self.n:
            raise ValueError(f"qubit index {q} out of range [0, {self.n})")

    def _rowsum(self, h_row: int, i_row: int) -> None:
        """Set row h to (row h * row i) -- left-multiply h by i. Updates sign.

        From Aaronson-Gottesman: when multiplying two Pauli strings P1 P2,
        we compute the new sign as r1 + r2 + sum over qubits g(x1, z1, x2, z2)
        where g is +1, 0, or -1 mod 4 depending on the Pauli pair on each qubit.

        This is the inner loop of measurement; called O(n) times per measurement
        and O(n) work per call, so O(n^2) total per measurement. Fully
        vectorized.
        """
        n = self.n
        # Use views (no copies). M is uint8, so signed arithmetic works
        # within int8 range for these small values.
        x1 = self.M[i_row, 0:n]
        z1 = self.M[i_row, n : 2 * n]
        x2 = self.M[h_row, 0:n]
        z2 = self.M[h_row, n : 2 * n]

        # Closed form for g(x1, z1, x2, z2) using int arithmetic:
        # g = (1 - 2*z1) * x2 * z1   if x1=0, z1=1   -> x2*(1-2*z2) actually
        # Better: enumerate four cases compactly.
        # g_case_X (x1=1, z1=0): z2*(2*x2 - 1)
        # g_case_Z (x1=0, z1=1): x2*(1 - 2*z2)
        # g_case_Y (x1=1, z1=1): z2 - x2
        # g_case_I (x1=0, z1=0): 0
        # We can compute using indicator products without masking:
        # Let X1 = x1 (uint8), Z1 = z1 (uint8). Then:
        #   only_x = X1 & ~Z1
        #   only_z = ~X1 & Z1
        #   xy = X1 & Z1
        # Coerce to int16 once for the summed quantity.
        x1i = x1.astype(np.int16)
        z1i = z1.astype(np.int16)
        x2i = x2.astype(np.int16)
        z2i = z2.astype(np.int16)
        only_x = x1i * (1 - z1i)  # 1 where x1=1, z1=0
        only_z = (1 - x1i) * z1i  # 1 where x1=0, z1=1
        xy = x1i * z1i  # 1 where x1=1, z1=1
        g_total = int(
            np.sum(
                only_x * (z2i * (2 * x2i - 1))
                + only_z * (x2i * (1 - 2 * z2i))
                + xy * (z2i - x2i)
            )
        )

        total = (
            2 * int(self.M[h_row, 2 * n]) + 2 * int(self.M[i_row, 2 * n]) + g_total
        ) % 4
        if total == 0:
            self.M[h_row, 2 * n] = 0
        elif total == 2:
            self.M[h_row, 2 * n] = 1
        else:
            raise RuntimeError(
                f"rowsum produced odd total {total}; tableau is corrupted"
            )
        # XOR the X and Z parts (in-place uint8 XOR)
        np.bitwise_xor(self.M[h_row, 0:n], self.M[i_row, 0:n], out=self.M[h_row, 0:n])
        np.bitwise_xor(
            self.M[h_row, n : 2 * n],
            self.M[i_row, n : 2 * n],
            out=self.M[h_row, n : 2 * n],
        )


GATE_TABLEAU_METHODS = {
    "H": "h",
    "S": "s",
    "SDG": "sdg",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "CNOT": "cnot",
    "CX": "cnot",
    "CZ": "cz",
    "SWAP": "swap",
}


class StabilizerSimulator:
    name = "stabilizer"
    MAX_QUBITS = MAX_QUBITS
    MAX_OPS = MAX_OPS

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
        if not isinstance(ops, list) or len(ops) > self.MAX_OPS:
            return False
        # Every op must be a Clifford gate. Reject if any non-Clifford
        # gate is present.
        for op in ops:
            if not isinstance(op, dict):
                return False
            gate = str(op.get("gate", "")).upper()
            if gate not in CLIFFORD_GATES:
                return False
        # Task must be sample or expectation (probabilities of all 2^n
        # outcomes is exponential and shouldn't be requested from this
        # engine).
        task = p.get("task", "sample")
        if task not in ("sample", "expectation_z"):
            return False
        return True

    def estimate_cost(self, problem: Problem) -> float:
        """Tableau gate cost is O(n) per single-qubit gate, O(n) per two-qubit.
        Measurement is O(n^2). Far cheaper than 2^n, so this engine wins
        whenever it's eligible."""
        p = problem.payload
        n = int(p["n_qubits"])
        n_ops = len(p["ops"])
        task = p.get("task", "sample")
        # Per gate: ~5 numpy ops on length-n vectors. Conservative ~ n / 1e8 sec.
        gate_cost = n_ops * n / 1e8
        # Sampling: each shot is n measurements, each O(n^2). For 1000 shots
        # at n=100 that's 10^7 ops.
        if task == "sample":
            n_shots = problem.payload.get("task_args", {}).get("n_shots", 1000)
            sample_cost = n_shots * n * n / 1e8
        else:
            sample_cost = n / 1e6
        return 0.001 + gate_cost + sample_cost

    def solve(self, problem: Problem) -> Solution:
        t0 = time.perf_counter()
        p = problem.payload
        n = int(p["n_qubits"])
        if n < 1 or n > self.MAX_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {self.MAX_QUBITS}]")
        ops = p["ops"]
        if not isinstance(ops, list) or len(ops) > self.MAX_OPS:
            raise ValueError(f"ops must be list of length <= {self.MAX_OPS}")
        task = p.get("task", "sample")
        if task not in ("sample", "expectation_z"):
            raise ValueError(
                f"stabilizer engine supports task in (sample, expectation_z), got {task}"
            )

        tab = Tableau(n)
        # Apply gates
        for op in ops:
            self._apply_op(tab, op, n)

        rng_seed = p.get("task_args", {}).get("seed")
        rng = np.random.default_rng(rng_seed)

        if task == "sample":
            n_shots_raw = p.get("task_args", {}).get("n_shots", 1000)
            if (
                not isinstance(n_shots_raw, int)
                or n_shots_raw < 1
                or n_shots_raw > 1_000_000
            ):
                raise ValueError(
                    f"n_shots must be int in [1, 1000000], got {n_shots_raw}"
                )
            counts: dict[str, int] = {}
            for _ in range(n_shots_raw):
                # Each shot needs an independent measurement: use a copy of
                # the tableau because measurement collapses the state.
                tab_copy = self._clone_tableau(tab)
                bits = ""
                for q in range(n):
                    bit = tab_copy.measure_z(q, rng)
                    bits += str(bit)
                counts[bits] = counts.get(bits, 0) + 1
            value = {"counts": counts, "n_shots": n_shots_raw}
        elif task == "expectation_z":
            # Expectation value <Z_q> in [-1, 1].
            # For a stabilizer state, <Z_q> is +1 if Z_q is in the stabilizer
            # group (with sign +), -1 if in the group with sign -, and
            # 0 otherwise. We approximate via sampling.
            qubit = int(p.get("task_args", {}).get("qubit", 0))
            if qubit < 0 or qubit >= n:
                raise ValueError(f"qubit {qubit} out of range")
            n_shots = p.get("task_args", {}).get("n_shots", 10000)
            tab_copy = self._clone_tableau(tab)
            outcomes = np.zeros(n_shots, dtype=np.int32)
            for s in range(n_shots):
                tab_for_shot = self._clone_tableau(tab)
                outcomes[s] = tab_for_shot.measure_z(qubit, rng)
            # outcome 0 -> +1, outcome 1 -> -1
            expectation = float(np.mean(1 - 2 * outcomes))
            value = {"expectation_z": expectation, "qubit": qubit, "n_shots": n_shots}

        elapsed = time.perf_counter() - t0
        return Solution(
            value=value,
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"n_qubits": n, "n_ops": len(ops), "task": task},
        )

    @staticmethod
    def _clone_tableau(tab: Tableau) -> Tableau:
        """Deep-copy a tableau so measurement doesn't affect the original."""
        new = Tableau.__new__(Tableau)
        new.n = tab.n
        new.M = tab.M.copy()
        return new

    @staticmethod
    def _apply_op(tab: Tableau, op: dict, n: int) -> None:
        if not isinstance(op, dict):
            raise ValueError("op must be a dict")
        gate = str(op.get("gate", "")).upper()
        if gate not in CLIFFORD_GATES:
            raise ValueError(
                f"non-Clifford gate {gate} unsupported in stabilizer engine"
            )
        method_name = GATE_TABLEAU_METHODS[gate]
        method = getattr(tab, method_name)
        qubits = op.get("qubits", [])
        if not isinstance(qubits, list):
            raise ValueError("qubits must be a list")
        for q in qubits:
            if not isinstance(q, int) or isinstance(q, bool):
                raise ValueError(f"qubit indices must be ints, got {q!r}")
            if q < 0 or q >= n:
                raise ValueError(f"qubit index {q} out of range [0, {n})")
        if gate in TWO_QUBIT:
            if len(qubits) != 2:
                raise ValueError(f"{gate} requires 2 qubits")
            if qubits[0] == qubits[1]:
                raise ValueError(f"{gate} requires distinct qubits")
            method(qubits[0], qubits[1])
        else:
            if len(qubits) != 1:
                raise ValueError(f"{gate} requires exactly 1 qubit")
            method(qubits[0])
