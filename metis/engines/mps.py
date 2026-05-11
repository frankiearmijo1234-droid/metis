"""Matrix Product State (MPS) tensor-network quantum simulator.

For a state-vector simulator, an n-qubit state needs 2^n complex amplitudes.
At n=30 that's 16 GB; at n=50 it's 16 PB. Hard memory wall.

MPS represents the state as a chain of small tensors:

    A^[0] -- A^[1] -- A^[2] -- ... -- A^[n-1]
       |       |       |                |
     site 0  site 1  site 2          site n-1

Each tensor A^[i] has shape (chi_left, 2, chi_right). The bond dimension
chi caps how much entanglement the representation can hold. Memory scales
as O(n * chi^2), so n=100 with chi=64 fits in <1 MB.

This wins over state-vector simulation when:
- Entanglement stays bounded (1D physics, shallow circuits, etc.).
- The user picks a bond dimension large enough for their state but smaller
  than 2^(n/2).

It loses (gives bad answers) when entanglement grows past chi. We honestly
cap that and warn.

Reference: Schollwöck, "The density-matrix renormalization group in the
age of matrix product states" (Annals of Physics 326, 2011).

Supports: H, S, S†, T, T†, X, Y, Z, RX, RY, RZ, CNOT, CZ, SWAP.
Two-qubit gates only directly applied to *adjacent* qubits. Non-adjacent
gates use an explicit SWAP network. We hide that wiring behind the
canonical apply_op interface.

Cap: 200 qubits, bond_dim 256. That keeps tensors under ~50 MB total.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

MAX_QUBITS = 200
MAX_OPS = 100_000
MAX_BOND_DIM = 256
DEFAULT_BOND_DIM = 32

# Single-qubit gate matrices
_SQRT2 = 1.0 / np.sqrt(2.0)
GATE_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": _SQRT2 * np.array([[1, 1], [1, -1]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "SDG": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    "TDG": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128),
}
SINGLE_QUBIT = set(GATE_MATRICES.keys())
ROTATION_GATES = {"RX", "RY", "RZ", "PHASE"}
TWO_QUBIT = {"CNOT", "CX", "CZ", "SWAP"}
ALL_GATES = SINGLE_QUBIT | ROTATION_GATES | TWO_QUBIT


def rotation_matrix(name: str, theta: float) -> np.ndarray:
    """Build a rotation gate matrix from a name and angle."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    if name == "RX":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if name == "RY":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=np.complex128,
        )
    if name == "PHASE":
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
    raise ValueError(f"unknown rotation gate {name}")


# Two-qubit unitaries as 4x4 matrices (in computational basis ordering)
def _twoq_unitary(name: str) -> np.ndarray:
    if name in ("CNOT", "CX"):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if name == "SWAP":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"unknown two-qubit gate {name}")


class MPS:
    """Matrix Product State for n qubits.

    Internal layout: list of n tensors, tensor i has shape
    (chi_left, 2, chi_right). For the |0...0> initial state, all tensors
    have shape (1, 2, 1) with the value [[[1], [0]]] (i.e., bond dim 1).
    """

    def __init__(self, n: int, bond_dim: int = DEFAULT_BOND_DIM):
        if n < 1 or n > MAX_QUBITS:
            raise ValueError(f"n must be in [1, {MAX_QUBITS}], got {n}")
        if bond_dim < 2 or bond_dim > MAX_BOND_DIM:
            raise ValueError(f"bond_dim must be in [2, {MAX_BOND_DIM}], got {bond_dim}")
        self.n = n
        self.max_bond_dim = bond_dim
        # Initialize |0...0>
        self.tensors: list[np.ndarray] = []
        for _ in range(n):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)
        # Track maximum bond dim actually used (for diagnostics)
        self.max_bond_used = 1
        self.truncation_error = 0.0  # accumulated truncation L2 error squared

    def apply_single(self, gate: np.ndarray, q: int) -> None:
        """Apply a 2x2 unitary to qubit q. No bond-dim change."""
        if q < 0 or q >= self.n:
            raise ValueError(f"qubit {q} out of range [0, {self.n})")
        T = self.tensors[q]  # (chi_l, 2, chi_r)
        # Contract gate's input index with T's physical index (axis 1).
        # T'[a, p_out, b] = sum_{p_in} G[p_out, p_in] T[a, p_in, b]
        self.tensors[q] = np.einsum("ij,ajb->aib", gate, T)

    def apply_two_adjacent(self, gate4x4: np.ndarray, q: int) -> None:
        """Apply a 4x4 unitary to adjacent qubits q and q+1.

        Uses SVD to update the two affected tensors and truncate back to
        bond_dim.
        """
        if q < 0 or q >= self.n - 1:
            raise ValueError(f"adjacent gate at q={q} out of range")
        Tl = self.tensors[q]  # (chi_l, 2, chi_m)
        Tr = self.tensors[q + 1]  # (chi_m, 2, chi_r)
        chi_l, _, chi_m = Tl.shape
        _, _, chi_r = Tr.shape
        # Combine into a (chi_l, 2, 2, chi_r) tensor
        # Step 1: contract Tl and Tr along their middle bond.
        theta = np.einsum("aib,bjc->aijc", Tl, Tr)
        # Step 2: apply gate. Reshape gate to (2, 2, 2, 2): gate[i_out, j_out, i_in, j_in]
        G = gate4x4.reshape(2, 2, 2, 2)
        # Apply: theta'[a, i', j', c] = sum_{i, j} G[i', j', i, j] theta[a, i, j, c]
        theta = np.einsum("xyij,aijc->axyc", G, theta)
        # Step 3: SVD-truncate. Reshape theta to (chi_l*2, 2*chi_r) matrix.
        mat = theta.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        # Truncate to max_bond_dim singular values
        keep = min(self.max_bond_dim, S.size)
        # Track truncation error = sum of discarded singular values squared
        if S.size > keep:
            disc = float(np.sum(S[keep:] ** 2))
            self.truncation_error += disc
        S = S[:keep]
        U = U[:, :keep]
        Vh = Vh[:keep, :]
        # Renormalize after truncation (small loss but keeps state norm 1)
        norm = np.linalg.norm(S)
        if norm > 0:
            S = S / norm
        # Step 4: split back into Tl' = U * sqrt(S) and Tr' = sqrt(S) * Vh.
        # Use left-canonical convention: U absorbs no S, Tr' = diag(S) @ Vh.
        # But that requires propagating norms; standard practice splits
        # symmetrically.
        sqrt_S = np.sqrt(S)
        new_Tl = (U * sqrt_S).reshape(chi_l, 2, keep)
        new_Tr = (sqrt_S[:, None] * Vh).reshape(keep, 2, chi_r)
        self.tensors[q] = new_Tl
        self.tensors[q + 1] = new_Tr
        if keep > self.max_bond_used:
            self.max_bond_used = keep

    def swap(self, q1: int, q2: int) -> None:
        """Apply SWAP between any two qubits via SWAPs along a chain.

        For adjacent SWAP, this is a single two-qubit gate.
        For distant SWAP, we perform a chain of adjacent SWAPs.
        """
        if q1 == q2:
            return
        a, b = sorted([q1, q2])
        S = _twoq_unitary("SWAP")
        # Move qubit at a rightward to position b, then back to a
        # (this leaves the state with qubits effectively swapped at endpoints).
        # Simpler: apply adjacent SWAPs (a,a+1), (a+1,a+2), ..., (b-1,b).
        # That moves whatever was at a to position b, and shifts everything in
        # between left by one. Then to swap, we need to bring the original
        # qubit-b content back to position a, which requires another chain.
        # Cleanest: do a -> b (forward), then b-1 -> a (backward, on the fact
        # that the original b content is now at b-1).
        for k in range(a, b):
            self.apply_two_adjacent(S, k)
        for k in range(b - 1, a, -1):
            self.apply_two_adjacent(S, k - 1)

    def apply_two_qubit(self, gate_name: str, q1: int, q2: int) -> None:
        """Apply a 2-qubit gate by name to (possibly non-adjacent) qubits.

        For adjacent qubits we directly apply. For non-adjacent we use a
        SWAP network: bring q1 adjacent to q2, apply gate, swap back.
        """
        if q1 == q2:
            raise ValueError("two-qubit gate requires distinct qubits")
        gate = _twoq_unitary(gate_name)
        if abs(q1 - q2) == 1:
            # Already adjacent. Apply with the lower index first.
            if q1 < q2:
                self.apply_two_adjacent(gate, q1)
            else:
                # Need to apply gate with the convention swapped.
                # SWAP-conjugate: G' = SWAP * G * SWAP.
                S = _twoq_unitary("SWAP")
                G_prime = S @ gate @ S
                self.apply_two_adjacent(G_prime, q2)
            return
        # Non-adjacent: use SWAP network. Bring qubit q1 adjacent to q2 by
        # SWAPping it stepwise. We always SWAP q1 toward q2 by one step at
        # a time, apply, then unwind.
        a, b = q1, q2
        # Move a toward b
        if a < b:
            for k in range(a, b - 1):
                self.apply_two_adjacent(_twoq_unitary("SWAP"), k)
            # Now a is at position b-1 (adjacent). Apply gate at (b-1, b).
            self.apply_two_adjacent(gate, b - 1)
            # Unwind
            for k in range(b - 2, a - 1, -1):
                self.apply_two_adjacent(_twoq_unitary("SWAP"), k)
        else:
            # a > b: mirror
            for k in range(a - 1, b, -1):
                self.apply_two_adjacent(_twoq_unitary("SWAP"), k)
            self.apply_two_adjacent(gate, b)
            for k in range(b + 1, a):
                self.apply_two_adjacent(_twoq_unitary("SWAP"), k)

    def to_state_vector(self) -> np.ndarray:
        """Contract all tensors into a 2^n state vector. Memory expensive!

        Only safe at small n. Used for tests and small-circuit verification.
        """
        if self.n > 20:
            raise ValueError(f"to_state_vector at n={self.n} would need 2^n amplitudes")
        # Contract from left: state is (chi_left, prod_phys, chi_right)
        result = self.tensors[0]  # (1, 2, chi)
        for k in range(1, self.n):
            T = self.tensors[k]
            # Contract the right bond of result with the left bond of T.
            # result shape: (1, 2^k, chi_old)
            # T shape: (chi_old, 2, chi_new)
            # New result: (1, 2^k * 2, chi_new)
            result = np.einsum("aib,bjc->aijc", result, T)
            new_shape = (result.shape[0], result.shape[1] * 2, result.shape[3])
            result = result.reshape(new_shape)
        # result is (1, 2^n, 1)
        sv = result.reshape(2**self.n)
        # Normalize (truncation may have caused a tiny norm drift)
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        return sv

    def probabilities(self) -> np.ndarray:
        """Return probabilities of all 2^n outcomes. Only sane for small n."""
        sv = self.to_state_vector()
        return np.abs(sv) ** 2

    def sample(self, n_shots: int, rng: np.random.Generator) -> list[str]:
        """Sample bitstrings from the state.

        For small n, sample directly from the full probabilities.
        For larger n, use sequential conditional sampling per qubit (still
        works without materializing 2^n amps).
        """
        if self.n <= 20:
            probs = self.probabilities()
            indices = rng.choice(2**self.n, size=n_shots, p=probs)
            # Bit order: leftmost char of the result string is qubit 0.
            # `format(i, '0Nb')` is MSB-first, which matches our convention
            # (in to_state_vector, qubit 0 is the outermost site -> MSB).
            return [format(int(i), f"0{self.n}b") for i in indices]
        # Sequential sampling: for each shot, walk the chain qubit by qubit.
        # At each site, marginalize remaining qubits, condition on prior bits.
        results = []
        for _ in range(n_shots):
            results.append(self._sample_one(rng))
        return results

    def _sample_one(self, rng: np.random.Generator) -> str:
        """Single-shot sequential sampling -- O(n * chi^3) per shot.

        Walks the MPS chain left-to-right, computing the marginal probability
        of bit i given bits 0..i-1, sampling, and absorbing the result into
        a running left environment.
        """
        n = self.n
        # Build right environments R[i] = contraction of all tensors[i..n-1]
        # against their conjugates, leaving open left bonds. Shape: (chi_i, chi_i).
        # R[n] = 1 (1x1 trivial)
        right_envs: list[np.ndarray] = [None] * (n + 1)  # type: ignore
        right_envs[n] = np.array([[1.0]], dtype=np.complex128)
        for i in range(n - 1, -1, -1):
            T = self.tensors[i]  # (chi_l, 2, chi_r)
            R_next = right_envs[i + 1]  # (chi_r, chi_r')
            # tmp[a, p, c] = sum_b T[a, p, b] R_next[b, c]
            tmp = np.einsum("apb,bc->apc", T, R_next)
            # R[i][a, d] = sum_{p, c} tmp[a, p, c] conj(T)[d, p, c]
            R = np.einsum("apc,dpc->ad", tmp, np.conjugate(T))
            right_envs[i] = R

        # Walk left-to-right. Maintain L[a, a'] (running left environment)
        L = np.array([[1.0]], dtype=np.complex128)
        bits: list[str] = []
        for i in range(n):
            T = self.tensors[i]
            R = right_envs[i + 1]
            # Single-site reduced density matrix:
            # rho[p, q] = sum L[a, d] T[a, p, b] R[b, c] conj(T)[d, q, c]
            tmp = np.einsum("ad,apb->dpb", L, T)
            tmp = np.einsum("dpb,bc->dpc", tmp, R)
            rho = np.einsum("dpc,dqc->pq", tmp, np.conjugate(T))
            p0 = float(np.real(rho[0, 0]))
            p1 = float(np.real(rho[1, 1]))
            total = max(p0 + p1, 1e-14)
            p0 /= total
            r = rng.random()
            bit = 0 if r < p0 else 1
            bits.append(str(bit))
            # Project: take the slice of T at the measured bit and update L.
            T_slice = T[:, bit, :]  # (chi_l, chi_r)
            # New L[b, c] = sum_{a, d} L[a, d] T_slice[a, b] conj(T_slice)[d, c]
            L_new = np.einsum("ad,ab->db", L, T_slice)
            L_new = np.einsum("db,dc->bc", L_new, np.conjugate(T_slice))
            tr = np.real(np.trace(L_new))
            if tr > 1e-14:
                L_new = L_new / tr
            L = L_new
        return "".join(bits)


# ---------- Engine wrapper ----------


class MPSSimulator:
    name = "mps"
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
        # Validate gates
        for op in ops:
            if not isinstance(op, dict):
                return False
            gate = str(op.get("gate", "")).upper()
            if gate not in ALL_GATES:
                return False
        # Tasks supported
        task = p.get("task", "sample")
        if task not in ("sample", "expectation_z", "probabilities"):
            return False
        # Probabilities only sensible at small n (we'd need 2^n amplitudes)
        if task == "probabilities" and n > 20:
            return False
        # MPS is opt-in for now: routing should prefer state-vector at small
        # n (it's faster) and stabilizer for Clifford circuits. The user
        # signals MPS preference via prefer_mps=True.
        prefer = problem.hints.get("prefer_mps", False)
        if not prefer:
            return False
        return True

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        n = int(p["n_qubits"])
        n_ops = len(p["ops"])
        bond_dim = problem.hints.get("bond_dim", DEFAULT_BOND_DIM)
        # Per single-qubit gate: O(chi^2) single-site contraction
        # Per two-qubit gate: O(chi^3) for the SVD
        # Distant two-qubit gates need O(distance * chi^3) for swap network
        # Rough: each op contributes ~5e-7 * chi^3 seconds.
        per_gate = 5e-7 * (bond_dim**3)
        return 0.005 + n_ops * per_gate

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
        if task not in ("sample", "expectation_z", "probabilities"):
            raise ValueError(f"unknown task: {task}")
        bond_dim = int(problem.hints.get("bond_dim", DEFAULT_BOND_DIM))

        mps = MPS(n, bond_dim=bond_dim)
        for op in ops:
            self._apply_op(mps, op, n)

        rng_seed = p.get("task_args", {}).get("seed")
        rng = np.random.default_rng(rng_seed)

        if task == "sample":
            n_shots = p.get("task_args", {}).get("n_shots", 1000)
            if not isinstance(n_shots, int) or n_shots < 1 or n_shots > 1_000_000:
                raise ValueError(f"n_shots must be int in [1, 1000000], got {n_shots}")
            samples = mps.sample(n_shots, rng)
            counts: dict[str, int] = {}
            for s in samples:
                counts[s] = counts.get(s, 0) + 1
            value = {
                "counts": counts,
                "n_shots": n_shots,
                "bond_dim_used": mps.max_bond_used,
                "bond_dim_max": mps.max_bond_dim,
                "truncation_error": mps.truncation_error,
            }
        elif task == "probabilities":
            value = {
                "probabilities": mps.probabilities().tolist(),
                "bond_dim_used": mps.max_bond_used,
                "truncation_error": mps.truncation_error,
            }
        else:  # expectation_z
            qubit = int(p.get("task_args", {}).get("qubit", 0))
            if qubit < 0 or qubit >= n:
                raise ValueError(f"qubit {qubit} out of range")
            # Sample-based estimate
            n_shots = p.get("task_args", {}).get("n_shots", 10000)
            samples = mps.sample(n_shots, rng)
            outcomes = np.array([int(s[qubit]) for s in samples])
            expectation = float(np.mean(1 - 2 * outcomes))
            value = {
                "expectation_z": expectation,
                "qubit": qubit,
                "n_shots": n_shots,
                "bond_dim_used": mps.max_bond_used,
                "truncation_error": mps.truncation_error,
            }

        elapsed = time.perf_counter() - t0
        return Solution(
            value=value,
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={
                "n_qubits": n,
                "n_ops": len(ops),
                "task": task,
                "bond_dim_max": mps.max_bond_dim,
                "bond_dim_used": mps.max_bond_used,
                "truncation_error": mps.truncation_error,
            },
        )

    @staticmethod
    def _apply_op(mps: MPS, op: dict, n: int) -> None:
        if not isinstance(op, dict):
            raise ValueError("op must be a dict")
        gate = str(op.get("gate", "")).upper()
        if gate not in ALL_GATES:
            raise ValueError(f"unknown or unsupported gate: {gate}")
        qubits = op.get("qubits", [])
        if not isinstance(qubits, list):
            raise ValueError("qubits must be a list")
        for q in qubits:
            if not isinstance(q, int) or isinstance(q, bool):
                raise ValueError(f"qubit indices must be ints, got {q!r}")
            if q < 0 or q >= n:
                raise ValueError(f"qubit index {q} out of range [0, {n})")
        params = op.get("params", [])
        if not isinstance(params, list):
            raise ValueError("params must be a list")
        for prm in params:
            if not isinstance(prm, (int, float)) or isinstance(prm, bool):
                raise ValueError(f"param must be a number, got {prm!r}")
            if not (-1e9 < float(prm) < 1e9):
                raise ValueError(f"param out of safe range: {prm}")

        if gate in SINGLE_QUBIT:
            if len(qubits) != 1:
                raise ValueError(f"{gate} requires 1 qubit")
            mps.apply_single(GATE_MATRICES[gate], qubits[0])
        elif gate in ROTATION_GATES:
            if len(params) != 1 or len(qubits) != 1:
                raise ValueError(f"{gate} requires 1 param and 1 qubit")
            mps.apply_single(rotation_matrix(gate, params[0]), qubits[0])
        elif gate in TWO_QUBIT:
            if len(qubits) != 2 or qubits[0] == qubits[1]:
                raise ValueError(f"{gate} requires 2 distinct qubits")
            mps.apply_two_qubit(gate, qubits[0], qubits[1])
        else:
            raise ValueError(f"unhandled gate: {gate}")
