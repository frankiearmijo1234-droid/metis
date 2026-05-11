"""QAOA: Quantum Approximate Optimization Algorithm.

Hybrid quantum/classical algorithm for QUBO. Builds a parameterized
quantum circuit, simulates it (via qmlx), and lets a classical optimizer
search for the parameters that minimize <H_C> (expected cost). After the
classical loop finishes, samples from the optimized circuit; the most
frequent bitstring is the answer.

Reference: Farhi, Goldstone, Gutmann (2014) "A Quantum Approximate
Optimization Algorithm" (arXiv:1411.4028).

Math, briefly
-------------
For a QUBO with binary x in {0,1}^n minimizing x^T Q x, we transform to
spin variables s_i = 1 - 2*x_i in {-1, +1}. The cost expressed on Z
operators is:

    H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + const

Cost layer: U_C(gamma) = exp(-i * gamma * H_C). Since each term commutes
with itself, U_C is implemented as RZ(2*gamma*h_i) on each qubit i and
ZZ(2*gamma*J_{ij}) on each pair. ZZ(theta) = CNOT, RZ(theta), CNOT.

Mixer layer: U_B(beta) = exp(-i * beta * sum_i X_i) = product of RX(2*beta)
on each qubit.

Initial state: |+>^n. After p layers of alternating U_C, U_B, we measure.

Parameter search: scipy.optimize over (gamma_1..gamma_p, beta_1..beta_p)
to minimize <H_C>. Computing <H_C> requires running the simulator and
extracting Z and ZZ expectation values from the state vector.

This engine handles QUBO problems up to ~14 qubits in a useful time budget
on a CPU. With MLX or a GPU it could go higher; we default-cap at 18 to
be honest about runtime.

Handles ProblemKind.OPTIMIZATION with QUBO payload.
Hints respected:
    - p: int                 QAOA depth (default 2)
    - max_iter: int          classical optimizer iterations (default 50)
    - n_shots: int           samples for the final answer (default 2048)
    - optimizer: str         scipy method name (default "COBYLA")
    - seed: int
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Problem, ProblemKind, Solution

MAX_QUBITS = 18  # 2^18 amplitudes, ~2MB; bigger gets slow
MAX_P = 8  # circuit depth
DEFAULT_MAX_ITER = 50
DEFAULT_N_SHOTS = 2048


class QAOA:
    name = "qaoa"
    MAX_QUBITS = MAX_QUBITS

    def can_handle(self, problem: Problem) -> bool:
        if problem.kind != ProblemKind.OPTIMIZATION:
            return False
        p = problem.payload
        if p.get("linear_constraints") or p.get("ilp_solve"):
            return False
        if "qubo_Q" not in p or not p.get("qubo_solve", False):
            return False
        try:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
        except Exception:
            return False
        if not (1 <= n <= MAX_QUBITS):
            return False
        # QAOA is opt-in: it's slower than SA/PT for typical QUBO. The user
        # signals intent by passing prefer_qaoa=True or method="qaoa".
        prefer = problem.hints.get("prefer_qaoa", False)
        method = str(problem.hints.get("method", "")).lower()
        if not prefer and method != "qaoa":
            return False
        return True

    def estimate_cost(self, problem: Problem) -> float:
        p = problem.payload
        try:
            n = int(np.asarray(p["qubo_Q"]).shape[0])
        except Exception:
            return float("inf")
        if n > MAX_QUBITS:
            return float("inf")
        depth = problem.hints.get("p", 2)
        max_iter = problem.hints.get("max_iter", DEFAULT_MAX_ITER)
        # Each classical iteration runs the simulator once, which is
        # 2^n complex amplitudes * O(p * n^2) ops for the cost+mixer layers.
        # Calibrated very roughly on this CPU.
        amps = 2**n
        gates_per_layer = n + n * (n - 1) // 2 * 3  # n RX + n*(n-1)/2 * (CNOT,RZ,CNOT)
        total_per_iter = amps * depth * gates_per_layer * 1e-9
        return 0.05 + max_iter * total_per_iter

    def solve(self, problem: Problem) -> Solution:
        from qmlx import Circuit  # state-vector simulator
        from scipy.optimize import minimize as scipy_min

        t0 = time.perf_counter()
        p = problem.payload

        # Validate Q
        Q_in = np.asarray(p["qubo_Q"])
        if np.iscomplexobj(Q_in):
            raise ValueError("QUBO Q must be real-valued, got complex dtype")
        Q = np.asarray(Q_in, dtype=np.float64)
        n = Q.shape[0]
        if Q.ndim != 2 or n != Q.shape[1]:
            raise ValueError(f"QUBO Q must be square, got {Q.shape}")
        if n < 1 or n > MAX_QUBITS:
            raise ValueError(f"n={n} outside [1, {MAX_QUBITS}]")
        if not np.all(np.isfinite(Q)):
            raise ValueError("QUBO Q contains NaN or inf values")
        Qs = (Q + Q.T) / 2.0

        # Hints
        depth = int(problem.hints.get("p", 2))
        max_iter = int(problem.hints.get("max_iter", DEFAULT_MAX_ITER))
        n_shots = int(problem.hints.get("n_shots", DEFAULT_N_SHOTS))
        optimizer_name = str(problem.hints.get("optimizer", "COBYLA"))
        seed = problem.hints.get("seed")

        if depth < 1 or depth > MAX_P:
            raise ValueError(f"p must be in [1, {MAX_P}], got {depth}")
        if max_iter < 1 or max_iter > 1000:
            raise ValueError("max_iter must be in [1, 1000]")
        if n_shots < 1 or n_shots > 1_000_000:
            raise ValueError("n_shots must be in [1, 1000000]")

        # Convert QUBO -> Ising (Z and ZZ coefficients). For binary
        # x_i in {0,1}, substitute x_i = (1 - z_i) / 2 with z_i in {-1,+1}.
        # x^T Q x = sum_ij Q_ij x_i x_j
        #        = sum_ij Q_ij * (1-z_i)(1-z_j)/4
        # Expanding: (constant) + (linear z_i terms) + (z_i z_j terms)
        # Linear: h_i = -1/4 * sum_j (Q_ij + Q_ji) = -1/2 * sum_j Qs_ij
        # Quadratic: J_ij = 1/4 * (Qs_ij + Qs_ji) = Qs_ij / 2 (for i<j we
        # combine both off-diagonal terms; the i=j case folds into linear).
        h = np.zeros(n)
        # Diagonal: Qs[i,i]*x_i^2 = Qs[i,i]*x_i (since x_i is binary)
        # x_i = (1 - z_i)/2, so this contributes -Qs[i,i]/2 * z_i + Qs[i,i]/2
        for i in range(n):
            h[i] = -Qs[i, i] / 2.0
            for j in range(n):
                if j != i:
                    # Off-diagonal contributes via the (1-z_i)(1-z_j)/4 expansion.
                    # The linear part picks up -Qs[i,j]/2 on z_i (since pair i,j
                    # contributes Qs[i,j]/2 to coefficient of -z_i overall).
                    h[i] -= Qs[i, j] / 2.0
        # Quadratic ZZ couplings: J[i,j] for i<j
        J: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                # x_i x_j contribution to Q-form: 2 * Qs[i,j] (Qs symmetric)
                # In Z-form this becomes Qs[i,j]/2 * z_i z_j
                Jij = Qs[i, j] / 2.0
                if abs(Jij) > 1e-15:
                    J[(i, j)] = Jij

        # Map non-zero couplings to a list for circuit construction
        zz_pairs = [(i, j, Jij) for (i, j), Jij in J.items()]

        # Build the QAOA circuit at parameters (gammas, betas), simulate,
        # return the state vector.
        def build_state(params: np.ndarray):
            gammas = params[:depth]
            betas = params[depth:]
            circuit = Circuit(n)
            # Initial state |+>^n
            for i in range(n):
                circuit.h(i)
            # p layers
            for layer in range(depth):
                gamma = float(gammas[layer])
                beta = float(betas[layer])
                # Cost layer U_C(gamma) = exp(-i * gamma * H_C)
                # Linear part: RZ(2*gamma*h_i) on each qubit
                for i in range(n):
                    if abs(h[i]) > 1e-15:
                        circuit.rz(2 * gamma * h[i], i)
                # Quadratic part: ZZ(2*gamma*J_ij) on each pair
                for i, j, Jij in zz_pairs:
                    angle = 2 * gamma * Jij
                    circuit.cnot(i, j)
                    circuit.rz(angle, j)
                    circuit.cnot(i, j)
                # Mixer layer U_B(beta) = product of RX(2*beta) on each qubit
                for i in range(n):
                    circuit.rx(2 * beta, i)
            return circuit.run()

        # Compute <H_C> for a given state vector. For our QUBO form this
        # equals sum_i h_i <Z_i> + sum_{i<j} J_ij <Z_i Z_j> + constant offset.
        # The constant doesn't affect optimization; we add it back at the end
        # to report the original QUBO value.
        constant = float(np.sum(Qs) / 4.0 + np.trace(Qs) / 4.0)
        # Wait: let me re-derive constant. Take all the bits (1-z)/2
        # substitution: x_i x_j = (1-z_i)(1-z_j)/4. Sum:
        #   sum_ij Qs_ij x_i x_j = sum_ij Qs_ij * (1 - z_i - z_j + z_i z_j) / 4
        # Constant: sum_ij Qs_ij / 4
        # i ≠ j Z_i Z_j contribution: sum_{i<j} Qs_ij / 2 (after combining)
        # Diagonal i=j: Qs_ii x_i = Qs_ii (1-z_i)/2 -> Qs_ii/2 - Qs_ii z_i / 2
        # We already included diagonal in h. The "sum_ij Qs_ij / 4" double-
        # counts the diagonal. Let me just compute the constant directly.
        constant = float(np.sum(Qs) / 4.0 + np.sum(np.diag(Qs)) / 4.0)

        # Recompute constant carefully:
        # sum_ij Qs_ij x_i x_j with x = (1-z)/2:
        #   = (1/4) sum_ij Qs_ij - (1/4) sum_ij Qs_ij z_j ... etc.
        # Expanding fully:
        #   x_i x_j = 1/4 (1 - z_i - z_j + z_i z_j)
        # Sum over i,j:
        #   = (1/4)[sum_ij Qs_ij - sum_j(sum_i Qs_ij) z_j - sum_i (sum_j Qs_ij) z_i + sum_ij Qs_ij z_i z_j]
        # By symmetry of Qs:
        #   = (1/4) sum_ij Qs_ij - (1/2) sum_j (sum_i Qs_ij) z_j + (1/4) sum_ij Qs_ij z_i z_j
        # Linear: h_j = -(1/2) sum_i Qs_ij
        # Quadratic: (1/4) sum_ij Qs_ij z_i z_j. For i=j, z_i^2 = 1, so that
        # contributes (1/4) sum_i Qs_ii (a constant). For i!=j, contributes
        # (1/4)*2 * sum_{i<j} Qs_ij z_i z_j = (1/2) sum_{i<j} Qs_ij z_i z_j.
        # So J_ij for i<j is Qs_ij / 2.
        # And constant = (1/4) sum_ij Qs_ij + (1/4) sum_i Qs_ii.
        constant = float(Qs.sum() / 4.0 + np.diag(Qs).sum() / 4.0)
        # h: h_j = -(1/2) sum_i Qs_ij. We had set h[i] using a different
        # decomposition above; redo cleanly:
        h = -0.5 * Qs.sum(axis=1)
        # zz_pairs unchanged: J[i,j] = Qs[i,j]/2 for i<j

        def expected_energy(state_vec) -> float:
            """<psi| H_C |psi> = sum_i h_i <Z_i> + sum_{i<j} J_ij <Z_i Z_j> + const."""
            E = constant
            # <Z_i>: probabilities times Z eigenvalue
            probs = state_vec.probabilities()  # length 2^n
            # For each qubit i, <Z_i> = sum_states (prob if bit i = 0) - (prob if bit i = 1)
            # Equivalent: sum over states of prob * (1 - 2*bit_i)
            # Vectorized:
            # Compute bit i of each state index using bit operations.
            indices = np.arange(2**n)
            for i in range(n):
                bit_i = (indices >> i) & 1  # 0 or 1
                z_i = 1 - 2 * bit_i  # +1 or -1
                E += float(h[i] * np.sum(probs * z_i))
            for i, j, Jij in zz_pairs:
                bit_i = (indices >> i) & 1
                bit_j = (indices >> j) & 1
                zz = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                E += float(Jij * np.sum(probs * zz))
            return E

        # Starting parameters: small random in [0, pi]
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, np.pi, size=2 * depth)

        # Track the iteration cost for diagnostics
        eval_history: list[float] = []

        def loss(params: np.ndarray) -> float:
            sv = build_state(params)
            E = expected_energy(sv)
            eval_history.append(E)
            return E

        # Classical optimizer
        try:
            result = scipy_min(
                loss,
                x0,
                method=optimizer_name,
                options={"maxiter": max_iter, "rhobeg": 0.5},
            )
            best_params = result.x
            best_energy = float(result.fun)
            opt_success = bool(result.success)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return Solution(
                value={
                    "x": None,
                    "fun": float("inf"),
                    "method": "qaoa",
                    "warning": f"classical optimizer failed: {e}",
                },
                engine_name=self.name,
                elapsed_sec=elapsed,
                metadata={},
            )

        # Sample the optimized circuit and pick the best bitstring observed
        sv_opt = build_state(best_params)
        samples = sv_opt.sample(n_shots, seed=seed)
        # samples: array of integers in [0, 2^n). Decode bit i of each.
        # Find the bitstring that minimizes the actual QUBO objective.
        # (Not necessarily the most frequent; we evaluate every distinct sample.)
        unique, counts = np.unique(samples, return_counts=True)
        best_x = None
        best_qubo = float("inf")
        for s, c in zip(unique, counts):
            x = np.array([(int(s) >> i) & 1 for i in range(n)], dtype=float)
            v = float(x @ Qs @ x)
            if v < best_qubo:
                best_qubo = v
                best_x = x

        elapsed = time.perf_counter() - t0
        return Solution(
            value={
                "x": best_x,
                "fun": best_qubo,
                "method": "qaoa",
                "p": depth,
                "n_shots": n_shots,
                "qaoa_expected_energy": best_energy,
                "optimizer_iters": len(eval_history),
                "optimizer_success": opt_success,
            },
            engine_name=self.name,
            elapsed_sec=elapsed,
            metadata={"size": n, "p": depth},
        )
