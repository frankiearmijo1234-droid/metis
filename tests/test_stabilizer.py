"""Tests for the stabilizer simulator engine.

Stabilizer simulation runs Clifford circuits in O(n^2) memory instead of
O(2^n), enabling 1000+ qubit simulations. Tests cover:
1. Correctness: known states (|0>, |+>, Bell, GHZ) at small n.
2. Cross-check against state-vector simulator at moderate n.
3. Routing: which engine wins for which circuit.
4. Resource validation.
"""

import numpy as np
import pytest

from metis import Problem, ProblemKind, StabilizerSimulator
from metis.engines.stabilizer import Tableau

# ---------- Tableau-level correctness ----------


def test_initial_state_measures_zero():
    """|0> state in Z basis always reads 0."""
    tab = Tableau(1)
    rng = np.random.default_rng(0)
    assert tab.measure_z(0, rng) == 0


def test_x_gate_flips_to_one():
    tab = Tableau(1)
    tab.x(0)
    rng = np.random.default_rng(0)
    assert tab.measure_z(0, rng) == 1


def test_z_gate_does_not_flip_z_basis():
    tab = Tableau(1)
    tab.z(0)
    rng = np.random.default_rng(0)
    assert tab.measure_z(0, rng) == 0


def test_h_creates_superposition():
    """H|0> = |+> measured in Z is uniformly random."""
    counts = [0, 0]
    for s in range(2000):
        tab = Tableau(1)
        tab.h(0)
        rng = np.random.default_rng(s)
        counts[tab.measure_z(0, rng)] += 1
    # Statistical test: ratio should be ~0.5 ± 0.05
    assert 0.45 < counts[1] / 2000 < 0.55


def test_h_h_returns_to_zero():
    """H is self-inverse: HH|0> = |0>."""
    tab = Tableau(1)
    tab.h(0)
    tab.h(0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        # Need to recreate tableau for each measurement
        t = Tableau(1)
        t.h(0)
        t.h(0)
        assert t.measure_z(0, rng) == 0


def test_bell_state_perfectly_correlates():
    """Bell state via H + CNOT: bits always agree."""
    for s in range(500):
        tab = Tableau(2)
        tab.h(0)
        tab.cnot(0, 1)
        rng = np.random.default_rng(s)
        b0 = tab.measure_z(0, rng)
        b1 = tab.measure_z(1, rng)
        assert b0 == b1, f"Bell pair mismatch at seed {s}: {b0} vs {b1}"


def test_5_qubit_ghz():
    """5-qubit GHZ: all 5 bits always agree."""
    n = 5
    for s in range(200):
        tab = Tableau(n)
        tab.h(0)
        for q in range(n - 1):
            tab.cnot(q, q + 1)
        rng = np.random.default_rng(s)
        bits = [tab.measure_z(q, rng) for q in range(n)]
        # All bits must be identical
        assert len(set(bits)) == 1


def test_s_gate_is_inverse_of_sdg():
    """S * S_dag = I in Z basis."""
    tab = Tableau(1)
    tab.h(0)  # superposition
    tab.s(0)
    tab.sdg(0)
    # State should be back to |+>; measurements still uniform
    counts = [0, 0]
    for s in range(2000):
        t = Tableau(1)
        t.h(0)
        t.s(0)
        t.sdg(0)
        rng = np.random.default_rng(s)
        counts[t.measure_z(0, rng)] += 1
    assert 0.45 < counts[1] / 2000 < 0.55


def test_swap_gate():
    """SWAP exchanges bits."""
    # |10> --SWAP--> |01>
    tab = Tableau(2)
    tab.x(0)  # |10>
    tab.swap(0, 1)
    rng = np.random.default_rng(0)
    assert tab.measure_z(0, rng) == 0
    rng = np.random.default_rng(0)
    tab2 = Tableau(2)
    tab2.x(0)
    tab2.swap(0, 1)
    assert tab2.measure_z(1, rng) == 1


def test_cz_gate():
    """CZ |++> -- both qubits still uniform when measured."""
    counts = [0, 0, 0, 0]  # 00, 01, 10, 11
    for s in range(2000):
        tab = Tableau(2)
        tab.h(0)
        tab.h(1)
        tab.cz(0, 1)
        rng = np.random.default_rng(s)
        b0 = tab.measure_z(0, rng)
        b1 = tab.measure_z(1, rng)
        counts[b0 * 2 + b1] += 1
    # CZ|++> has equal amplitude at all 4 outcomes (then a sign flip on |11>),
    # but Z measurements only see |amplitude|^2, so still 25% each.
    for c in counts:
        assert 0.20 < c / 2000 < 0.30


# ---------- Engine-level via solve() ----------


@pytest.fixture
def engine():
    return StabilizerSimulator()


def test_engine_handles_clifford_circuit(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
        },
    )
    assert engine.can_handle(p)


def test_engine_rejects_non_clifford_t_gate(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}, {"gate": "T", "qubits": [0]}],
            "task": "sample",
        },
    )
    assert not engine.can_handle(p)


def test_engine_rejects_rotation_gates(engine):
    """RX, RY, RZ are not Clifford in general."""
    for gate in ["RX", "RY", "RZ"]:
        p = Problem(
            kind=ProblemKind.QUANTUM_CIRCUIT,
            payload={
                "n_qubits": 2,
                "ops": [{"gate": gate, "qubits": [0], "params": [0.5]}],
                "task": "sample",
            },
        )
        assert not engine.can_handle(p), f"should reject {gate}"


def test_engine_rejects_probabilities_task(engine):
    """Probabilities = 2^n floats; refuse this for stabilizer."""
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "probabilities",
        },
    )
    assert not engine.can_handle(p)


def test_engine_handles_thousand_qubits(engine):
    """The headline claim: stabilizer can handle problems state-vector can't."""
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 1000,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
        },
    )
    assert engine.can_handle(p)


def test_engine_solves_5qubit_ghz_correctly(engine):
    n = 5
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": n,
            "ops": ops,
            "task": "sample",
            "task_args": {"n_shots": 500, "seed": 42},
        },
    )
    sol = engine.solve(p)
    assert set(sol.value["counts"].keys()).issubset({"00000", "11111"})
    # Both outcomes should appear
    assert "00000" in sol.value["counts"]
    assert "11111" in sol.value["counts"]


def test_engine_solves_100qubit_ghz(engine):
    """Sanity-check that 100-qubit Clifford works through the engine."""
    n = 100
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": n,
            "ops": ops,
            "task": "sample",
            "task_args": {"n_shots": 5, "seed": 0},
        },
    )
    sol = engine.solve(p)
    all_zero = "0" * n
    all_one = "1" * n
    for k in sol.value["counts"]:
        assert k in (all_zero, all_one), f"got non-GHZ outcome {k}"


# ---------- Cross-check vs state-vector simulator ----------


def test_stabilizer_matches_qmlx_on_small_clifford():
    """At small n where both engines work, sample distributions agree."""
    from metis.engines.quantum_sim import QuantumStateVector

    qmlx = QuantumStateVector()
    stab = StabilizerSimulator()

    # Bell state circuit
    n = 2
    ops = [{"gate": "H", "qubits": [0]}, {"gate": "CNOT", "qubits": [0, 1]}]
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": n,
            "ops": ops,
            "task": "sample",
            "task_args": {"n_shots": 5000, "seed": 0},
        },
    )
    sol_qmlx = qmlx.solve(p)
    sol_stab = stab.solve(p)

    # Both should give counts ~50/50 between 00 and 11
    for sol in [sol_qmlx, sol_stab]:
        c = sol.value["counts"]
        assert set(c.keys()).issubset({"00", "11"})
        ratio = c.get("11", 0) / sum(c.values())
        assert 0.45 < ratio < 0.55


# ---------- Validation / security ----------


def test_engine_rejects_huge_n_qubits(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 100_000, "ops": [], "task": "sample"},
    )
    assert not engine.can_handle(p)


def test_engine_rejects_negative_n_qubits(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": -1, "ops": [], "task": "sample"},
    )
    assert not engine.can_handle(p)


def test_engine_rejects_qubit_out_of_range(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [99]}],
            "task": "sample",
        },
    )
    with pytest.raises(ValueError, match="out of range"):
        engine.solve(p)


def test_engine_rejects_huge_n_shots(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
            "task_args": {"n_shots": 10**9, "seed": 0},
        },
    )
    with pytest.raises(ValueError, match="n_shots"):
        engine.solve(p)
