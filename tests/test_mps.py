"""Tests for the MPS tensor-network simulator."""

import numpy as np
import pytest

from metis import (
    MPSSimulator,
    Problem,
    ProblemKind,
    QuantumStateVector,
    default_router,
)
from metis.engines.mps import GATE_MATRICES, MPS, _twoq_unitary

# ---------- Tableau-equivalent (Tableau-level) tests ----------


def test_initial_state_is_zero():
    mps = MPS(3, bond_dim=4)
    sv = mps.to_state_vector()
    expected = np.zeros(8)
    expected[0] = 1
    np.testing.assert_allclose(sv, expected)


def test_x_gate_flips_qubit():
    mps = MPS(3, bond_dim=4)
    mps.apply_single(GATE_MATRICES["X"], 1)
    sv = mps.to_state_vector()
    # X on qubit 1 -> |010>; in MSB-first reading qubit 0 is leftmost
    # Index 2 = '010' has qubit 1 set
    expected = np.zeros(8)
    expected[2] = 1
    np.testing.assert_allclose(sv, expected)


def test_h_creates_plus():
    mps = MPS(1, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    sv = mps.to_state_vector()
    expected = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected)


def test_bell_state_via_adjacent_cnot():
    mps = MPS(2, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 0)
    sv = mps.to_state_vector()
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected)


def test_ghz_3qubit():
    mps = MPS(3, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 1)
    sv = mps.to_state_vector()
    expected = np.zeros(8)
    expected[0] = expected[7] = 1 / np.sqrt(2)
    np.testing.assert_allclose(sv, expected)


def test_non_adjacent_cnot_via_swap_network():
    """CNOT between qubits 0 and 2 with qubit 1 in between."""
    mps = MPS(3, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    mps.apply_two_qubit("CNOT", 0, 2)
    sv = mps.to_state_vector()
    # Expected: (|000> + |101>)/sqrt(2) -- index 0 and index 5 (binary 101)
    expected = np.zeros(8)
    expected[0] = expected[5] = 1 / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-10)


def test_probabilities_match_state_vector_squared():
    mps = MPS(3, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 1)
    probs = mps.probabilities()
    expected = np.zeros(8)
    expected[0] = expected[7] = 0.5
    np.testing.assert_allclose(probs, expected)


# ---------- Sampling ----------


def test_small_n_sampling_bell_state():
    mps = MPS(2, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    mps.apply_two_adjacent(_twoq_unitary("CNOT"), 0)
    samples = mps.sample(2000, np.random.default_rng(0))
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    # Only 00 and 11 should appear
    assert set(counts.keys()) == {"00", "11"}
    # Roughly 50/50
    ratio = counts["11"] / 2000
    assert 0.45 < ratio < 0.55


def test_sequential_sampling_at_n21_ghz():
    """Sequential sampling triggers when n > 20. Test on 21-qubit GHZ."""
    n = 21
    mps = MPS(n, bond_dim=4)
    mps.apply_single(GATE_MATRICES["H"], 0)
    for q in range(n - 1):
        mps.apply_two_adjacent(_twoq_unitary("CNOT"), q)
    samples = mps.sample(50, np.random.default_rng(0))
    all_zero = "0" * n
    all_one = "1" * n
    # Every sample must be all-0 or all-1
    for s in samples:
        assert s in (all_zero, all_one)
    # Both outcomes should appear (with 50 shots)
    assert all_zero in samples
    assert all_one in samples


# ---------- Engine wrapper ----------


@pytest.fixture
def engine():
    return MPSSimulator()


def test_engine_requires_opt_in(engine):
    """MPS shouldn't be picked unless explicitly requested."""
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 5,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
        },
    )
    assert not engine.can_handle(p)


def test_engine_handles_with_prefer_mps(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 5,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
        },
        hints={"prefer_mps": True},
    )
    assert engine.can_handle(p)


def test_engine_rejects_oversized_n(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 1000, "ops": [], "task": "sample"},
        hints={"prefer_mps": True},
    )
    assert not engine.can_handle(p)


def test_engine_rejects_probabilities_at_large_n(engine):
    """probabilities returns 2^n floats; should refuse at large n."""
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": 50, "ops": [], "task": "probabilities"},
        hints={"prefer_mps": True},
    )
    assert not engine.can_handle(p)


def test_engine_solves_50qubit_ghz_quickly(engine):
    """The headline use case: low-entanglement state at n far beyond
    state-vector capacity."""
    n = 50
    ops = [{"gate": "H", "qubits": [0]}]
    for q in range(n - 1):
        ops.append({"gate": "CNOT", "qubits": [q, q + 1]})
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": n,
            "ops": ops,
            "task": "sample",
            "task_args": {"n_shots": 10, "seed": 0},
        },
        hints={"prefer_mps": True, "bond_dim": 8},
    )
    sol = engine.solve(p)
    all_zero = "0" * n
    all_one = "1" * n
    for k in sol.value["counts"]:
        assert k in (all_zero, all_one)
    # GHZ has bond dim exactly 2
    assert sol.value["bond_dim_used"] == 2


def test_mps_matches_state_vector_at_small_n_with_rotations():
    """For circuits that fit in state vector, MPS at full bond dim should
    produce the same probabilities (up to numerical precision)."""
    qmlx = QuantumStateVector()
    mps_eng = MPSSimulator()
    n = 6
    rng = np.random.default_rng(42)
    ops = []
    for layer in range(3):
        for q in range(n):
            gate = rng.choice(["RX", "RY", "RZ"])
            theta = float(rng.uniform(0, 2 * np.pi))
            ops.append({"gate": gate, "qubits": [q], "params": [theta]})
        for q in range(0, n - 1, 2):
            ops.append({"gate": "CNOT", "qubits": [q, q + 1]})

    p_qmlx = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": n, "ops": ops, "task": "probabilities"},
    )
    p_mps = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={"n_qubits": n, "ops": ops, "task": "probabilities"},
        hints={"prefer_mps": True, "bond_dim": 64},
    )
    probs_qmlx = np.array(qmlx.solve(p_qmlx).value["probabilities"])
    probs_mps = np.array(mps_eng.solve(p_mps).value["probabilities"])
    np.testing.assert_allclose(probs_qmlx, probs_mps, atol=1e-6)


def test_router_does_not_pick_mps_by_default():
    router = default_router()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 5,
            "ops": [{"gate": "H", "qubits": [0]}],
            "task": "sample",
        },
    )
    sol = router.solve(p)
    assert sol.engine_name != "mps"


# ---------- Validation ----------


def test_engine_rejects_non_clifford_t_gate_no_wait_t_works():
    """MPS DOES support T (it's a general 2x2 unitary). Just verifying it
    actually accepts T gates (unlike stabilizer)."""
    eng = MPSSimulator()
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "H", "qubits": [0]}, {"gate": "T", "qubits": [0]}],
            "task": "sample",
        },
        hints={"prefer_mps": True},
    )
    assert eng.can_handle(p)


def test_engine_rejects_unknown_gate(engine):
    p = Problem(
        kind=ProblemKind.QUANTUM_CIRCUIT,
        payload={
            "n_qubits": 3,
            "ops": [{"gate": "FAKEGATE", "qubits": [0]}],
            "task": "sample",
        },
        hints={"prefer_mps": True},
    )
    assert not engine.can_handle(p)


def test_mps_constructor_rejects_bad_bond_dim():
    with pytest.raises(ValueError, match="bond_dim"):
        MPS(5, bond_dim=10000)
    with pytest.raises(ValueError, match="bond_dim"):
        MPS(5, bond_dim=0)


def test_mps_constructor_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be"):
        MPS(0, bond_dim=4)
    with pytest.raises(ValueError, match="n must be"):
        MPS(10000, bond_dim=4)
