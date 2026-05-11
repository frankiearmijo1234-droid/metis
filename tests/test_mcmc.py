"""Tests for the MCMC engine.

Tests cover:
1. can_handle / estimate_cost on both payload shapes (continuous and QUBO).
2. Correctness: known distributions reproduce known statistics.
3. Routing: SAMPLING problems go to MCMC.
4. Validation: hostile inputs rejected.
"""

import numpy as np
import pytest

from metis import (
    MCMCEngine,
    Problem,
    ProblemKind,
    default_router,
)


@pytest.fixture
def engine():
    return MCMCEngine()


# ---------- can_handle ----------


def test_handles_continuous_mh(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={
            "log_density": lambda x: -0.5 * float(x[0] ** 2),
            "x0": np.array([0.0]),
            "n_samples": 100,
        },
    )
    assert engine.can_handle(p)


def test_handles_qubo_gibbs(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={
            "qubo_Q": np.eye(5),
            "qubo_sample": True,
            "T": 1.0,
            "n_samples": 100,
        },
    )
    assert engine.can_handle(p)


def test_rejects_optimization(engine):
    """MCMC handles SAMPLING, not OPTIMIZATION."""
    p = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": np.eye(5), "qubo_solve": True},
    )
    assert not engine.can_handle(p)


def test_rejects_quantum_circuit(engine):
    p = Problem(kind=ProblemKind.QUANTUM_CIRCUIT, payload={})
    assert not engine.can_handle(p)


def test_rejects_qubo_without_T(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(5), "qubo_sample": True, "n_samples": 100},
    )
    assert not engine.can_handle(p)


# ---------- correctness: continuous MH ----------


def test_mh_recovers_unit_normal_mean_and_std(engine):
    """Standard normal: mean ≈ 0, std ≈ 1."""

    def log_pdf(x):
        return -0.5 * float(x[0] ** 2)

    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"log_density": log_pdf, "x0": np.array([0.0]), "n_samples": 5000},
        hints={"n_chains": 4, "burn_in": 1000, "proposal_scale": 1.0, "seed": 0},
    )
    sol = engine.solve(p)
    samples = sol.value["samples"].reshape(-1)
    assert abs(samples.mean()) < 0.1
    assert 0.85 < samples.std() < 1.15


def test_mh_recovers_2d_covariance(engine):
    """2D Gaussian with known covariance."""
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    inv_cov = np.linalg.inv(cov)

    def log_pdf(x):
        return -0.5 * float(x @ inv_cov @ x)

    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"log_density": log_pdf, "x0": np.array([0.0, 0.0]), "n_samples": 5000},
        hints={"n_chains": 4, "burn_in": 2000, "proposal_scale": 1.0, "seed": 0},
    )
    sol = engine.solve(p)
    samples = sol.value["samples"].reshape(-1, 2)
    emp_cov = np.cov(samples.T)
    np.testing.assert_allclose(emp_cov, cov, atol=0.15)


def test_mh_records_acceptance_rate(engine):
    def log_pdf(x):
        return -0.5 * float(x[0] ** 2)

    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"log_density": log_pdf, "x0": np.array([0.0]), "n_samples": 500},
        hints={"n_chains": 2, "burn_in": 100, "proposal_scale": 1.0, "seed": 0},
    )
    sol = engine.solve(p)
    rate = sol.value["acceptance_rate"]
    # With proposal_scale=1 on N(0,1), acceptance rate should be ~0.5-0.7
    assert 0.3 < rate < 0.9


def test_mh_handles_log_density_returning_inf():
    """If user's log_density returns -inf or raises, that proposal is rejected."""
    eng = MCMCEngine()
    counter = {"crashes": 0}

    def hostile_log_pdf(x):
        # 50% chance of returning bad value
        if x[0] > 0.5:
            counter["crashes"] += 1
            raise RuntimeError("simulated user bug")
        return -0.5 * float(x[0] ** 2)

    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={
            "log_density": hostile_log_pdf,
            "x0": np.array([-1.0]),
            "n_samples": 200,
        },
        hints={"n_chains": 1, "burn_in": 100, "proposal_scale": 0.5, "seed": 0},
    )
    sol = eng.solve(p)
    # Engine should not have crashed; samples should stay below 0.5
    samples = sol.value["samples"].reshape(-1)
    assert samples.max() < 0.5  # never accepted a bad proposal
    assert counter["crashes"] > 0  # confirm hostile path was hit


# ---------- correctness: Gibbs on QUBO ----------


def test_gibbs_low_T_concentrates_at_ground_state(engine):
    """At T → 0, samples should concentrate at the ground state."""
    n = 6
    Q = -np.eye(n)  # ground = all-ones with E = -n
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": Q, "qubo_sample": True, "T": 0.01, "n_samples": 200},
        hints={"n_chains": 4, "burn_in": 1000, "seed": 0},
    )
    sol = engine.solve(p)
    energies = sol.value["energies"]
    # All samples should be at ground state at very low T
    assert energies.min() == pytest.approx(-n)
    # At least 95% of samples should be exactly at ground
    fraction = float((energies == -n).sum() / energies.size)
    assert fraction > 0.95


def test_gibbs_high_T_explores_widely(engine):
    """At very high T, samples should be approximately uniform."""
    n = 4
    Q = -np.eye(n)
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": Q, "qubo_sample": True, "T": 1000.0, "n_samples": 5000},
        hints={"n_chains": 4, "burn_in": 1000, "seed": 0},
    )
    sol = engine.solve(p)
    samples = sol.value["samples"].reshape(-1, n)
    # Each bit should be ~50/50
    bit_means = samples.mean(axis=0)
    np.testing.assert_allclose(bit_means, 0.5, atol=0.05)


def test_gibbs_metadata_recorded(engine):
    n = 4
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(n), "qubo_sample": True, "T": 1.0, "n_samples": 100},
        hints={"n_chains": 2, "burn_in": 50, "seed": 0},
    )
    sol = engine.solve(p)
    assert sol.value["method"] == "gibbs_qubo"
    assert sol.value["T"] == 1.0
    assert sol.value["n_chains"] == 2
    assert "mean_energy" in sol.value
    assert "min_energy_seen" in sol.value


# ---------- routing ----------


def test_router_picks_mcmc_for_sampling():
    router = default_router()
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(4), "qubo_sample": True, "T": 1.0, "n_samples": 50},
        hints={"n_chains": 2, "burn_in": 20, "seed": 0},
    )
    sol = router.solve(p)
    assert sol.engine_name == "mcmc"


# ---------- validation ----------


def test_rejects_nan_qubo(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={
            "qubo_Q": np.full((4, 4), np.nan),
            "qubo_sample": True,
            "T": 1.0,
            "n_samples": 10,
        },
    )
    with pytest.raises(ValueError, match="non-finite"):
        engine.solve(p)


def test_rejects_zero_temperature(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(4), "qubo_sample": True, "T": 0.0, "n_samples": 10},
    )
    with pytest.raises(ValueError, match="positive"):
        engine.solve(p)


def test_rejects_negative_temperature(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(4), "qubo_sample": True, "T": -1.0, "n_samples": 10},
    )
    with pytest.raises(ValueError, match="positive"):
        engine.solve(p)


def test_rejects_huge_n_samples(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={
            "qubo_Q": np.eye(4),
            "qubo_sample": True,
            "T": 1.0,
            "n_samples": 10**9,
        },
    )
    with pytest.raises(ValueError, match="n_samples"):
        engine.solve(p)


def test_rejects_huge_n_chains(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(4), "qubo_sample": True, "T": 1.0, "n_samples": 100},
        hints={"n_chains": 1000, "burn_in": 10, "seed": 0},
    )
    with pytest.raises(ValueError, match="n_chains"):
        engine.solve(p)


def test_rejects_huge_burn_in(engine):
    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"qubo_Q": np.eye(4), "qubo_sample": True, "T": 1.0, "n_samples": 10},
        hints={"n_chains": 2, "burn_in": 10**9, "seed": 0},
    )
    with pytest.raises(ValueError, match="burn_in"):
        engine.solve(p)


def test_rejects_negative_proposal_scale(engine):
    def log_pdf(x):
        return -0.5 * float(x[0] ** 2)

    p = Problem(
        kind=ProblemKind.SAMPLING,
        payload={"log_density": log_pdf, "x0": np.array([0.0]), "n_samples": 100},
        hints={"n_chains": 2, "burn_in": 10, "proposal_scale": -1.0, "seed": 0},
    )
    with pytest.raises(ValueError, match="proposal_scale"):
        engine.solve(p)
