"""Tests for the router: dispatch logic, error handling, decisions."""

import pytest

from metis import (
    NoEngineAvailableError,
    Problem,
    ProblemKind,
    Router,
    Solution,
)

# ---------- Stub engines for unit tests ----------


class _AcceptAll:
    name = "accept_all"

    def __init__(self, cost=1.0, label="ok"):
        self._cost = cost
        self._label = label

    def can_handle(self, problem):
        return True

    def estimate_cost(self, problem):
        return self._cost

    def solve(self, problem):
        return Solution(
            value={"label": self._label}, engine_name=self.name, elapsed_sec=0.0
        )


class _RejectAll:
    name = "reject_all"

    def can_handle(self, problem):
        return False

    def estimate_cost(self, problem):
        return float("inf")

    def solve(self, problem):
        raise AssertionError("should never be called")


class _Crashy:
    name = "crashy"

    def can_handle(self, problem):
        raise RuntimeError("boom")

    def estimate_cost(self, problem):
        raise AssertionError("not reached")

    def solve(self, problem):
        raise AssertionError("not reached")


class _RefusesByInfCost:
    name = "infinite_cost"

    def can_handle(self, problem):
        return True  # passes filter...

    def estimate_cost(self, problem):
        return float("inf")  # ...but declines via cost

    def solve(self, problem):
        raise AssertionError("should not be called")


def _trivial_problem():
    return Problem(kind=ProblemKind.OPTIMIZATION, payload={})


# ---------- Tests ----------


def test_register_returns_self_for_chaining():
    r = Router()
    assert r.register(_AcceptAll()) is r


def test_duplicate_engine_name_rejected():
    r = Router().register(_AcceptAll(label="a"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(_AcceptAll(label="b"))


def test_no_eligible_engine_raises():
    r = Router().register(_RejectAll())
    with pytest.raises(NoEngineAvailableError):
        r.solve(_trivial_problem())


def test_crashing_engine_excluded_others_proceed():
    r = Router().register(_Crashy()).register(_AcceptAll())
    sol = r.solve(_trivial_problem())
    assert sol.engine_name == "accept_all"
    decision = sol.metadata["routing_decision"]
    assert any("crashy" in name for name, _ in decision.rejected)


def test_lowest_cost_wins():
    expensive = _AcceptAll(cost=10.0, label="expensive")
    cheap = _AcceptAll(cost=0.1, label="cheap")
    cheap.name = "cheap_one"
    expensive.name = "expensive_one"
    r = Router().register(expensive).register(cheap)
    sol = r.solve(_trivial_problem())
    assert sol.engine_name == "cheap_one"
    assert sol.value["label"] == "cheap"


def test_tie_breaks_by_registration_order():
    a = _AcceptAll(cost=1.0)
    a.name = "first"
    b = _AcceptAll(cost=1.0)
    b.name = "second"
    r = Router().register(a).register(b)
    sol = r.solve(_trivial_problem())
    assert sol.engine_name == "first"


def test_inf_cost_treated_as_decline():
    decliner = _RefusesByInfCost()
    accepter = _AcceptAll(cost=1.0)
    r = Router().register(decliner).register(accepter)
    sol = r.solve(_trivial_problem())
    assert sol.engine_name == "accept_all"
    decision = sol.metadata["routing_decision"]
    rejected_names = [name for name, _ in decision.rejected]
    assert "infinite_cost" in rejected_names


def test_routing_decision_lists_candidates():
    a = _AcceptAll(cost=1.0)
    a.name = "a"
    b = _AcceptAll(cost=2.0)
    b.name = "b"
    r = Router().register(a).register(b)
    sol = r.solve(_trivial_problem())
    decision = sol.metadata["routing_decision"]
    candidate_names = [name for name, _ in decision.candidates]
    assert "a" in candidate_names and "b" in candidate_names
    assert decision.chosen == "a"


def test_solution_records_outer_elapsed_time():
    r = Router().register(_AcceptAll())
    sol = r.solve(_trivial_problem())
    assert sol.elapsed_sec >= 0.0


def test_route_does_not_solve():
    """route() should pick an engine without running it."""

    class Counter:
        name = "counter"
        solve_called = 0

        def can_handle(self, p):
            return True

        def estimate_cost(self, p):
            return 1.0

        def solve(self, p):
            Counter.solve_called += 1
            return Solution(value=None, engine_name=self.name, elapsed_sec=0.0)

    r = Router().register(Counter())
    engine, decision = r.route(_trivial_problem())
    assert engine.name == "counter"
    assert Counter.solve_called == 0
    # Now actually solve and verify it runs:
    r.solve(_trivial_problem())
    assert Counter.solve_called == 1
