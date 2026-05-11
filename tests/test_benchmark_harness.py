"""Tests for the benchmark harness.

The harness is itself code that needs to work. We test:
1. Successful runs report sensible timings.
2. Ineligible engines are skipped, not failed.
3. Engines that crash are reported with the error.
4. Timeouts are honored (smoke test).
5. JSON serialization round-trips.
"""

import json
import os
import tempfile
import time

from benchmarks.harness import (
    CRASHED,
    INELIGIBLE,
    TIMEOUT,
    Benchmark,
    render_markdown_summary,
    run_one,
    run_suite,
    write_results_json,
)
from metis import Problem, ProblemKind, Router, Solution


class _FastEngine:
    name = "fast"

    def can_handle(self, p):
        return True

    def estimate_cost(self, p):
        return 0.001

    def solve(self, p):
        return Solution(value={"fun": 1.0}, engine_name=self.name, elapsed_sec=0.0)


class _RefuseEngine:
    name = "refuse"

    def can_handle(self, p):
        return False

    def estimate_cost(self, p):
        return float("inf")

    def solve(self, p):
        raise AssertionError("should not be called")


class _CrashEngine:
    name = "crasher"

    def can_handle(self, p):
        return True

    def estimate_cost(self, p):
        return 0.001

    def solve(self, p):
        raise RuntimeError("simulated crash")


class _SlowEngine:
    name = "slow"

    def can_handle(self, p):
        return True

    def estimate_cost(self, p):
        return 0.001

    def solve(self, p):
        time.sleep(2.0)
        return Solution(value={"fun": 1.0}, engine_name=self.name, elapsed_sec=2.0)


def _trivial_problem():
    return Problem(kind=ProblemKind.OPTIMIZATION, payload={})


def _trivial_benchmark(timeout=5.0, n_trials=2):
    return Benchmark(
        problem_id="trivial",
        problem=_trivial_problem(),
        n_trials=n_trials,
        timeout_sec=timeout,
    )


def test_run_one_records_timing():
    r = run_one(_FastEngine(), _trivial_benchmark())
    assert r.status == "ok"
    assert r.median_sec >= 0
    assert r.engine == "fast"
    assert len(r.times_sec) == 2


def test_ineligible_engine_records_skip():
    r = run_one(_RefuseEngine(), _trivial_benchmark())
    assert r.status == INELIGIBLE
    assert r.median_sec is None
    assert r.times_sec == []


def test_crashing_engine_records_error():
    r = run_one(_CrashEngine(), _trivial_benchmark())
    assert r.status == CRASHED
    assert "simulated crash" in r.error_message
    assert "RuntimeError" in r.error_message


def test_timeout_honored():
    """SlowEngine sleeps 2s; benchmark times out at 0.5s."""
    r = run_one(_SlowEngine(), _trivial_benchmark(timeout=0.5))
    assert r.status == TIMEOUT


def test_objective_extraction():
    bench = Benchmark(
        problem_id="obj_test",
        problem=_trivial_problem(),
        extract_objective=lambda sol: sol.value["fun"],
        n_trials=2,
    )
    r = run_one(_FastEngine(), bench)
    assert r.status == "ok"
    assert r.objective == 1.0


def test_run_suite_with_mixed_engines():
    router = (
        Router()
        .register(_FastEngine())
        .register(_RefuseEngine())
        .register(_CrashEngine())
    )
    results = run_suite(router, [_trivial_benchmark()], quiet=True)
    assert len(results) == 3
    by_engine = {r.engine: r for r in results}
    assert by_engine["fast"].status == "ok"
    assert by_engine["refuse"].status == INELIGIBLE
    assert by_engine["crasher"].status == CRASHED


def test_json_roundtrip():
    router = Router().register(_FastEngine())
    results = run_suite(router, [_trivial_benchmark()], quiet=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        write_results_json(results, path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["engine"] == "fast"
        assert data["results"][0]["status"] == "ok"
    finally:
        os.unlink(path)


def test_markdown_summary_contains_engines_and_problems():
    router = Router().register(_FastEngine()).register(_RefuseEngine())
    results = run_suite(router, [_trivial_benchmark()], quiet=True)
    md = render_markdown_summary(results)
    assert "fast" in md
    assert "refuse" in md
    assert "trivial" in md
    # Markdown table sanity
    assert "|" in md
