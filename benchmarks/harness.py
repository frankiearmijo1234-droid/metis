"""Benchmark harness: measure engine performance reproducibly.

Design notes:
- Each benchmark runs N trials and reports median (robust to outliers from
  GC pauses or thermal throttling). Mean and stdev also kept for completeness.
- Engines that refuse a problem (can_handle=False or estimate_cost=inf)
  are recorded as "ineligible," not as a failure.
- Engines that crash are recorded with the exception type, not silently
  swallowed.
- A timeout protects against pathological cases. We use a thread-based
  watchdog rather than signals so it works in any environment.
- Results are JSON-serializable for downstream plotting/comparison.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from metis import Problem, Router

# Sentinel: returned by run_one when engine refuses or times out.
INELIGIBLE = "INELIGIBLE"
TIMEOUT = "TIMEOUT"
CRASHED = "CRASHED"


@dataclass
class BenchmarkResult:
    """Outcome of a single (engine, problem) measurement."""

    engine: str
    problem_id: str
    status: str  # "ok", INELIGIBLE, TIMEOUT, CRASHED
    times_sec: list[float] = field(default_factory=list)
    median_sec: float | None = None
    mean_sec: float | None = None
    stdev_sec: float | None = None
    objective: float | None = None  # solution quality (lower better for min)
    error_message: str | None = None
    # Free-form extras (e.g., is_optimal, n_qubits, etc.)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Benchmark:
    """A single problem to run against each engine.

    The `extract_objective` callback turns a Solution into a comparable
    scalar (lower-is-better convention). For QUBO this is fun; for quantum
    circuits it's typically a count check.
    """

    problem_id: str
    problem: Problem
    extract_objective: Callable[[Any], float] | None = None
    n_trials: int = 3
    timeout_sec: float = 60.0


def _time_one_solve(engine, problem, timeout_sec: float) -> tuple[str, float, Any]:
    """Run engine.solve(problem) once with a thread-based timeout.

    Returns (status, elapsed_sec, solution_or_error).
    """
    import threading

    box: dict[str, Any] = {}

    def target():
        try:
            t0 = time.perf_counter()
            sol = engine.solve(problem)
            box["elapsed"] = time.perf_counter() - t0
            box["solution"] = sol
        except Exception as e:
            box["error"] = e
            box["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        # Note: we can't actually kill a Python thread cleanly. The thread
        # keeps running but we move on. For benchmark purposes, the wall-
        # clock-bound case is what matters.
        return TIMEOUT, timeout_sec, None
    if "error" in box:
        return CRASHED, 0.0, (box["error"], box["traceback"])
    return "ok", box["elapsed"], box["solution"]


def run_one(engine, benchmark: Benchmark) -> BenchmarkResult:
    """Run a single engine against a single benchmark."""
    # Eligibility check (fast, no actual solve).
    try:
        eligible = engine.can_handle(benchmark.problem)
    except Exception as e:
        return BenchmarkResult(
            engine=engine.name,
            problem_id=benchmark.problem_id,
            status=CRASHED,
            error_message=f"can_handle raised: {type(e).__name__}: {e}",
        )
    if not eligible:
        return BenchmarkResult(
            engine=engine.name,
            problem_id=benchmark.problem_id,
            status=INELIGIBLE,
        )
    try:
        cost = float(engine.estimate_cost(benchmark.problem))
        if not math.isfinite(cost):
            return BenchmarkResult(
                engine=engine.name,
                problem_id=benchmark.problem_id,
                status=INELIGIBLE,
                error_message="estimate_cost returned inf (engine declined)",
            )
    except Exception as e:
        return BenchmarkResult(
            engine=engine.name,
            problem_id=benchmark.problem_id,
            status=CRASHED,
            error_message=f"estimate_cost raised: {type(e).__name__}: {e}",
        )

    # Measure. Discard the first run as a warmup if we have enough trials.
    times: list[float] = []
    last_solution = None
    for trial in range(benchmark.n_trials):
        status, elapsed, payload = _time_one_solve(
            engine, benchmark.problem, benchmark.timeout_sec
        )
        if status == TIMEOUT:
            return BenchmarkResult(
                engine=engine.name,
                problem_id=benchmark.problem_id,
                status=TIMEOUT,
                times_sec=times,
                error_message=f"exceeded {benchmark.timeout_sec}s timeout",
            )
        if status == CRASHED:
            err, tb = payload
            return BenchmarkResult(
                engine=engine.name,
                problem_id=benchmark.problem_id,
                status=CRASHED,
                error_message=f"{type(err).__name__}: {err}",
                extras={"traceback": tb},
            )
        times.append(elapsed)
        last_solution = payload

    # First-run warmup discard (if we ran more than 1 trial)
    timings = times[1:] if len(times) > 1 else times

    obj = None
    if benchmark.extract_objective is not None and last_solution is not None:
        try:
            obj = benchmark.extract_objective(last_solution)
        except Exception:
            obj = None

    extras: dict[str, Any] = {}
    if last_solution is not None and hasattr(last_solution, "value"):
        v = last_solution.value
        if isinstance(v, dict):
            for k in (
                "is_optimal",
                "status",
                "method",
                "n_shots",
                "n_constraints",
                "warning",
            ):
                if k in v:
                    extras[k] = v[k]

    return BenchmarkResult(
        engine=engine.name,
        problem_id=benchmark.problem_id,
        status="ok",
        times_sec=times,
        median_sec=statistics.median(timings),
        mean_sec=statistics.mean(timings),
        stdev_sec=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        objective=obj,
        extras=extras,
    )


def run_suite(
    router: Router,
    benchmarks: list[Benchmark],
    quiet: bool = False,
) -> list[BenchmarkResult]:
    """Run every benchmark against every eligible engine in the router.

    Returns one BenchmarkResult per (engine, benchmark) pair. Status field
    tells you whether it actually ran or was skipped.
    """
    results: list[BenchmarkResult] = []
    engines = router.engines()
    total = len(engines) * len(benchmarks)
    completed = 0

    for benchmark in benchmarks:
        for engine in engines:
            completed += 1
            if not quiet:
                print(
                    f"  [{completed}/{total}] {engine.name} on "
                    f"{benchmark.problem_id}...",
                    end="",
                    flush=True,
                )
            r = run_one(engine, benchmark)
            results.append(r)
            if not quiet:
                if r.status == "ok":
                    obj = f" obj={r.objective:.4g}" if r.objective is not None else ""
                    print(f" {r.median_sec*1000:.1f}ms{obj}")
                elif r.status == INELIGIBLE:
                    print(" -")
                else:
                    print(f" {r.status}")

    return results


def write_results_json(results: list[BenchmarkResult], path: str) -> None:
    """Save results to a JSON file."""
    payload = {
        "version": 1,
        "timestamp": time.time(),
        "results": [r.to_dict() for r in results],
    }
    # Drop tracebacks from the JSON to keep file size reasonable.
    for r in payload["results"]:
        if "traceback" in r.get("extras", {}):
            del r["extras"]["traceback"]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def render_markdown_summary(results: list[BenchmarkResult]) -> str:
    """Build a markdown table summarizing results.

    Rows are problems; columns are engines; cells are median time (or status).
    A second table shows objective quality where applicable.
    """
    # Collect engines and problems in encounter order
    engines: list[str] = []
    problems: list[str] = []
    seen_e: set[str] = set()
    seen_p: set[str] = set()
    for r in results:
        if r.engine not in seen_e:
            engines.append(r.engine)
            seen_e.add(r.engine)
        if r.problem_id not in seen_p:
            problems.append(r.problem_id)
            seen_p.add(r.problem_id)

    # Build a 2-D map: results_map[problem][engine] = result
    results_map: dict[str, dict[str, BenchmarkResult]] = {p: {} for p in problems}
    for r in results:
        results_map[r.problem_id][r.engine] = r

    out = []
    out.append("# metis benchmark results")
    out.append("")
    out.append(
        f"_{len(results)} measurements across {len(engines)} engines and {len(problems)} problems._"
    )
    out.append("")
    out.append("## Times (median, ms)")
    out.append("")
    header = "| Problem | " + " | ".join(engines) + " |"
    sep = "|---" * (len(engines) + 1) + "|"
    out.append(header)
    out.append(sep)
    for p in problems:
        row = [p]
        for e in engines:
            r = results_map[p].get(e)
            if r is None:
                row.append("-")
            elif r.status == INELIGIBLE:
                row.append("—")
            elif r.status == TIMEOUT:
                row.append("⏱")
            elif r.status == CRASHED:
                row.append("✗")
            else:
                t_ms = r.median_sec * 1000
                if t_ms < 1:
                    row.append(f"{t_ms:.2f}")
                elif t_ms < 1000:
                    row.append(f"{t_ms:.0f}")
                else:
                    row.append(f"{t_ms/1000:.1f}s")
        out.append("| " + " | ".join(row) + " |")

    # Quality table only if at least one result has an objective
    has_obj = any(r.objective is not None for r in results)
    if has_obj:
        out.append("")
        out.append("## Objective values (lower = better, '✓' = proved optimal)")
        out.append("")
        out.append(header)
        out.append(sep)
        for p in problems:
            row = [p]
            for e in engines:
                r = results_map[p].get(e)
                if r is None or r.objective is None:
                    row.append("-")
                else:
                    is_opt = r.extras.get("is_optimal")
                    suffix = " ✓" if is_opt else ""
                    row.append(f"{r.objective:.4g}{suffix}")
            out.append("| " + " | ".join(row) + " |")

    out.append("")
    out.append(
        "Legend: `—` ineligible · `⏱` timeout · `✗` crashed · `✓` optimum proven"
    )
    return "\n".join(out)
