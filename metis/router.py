"""The Router: matches Problems to Engines.

Routing strategy:
1. Filter engines by `can_handle()` -- the eligible set.
2. Ask each eligible engine for a cost estimate.
3. Pick the lowest cost. Break ties by preference order in the registry.
4. Run it. Record the decision in the Solution metadata.

Future extensions (intentionally not built yet):
- Learned routing from past performance
- Multi-engine racing (run two engines in parallel, take the first answer)
- Fallback chain (if engine A fails, retry on B)

These can all be added without changing the engine protocol.
"""

from __future__ import annotations

import math
import time

from .types import Engine, Problem, RoutingDecision, Solution


class NoEngineAvailableError(RuntimeError):
    """Raised when no registered engine can handle the problem."""


class Router:
    def __init__(self):
        self._engines: list[Engine] = []

    def register(self, engine: Engine) -> Router:
        """Add an engine. Order matters only for tie-breaking."""
        if not hasattr(engine, "name"):
            raise TypeError("engine must have a `name` attribute")
        # Reject duplicate names so logs stay sensible.
        if any(e.name == engine.name for e in self._engines):
            raise ValueError(f"engine name '{engine.name}' already registered")
        self._engines.append(engine)
        return self

    def engines(self) -> list[Engine]:
        return list(self._engines)

    def route(self, problem: Problem) -> tuple[Engine, RoutingDecision]:
        """Pick an engine for `problem`. Does not solve it."""
        eligible: list[tuple[Engine, float]] = []
        rejected: list[tuple[str, str]] = []

        for engine in self._engines:
            try:
                ok = engine.can_handle(problem)
            except Exception as e:
                rejected.append((engine.name, f"can_handle raised: {e}"))
                continue
            if not ok:
                rejected.append((engine.name, "can_handle=False"))
                continue
            try:
                cost = float(engine.estimate_cost(problem))
            except Exception as e:
                rejected.append((engine.name, f"estimate_cost raised: {e}"))
                continue
            if not math.isfinite(cost):
                rejected.append((engine.name, "cost=inf (declined)"))
                continue
            eligible.append((engine, cost))

        if not eligible:
            raise NoEngineAvailableError(
                f"No engine accepted problem of kind={problem.kind.value}. "
                f"Rejected: {rejected}"
            )

        # Sort by cost ascending; on tie, preserve registration order
        eligible.sort(key=lambda pair: pair[1])
        chosen, chosen_cost = eligible[0]

        decision = RoutingDecision(
            chosen=chosen.name,
            candidates=[(e.name, c) for e, c in eligible],
            rejected=rejected,
            reason=(
                f"{chosen.name} had lowest estimated cost ({chosen_cost:.4g}s) "
                f"among {len(eligible)} eligible engines"
            ),
        )
        return chosen, decision

    def solve(self, problem: Problem, *, fallback: bool = False) -> Solution:
        """Pick an engine and run it. The common-case entry point.

        If `fallback=True` and the chosen engine raises during solve(), the
        router will try the next-cheapest engine, and so on. Default is
        False because in most cases the user wants the crash to surface
        rather than silently get a worse engine's answer.
        """
        engine, decision = self.route(problem)

        if not fallback:
            t0 = time.perf_counter()
            solution = engine.solve(problem)
            elapsed = time.perf_counter() - t0
            solution.metadata.setdefault(
                "engine_internal_elapsed_sec", solution.elapsed_sec
            )
            solution.elapsed_sec = elapsed
            solution.metadata["routing_decision"] = decision
            return solution

        # Fallback path: try engines in cost order, swallow exceptions
        # until one succeeds.
        eligible_pairs = list(decision.candidates)  # already sorted by cost
        engines_by_name = {e.name: e for e in self._engines}
        last_error: Exception | None = None
        attempts: list[tuple[str, str]] = []

        for name, _cost in eligible_pairs:
            eng = engines_by_name[name]
            t0 = time.perf_counter()
            try:
                solution = eng.solve(problem)
            except Exception as e:
                last_error = e
                attempts.append((name, f"raised {type(e).__name__}: {e}"))
                continue
            elapsed = time.perf_counter() - t0
            solution.metadata.setdefault(
                "engine_internal_elapsed_sec", solution.elapsed_sec
            )
            solution.elapsed_sec = elapsed
            solution.metadata["routing_decision"] = decision
            solution.metadata["fallback_attempts"] = attempts
            return solution

        # All engines failed
        raise RuntimeError(
            f"All eligible engines failed. Attempts: {attempts}. "
            f"Last error: {last_error}"
        )
