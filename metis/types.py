"""Core types for metis: Problem, Solution, Engine.

The architecture is:
1. A user (or Claude) describes a Problem.
2. The Router picks an Engine that can handle it.
3. The Engine returns a Solution.

Engines are interchangeable as long as they implement the Engine protocol.
This is the abstraction that lets us add new backends (stabilizer simulator,
tensor network, neural solvers, etc.) without changing the router or the
caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ---------- Problem types ----------


class ProblemKind(str, Enum):
    """High-level problem categories the router dispatches on."""

    OPTIMIZATION = "optimization"  # find argmin/argmax of an objective
    SAMPLING = "sampling"  # draw samples from a distribution
    QUANTUM_CIRCUIT = "quantum_circuit"  # run a quantum circuit
    SEARCH = "search"  # find x such that f(x) is True
    SIMULATION = "simulation"  # simulate a physical system


@dataclass
class Problem:
    """A unit of work to send to an engine.

    The `kind` determines which engines are eligible. The `payload` carries
    problem-specific data. The `hints` carry router hints (size, structure,
    desired precision, time budget) that influence which engine is picked.
    """

    kind: ProblemKind
    payload: dict[str, Any]
    hints: dict[str, Any] = field(default_factory=dict)

    # Common hints (not exhaustive; engines may inspect their own):
    # - size: int        problem size (n_vars, n_qubits, n_dimensions)
    # - time_budget_s: float
    # - precision: "exact" | "approximate"
    # - structure: "dense" | "sparse" | "clifford" | "low_entanglement"
    # - n_constraints: int


@dataclass
class Solution:
    """What an engine returns.

    `value` is the answer (typed by problem kind). `metadata` is everything
    the caller might want to know about how it was computed.
    """

    value: Any
    engine_name: str
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------- Engine protocol ----------


@runtime_checkable
class Engine(Protocol):
    """The interface every engine must implement.

    Engines are stateless objects. The router holds a registry of engines
    and asks each one whether it can handle a given problem.
    """

    name: str

    def can_handle(self, problem: Problem) -> bool:
        """Return True if this engine can run this problem.

        Should be fast (no actual computation). The router calls this on
        every registered engine to build the candidate set.
        """
        ...

    def estimate_cost(self, problem: Problem) -> float:
        """Estimate the wall-clock cost in seconds.

        Used by the router to pick among multiple eligible engines.
        Lower is better. Inf means "I refuse this" (equivalent to
        can_handle returning False, but more granular for routing).
        Should not actually solve the problem -- just inspect size/structure.
        """
        ...

    def solve(self, problem: Problem) -> Solution:
        """Actually solve the problem and return a Solution."""
        ...


# ---------- Router decision record ----------


@dataclass
class RoutingDecision:
    """Audit trail for why the router picked a given engine.

    Stored on Solution.metadata so users can see why their problem went to
    one engine vs another. Critical for debugging and for trust.
    """

    chosen: str
    candidates: list[tuple[str, float]]  # (engine_name, estimated_cost)
    rejected: list[tuple[str, str]]  # (engine_name, reason)
    reason: str
