"""metis: Local compute orchestrator for Apple Silicon.

Quick start:
    from metis import default_router, Problem, ProblemKind

    router = default_router()
    problem = Problem(
        kind=ProblemKind.OPTIMIZATION,
        payload={"qubo_Q": Q_matrix, "qubo_solve": True},
        hints={"size": 50},
    )
    solution = router.solve(problem)
    print(solution.value)
    print(solution.metadata["routing_decision"].reason)
"""

from .engines.classical import ClassicalOptimizer
from .engines.mcmc import MCMCEngine
from .engines.mps import MPSSimulator
from .engines.parallel_tempering import ParallelTempering
from .engines.qaoa import QAOA
from .engines.quantum_sim import QuantumStateVector
from .engines.simulated_annealing import SimulatedAnnealing
from .engines.simulated_annealing_mlx import (
    SimulatedAnnealingMLX,
    is_mlx_available,
)
from .engines.stabilizer import StabilizerSimulator
from .router import NoEngineAvailableError, Router
from .types import Engine, Problem, ProblemKind, RoutingDecision, Solution

# Optional engines (only available when their backing libraries are installed)
try:
    from .engines.ortools_engine import ORTools

    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False
    ORTools = None  # type: ignore

__version__ = "0.6.0"


def default_router() -> Router:
    """Return a Router with the default set of engines registered."""
    router = (
        Router()
        .register(ClassicalOptimizer())
        .register(SimulatedAnnealing())
        .register(SimulatedAnnealingMLX())
        .register(ParallelTempering())
        .register(QAOA())  # opt-in via prefer_qaoa
        .register(MPSSimulator())  # opt-in via prefer_mps
        .register(StabilizerSimulator())
        .register(QuantumStateVector())
        .register(MCMCEngine())  # only engine for SAMPLING
    )
    if _ORTOOLS_AVAILABLE:
        router.register(ORTools())
    return router


__all__ = [
    "Problem",
    "ProblemKind",
    "Solution",
    "Engine",
    "RoutingDecision",
    "Router",
    "NoEngineAvailableError",
    "ClassicalOptimizer",
    "SimulatedAnnealing",
    "SimulatedAnnealingMLX",
    "ParallelTempering",
    "QAOA",
    "MCMCEngine",
    "QuantumStateVector",
    "StabilizerSimulator",
    "MPSSimulator",
    "ORTools",
    "is_mlx_available",
    "default_router",
    "__version__",
]
