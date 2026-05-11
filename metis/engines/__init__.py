"""Engine implementations.

Each module exposes a single class implementing the Engine protocol
defined in metis.types.
"""

from .classical import ClassicalOptimizer
from .mcmc import MCMCEngine
from .mps import MPSSimulator
from .parallel_tempering import ParallelTempering
from .qaoa import QAOA
from .quantum_sim import QuantumStateVector
from .simulated_annealing import SimulatedAnnealing
from .simulated_annealing_mlx import SimulatedAnnealingMLX, is_mlx_available
from .stabilizer import StabilizerSimulator

# Optional engines: only available when their backing libraries are installed.
try:
    from .ortools_engine import ORTools

    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False
    ORTools = None  # type: ignore

__all__ = [
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
]
