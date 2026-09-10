"""Quantum algorithm implementations."""

from .adapt_vqe import AdaptResult, AdaptVQE
from .amplitude_estimation import AmplitudeEstimation, AmplitudeEstimationResult
from .base import Algorithm, AlgorithmResult
from .bernstein_vazirani import BernsteinVazirani, BVResult
from .deutsch_jozsa import DeutschJozsa, DJResult
from .grover import GroverResult, GroverSearch
from .hamiltonian_simulation import HamiltonianSimulation, TrotterResult
from .hhl import HHL, HHLResult
from .phase_estimation import PhaseEstimation, PhaseEstimationResult
from .qaoa import QAOA
from .qft import QFT, inverse_qft_circuit, qft_circuit
from .quantum_walk import (
    ContinuousQuantumWalk,
    DiscreteQuantumWalk,
    QuantumWalkResult,
)
from .shor import ShorResult, ShorsAlgorithm
from .simulation_enhanced import (
    fourth_order_simulation,
    qdrift_simulation,
)
from .vqd import VQD, VQDResult
from .vqe import VQE, VQEResult

__all__ = [
    "Algorithm",
    "AlgorithmResult",
    "VQE",
    "VQEResult",
    "VQD",
    "VQDResult",
    "QAOA",
    "GroverSearch",
    "GroverResult",
    "AmplitudeEstimation",
    "AmplitudeEstimationResult",
    "HamiltonianSimulation",
    "TrotterResult",
    "QFT",
    "qft_circuit",
    "inverse_qft_circuit",
    "PhaseEstimation",
    "PhaseEstimationResult",
    "AdaptVQE",
    "AdaptResult",
    "DiscreteQuantumWalk",
    "ContinuousQuantumWalk",
    "QuantumWalkResult",
    "ShorsAlgorithm",
    "ShorResult",
    "BernsteinVazirani",
    "BVResult",
    "DeutschJozsa",
    "DJResult",
    "HHL",
    "HHLResult",
    "fourth_order_simulation",
    "qdrift_simulation",
]
