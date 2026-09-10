"""Data encoding circuits for quantum machine learning.

Provides multiple strategies for encoding classical data into quantum
circuits, each with different expressivity and entangling properties.

Encoding types:
- AngleEncoding: Each feature mapped to a rotation gate on one qubit
- AmplitudeEncoding: Features stored as amplitudes of a quantum state
- IQPEncoding: Interleaved data encoding with entangling layers
- ZFeatureMap: Single-qubit Z-rotations with CNOT entangler
"""
from .encoding import (
    AngleEncoding,
    AmplitudeEncoding,
    IQPEncoding,
    ZFeatureMap,
)
from .kernels import QuantumKernel
from .classifier import VariationalClassifier, ClassifierResult

__all__ = [
    "AngleEncoding",
    "AmplitudeEncoding",
    "IQPEncoding",
    "ZFeatureMap",
    "QuantumKernel",
    "VariationalClassifier",
    "ClassifierResult",
]
