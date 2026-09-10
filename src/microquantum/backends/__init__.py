"""Execution backends for quantum circuits."""

from .adapter import BackendAdapter
from .array_backend import (
    available_backends,
    get_array_backend,
    gpu_available,
    is_gpu,
    set_array_backend,
)
from .base import Backend, BackendResult, Job, JobStatus
from .capabilities import (
    CIRCUIT_FEATURES,
    EXECUTION_CAPABILITIES,
    EXECUTION_DENSITY_MATRIX,
    EXECUTION_EXPECTATION_VALUES,
    EXECUTION_SAMPLING,
    EXECUTION_SHOTS,
    EXECUTION_STATEVECTOR,
    EXECUTION_UNITARY,
    FEATURE_CONTROLLED_OPERATIONS,
    FEATURE_CUSTOM_GATES,
    FEATURE_MEASUREMENT,
    FEATURE_MID_CIRCUIT_MEASUREMENT,
    FEATURE_PARAMETERIZED_CIRCUITS,
    FEATURE_RESET,
    BackendCapabilities,
    TargetClass,
)
from .density_matrix import DensityMatrixBackend
from .executor import Executor, ExecutorResult
from .local import LocalSimulatorBackend
from .mock import MockBackend
from .mps import MatrixProductState, MPSBackend
from .noise import NoiseChannel, NoiseModel
from .provider import LocalProvider, Provider
from .registry import BackendRegistry, default_registry
from .statevector import StatevectorBackend
from .tensor_network import TreeTensorNetwork, TreeTensorNetworkBackend

__all__ = [
    "Backend",
    "BackendAdapter",
    "BackendCapabilities",
    "BackendResult",
    "BackendRegistry",
    "CIRCUIT_FEATURES",
    "EXECUTION_CAPABILITIES",
    "EXECUTION_DENSITY_MATRIX",
    "EXECUTION_EXPECTATION_VALUES",
    "EXECUTION_SAMPLING",
    "EXECUTION_SHOTS",
    "EXECUTION_STATEVECTOR",
    "EXECUTION_UNITARY",
    "FEATURE_CONTROLLED_OPERATIONS",
    "FEATURE_CUSTOM_GATES",
    "FEATURE_MEASUREMENT",
    "FEATURE_MID_CIRCUIT_MEASUREMENT",
    "FEATURE_PARAMETERIZED_CIRCUITS",
    "FEATURE_RESET",
    "Job",
    "JobStatus",
    "LocalProvider",
    "LocalSimulatorBackend",
    "MockBackend",
    "Provider",
    "StatevectorBackend",
    "DensityMatrixBackend",
    "MatrixProductState",
    "MPSBackend",
    "TreeTensorNetwork",
    "TreeTensorNetworkBackend",
    "NoiseModel",
    "NoiseChannel",
    "Executor",
    "ExecutorResult",
    "default_registry",
    "set_array_backend",
    "get_array_backend",
    "available_backends",
    "gpu_available",
    "is_gpu",
    "TargetClass",
]
