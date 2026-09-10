"""Core quantum computing primitives for microquantum."""

from .circuit import QuantumCircuit
from .coupling import CouplingMap
from .dynamic import ClassicalRegister, DynamicCircuit, DynamicCircuitResult
from .engine import apply_gate
from .gradient import gradient, parameter_shift_gradient
from .measurement import (
    MeasurementResult,
    expectation_value,
    measure_and_collapse,
    measure_qubits,
    sample_state,
)
from .operators import Operator
from .optimization import (
    cancel_inverse_pairs,
    circuit_stats,
    fuse_single_qubit_gates,
    remove_identity_gates,
    simplify_circuit,
    transpile,
)
from .parameter import Parameter, ParameterExpression
from .pauli import PauliString, PauliSum
from .registers import ClassicalRegister as ClassicalRegisterBase, QuantumRegister
from .resources import ResourceEstimator, ResourceEstimate
from .state import StateVector
from .tensor import expand_operator, tensor
from .transpiler import Pass, PassManager, TargetGateSet

__all__ = [
    "QuantumCircuit",
    "CouplingMap",
    "StateVector",
    "apply_gate",
    "Operator",
    "tensor",
    "expand_operator",
    "MeasurementResult",
    "sample_state",
    "measure_qubits",
    "measure_and_collapse",
    "expectation_value",
    "Parameter",
    "ParameterExpression",
    "parameter_shift_gradient",
    "gradient",
    "PauliString",
    "PauliSum",
    "QuantumRegister",
    "ClassicalRegisterBase",
    "fuse_single_qubit_gates",
    "remove_identity_gates",
    "cancel_inverse_pairs",
    "simplify_circuit",
    "transpile",
    "circuit_stats",
    # Transpiler
    "Pass",
    "PassManager",
    "TargetGateSet",
    # Dynamic circuits
    "DynamicCircuit",
    "DynamicCircuitResult",
    "ClassicalRegister",
    # Resources
    "ResourceEstimator",
    "ResourceEstimate",
]
