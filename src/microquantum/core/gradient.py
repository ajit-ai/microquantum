"""Gradient computation via the Parameter-Shift Rule."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .circuit import QuantumCircuit
from .operators import Operator
from .parameter import Parameter, ParameterExpression
from .state import StateVector


def _param_matches(
    gate_param: Union[Parameter, ParameterExpression],
    target_param: Parameter,
) -> bool:
    """Check if a gate's parameter matches the target parameter."""
    if isinstance(gate_param, Parameter):
        return gate_param is target_param
    if isinstance(gate_param, ParameterExpression):
        return gate_param.parameter is target_param
    return False


def _resolve_gate_param(
    gate_param: Union[Parameter, ParameterExpression, float],
    param_values: dict[Union[str, Parameter], float],
) -> float:
    """Resolve a gate parameter to a float given a binding map."""
    if isinstance(gate_param, (int, float)):
        return float(gate_param)
    if isinstance(gate_param, Parameter):
        if gate_param in param_values:
            return float(param_values[gate_param])
        if gate_param.name in param_values:
            return float(param_values[gate_param.name])
        raise KeyError(f"Parameter '{gate_param.name}' not in param_values")
    if isinstance(gate_param, ParameterExpression):
        return gate_param.evaluate(param_values)
    raise TypeError(f"Unexpected gate param type: {type(gate_param)}")


def _build_circuit_with_single_shift(
    circuit: QuantumCircuit,
    gate_idx: int,
    shift_amount: float,
    param_values: dict[Union[str, Parameter], float],
) -> QuantumCircuit:
    """Build a circuit where only the parameterized gate at gate_idx is shifted.

    All other parameterized gates are bound to their values from param_values.
    This is used to compute gradient contributions from individual gate
    occurrences when a parameter appears in multiple gates.
    """
    gate_map = {"rx": Operator.Rx, "ry": Operator.Ry, "rz": Operator.Rz}
    resolved = QuantumCircuit(circuit._num_qubits)

    for i, instr in enumerate(circuit._gate_instructions):
        if not QuantumCircuit._is_parameterized_gate(instr):
            resolved._gate_instructions.append(instr)
            continue

        gate_type = instr[0]
        param = instr[1]
        target = instr[2]

        if i == gate_idx:
            angle = _resolve_gate_param(param, param_values) + shift_amount
        else:
            angle = _resolve_gate_param(param, param_values)

        op = gate_map[gate_type](angle)
        resolved._gate_instructions.append((op, [target]))

    return resolved


def parameter_shift_gradient(
    circuit: QuantumCircuit,
    observable: Operator,
    param: Parameter,
    param_values: dict[Union[str, Parameter], float],
    shift: float = np.pi / 2,
    targets: Optional[list[int]] = None,
) -> float:
    """Compute the analytical partial derivative using the Parameter-Shift Rule.

    For a parameterized rotation gate with unitary U(theta) = exp(-i*theta*G/2)
    where G is the generator (Pauli), the gradient of an expectation value
    is:

        d<O>/d theta = (<O>_{theta + s} - <O>_{theta - s}) / (2 * sin(s))

    When a parameter appears in multiple gates, the total gradient is the
    sum of the individual contributions from each occurrence (product rule).

    Args:
        circuit: Parameterized quantum circuit.
        observable: Hermitian observable operator.
        param: The Parameter to differentiate with respect to.
        param_values: Current numeric values for all parameters.
        shift: Angle shift for the rule (default: pi/2).
        targets: Optional qubit indices the observable acts on.

    Returns:
        The analytical partial derivative d<O>/d(param).

    Raises:
        ValueError: If shift is 0 or pi (sin(s) = 0).
        KeyError: If param is not in param_values.
    """
    sin_shift = np.sin(shift)
    if abs(sin_shift) < 1e-12:
        raise ValueError(
            f"Shift value {shift} gives sin(s) ≈ 0; "
            f"use a shift that is not a multiple of pi"
        )

    # Verify param exists in values
    if param not in param_values and param.name not in param_values:
        raise KeyError(f"Parameter '{param.name}' not found in param_values")

    # Find all gate occurrences of this parameter
    occurrences: list[int] = []
    for i, instr in enumerate(circuit._gate_instructions):
        if QuantumCircuit._is_parameterized_gate(instr):
            gate_param = instr[1]
            if _param_matches(gate_param, param):
                occurrences.append(i)

    if not occurrences:
        raise KeyError(
            f"Parameter '{param.name}' not found in any gate in circuit"
        )

    total_grad = 0.0
    for gate_idx in occurrences:
        circuit_plus = _build_circuit_with_single_shift(
            circuit, gate_idx, shift, param_values
        )
        state_plus = circuit_plus.run()
        exp_plus = _expectation_value_raw(state_plus, observable, targets)

        circuit_minus = _build_circuit_with_single_shift(
            circuit, gate_idx, -shift, param_values
        )
        state_minus = circuit_minus.run()
        exp_minus = _expectation_value_raw(state_minus, observable, targets)

        total_grad += (exp_plus - exp_minus) / (2 * sin_shift)

    return float(total_grad)


def gradient(
    circuit: QuantumCircuit,
    observable: Operator,
    param_values: dict[Union[str, Parameter], float],
    shift: float = np.pi / 2,
    targets: Optional[list[int]] = None,
) -> dict[Parameter, float]:
    """Compute the full gradient vector for all parameters.

    Evaluates d<O>/d(param_i) for every Parameter in the circuit.
    Handles parameters that appear in multiple gates by summing
    gradient contributions from each occurrence.

    Args:
        circuit: Parameterized quantum circuit.
        observable: Hermitian observable operator.
        param_values: Current numeric values for all parameters.
        shift: Angle shift for the parameter-shift rule (default: pi/2).
        targets: Optional qubit indices the observable acts on.

    Returns:
        Dictionary mapping each Parameter to its gradient component.
    """
    grad: dict[Parameter, float] = {}
    for param in circuit.parameters:
        grad[param] = parameter_shift_gradient(
            circuit, observable, param, param_values,
            shift=shift, targets=targets,
        )
    return grad


def _expectation_value_raw(
    state: StateVector,
    observable: Operator,
    targets: Optional[list[int]] = None,
) -> float:
    """Compute <state|observable|state> as a real float."""
    from .measurement import expectation_value as _exp
    return float(_exp(state, observable, targets=targets))
