"""Multi-qubit tensor contraction engine for quantum gate application."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .state import StateVector


def apply_gate(
    state: StateVector,
    gate_matrix: NDArray[np.complex128],
    target_qubits: list[int],
) -> StateVector:
    """Apply a quantum gate to specified qubits via tensor contraction.

    Reshapes the state vector into an N-dimensional tensor and uses einsum
    to contract the gate matrix onto the target qubit axes, avoiding
    construction of the full 2^N x 2^N dense operator matrix.

    Args:
        state: The input quantum state vector.
        gate_matrix: Unitary gate matrix of shape (2^k, 2^k) where
            k = len(target_qubits).
        target_qubits: List of qubit indices the gate acts on.

    Returns:
        A new StateVector with the gate applied.

    Raises:
        ValueError: If target qubit indices are out of range, duplicated,
            or if gate matrix dimensions are incorrect.
    """
    n = state.num_qubits
    k = len(target_qubits)

    _validate_target_qubits(target_qubits, n)
    _validate_gate_matrix(gate_matrix, k)

    amp_tensor = state.amplitudes.reshape((2,) * n)
    gate_tensor = np.array(gate_matrix, dtype=np.complex128).reshape((2,) * (2 * k))

    result_tensor = _contract(amp_tensor, gate_tensor, target_qubits, n, k)

    new_amplitudes = result_tensor.flatten().astype(np.complex128)
    return StateVector(num_qubits=n, amplitudes=new_amplitudes)


def _validate_target_qubits(target_qubits: list[int], num_qubits: int) -> None:
    """Validate target qubit indices."""
    if not target_qubits:
        raise ValueError("target_qubits must not be empty")
    for q in target_qubits:
        if q < 0 or q >= num_qubits:
            raise ValueError(
                f"Target qubit index {q} out of range for "
                f"{num_qubits}-qubit state (valid: 0..{num_qubits - 1})"
            )
    if len(target_qubits) != len(set(target_qubits)):
        raise ValueError("target_qubits must contain distinct indices")


def _validate_gate_matrix(
    gate_matrix: NDArray[np.complex128], k: int
) -> None:
    """Validate gate matrix dimensions."""
    expected = 2**k
    if gate_matrix.shape != (expected, expected):
        raise ValueError(
            f"Gate matrix must have shape ({expected}, {expected}) for "
            f"{k} target qubit(s), got {gate_matrix.shape}"
        )


def _contract(
    amp_tensor: NDArray[np.complex128],
    gate_tensor: NDArray[np.complex128],
    target_qubits: list[int],
    n: int,
    k: int,
) -> NDArray[np.complex128]:
    """Perform tensor contraction of gate onto state tensor.

    The gate matrix axes are laid out as:
        [output_q0, output_q1, ..., input_q0, input_q1, ...]
    where output/input indices correspond to the target qubits in order.

    Gate input indices contract with the state axes at the target qubit
    positions. Gate output indices introduce new letters that replace
    those axes in the result.
    """
    letters = [chr(ord("a") + i) for i in range(n)]
    new_letters = [chr(ord("a") + n + i) for i in range(k)]

    state_idx = list(letters)

    gate_output_idx = list(new_letters)
    gate_input_idx = [letters[t] for t in target_qubits]

    gate_idx = gate_output_idx + gate_input_idx

    result_idx = list(letters)
    for i, t in enumerate(target_qubits):
        result_idx[t] = new_letters[i]

    input_str = "".join(state_idx) + "," + "".join(gate_idx)
    output_str = "".join(result_idx)
    einsum_str = f"{input_str}->{output_str}"

    return np.asarray(np.einsum(einsum_str, amp_tensor, gate_tensor), dtype=np.complex128)
