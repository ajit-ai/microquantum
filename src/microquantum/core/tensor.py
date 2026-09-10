"""Tensor products and gate expansion into full Hilbert space."""

from __future__ import annotations

from typing import Union

import numpy as np

from .operators import Operator
from .state import StateVector


def tensor(
    *items: Union[StateVector, Operator],
) -> Union[StateVector, Operator]:
    """Compute the Kronecker product (tensor product) of quantum objects.

    All items must be of the same type (all StateVector or all Operator).

    Args:
        *items: StateVector or Operator instances to tensor together.

    Returns:
        A new StateVector or Operator representing the combined system.

    Raises:
        TypeError: If items are of mixed types, the sequence is empty,
            or items are not StateVector / Operator instances.
    """
    if not items:
        raise TypeError("tensor() requires at least one argument")

    first = items[0]
    if not isinstance(first, (StateVector, Operator)):
        raise TypeError(
            f"Items must be StateVector or Operator, got {type(first).__name__}"
        )
    if not all(isinstance(item, type(first)) for item in items):
        raise TypeError("All items must be of the same type")

    if isinstance(first, StateVector):
        return _tensor_states(items)  # type: ignore[arg-type]
    return _tensor_operators(items)  # type: ignore[arg-type]


def _tensor_states(items: tuple[StateVector, ...]) -> StateVector:
    """Kronecker product of state vectors."""
    result_amps = items[0].amplitudes
    total_qubits = items[0].num_qubits
    for item in items[1:]:
        result_amps = np.asarray(np.kron(result_amps, item.amplitudes), dtype=np.complex128)
        total_qubits += item.num_qubits
    return StateVector(
        num_qubits=total_qubits,
        amplitudes=np.asarray(result_amps, dtype=np.complex128),
    )


def _tensor_operators(items: tuple[Operator, ...]) -> Operator:
    """Kronecker product of operators."""
    result_matrix = items[0].matrix
    for item in items[1:]:
        result_matrix = np.asarray(np.kron(result_matrix, item.matrix), dtype=np.complex128)
    return Operator(np.asarray(result_matrix, dtype=np.complex128))


def expand_operator(
    op: Operator,
    targets: list[int],
    num_qubits: int,
) -> Operator:
    """Embed an operator into the full N-qubit Hilbert space.

    Places the operator onto the specified target qubits within an
    N-qubit system, applying the identity to all other qubits.

    Args:
        op: The operator to embed (dimension 2^k x 2^k).
        targets: List of qubit indices the operator acts on.
            Qubit 0 is the most significant qubit (MSB).
        num_qubits: Total number of qubits in the system.

    Returns:
        A new Operator of dimension 2^N x 2^N.

    Raises:
        ValueError: If target qubit indices are out of bounds,
            or if the number of targets doesn't match op.num_qubits.
    """
    n = num_qubits
    k = op.num_qubits

    if len(targets) != k:
        raise ValueError(
            f"Operator acts on {k} qubit(s) but {len(targets)} "
            f"target(s) were given"
        )
    for t in targets:
        if t < 0 or t >= n:
            raise ValueError(
                f"Target qubit index {t} out of range for "
                f"{n}-qubit system (valid: 0..{n - 1})"
            )
    if len(targets) != len(set(targets)):
        raise ValueError("target qubits must be distinct")

    if k == n and sorted(targets) == list(range(n)):
        return op

    non_targets = sorted(set(range(n)) - set(targets))
    perm = list(targets) + non_targets

    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    if k < n:
        eye_right = Operator(np.eye(2 ** (n - k), dtype=np.complex128))
        natural_op = _kron_operator(op, eye_right)
    else:
        natural_op = op

    full_perm = inv_perm + [p + n for p in inv_perm]
    tensor = natural_op.matrix.reshape((2,) * (2 * n))
    transposed = tensor.transpose(full_perm)
    result = transposed.reshape((2**n, 2**n))

    return Operator(np.asarray(result, dtype=np.complex128))


def _kron_operator(a: Operator, b: Operator) -> Operator:
    """Kronecker product of two operators (internal helper)."""
    return Operator(np.asarray(np.kron(a.matrix, b.matrix), dtype=np.complex128))
