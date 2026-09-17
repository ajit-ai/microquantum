"""Standard library: common quantum state factories (Phase 117).

Factories build the state vectors that recur across examples, benchmarks and
application code, on top of :class:`~microquantum.StateVector`.  They reuse
the SDK's dense-allocation guard through :func:`core._limits.check_dense_allocation`,
so they fail fast beyond the dense-simulation budget exactly like the core
constructor.

All returned states are normalized to unit norm and read back as ordinary
:class:`~microquantum.StateVector` objects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core._limits import check_dense_allocation
from ..core.state import StateVector


def _allocate(num_qubits: int) -> NDArray[np.complex128]:
    """Allocate a zero complex128 state of *num_qubits*, honoring the budget."""
    check_dense_allocation(num_qubits, density=False, kind="state vector")
    return np.zeros(1 << num_qubits, dtype=np.complex128)


def _validate_index(index: Any, *, size: int, name: str) -> None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{name} must be an int, got {type(index).__name__}")
    if index < 0 or index >= size:
        raise ValueError(f"{name} must be in 0..{size - 1}, got {index}")


def basis_state(num_qubits: int, index: int) -> StateVector:
    """Computational basis state ``|index>`` of ``num_qubits`` qubits.

    Args:
        num_qubits: Number of qubits (>= 1, within the dense budget).
        index: Basis index in ``0..2**num_qubits - 1``.

    Returns:
        The state vector with exactly one unit amplitude at ``index``.

    Raises:
        TypeError: If ``num_qubits``/``index`` is not an ``int``.
        ValueError: If ``num_qubits`` is invalid, exceeds the dense budget,
            or ``index`` is out of range.
    """
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    _validate_index(index, size=state.dim, name="index")
    state.amplitudes[index] = complex(1.0)
    return state


def uniform_superposition(num_qubits: int) -> StateVector:
    """Equal-weight superposition of all computational basis states.

    Args:
        num_qubits: Number of qubits (>= 1, within the dense budget).

    Returns:
        ``sum_i |i> / sqrt(2**num_qubits)``.

    Raises:
        TypeError: If ``num_qubits`` is not an ``int``.
        ValueError: If ``num_qubits`` is invalid or exceeds the dense budget.
    """
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    state.amplitudes.fill(complex(1.0 / math.sqrt(state.dim)))
    return state


def bell_state(index: int = 0) -> StateVector:
    """Bell (EPR) state on two qubits.

    Args:
        index: Which Bell state to build.

            * ``0`` — ``(|00> + |11>) / sqrt(2)``
            * ``1`` — ``(|00> - |11>) / sqrt(2)``
            * ``2`` — ``(|01> + |10>) / sqrt(2)``
            * ``3`` — ``(|01> - |10>) / sqrt(2)``

    Returns:
        The chosen maximally entangled two-qubit state vector.

    Raises:
        TypeError: If ``index`` is not an ``int``.
        ValueError: If ``index`` is outside ``0..3``.
    """
    state = StateVector(2, amplitudes=_allocate(2))
    _validate_index(index, size=4, name="index")
    scale = complex(1.0 / math.sqrt(2.0))
    if index in (0, 1):
        state.amplitudes[0] = scale
        state.amplitudes[3] = scale if index == 0 else -scale
    else:
        state.amplitudes[1] = scale
        state.amplitudes[2] = scale if index == 2 else -scale
    return state


def ghz_state(num_qubits: int) -> StateVector:
    """Greenberger–Horne–Zeilinger (GHZ) state.

    Args:
        num_qubits: Number of qubits (>= 1, within the dense budget).

    Returns:
        ``(|0...0> + |1...1>) / sqrt(2)``.  For one qubit this is the
        ``|+>`` state.

    Raises:
        TypeError: If ``num_qubits`` is not an ``int``.
        ValueError: If ``num_qubits`` is invalid or exceeds the dense budget.
    """
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    scale = complex(1.0 / math.sqrt(2.0))
    state.amplitudes[0] = scale
    state.amplitudes[state.dim - 1] = scale
    return state


def w_state(num_qubits: int) -> StateVector:
    """Symmetric single-excitation (W) state.

    Args:
        num_qubits: Number of qubits (>= 1, within the dense budget).

    Returns:
        ``(|10...0> + |01...0> + ... + |00...1>) / sqrt(num_qubits)`` — the
        equal-weight superposition of all states with exactly one ``1``.

    Raises:
        TypeError: If ``num_qubits`` is not an ``int``.
        ValueError: If ``num_qubits`` is invalid or exceeds the dense budget.
    """
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    scale = complex(1.0 / math.sqrt(num_qubits))
    for qubit in range(num_qubits):
        state.amplitudes[1 << qubit] = scale
    return state


__all__ = [
    "basis_state",
    "bell_state",
    "ghz_state",
    "uniform_superposition",
    "w_state",
]