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


def dicke_state(num_qubits: int, weight: int) -> StateVector:
    """Dicke state: uniform superposition over Hamming-weight states.

    Args:
        num_qubits: Number of qubits (>= 1, within the dense budget).
        weight: Number of ``1`` bits (0 <= weight <= num_qubits).

    Returns:
        ``sum_{|x|=weight} |x> / sqrt(C(num_qubits, weight))`` with
        qubit 0 read as the most-significant bit.

    Raises:
        TypeError: If arguments are not ``int``.
        ValueError: If out of range or beyond the dense budget.
    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError(f"num_qubits must be an int, got {type(num_qubits).__name__}")
    if isinstance(weight, bool) or not isinstance(weight, int):
        raise TypeError(f"weight must be an int, got {type(weight).__name__}")
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if not 0 <= weight <= num_qubits:
        raise ValueError(f"weight must be in 0..{num_qubits}, got {weight}")
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    members = [index for index in range(state.dim) if bin(index).count("1") == weight]
    scale = complex(1.0 / math.sqrt(len(members)))
    for index in members:
        state.amplitudes[index] = scale
    return state


def graph_state(edges: list[tuple[int, int]], num_qubits: int) -> StateVector:
    """Graph state for *edges* on ``num_qubits`` qubits.

    Prepares ``|+>^⊗n`` then applies CZ for every edge.  Qubit ``q``
    is read as bit ``(num_qubits - 1 - q)`` (MSB-first, matching the
    core circuit convention).

    Args:
        edges: Undirected ``(a, b)`` pairs with ``a != b``.
        num_qubits: Number of qubits (>= 1, within the dense budget).

    Returns:
        The normalized graph state vector.
    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError(f"num_qubits must be an int, got {type(num_qubits).__name__}")
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    normalized: list[tuple[int, int]] = []
    for edge in edges:
        first, second = edge
        if first == second:
            raise ValueError(f"Self-loop edge {edge} is not allowed")
        if not 0 <= first < num_qubits or not 0 <= second < num_qubits:
            raise ValueError(f"Edge {edge} out of range for {num_qubits} qubits")
        normalized.append((min(first, second), max(first, second)))
    state = StateVector(num_qubits, amplitudes=_allocate(num_qubits))
    scale = complex(1.0 / math.sqrt(state.dim))
    for index in range(state.dim):
        sign = 1.0
        for first, second in normalized:
            bit_first = (index >> (num_qubits - 1 - first)) & 1
            bit_second = (index >> (num_qubits - 1 - second)) & 1
            if bit_first and bit_second:
                sign = -sign
        state.amplitudes[index] = complex(sign * scale.real)
    return state


__all__ = [
    "basis_state",
    "bell_state",
    "dicke_state",
    "ghz_state",
    "graph_state",
    "uniform_superposition",
    "w_state",
]