"""Pauli algebra layer: strings, sums, multiplication and commutation.

Re-exports the canonical :class:`PauliString` / :class:`PauliSum` from
``microquantum.core.pauli`` and adds pure-algebra helpers (single-qubit
multiplication table, commutation checks, phase handling) reusable by
QEC, chemistry, optimization, algorithms and QML.
"""

from __future__ import annotations

from ._model import PauliString, PauliSum

__all__ = [
    "PauliString",
    "PauliSum",
    "multiply_labels",
    "commutes",
    "anticommutes",
    "pauli_matrices",
]

import numpy as np
from numpy.typing import NDArray

_PAULI_MATS: dict[str, NDArray[np.complex128]] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

# (result_label, phase) for single-qubit Pauli multiplication.
_MULT_TABLE: dict[tuple[str, str], tuple[str, complex]] = {
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),
    ("Y", "I"): ("Y", 1),
    ("Z", "I"): ("Z", 1),
    ("X", "X"): ("I", 1),
    ("Y", "Y"): ("I", 1),
    ("Z", "Z"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}


def pauli_matrices() -> dict[str, NDArray[np.complex128]]:
    """Return copies of the single-qubit Pauli matrices."""
    return {k: v.copy() for k, v in _PAULI_MATS.items()}


def multiply_labels(a: str, b: str) -> tuple[str, complex]:
    """Multiply two Pauli labels: returns ``(label, phase)``.

    Labels use leftmost = qubit 0 convention, matching
    :class:`PauliString`.
    """
    a = a.upper()
    b = b.upper()
    if len(a) != len(b) or not a:
        raise ValueError(f"Pauli labels must be non-empty and equal length: {a!r} vs {b!r}")
    out: list[str] = []
    phase = complex(1)
    for ca, cb in zip(a, b, strict=False):
        if ca not in ("I", "X", "Y", "Z") or cb not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Invalid Pauli characters: {ca!r}, {cb!r}")
        res, ph = _MULT_TABLE[(ca, cb)]
        out.append(res)
        phase *= ph
    return "".join(out), phase


def _num_anticommuting_positions(a: str, b: str) -> int:
    count = 0
    for ca, cb in zip(a.upper(), b.upper(), strict=False):
        if ca != "I" and cb != "I" and ca != cb:
            count += 1
    return count


def commutes(a: PauliString, b: PauliString) -> bool:
    """Return True when Pauli strings commute."""
    if a.num_qubits != b.num_qubits:
        raise ValueError("Pauli strings act on different qubit counts")
    return _num_anticommuting_positions(a.label, b.label) % 2 == 0


def anticommutes(a: PauliString, b: PauliString) -> bool:
    """Return True when Pauli strings anticommute."""
    if a.num_qubits != b.num_qubits:
        raise ValueError("Pauli strings act on different qubit counts")
    return _num_anticommuting_positions(a.label, b.label) % 2 == 1
