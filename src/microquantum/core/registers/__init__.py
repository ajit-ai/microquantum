"""Explicit register abstractions with stable qubit/bit addressing.

The canonical :class:`QuantumRegister` / :class:`ClassicalRegister`
implementations live in ``microquantum.core.registers._model`` (moved
from the stable 1.0.0 flat module) and are re-exported unchanged to
preserve the public API.  This package additionally provides the
:class:`Register` base, :class:`Qubit` / :class:`Clbit` handles and
JSON helpers for circuit integration and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._model import ClassicalRegister, QuantumRegister

__all__ = [
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "Qubit",
    "Clbit",
    "register_to_dict",
    "register_from_dict",
]


@dataclass(frozen=True)
class Qubit:
    """A single qubit handle bound to a named register."""

    register: str
    index: int

    def __post_init__(self) -> None:
        if not self.register:
            raise ValueError("register name must be non-empty")
        if self.index < 0:
            raise ValueError("Qubit index must be >= 0")


@dataclass(frozen=True)
class Clbit:
    """A single classical bit handle bound to a named register."""

    register: str
    index: int

    def __post_init__(self) -> None:
        if not self.register:
            raise ValueError("register name must be non-empty")
        if self.index < 0:
            raise ValueError("Clbit index must be >= 0")


class Register:
    """Base class for sized, named registers.

    The concrete :class:`QuantumRegister` / :class:`ClassicalRegister`
    remain the canonical circuit-integration types; this base exists for
    ``isinstance`` checks and shared validation.
    """

    def __init__(self, size: int, name: str) -> None:
        if size < 1:
            raise ValueError(f"Register size must be >= 1, got {size}")
        if not name:
            raise ValueError("Register name must be non-empty")
        self._size = size
        self._name = name

    @property
    def size(self) -> int:
        """Number of entries in the register."""
        return self._size

    @property
    def name(self) -> str:
        """Register name."""
        return self._name

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self._name}', size={self._size})"


def register_to_dict(register: Any) -> dict[str, Any]:
    """Serialize a quantum or classical register to a JSON-safe dict."""
    kind = "quantum" if isinstance(register, QuantumRegister) else "classical"
    if not isinstance(register, (QuantumRegister, ClassicalRegister)):
        raise TypeError(f"Expected a register, got {type(register).__name__}")
    return {"kind": kind, "name": register.name, "size": register.size}


def register_from_dict(data: dict[str, Any]) -> Any:
    """Deserialize a register produced by :func:`register_to_dict`."""
    if not isinstance(data, dict) or data.get("kind") not in ("quantum", "classical"):
        raise ValueError(f"Invalid register payload: {data!r}")
    if data["kind"] == "quantum":
        return QuantumRegister(str(data["name"]), int(data["size"]))
    return ClassicalRegister(str(data["name"]), int(data["size"]))


def validate_registers(registers: list[Any], num_qubits: int, num_clbits: int = 0) -> None:
    """Validate that registers fit within the given qubit/bit counts."""
    for reg in registers:
        if isinstance(reg, QuantumRegister):
            if reg.size > num_qubits:
                raise ValueError(f"Register '{reg.name}' needs {reg.size} qubits, have {num_qubits}")
        elif isinstance(reg, ClassicalRegister):
            if reg.size > num_clbits:
                raise ValueError(f"Register '{reg.name}' needs {reg.size} clbits, have {num_clbits}")
        else:
            raise TypeError(f"Expected a register, got {type(reg).__name__}")
