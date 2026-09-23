"""Syndrome types and decoder interface for QEC codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Syndrome",
    "Decoder",
    "LookupDecoder",
]


@dataclass(frozen=True)
class Syndrome:
    """A measured error syndrome.

    Attributes:
        bits: Syndrome bits (one per stabilizer measurement).
        code_name: Name of the code that produced the syndrome.
        round: Measurement round index (for repeated readout).
    """

    bits: tuple[int, ...]
    code_name: str = ""
    round: int = 0

    def __post_init__(self) -> None:
        if any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("Syndrome bits must be 0 or 1")
        if self.round < 0:
            raise ValueError("round must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {"bits": list(self.bits), "code_name": self.code_name, "round": self.round}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Syndrome:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            bits=tuple(int(b) for b in data.get("bits", [])),
            code_name=str(data.get("code_name", "")),
            round=int(data.get("round", 0)),
        )


class Decoder(Protocol):
    """Interface for syndrome decoders."""

    def decode(self, syndrome: Syndrome) -> list[tuple[int, str]]:
        """Return ``(qubit, pauli)`` corrections for *syndrome*."""
        ...  # pragma: no cover - protocol stub


@dataclass
class LookupDecoder:
    """Table decoder mapping syndrome bits to corrections.

    Attributes:
        table: Mapping of syndrome-bit tuples to correction lists.
            The all-zero syndrome should map to ``[]`` (no error).
        code_name: Code this table was built for (checked on decode).
    """

    table: dict[tuple[int, ...], list[tuple[int, str]]] = field(default_factory=dict)
    code_name: str = ""

    def decode(self, syndrome: Syndrome) -> list[tuple[int, str]]:
        """Look up corrections for *syndrome* (unknown → ``[]``)."""
        if self.code_name and syndrome.code_name and syndrome.code_name != self.code_name:
            raise ValueError(
                f"Decoder built for '{self.code_name}' "
                f"cannot decode '{syndrome.code_name}'"
            )
        return list(self.table.get(tuple(syndrome.bits), []))

    def covers_single_qubit_errors(self, num_qubits: int, paulis: tuple[str, ...] = ("X", "Z")) -> bool:
        """True when every single-qubit error has a non-empty entry."""
        return all(
            any(correction and correction[0][0] == qubit for correction in self.table.values())
            for qubit in range(num_qubits)
            for _ in paulis
        )
