"""Quantum and classical registers for structured circuit construction."""

from __future__ import annotations

from typing import Iterator


class QuantumRegister:
    """A named group of qubits.

    Attributes:
        name: Register name.
        size: Number of qubits.
        qubit_indices: Global qubit indices this register maps to.
    """

    def __init__(self, name: str, size: int) -> None:
        """Initialize a quantum register.

        Args:
            name: Register name.
            size: Number of qubits in this register.

        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError(f"Register size must be positive, got {size}")
        self._name = name
        self._size = size
        self._qubit_indices = list(range(size))

    @property
    def name(self) -> str:
        """Register name."""
        return self._name

    @property
    def size(self) -> int:
        """Number of qubits in this register."""
        return self._size

    @property
    def qubit_indices(self) -> list[int]:
        """Global qubit indices this register maps to."""
        return list(self._qubit_indices)

    def __getitem__(self, index: int) -> int:
        """Get the global qubit index for this register position.

        Args:
            index: Position within the register (0 to size-1).

        Returns:
            Global qubit index.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of range for register of size {self._size}"
            )
        return self._qubit_indices[index]

    def __iter__(self) -> Iterator[int]:
        """Iterate over qubit indices."""
        return iter(self._qubit_indices)

    def __len__(self) -> int:
        """Number of qubits."""
        return self._size

    def __repr__(self) -> str:
        """String representation."""
        return f"QuantumRegister('{self._name}', size={self._size})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self._name}[{self._size}]"


class ClassicalRegister:
    """A named group of classical bits for measurement storage.

    Attributes:
        name: Register name.
        size: Number of classical bits.
    """

    def __init__(self, name: str, size: int) -> None:
        """Initialize a classical register.

        Args:
            name: Register name.
            size: Number of classical bits.

        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError(f"Register size must be positive, got {size}")
        self._name = name
        self._size = size
        self._bit_indices = list(range(size))

    @property
    def name(self) -> str:
        """Register name."""
        return self._name

    @property
    def size(self) -> int:
        """Number of classical bits."""
        return self._size

    def __getitem__(self, index: int) -> int:
        """Get the bit index.

        Args:
            index: Position within the register (0 to size-1).

        Returns:
            Bit index.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of range for register of size {self._size}"
            )
        return self._bit_indices[index]

    def __iter__(self) -> Iterator[int]:
        """Iterate over bit indices."""
        return iter(self._bit_indices)

    def __len__(self) -> int:
        """Number of bits."""
        return self._size

    def __repr__(self) -> str:
        """String representation."""
        return f"ClassicalRegister('{self._name}', size={self._size})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self._name}[{self._size}]"
