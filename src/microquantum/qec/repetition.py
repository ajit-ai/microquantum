"""Repetition code, bit-flip code, and phase-flip code.

RepetitionCode: encodes 1 logical qubit into n physical qubits
using the repetition [[n, 1, n-1]] code. Protects against up to
floor((n-1)/2) bit-flip errors.

BitFlipCode: 3-qubit repetition code [[3,1,1]]. Corrects 1 bit-flip.

PhaseFlipCode: 3-qubit phase-flip code. Corrects 1 phase-flip by
transferring to the X-basis.
"""

from __future__ import annotations

from typing import Optional

from ..core.circuit import QuantumCircuit


class RepetitionCode:
    """n-qubit repetition code [[n, 1, n-1]].

    Encodes |psi> = a|0> + b|1> into:
        a|00...0> + b|11...1>

    Detects and corrects up to floor((n-1)/2) bit-flip errors.

    Args:
        num_data_qubits: Number of physical qubits (n).
    """

    def __init__(self, num_data_qubits: int = 3) -> None:
        if num_data_qubits < 3:
            raise ValueError(f"Need >= 3 data qubits, got {num_data_qubits}")
        if num_data_qubits % 2 == 0:
            raise ValueError(
                f"Data qubits should be odd for unique decoding, got {num_data_qubits}"
            )
        self._n = num_data_qubits
        # Syndrome qubits are placed after data qubits
        self._num_syndrome = num_data_qubits - 1

    @property
    def num_data_qubits(self) -> int:
        """Number of physical data qubits."""
        return self._n

    @property
    def num_syndrome_qubits(self) -> int:
        """Number of syndrome measurement qubits."""
        return self._num_syndrome

    @property
    def total_qubits(self) -> int:
        """Total qubits needed (data + syndrome)."""
        return self._n + self._num_syndrome

    @property
    def distance(self) -> int:
        """Code distance (number of correctable errors = (d-1)/2)."""
        return self._n

    def encode_circuit(self) -> QuantumCircuit:
        """Create a circuit that encodes 1 logical qubit into n physical qubits.

        Uses CNOT gates from qubit 0 to each other data qubit.

        Returns:
            Circuit that encodes the logical qubit.
        """
        qc = QuantumCircuit(self._n)
        for i in range(1, self._n):
            qc.cx(0, i)
        return qc

    def syndrome_circuit(self) -> QuantumCircuit:
        """Create a circuit to measure bit-flip syndromes.

        Uses ancilla qubits to measure parity checks between
        adjacent data qubits. Returns a circuit on total_qubits
        (data + syndrome qubits).

        Returns:
            Circuit for syndrome extraction.
        """
        n = self._n
        n_anc = self._num_syndrome
        qc = QuantumCircuit(n + n_anc)

        # CNOT from each data qubit to its syndrome qubit
        for i in range(n_anc):
            # Parity of data qubits i and i+1
            qc.cx(i, n + i)       # data[i] -> syndrome[i]
            qc.cx(i + 1, n + i)   # data[i+1] -> syndrome[i]

        return qc

    def decode_syndrome(self, syndrome_bits: list[int]) -> Optional[int]:
        """Decode syndrome bits to find the error location.

        For the repetition code with n data qubits, syndrome bit i checks
        parity of data qubits i and i+1. Error on qubit j produces 1s in
        syndrome positions j-1 and j (where valid). We match the observed
        syndrome against each possible single-qubit error pattern.

        Args:
            syndrome_bits: List of 0/1 syndrome measurement results.

        Returns:
            Index of the qubit with the detected error, or None if no error.
        """
        if len(syndrome_bits) != self._num_syndrome:
            raise ValueError(
                f"Expected {self._num_syndrome} syndrome bits, "
                f"got {len(syndrome_bits)}"
            )

        if all(b == 0 for b in syndrome_bits):
            return None

        for j in range(self._n):
            expected = [0] * self._num_syndrome
            if j > 0:
                expected[j - 1] = 1
            if j < self._n - 1:
                expected[j] = 1
            if list(syndrome_bits) == expected:
                return j

        return None

    def error_locations(self, syndrome_bits: list[int]) -> list[int]:
        """Find all error locations from syndrome (for multiple errors).

        Decomposes syndrome into a sum of single-qubit error patterns
        by greedily matching known patterns.

        Args:
            syndrome_bits: Syndrome measurement results.

        Returns:
            List of qubit indices with detected errors.
        """
        if len(syndrome_bits) != self._num_syndrome:
            raise ValueError(
                f"Expected {self._num_syndrome} syndrome bits, "
                f"got {len(syndrome_bits)}"
            )

        if all(b == 0 for b in syndrome_bits):
            return []

        remaining = list(syndrome_bits)
        locations: list[int] = []

        for j in range(self._n):
            expected = [0] * self._num_syndrome
            if j > 0:
                expected[j - 1] = 1
            if j < self._n - 1:
                expected[j] = 1
            if all(remaining[i] >= expected[i] for i in range(self._num_syndrome)):
                locations.append(j)
                remaining = [remaining[i] - expected[i]
                             for i in range(self._num_syndrome)]

        return locations

    def __repr__(self) -> str:
        return (
            f"RepetitionCode(data={self._n}, "
            f"syndrome={self._num_syndrome}, "
            f"distance={self.distance})"
        )


class BitFlipCode(RepetitionCode):
    """3-qubit bit-flip code [[3, 1, 1]].

    Encodes |psi> = a|0> + b|1> into a|000> + b|111>.
    Corrects any single bit-flip (X) error.
    """

    def __init__(self) -> None:
        super().__init__(num_data_qubits=3)

    def __repr__(self) -> str:
        return "BitFlipCode([[3,1,1]])"


class PhaseFlipCode:
    """3-qubit phase-flip code [[3, 1, 1]].

    Corrects any single phase-flip (Z) error by working in the
    Hadamard (X) basis: maps Z errors to X errors via H gates,
    then applies standard bit-flip correction.

    |psi> = a|0> + b|1> -> a|+++> + b|--->

    Args:
        (none - fixed 3-qubit code)
    """

    def __init__(self) -> None:
        self._n = 3
        self._num_syndrome = 2

    @property
    def num_data_qubits(self) -> int:
        return self._n

    @property
    def num_syndrome_qubits(self) -> int:
        return self._num_syndrome

    @property
    def total_qubits(self) -> int:
        return self._n + self._num_syndrome

    @property
    def distance(self) -> int:
        return 3

    def encode_circuit(self) -> QuantumCircuit:
        """Encode in the X-basis: H -> CNOT chain -> Hadamard.

        Circuit:
            |psi> - H - CNOT(0,1) - CNOT(0,2) - H - H
                                    -> a|000> + b|111> in X-basis
        """
        qc = QuantumCircuit(self._n)

        # Transform to X-basis
        qc.h(0)
        qc.h(1)
        qc.h(2)

        # Bit-flip encoding in X-basis
        qc.cx(0, 1)
        qc.cx(0, 2)

        return qc

    def syndrome_circuit(self) -> QuantumCircuit:
        """Syndrome measurement circuit.

        First transform to computational basis (H on all data qubits),
        then measure parity checks for bit-flip errors in the X-basis.
        """
        n = self._n
        n_anc = self._num_syndrome
        qc = QuantumCircuit(n + n_anc)

        # Transform data qubits to computational basis
        qc.h(0)
        qc.h(1)
        qc.h(2)

        # Syndrome extraction (same as bit-flip code)
        qc.cx(0, n + 0)
        qc.cx(1, n + 0)
        qc.cx(1, n + 1)
        qc.cx(2, n + 1)

        # Transform back to X-basis
        qc.h(0)
        qc.h(1)
        qc.h(2)

        return qc

    def decode_syndrome(self, syndrome_bits: list[int]) -> Optional[int]:
        """Decode syndrome for phase-flip error location."""
        if len(syndrome_bits) != self._num_syndrome:
            raise ValueError(
                f"Expected {self._num_syndrome} syndrome bits, "
                f"got {len(syndrome_bits)}"
            )

        if syndrome_bits == [0, 0]:
            return None
        elif syndrome_bits == [1, 0]:
            return 0
        elif syndrome_bits == [1, 1]:
            return 1
        elif syndrome_bits == [0, 1]:
            return 2
        return None

    def error_locations(self, syndrome_bits: list[int]) -> list[int]:
        """Find error locations from syndrome."""
        loc = self.decode_syndrome(syndrome_bits)
        return [loc] if loc is not None else []

    def __repr__(self) -> str:
        return "PhaseFlipCode([[3,1,1]])"
