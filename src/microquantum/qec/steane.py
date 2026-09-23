"""Steane's 7-qubit quantum error correction code [[7, 1, 3]].

A CSS code built from the classical [7, 4, 3] Hamming code: three
X-type and three Z-type stabilizers on the Hamming parity supports.
Corrects any single-qubit X, Z (hence Y) error; syndromes decode by
binary position, exactly like Hamming decoding.

Stabilizers (0-indexed qubits)::

    X_3456 = X3 X4 X5 X6        Z_3456 = Z3 Z4 Z5 Z6
    X_1256 = X1 X2 X5 X6        Z_1256 = Z1 Z2 Z5 Z6
    X_0246 = X0 X2 X4 X6        Z_0246 = Z0 Z2 Z4 Z6

Syndrome layout (6 bits)::

    [z0, z1, z2, x0, x1, x2]

where ``z*`` come from the Z stabilizers (detect X errors) and ``x*``
from the X stabilizers (detect Z errors).  A non-zero triple reads, in
binary, as the 1-indexed error position (0 = no error).
"""

from __future__ import annotations

from typing import Optional

from ..core.circuit import QuantumCircuit

__all__ = [
    "SteaneCode",
    "STEANE_X_STABILIZERS",
    "STEANE_Z_STABILIZERS",
]

STEANE_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)

STEANE_Z_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)

_DATA_QUBITS = (2, 4, 5, 6)
_PARITY_CNOTS = (
    (2, 0),
    (4, 0),
    (6, 0),
    (2, 1),
    (5, 1),
    (6, 1),
    (4, 3),
    (5, 3),
    (6, 3),
)


class SteaneCode:
    """Steane's [[7, 1, 3]] code.

    Properties:
        num_data_qubits: 7
        num_syndrome_qubits: 6
        distance: 3
    """

    def __init__(self) -> None:
        self._n = 7
        self._num_syndrome = 6

    @property
    def num_data_qubits(self) -> int:
        """Number of data qubits."""
        return self._n

    @property
    def num_syndrome_qubits(self) -> int:
        """Number of syndrome ancillas."""
        return self._num_syndrome

    @property
    def total_qubits(self) -> int:
        """Data plus syndrome qubits."""
        return self._n + self._num_syndrome

    @property
    def distance(self) -> int:
        """Code distance."""
        return 3

    @property
    def x_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """X-type stabilizer supports."""
        return STEANE_X_STABILIZERS

    @property
    def z_stabilizers(self) -> tuple[tuple[int, ...], ...]:
        """Z-type stabilizer supports."""
        return STEANE_Z_STABILIZERS

    def encode_circuit(self) -> QuantumCircuit:
        """Encode ``|0>`` into the logical ``|0>_L`` state.

        Prepares the uniform superposition over Hamming codewords:
        Hadamards on the data qubits followed by parity CNOTs, so the
        output is stabilized by all six stabilizers.

        Returns:
            7-qubit encoding circuit.
        """
        qc = QuantumCircuit(self._n)
        for qubit in _DATA_QUBITS:
            qc.h(qubit)
        for control, target in _PARITY_CNOTS:
            qc.cx(control, target)
        return qc

    def syndrome_circuit(self) -> QuantumCircuit:
        """Create the syndrome extraction circuit.

        Ancillas 7..9 measure the Z stabilizers (X-error syndrome) with
        CNOT(data → ancilla); ancillas 10..12 measure the X stabilizers
        (Z-error syndrome) in the Hadamard basis.  Follows the
        :class:`ShorCode` extraction pattern.

        Returns:
            13-qubit syndrome extraction circuit.
        """
        n = self._n
        qc = QuantumCircuit(n + self._num_syndrome)
        for index, support in enumerate(STEANE_Z_STABILIZERS):
            for qubit in support:
                qc.cx(qubit, n + index)
        for index, support in enumerate(STEANE_X_STABILIZERS):
            ancilla = n + 3 + index
            for qubit in support:
                qc.h(qubit)
            for qubit in support:
                qc.cx(qubit, ancilla)
            for qubit in support:
                qc.h(qubit)
        return qc

    def _position(self, bits: tuple[int, int, int]) -> int:
        """Binary triple to 0-indexed qubit (0 triple → -1 = no error)."""
        value = bits[0] * 4 + bits[1] * 2 + bits[2]
        return value - 1

    def decode_syndrome(
        self, syndrome_bits: list[int]
    ) -> Optional[tuple[int, str]]:
        """Decode a 6-bit syndrome to ``(qubit, error_type)``.

        Layout ``[z0, z1, z2, x0, x1, x2]``.  Returns ``None`` when no
        error is detected.  Under the single-error model both triples
        agree; a ``Y`` error shows in both halves.

        Args:
            syndrome_bits: 6-bit syndrome measurement.

        Returns:
            Tuple of (qubit_index, error_type) with error_type in
            ``"X"``, ``"Z"``, ``"Y"``, or ``None`` if clean.
        """
        if len(syndrome_bits) != 6:
            raise ValueError(f"Expected 6 syndrome bits, got {len(syndrome_bits)}")
        if any(bit not in (0, 1) for bit in syndrome_bits):
            raise ValueError("Syndrome bits must be 0 or 1")
        x_error = self._position((syndrome_bits[0], syndrome_bits[1], syndrome_bits[2]))
        z_error = self._position((syndrome_bits[3], syndrome_bits[4], syndrome_bits[5]))
        if x_error < 0 and z_error < 0:
            return None
        if x_error >= 0 and z_error >= 0:
            if x_error == z_error:
                return (x_error, "Y")
            return (x_error, "X")
        if x_error >= 0:
            return (x_error, "X")
        return (z_error, "Z")

    def error_locations(
        self, syndrome_bits: list[int]
    ) -> list[tuple[int, str]]:
        """Find all error locations and types from a syndrome.

        Returns:
            List of (qubit_index, error_type) tuples (empty when clean;
            up to two entries when X and Z syndromes disagree).
        """
        if len(syndrome_bits) != 6:
            raise ValueError(f"Expected 6 syndrome bits, got {len(syndrome_bits)}")
        x_error = self._position((syndrome_bits[0], syndrome_bits[1], syndrome_bits[2]))
        z_error = self._position((syndrome_bits[3], syndrome_bits[4], syndrome_bits[5]))
        locations: list[tuple[int, str]] = []
        if x_error >= 0 and z_error >= 0 and x_error == z_error:
            return [(x_error, "Y")]
        if x_error >= 0:
            locations.append((x_error, "X"))
        if z_error >= 0:
            locations.append((z_error, "Z"))
        return locations

    def __repr__(self) -> str:
        return "SteaneCode([[7,1,3]])"
