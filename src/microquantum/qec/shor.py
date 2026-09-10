"""Shor's 9-qubit quantum error correction code [[9, 1, 3]].

The first quantum error correction code, capable of correcting
any arbitrary single-qubit error (X, Y, or Z).

Encoding: |psi> -> (|000> + |111>)(|000> + |111>)(|000> + |111>) / 2sqrt(2)

Structure: 3 blocks of 3 qubits each. Each block is a phase-flip code,
and the 3 blocks form a bit-flip code.
"""

from __future__ import annotations

from typing import Optional

from ..core.circuit import QuantumCircuit


class ShorCode:
    """Shor's 9-qubit code [[9, 1, 3]].

    Concatenates the 3-qubit phase-flip code with the 3-qubit bit-flip code.
    Can correct any single-qubit error (bit-flip, phase-flip, or both).

    Structure:
        Block 1: qubits 0, 1, 2
        Block 2: qubits 3, 4, 5
        Block 3: qubits 6, 7, 8

    Properties:
        num_data_qubits: 9
        num_syndrome_qubits: 8
        distance: 3
    """

    def __init__(self) -> None:
        self._n = 9
        self._num_syndrome = 8

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
        """Encode 1 logical qubit into 9 physical qubits.

        Encoding steps:
        1. Phase-flip encoding: CNOT(0,3), CNOT(0,6)
        2. H on qubits 0, 3, 6
        3. Bit-flip encoding within each block:
           Block 1: CNOT(0,1), CNOT(0,2)
           Block 2: CNOT(3,4), CNOT(3,5)
           Block 3: CNOT(6,7), CNOT(6,8)

        Returns:
            9-qubit encoding circuit.
        """
        qc = QuantumCircuit(self._n)

        # Step 1: Phase-flip encoding across blocks
        qc.cx(0, 3)
        qc.cx(0, 6)

        # Step 2: Transform block leaders to X-basis
        qc.h(0)
        qc.h(3)
        qc.h(6)

        # Step 3: Bit-flip encoding within each block
        # Block 1
        qc.cx(0, 1)
        qc.cx(0, 2)
        # Block 2
        qc.cx(3, 4)
        qc.cx(3, 5)
        # Block 3
        qc.cx(6, 7)
        qc.cx(6, 8)

        return qc

    def syndrome_circuit(self) -> QuantumCircuit:
        """Create syndrome measurement circuit.

        Uses 8 ancilla qubits:
        - 2 for bit-flip syndromes within each block (6 total)
        - 2 for phase-flip syndromes across blocks

        Returns:
            17-qubit syndrome extraction circuit.
        """
        n = self._n
        n_anc = self._num_syndrome
        qc = QuantumCircuit(n + n_anc)

        # Bit-flip syndrome within each block (6 syndrome qubits)
        # Block 1: parity checks on (0,1) and (1,2)
        qc.cx(0, n + 0)
        qc.cx(1, n + 0)
        qc.cx(1, n + 1)
        qc.cx(2, n + 1)

        # Block 2: parity checks on (3,4) and (4,5)
        qc.cx(3, n + 2)
        qc.cx(4, n + 2)
        qc.cx(4, n + 3)
        qc.cx(5, n + 3)

        # Block 3: parity checks on (6,7) and (7,8)
        qc.cx(6, n + 4)
        qc.cx(7, n + 4)
        qc.cx(7, n + 5)
        qc.cx(8, n + 5)

        # Phase-flip syndrome across blocks
        # Transform block leaders to X-basis
        qc.h(0)
        qc.h(3)
        qc.h(6)

        # Parity of block leaders: (0,3) and (3,6)
        qc.cx(0, n + 6)
        qc.cx(3, n + 6)
        qc.cx(3, n + 7)
        qc.cx(6, n + 7)

        # Transform back
        qc.h(0)
        qc.h(3)
        qc.h(6)

        return qc

    def decode_syndrome(
        self, syndrome_bits: list[int]
    ) -> Optional[tuple[int, str]]:
        """Decode syndrome to identify error location and type.

        Syndrome layout (8 bits):
            [bf_block1_01, bf_block1_12,
             bf_block2_34, bf_block2_45,
             bf_block3_67, bf_block3_78,
             pf_blocks_03, pf_blocks_36]

        Args:
            syndrome_bits: 8-bit syndrome measurement.

        Returns:
            Tuple of (qubit_index, error_type) or None if no error.
            error_type is one of "X", "Z", "Y", or "none".
        """
        if len(syndrome_bits) != 8:
            raise ValueError(f"Expected 8 syndrome bits, got {len(syndrome_bits)}")

        bf = syndrome_bits[:6]  # Bit-flip syndromes
        pf = syndrome_bits[6:]  # Phase-flip syndromes

        # Identify bit-flip error within each block
        bf_error_block = None
        bf_error_in_block = None

        for block in range(3):
            s0 = bf[block * 2]
            s1 = bf[block * 2 + 1]
            if s0 == 0 and s1 == 0:
                continue
            elif s0 == 1 and s1 == 0:
                bf_error_block = block
                bf_error_in_block = 0
            elif s0 == 1 and s1 == 1:
                bf_error_block = block
                bf_error_in_block = 1
            elif s0 == 0 and s1 == 1:
                bf_error_block = block
                bf_error_in_block = 2

        # Identify phase-flip error across blocks
        pf_block = None
        if pf[0] == 1 and pf[1] == 0:
            pf_block = 0
        elif pf[0] == 1 and pf[1] == 1:
            pf_block = 1
        elif pf[0] == 0 and pf[1] == 1:
            pf_block = 2

        # Determine error type and location
        if bf_error_block is not None and bf_error_in_block is not None and pf_block is not None:
            # Both bit-flip and phase-flip: Y error
            qubit = bf_error_block * 3 + bf_error_in_block
            return (qubit, "Y")
        elif bf_error_block is not None and bf_error_in_block is not None:
            # Bit-flip only: X error
            qubit = bf_error_block * 3 + bf_error_in_block
            return (qubit, "X")
        elif pf_block is not None:
            # Phase-flip only: Z error on block leader
            qubit = pf_block * 3
            return (qubit, "Z")

        return None

    def error_locations(
        self, syndrome_bits: list[int]
    ) -> list[tuple[int, str]]:
        """Find all error locations and types from syndrome.

        Returns:
            List of (qubit_index, error_type) tuples.
        """
        result = self.decode_syndrome(syndrome_bits)
        return [result] if result is not None else []

    def __repr__(self) -> str:
        return "ShorCode([[9,1,3]])"
