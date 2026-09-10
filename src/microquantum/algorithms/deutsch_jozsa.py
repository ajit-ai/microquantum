"""Deutsch-Jozsa Algorithm.

Determines whether a Boolean function f : {0,1}^n → {0,1} is constant
(same output for all inputs) or balanced (outputs 0 for exactly half
the inputs and 1 for the other half).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit


@dataclass
class DJResult(JSONSerializable):
    """Result container for Deutsch-Jozsa.

    Attributes:
        is_constant: True if function is constant, False if balanced.
        measurement: Raw measurement bitstring (as list of ints).
        circuit: The quantum circuit used.
    """

    is_constant: bool
    measurement: list[int]
    circuit: QuantumCircuit


class DeutschJozsa:
    """Deutsch-Jozsa algorithm for function property testing.

    Determines if *f* is constant or balanced with a single query.

    Args:
        n_qubits: Number of input qubits (>= 1).
        balanced: If True, uses a balanced oracle; if False, uses a
            constant oracle (all-zeros output).

    Example::

        dj = DeutschJozsa(n_qubits=3, balanced=True)
        result = dj.run()
        assert result.is_constant is False
    """

    def __init__(
        self, n_qubits: int, balanced: bool = True
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self._n = n_qubits
        self._balanced = balanced

    @property
    def num_qubits(self) -> int:
        """Number of input qubits."""
        return self._n

    def build_oracle(self) -> QuantumCircuit:
        """Build the oracle circuit Uf.

        For the balanced case, applies CNOT from each input qubit to
        the ancilla.  For the constant case, returns an empty circuit
        (identity).
        """
        total = self._n + 1
        qc = QuantumCircuit(total)

        if self._balanced:
            for q in range(self._n):
                qc.cx(q, self._n)

        return qc

    def build_circuit(self) -> QuantumCircuit:
        """Build the full Deutsch-Jozsa circuit.

        Circuit structure:
        1. X on ancilla (last qubit)
        2. Hadamard on all qubits
        3. Oracle
        4. Hadamard on input qubits
        5. Measure input qubits
        """
        total = self._n + 1
        qc = QuantumCircuit(total)

        # Step 1: X on ancilla
        qc.x(self._n)

        # Step 2: Hadamard on all qubits
        for q in range(total):
            qc.h(q)

        # Step 3: Oracle
        oracle = self.build_oracle()
        for op, targets in oracle.gates:
            qc.append(op, targets)

        # Step 4: Hadamard on input qubits
        for q in range(self._n):
            qc.h(q)

        return qc

    def run(self) -> DJResult:
        """Run the Deutsch-Jozsa algorithm.

        Returns:
            A ``DJResult`` with the determination.
        """
        qc = self.build_circuit()
        sv = qc.run()

        # Measure input qubits
        probs = np.zeros(2**self._n, dtype=np.float64)
        total_dim = len(sv.amplitudes)
        states_per_input = total_dim // (2**self._n)

        for input_val in range(2**self._n):
            for anc_val in range(states_per_input):
                idx = input_val * states_per_input + anc_val
                if idx < total_dim:
                    probs[input_val] += abs(sv.amplitudes[idx]) ** 2

        measured_val = int(np.argmax(probs))
        measurement = [
            (measured_val >> (self._n - 1 - i)) & 1
            for i in range(self._n)
        ]

        is_constant = all(b == 0 for b in measurement)

        return DJResult(
            is_constant=is_constant,
            measurement=measurement,
            circuit=qc,
        )

    def __repr__(self) -> str:
        kind = "balanced" if self._balanced else "constant"
        return f"DeutschJozsa(n_qubits={self._n}, {kind})"
