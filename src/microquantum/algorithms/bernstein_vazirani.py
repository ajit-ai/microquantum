"""Bernstein-Vazirani Algorithm.

Given a hidden bitstring s, the Bernstein-Vazirani algorithm finds s
using a single query to the oracle Uf where f(x) = s · x (mod 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.circuit import QuantumCircuit


@dataclass
class BVResult:
    """Result container for Bernstein-Vazirani.

    Attributes:
        secret_string: The hidden bitstring.
        measured: Measured bitstring from the algorithm.
        correct: Whether measurement matched the secret.
        circuit: The quantum circuit used.
    """

    secret_string: list[int]
    measured: list[int]
    correct: bool
    circuit: QuantumCircuit


class BernsteinVazirani:
    """Bernstein-Vazirani algorithm for hidden string discovery.

    Finds a hidden bitstring *s* using a single query to the oracle
    that computes f(x) = s · x (mod 2).

    Args:
        secret_string: The hidden bitstring as a list of 0s and 1s,
            or a binary string like ``"1011"``.

    Example::

        bv = BernsteinVazirani("1011")
        result = bv.run()
        assert result.measured == [1, 0, 1, 1]
    """

    def __init__(
        self, secret_string: str | list[int]
    ) -> None:
        if isinstance(secret_string, str):
            self._secret = [int(c) for c in secret_string]
        else:
            self._secret = list(secret_string)

        if not self._secret:
            raise ValueError("Secret string must be non-empty")
        if not all(b in (0, 1) for b in self._secret):
            raise ValueError(
                "Secret string must contain only 0s and 1s"
            )

    @property
    def secret_string(self) -> list[int]:
        """The hidden bitstring."""
        return list(self._secret)

    @property
    def num_qubits(self) -> int:
        """Number of qubits needed."""
        return len(self._secret) + 1

    def build_oracle(self) -> QuantumCircuit:
        """Build the oracle circuit Uf.

        The oracle applies CNOT from each qubit j where s[j] = 1
        to the ancilla qubit (last qubit).
        """
        n = len(self._secret)
        qc = QuantumCircuit(n + 1)

        for j, s_j in enumerate(self._secret):
            if s_j == 1:
                qc.cx(j, n)  # CNOT from qubit j to ancilla

        return qc

    def build_circuit(self) -> QuantumCircuit:
        """Build the full Bernstein-Vazirani circuit.

        Circuit structure:
        1. X on ancilla (last qubit)
        2. Hadamard on all qubits
        3. Oracle
        4. Hadamard on first n qubits
        5. Measure first n qubits
        """
        n = len(self._secret)
        total = n + 1
        qc = QuantumCircuit(total)

        # Step 1: X on ancilla
        qc.x(n)

        # Step 2: Hadamard on all qubits
        for q in range(total):
            qc.h(q)

        # Step 3: Oracle
        oracle = self.build_oracle()
        for op, targets in oracle.gates:
            qc.append(op, targets)

        # Step 4: Hadamard on data qubits
        for q in range(n):
            qc.h(q)

        return qc

    def run(self) -> BVResult:
        """Run the Bernstein-Vazirani algorithm.

        Returns:
            A ``BVResult`` with the measured bitstring.
        """
        qc = self.build_circuit()
        sv = qc.run()

        # Extract measurement probabilities for data qubits
        n = len(self._secret)
        probs = np.zeros(2**n, dtype=np.float64)

        total_dim = len(sv.amplitudes)
        states_per_data = total_dim // (2**n)

        for data_val in range(2**n):
            for anc_val in range(states_per_data):
                idx = data_val * states_per_data + anc_val
                if idx < total_dim:
                    probs[data_val] += abs(sv.amplitudes[idx]) ** 2

        measured_val = int(np.argmax(probs))
        measured = [
            (measured_val >> (n - 1 - i)) & 1 for i in range(n)
        ]

        return BVResult(
            secret_string=list(self._secret),
            measured=measured,
            correct=(measured == self._secret),
            circuit=qc,
        )

    def __repr__(self) -> str:
        s = "".join(str(b) for b in self._secret)
        return f"BernsteinVazirani(secret='{s}')"
