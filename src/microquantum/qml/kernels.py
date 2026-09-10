"""Quantum kernel estimation for quantum machine learning.

Implements quantum kernel methods where data is mapped to quantum states
via feature maps and the kernel is computed as the fidelity between
encoded states.

The quantum kernel K(x, x') = |<phi(x')|phi(x)>|^2 provides a
similarity measure between data points in a quantum feature space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.tensor import expand_operator
from .encoding import BaseEncoder, ZFeatureMap


class QuantumKernel:
    """Quantum kernel for similarity estimation between data points.

    Computes K(x, x') = |<phi(x')|phi(x)>|^2 where phi is the
    feature map circuit. Also supports the projected quantum kernel
    which measures local overlaps.

    Args:
        encoder: Feature map encoding circuit.
        entangle: Whether to add entangling gates between features
            in the kernel circuit (controlled-SWAP test).
    """

    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        entangle: bool = True,
    ) -> None:
        if encoder is None:
            encoder = ZFeatureMap(num_features=2)
        self._encoder = encoder
        self._entangle = entangle

    @property
    def num_qubits(self) -> int:
        return self._encoder.num_qubits

    @property
    def encoder(self) -> BaseEncoder:
        """The feature map encoder."""
        return self._encoder

    def build_kernel_circuit(
        self, x1: list[float], x2: list[float]
    ) -> QuantumCircuit:
        """Build the kernel circuit for two data points.

        The kernel circuit prepares |phi(x1)> on one register and
        |phi(x2)> on another, then computes their overlap.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            QuantumCircuit with ancilla for kernel computation.
        """
        n = self._encoder.num_qubits
        # 1 ancilla + 2 registers of n qubits each
        total = 1 + 2 * n

        qc = QuantumCircuit(total)

        # Hadamard on ancilla
        qc.h(0)

        # Controlled preparation of phi(x1) on register 1
        circ1 = self._encoder.encode(x1)
        for gate_instr in circ1._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            op = gate_instr[0]  # type: ignore[assignment]
            targets = gate_instr[1]  # type: ignore[assignment,union-attr]
            # Shift qubit indices by 1 (skip ancilla)
            shifted = [int(t) + 1 for t in targets]  # type: ignore[union-attr]
            expanded = expand_operator(op, shifted, total)  # type: ignore[arg-type]
            qc.append(expanded, list(range(total)))

        # Controlled-SWAP between registers
        # For each qubit i: controlled-SWAP(ancilla, reg1[i], reg2[i])
        if self._entangle:
            for i in range(n):
                reg1_q = 1 + i
                reg2_q = 1 + n + i
                # Controlled-SWAP using Fredkin decomposition
                # C-SWAP = CNOT(a2,a1) Toffoli(anc,a1,a2) CNOT(a2,a1)
                qc.cx(reg2_q, reg1_q)
                self._add_toffoli(qc, 0, reg1_q, reg2_q, total)
                qc.cx(reg2_q, reg1_q)

        # Controlled preparation of phi(x2)^dagger on register 2
        circ2 = self._encoder.encode(x2)
        for gate_instr in reversed(circ2._gate_instructions):
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            op = gate_instr[0]  # type: ignore[assignment]
            targets = gate_instr[1]  # type: ignore[assignment,union-attr]
            inv_op = op.inverse()  # type: ignore[union-attr]
            shifted = [int(t) + 1 + n for t in targets]  # type: ignore[union-attr]
            expanded = expand_operator(inv_op, shifted, total)  # type: ignore[arg-type]
            qc.append(expanded, list(range(total)))

        # Hadamard on ancilla
        qc.h(0)

        return qc

    def _add_toffoli(
        self, qc: QuantumCircuit, control: int, target1: int,
        target2: int, num_qubits: int
    ) -> None:
        """Add a Toffoli (CCX) gate using standard decomposition."""
        qc.cx(target2, target1)
        qc.h(target2)
        qc.cx(control, target2)
        qc.cx(target1, target2)
        qc.h(target2)
        qc.cx(target2, target1)

    def evaluate_single(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate kernel for a single pair of data points.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Kernel value K(x1, x2).
        """
        qc = self.build_kernel_circuit(x1, x2)
        state = qc.run()

        # Probability of ancilla being |0>
        probs = abs(state.amplitudes) ** 2
        n = 2 * self._encoder.num_qubits + 1
        dim = 2**n
        prob_zero = sum(probs[i] for i in range(dim) if not (i & 1))

        return float(prob_zero)

    def evaluate(
        self, X1: list[list[float]], X2: list[list[float]]
    ) -> np.ndarray:
        """Evaluate kernel matrix between two sets of data points.

        Args:
            X1: First set of feature vectors.
            X2: Second set of feature vectors.

        Returns:
            Kernel matrix of shape (len(X1), len(X2)).
        """
        m = len(X1)
        n = len(X2)
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = self.evaluate_single(X1[i], X2[j])
        return K

    def __repr__(self) -> str:
        return (
            f"QuantumKernel(encoder={self._encoder!r}, "
            f"entangle={self._entangle})"
        )
