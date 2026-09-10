"""Variational ansatz circuits for quantum chemistry.

Provides chemistry-specific ansatz circuits designed for VQE
calculations on molecular Hamiltonians.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.parameter import Parameter
from ..core.tensor import expand_operator, tensor


class HardwareEfficientAnsatz:
    """Hardware-efficient variational ansatz.

    Alternating layers of single-qubit rotations and entangling gates.
    This is a hardware-native ansatz that doesn't require problem-specific
    structure.

    Circuit structure per layer:
        Ry(theta_i) Rz(phi_i) on each qubit
        CX(i, i+1) for i in range(n-1)
        ...

    Args:
        num_qubits: Number of qubits.
        num_layers: Number of variational layers.
        entangler: Type of entangling gate ('cx', 'cz').
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        entangler: str = "cx",
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"Need >= 1 layer, got {num_layers}")
        valid_entanglers = {"cx", "cz"}
        if entangler not in valid_entanglers:
            raise ValueError(f"entangler must be one of {valid_entanglers}")
        self._num_qubits = num_qubits
        self._num_layers = num_layers
        self._entangler = entangler

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_parameters(self) -> int:
        """Number of parameters in this ansatz."""
        return self._num_layers * self._num_qubits * 2

    def build_circuit(
        self, params: Optional[dict[Parameter, float]] = None
    ) -> QuantumCircuit:
        """Build the ansatz circuit.

        Args:
            params: Parameter values. If None, uses zeros.

        Returns:
            QuantumCircuit implementing the ansatz.
        """
        n = self._num_qubits
        qc = QuantumCircuit(n)
        param_idx = 0

        for _ in range(self._num_layers):
            # Single-qubit rotation layer
            for i in range(n):
                theta = 0.0
                phi = 0.0
                if params is not None:
                    p_theta = Parameter(f"theta_{param_idx}")
                    p_phi = Parameter(f"phi_{param_idx}")
                    theta = params.get(p_theta, 0.0)
                    phi = params.get(p_phi, 0.0)
                qc.ry(theta, i)
                qc.rz(phi, i)
                param_idx += 1

            # Entangling layer
            for i in range(n - 1):
                if self._entangler == "cx":
                    qc.cx(i, i + 1)
                elif self._entangler == "cz":
                    qc.cz(i, i + 1)

        return qc

    def __repr__(self) -> str:
        return (
            f"HardwareEfficientAnsatz(qubits={self._num_qubits}, "
            f"layers={self._num_layers}, entangler={self._entangler!r})"
        )


class UCCSDAnsatz:
    """Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    The standard ansatz for quantum chemistry VQE. Uses particle-hole
    excitation operators to parameterize the wavefunction.

    The UCCSD ansatz is:
        |psi> = exp(T - T^dag) |reference>

    where T = T1 + T2 includes single and double excitations.

    Args:
        num_qubits: Number of qubits (spin-orbitals).
        num_electrons: Number of electrons.
        num_layers: Number of UCCSD layers (usually 1).
    """

    def __init__(
        self,
        num_qubits: int,
        num_electrons: int,
        num_layers: int = 1,
    ) -> None:
        if num_qubits < 2:
            raise ValueError(f"Need >= 2 qubits, got {num_qubits}")
        if num_electrons < 1:
            raise ValueError(f"Need >= 1 electron, got {num_electrons}")
        if num_electrons > num_qubits:
            raise ValueError("More electrons than qubits")
        self._num_qubits = num_qubits
        self._num_electrons = num_electrons
        self._num_layers = num_layers
        self._excitations = self._compute_excitations()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_electrons(self) -> int:
        return self._num_electrons

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the UCCSD ansatz."""
        return len(self._excitations) * self._num_layers

    def _compute_excitations(self) -> list[tuple[int, int, int, int]]:
        """Compute valid single and double excitations.

        Returns:
            List of (occupied_i, occupied_j, virtual_a, virtual_b) tuples.
        """
        n_occ = self._num_electrons // 2  # Number of occupied spatial orbitals
        n_vir = self._num_qubits // 2 - n_occ  # Number of virtual orbitals
        excitations = []

        # Single excitations: occ_i -> vir_a
        for i in range(n_occ):
            for a in range(n_vir):
                excitations.append((i, -1, n_occ + a, -1))

        # Double excitations: occ_i, occ_j -> vir_a, vir_b
        for i in range(n_occ):
            for j in range(i + 1, n_occ):
                for a in range(n_vir):
                    for b in range(a + 1, n_vir):
                        excitations.append(
                            (i, j, n_occ + a, n_occ + b)
                        )

        return excitations

    def build_circuit(
        self, params: Optional[dict[Parameter, float]] = None
    ) -> QuantumCircuit:
        """Build the UCCSD ansatz circuit.

        Uses the Trotterized form of the UCCSD operator with
        individual excitation operators as parameterized gates.

        Args:
            params: Parameter values for excitation amplitudes.

        Returns:
            QuantumCircuit implementing the UCCSD ansatz.
        """
        n = self._num_qubits
        qc = QuantumCircuit(n)

        # Prepare Hartree-Fock reference state
        for i in range(self._num_electrons):
            qc.x(i)

        param_idx = 0
        for _ in range(self._num_layers):
            for exc in self._excitations:
                occ_i, occ_j, vir_a, vir_b = exc
                theta = 0.0
                if params is not None:
                    p = Parameter(f"uccsd_{param_idx}")
                    theta = params.get(p, 0.0)

                # Build excitation operator
                if occ_j == -1:
                    # Single excitation: Ry(theta) on involved qubits
                    self._add_single_excitation(qc, occ_i, vir_a, theta)
                else:
                    # Double excitation
                    self._add_double_excitation(
                        qc, occ_i, occ_j, vir_a, vir_b, theta
                    )
                param_idx += 1

        return qc

    def _add_single_excitation(
        self, qc: QuantumCircuit, occ: int, vir: int, theta: float
    ) -> None:
        """Add a single excitation operator."""
        # Controlled rotation: CX(occ,vir) Ry(vir) CX(occ,vir)
        qc.cx(occ, vir)
        qc.ry(theta, vir)
        qc.cx(occ, vir)

    def _add_double_excitation(
        self,
        qc: QuantumCircuit,
        occ_i: int,
        occ_j: int,
        vir_a: int,
        vir_b: int,
        theta: float,
    ) -> None:
        """Add a double excitation operator."""
        # Simplified: multi-controlled rotation
        qc.cx(occ_i, vir_a)
        qc.cx(occ_j, vir_b)
        qc.ry(theta * 0.5, vir_a)
        qc.ry(theta * 0.5, vir_b)
        qc.cx(occ_i, vir_a)
        qc.cx(occ_j, vir_b)

    def __repr__(self) -> str:
        return (
            f"UCCSDAnsatz(qubits={self._num_qubits}, "
            f"electrons={self._num_electrons}, "
            f"parameters={self.num_parameters})"
        )
