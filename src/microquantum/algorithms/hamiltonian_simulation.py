"""Hamiltonian Simulation via Trotter-Suzuki product formulas.

Implements time evolution under a Hamiltonian H = Σ c_i P_i using
first- and second-order Trotter-Suzuki decomposition.  Each Pauli
term is exponentiated via a basis-change, parity-CNOT, Rz-rotation
circuit, and the full evolution is assembled as a product of these
term-wise unitaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.pauli import PauliSum
from ..core.state import StateVector

# ------------------------------------------------------------------ #
#  Result dataclass                                                    #
# ------------------------------------------------------------------ #

@dataclass
class TrotterResult:
    """Result container for a Trotter-Suzuki Hamiltonian simulation.

    Attributes:
        circuit: The quantum circuit implementing the Trotter evolution.
        num_qubits: Number of qubits in the system.
        num_steps: Number of Trotter steps used.
        trotter_order: Order of the Trotter-Suzuki decomposition (1 or 2).
        evolution_time: Total simulation time *t*.
    """

    circuit: QuantumCircuit
    num_qubits: int
    num_steps: int
    trotter_order: int
    evolution_time: float


# ------------------------------------------------------------------ #
#  Helper: single-term Pauli evolution circuit                         #
# ------------------------------------------------------------------ #

def _pauli_evolution_circuit(
    term_label: str,
    coefficient: complex,
    time: float,
    num_qubits: int,
) -> QuantumCircuit:
    """Build circuit for e^{-i * coefficient * P * time} where *P* is a Pauli string.

    The circuit:
        1. Conjugates each active qubit into the Z basis
           (X → H, Y → S†H, Z → nothing, I → skip).
        2. Chains CNOT gates from the highest active qubit down to
           the lowest active qubit to compute parity onto the
           lowest active qubit.
        3. Applies Rz(2 * coefficient * time) to the parity qubit.
        4. Reverses the CNOT ladder.
        5. Reverses the basis-change gates.

    Args:
        term_label: Pauli label, e.g. ``"XYZI"`` (qubit 0 = leftmost).
        coefficient: Scalar coefficient *c* of the Pauli term.
        time: Evolution time *t*.
        num_qubits: Total number of qubits in the system (label is
            right-padded with ``'I'`` to this length).

    Returns:
        QuantumCircuit implementing the single-term evolution.
    """
    padded_label = term_label.rjust(num_qubits, "I")
    n = num_qubits
    qc = QuantumCircuit(n)

    active_qubits: list[int] = []

    for i, pauli in enumerate(padded_label):
        qubit = n - 1 - i
        if pauli == "I":
            continue
        active_qubits.append(qubit)
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.s(qubit)
            qc.h(qubit)

    if not active_qubits:
        return qc

    active_qubits.sort()
    parity_qubit = active_qubits[0]

    for qubit in reversed(active_qubits):
        if qubit != parity_qubit:
            qc.cx(qubit, parity_qubit)

    angle = float(np.real(2.0 * coefficient * time))
    qc.rz(angle, parity_qubit)

    for qubit in active_qubits:
        if qubit != parity_qubit:
            qc.cx(qubit, parity_qubit)

    for i, pauli in enumerate(padded_label):
        qubit = n - 1 - i
        if pauli == "I":
            continue
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.h(qubit)
            qc.sdg(qubit)

    return qc


# ------------------------------------------------------------------ #
#  HamiltonianSimulation class                                         #
# ------------------------------------------------------------------ #

class HamiltonianSimulation:
    """Hamiltonian simulation via Trotter-Suzuki decomposition.

    Given a Hamiltonian H expressed as a :class:`PauliSum`, approximates
    the time-evolution operator U(t) = e^{-iHt} using a product formula
    of order 1 (Lie–Trotter) or order 2 (Suzuki symmetric).

    Args:
        hamiltonian: Hamiltonian to simulate, expressed as a PauliSum.
        evolution_time: Total simulation time *t* (> 0).
        num_steps: Number of Trotter steps (≥ 1).  More steps give
            a more accurate approximation.
        trotter_order: Order of the decomposition.  Must be 1 or 2.

    Raises:
        ValueError: If any argument is out of the valid range.
    """

    def __init__(
        self,
        hamiltonian: PauliSum,
        evolution_time: float,
        num_steps: int = 1,
        trotter_order: int = 1,
    ) -> None:
        if hamiltonian.num_terms == 0:
            raise ValueError("Hamiltonian must contain at least one Pauli term")
        if evolution_time <= 0:
            raise ValueError(
                f"evolution_time must be > 0, got {evolution_time}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if trotter_order not in (1, 2):
            raise ValueError(
                f"trotter_order must be 1 or 2, got {trotter_order}"
            )
        self._hamiltonian = hamiltonian
        self._evolution_time = float(evolution_time)
        self._num_steps = int(num_steps)
        self._trotter_order = int(trotter_order)

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def hamiltonian(self) -> PauliSum:
        """The Hamiltonian being simulated."""
        return self._hamiltonian

    @property
    def evolution_time(self) -> float:
        """Total evolution time *t*."""
        return self._evolution_time

    @property
    def num_steps(self) -> int:
        """Number of Trotter steps."""
        return self._num_steps

    @property
    def trotter_order(self) -> int:
        """Order of the Trotter-Suzuki decomposition."""
        return self._trotter_order

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the system."""
        return self._hamiltonian.num_qubits

    # ------------------------------------------------------------------ #
    #  Circuit construction                                               #
    # ------------------------------------------------------------------ #

    def build_circuit(self) -> QuantumCircuit:
        """Build the Trotter-Suzuki circuit for e^{-iHt}.

        Returns:
            QuantumCircuit implementing the approximate time evolution.

        Raises:
            ValueError: If the Hamiltonian is empty (should not occur
                after construction validation).
        """
        n = self._hamiltonian.num_qubits
        dt = self._evolution_time / self._num_steps
        terms = self._hamiltonian.terms

        if self._trotter_order == 1:
            step_circuit = QuantumCircuit(n)
            for term in terms:
                step_circuit = step_circuit + _pauli_evolution_circuit(
                    term.label, term.coefficient, dt, n
                )
            circuit = QuantumCircuit(n)
            for _ in range(self._num_steps):
                circuit = circuit + step_circuit
            return circuit

        half_dt = dt / 2.0
        step_circuit = QuantumCircuit(n)
        for term in terms:
            step_circuit = step_circuit + _pauli_evolution_circuit(
                term.label, term.coefficient, half_dt, n
            )

        step_circuit_rev = QuantumCircuit(n)
        for term in reversed(terms):
            step_circuit_rev = step_circuit_rev + _pauli_evolution_circuit(
                term.label, term.coefficient, half_dt, n
            )

        circuit = QuantumCircuit(n)
        for _ in range(self._num_steps):
            circuit = circuit + step_circuit + step_circuit_rev
        return circuit

    # ------------------------------------------------------------------ #
    #  Execution                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self, initial_state: Optional[StateVector] = None
    ) -> StateVector:
        """Execute the Trotter circuit on an initial state.

        Args:
            initial_state: Starting state vector.  If ``None``, the
                default |0…0⟩ state is used.

        Returns:
            Final state after Trotter evolution.
        """
        return self.build_circuit().run(initial_state)

    # ------------------------------------------------------------------ #
    #  Trotter convergence analysis                                       #
    # ------------------------------------------------------------------ #

    def run_trotter_convergence(
        self, max_steps: int = 20
    ) -> list[tuple[int, float]]:
        """Evaluate Trotter error as a function of the number of steps.

        Runs the simulation with step counts 1, 2, …, *max_steps* and
        compares each result against the exact matrix exponential of the
        Hamiltonian.

        Args:
            max_steps: Maximum number of Trotter steps to test.

        Returns:
            List of ``(num_steps, infidelity)`` pairs, where
            ``infidelity = 1 − |⟨ψ_exact|ψ_approx⟩|²``.
        """
        hamiltonian_op = self._hamiltonian.to_operator()
        h_matrix = hamiltonian_op.matrix

        eigenvalues, eigenvectors = np.linalg.eigh(h_matrix)
        exact_evolution = (
            eigenvectors
            @ np.diag(
                np.exp(-1j * eigenvalues * self._evolution_time).astype(
                    np.complex128
                )
            )
            @ eigenvectors.conj().T
        )

        initial = StateVector(self._hamiltonian.num_qubits)
        exact_state_vec = (
            exact_evolution @ initial.amplitudes
        ).astype(np.complex128)
        exact_state = StateVector(
            self._hamiltonian.num_qubits, amplitudes=exact_state_vec
        )

        results: list[tuple[int, float]] = []
        for steps in range(1, max_steps + 1):
            sim = HamiltonianSimulation(
                hamiltonian=self._hamiltonian,
                evolution_time=self._evolution_time,
                num_steps=steps,
                trotter_order=self._trotter_order,
            )
            approx_state = sim.run()
            fidelity = approx_state.fidelity(exact_state)
            infidelity = 1.0 - fidelity
            results.append((steps, infidelity))

        return results

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"HamiltonianSimulation("
            f"num_qubits={self._hamiltonian.num_qubits}, "
            f"evolution_time={self._evolution_time}, "
            f"num_steps={self._num_steps}, "
            f"trotter_order={self._trotter_order})"
        )
