"""Enhanced Hamiltonian Simulation with higher-order Trotter-Suzuki and qDRIFT.

Extends the base HamiltonianSimulation with:
- 4th-order Suzuki decomposition for higher accuracy
- qDRIFT (randomized product formula) for near-term devices
- Commuting-group term ordering for circuit depth reduction
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.pauli import PauliSum, PauliString
from .hamiltonian_simulation import TrotterResult, _pauli_evolution_circuit


def _fourth_order_suzuki(
    hamiltonian: PauliSum,
    time: float,
    num_steps: int,
    num_qubits: int,
) -> QuantumCircuit:
    """Build 4th-order Suzuki-Trotter circuit.

    S4 = S2(p1*t) * S2(p1*t) * S2(p2*t) * S2(p1*t) * S2(p1*t)
    where p1 = 1/(4-4^(1/3)), p2 = 1-4*p1
    and S2(t) = prod_j e^{-i*H_j*t/2} * prod_j e^{-i*H_j*t/2} (reversed)

    Args:
        hamiltonian: Target Hamiltonian.
        time: Total evolution time.
        num_steps: Number of Trotter steps.
        num_qubits: Number of qubits.

    Returns:
        QuantumCircuit implementing 4th-order evolution.
    """
    p1 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    p2 = 1.0 - 4.0 * p1
    coeffs = [p1, p1, p2, p1, p1]

    terms = []
    for ps in hamiltonian.terms:
        terms.append((ps.label, ps.coefficient, ps.num_qubits))

    qc = QuantumCircuit(num_qubits)
    dt = time / num_steps

    for _ in range(num_steps):
        for p in coeffs:
            step_time = p * dt
            # Forward sweep
            for label, coeff, nq in terms:
                sub = _pauli_evolution_circuit(
                    label, coeff, step_time, num_qubits
                )
                for op, targets in sub.gates:
                    qc.append(op, targets)
            # Reverse sweep
            for label, coeff, nq in reversed(terms):
                sub = _pauli_evolution_circuit(
                    label, coeff, step_time, num_qubits
                )
                for op, targets in reversed(sub.gates):
                    qc.append(op, targets)

    return qc


def _qdrift_circuit(
    hamiltonian: PauliSum,
    time: float,
    num_samples: int,
    num_qubits: int,
    seed: int | None = None,
) -> QuantumCircuit:
    """Build qDRIFT (randomized product formula) circuit.

    qDRIFT randomly selects Pauli terms with probability proportional
    to their absolute coefficient, applying each for time
    t = time * lambda / num_samples where lambda = sum|c_i|.

    Reference:
        E. Campbell, "Random Compiler for Fast Hamiltonian Simulation",
        Physical Review Letters, 2019.

    Args:
        hamiltonian: Target Hamiltonian.
        time: Total evolution time.
        num_samples: Number of random Pauli terms to sample.
        num_qubits: Number of qubits.
        seed: Random seed.

    Returns:
        QuantumCircuit implementing qDRIFT evolution.
    """
    rng = random.Random(seed)

    labels = []
    coefficients = []
    for ps in hamiltonian.terms:
        labels.append(ps.label)
        coefficients.append(abs(complex(ps.coefficient).real))

    total_norm = sum(coefficients)
    if total_norm == 0:
        return QuantumCircuit(num_qubits)

    probabilities = [c / total_norm for c in coefficients]
    step_time = total_norm * time / num_samples

    qc = QuantumCircuit(num_qubits)

    for _ in range(num_samples):
        idx = rng.choices(range(len(labels)), weights=probabilities, k=1)[0]
        coeff = complex(hamiltonian.terms[idx].coefficient).real
        sub = _pauli_evolution_circuit(
            labels[idx], coeff, step_time, num_qubits
        )
        for op, targets in sub.gates:
            qc.append(op, targets)

    return qc


def _commuting_groups(
    hamiltonian: PauliSum,
) -> list[list[PauliString]]:
    """Partition Pauli terms into commuting groups.

    Two Pauli strings commute if, for each qubit position, the
    number of non-identity positions where they differ is even.

    Uses a greedy coloring approach.

    Args:
        hamiltonian: Hamiltonian to partition.

    Returns:
        List of groups, each containing mutually commuting terms.
    """
    terms = list(hamiltonian.terms)
    groups: list[list[PauliString]] = []

    for term in terms:
        placed = False
        for group in groups:
            if all(_commutes(term, existing) for existing in group):
                group.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])

    return groups


def _commutes(a: PauliString, b: PauliString) -> bool:
    """Check if two Pauli strings commute."""
    label_a = a.label
    label_b = b.label

    # Pad to same length
    max_len = max(len(label_a), len(label_b))
    label_a = label_a.rjust(max_len, "I")
    label_b = label_b.rjust(max_len, "I")

    diff_count = 0
    for ca, cb in zip(label_a, label_b):
        if ca != "I" and cb != "I" and ca != cb:
            diff_count += 1

    return diff_count % 2 == 0


def fourth_order_simulation(
    hamiltonian: PauliSum,
    evolution_time: float = 1.0,
    num_steps: int = 1,
) -> TrotterResult:
    """Run 4th-order Suzuki-Trotter simulation.

    Args:
        hamiltonian: Target Hamiltonian.
        evolution_time: Total time t.
        num_steps: Number of Trotter steps.

    Returns:
        TrotterResult with circuit and metadata.
    """
    num_qubits = hamiltonian.num_qubits
    qc = _fourth_order_suzuki(
        hamiltonian, evolution_time, num_steps, num_qubits
    )
    return TrotterResult(
        circuit=qc,
        num_qubits=num_qubits,
        num_steps=num_steps,
        trotter_order=4,
        evolution_time=evolution_time,
    )


def qdrift_simulation(
    hamiltonian: PauliSum,
    evolution_time: float = 1.0,
    num_samples: int = 50,
    seed: int | None = None,
) -> TrotterResult:
    """Run qDRIFT randomized simulation.

    Args:
        hamiltonian: Target Hamiltonian.
        evolution_time: Total time t.
        num_samples: Number of random Pauli terms.
        seed: Random seed.

    Returns:
        TrotterResult with circuit and metadata.
    """
    num_qubits = hamiltonian.num_qubits
    qc = _qdrift_circuit(
        hamiltonian, evolution_time, num_samples, num_qubits, seed
    )
    return TrotterResult(
        circuit=qc,
        num_qubits=num_qubits,
        num_steps=num_samples,
        trotter_order=0,
        evolution_time=evolution_time,
    )
