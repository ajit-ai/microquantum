"""Circuit optimization: gate fusion, simplification, and transpilation.

Provides optimization passes that reduce circuit depth and gate count
by fusing adjacent single-qubit gates, canceling inverse pairs,
and removing identity operations.
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator


def fuse_single_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Fuse adjacent single-qubit gates on the same qubit into one.

    Scans through consecutive single-qubit gates acting on the same qubit
    and replaces them with a single gate equal to their product.

    Args:
        circuit: The input circuit to optimize.

    Returns:
        New optimized circuit with fused gates.
    """
    if circuit.is_parameterized:
        return circuit

    resolved = _resolve_circuit(circuit)
    n = circuit.num_qubits
    fused: list[tuple[Operator, list[int]]] = []

    i = 0
    while i < len(resolved):
        op, targets = resolved[i]

        if op.num_qubits != 1 or len(targets) != 1:
            fused.append((op, targets))
            i += 1
            continue

        # Collect consecutive single-qubit gates on the same qubit
        qubit = targets[0]
        product = op.matrix.copy()
        j = i + 1
        while j < len(resolved):
            next_op, next_targets = resolved[j]
            if (
                next_op.num_qubits == 1
                and len(next_targets) == 1
                and next_targets[0] == qubit
            ):
                product = next_op.matrix @ product
                j += 1
            else:
                break

        fused.append((Operator(np.asarray(product, dtype=np.complex128)), [qubit]))
        i = j

    return _build_circuit(circuit.num_qubits, fused)


def remove_identity_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove gates that are approximately identity matrices.

    Args:
        circuit: The input circuit to optimize.

    Returns:
        New circuit with identity gates removed.
    """
    if circuit.is_parameterized:
        return circuit

    resolved = _resolve_circuit(circuit)
    dim = 2
    filtered: list[tuple[Operator, list[int]]] = []

    for op, targets in resolved:
        if op.num_qubits == 1 and np.allclose(op.matrix, np.eye(dim), atol=1e-10):
            continue
        filtered.append((op, targets))

    return _build_circuit(circuit.num_qubits, filtered)


def cancel_inverse_pairs(circuit: QuantumCircuit) -> QuantumCircuit:
    """Cancel adjacent pairs of inverse gates.

    Scans for consecutive gates U followed by U^dagger on the same qubits
    and removes both.

    Args:
        circuit: The input circuit to optimize.

    Returns:
        New circuit with inverse pairs removed.
    """
    if circuit.is_parameterized:
        return circuit

    resolved = list(_resolve_circuit(circuit))
    changed = True

    while changed:
        changed = False
        new_resolved: list[tuple[Operator, list[int]]] = []
        i = 0
        while i < len(resolved):
            if i + 1 < len(resolved):
                op1, targets1 = resolved[i]
                op2, targets2 = resolved[i + 1]

                if (
                    targets1 == targets2
                    and np.allclose(
                        op1.matrix @ op2.matrix,
                        np.eye(2 ** op1.num_qubits),
                        atol=1e-10,
                    )
                ):
                    i += 2
                    changed = True
                    continue
            new_resolved.append(resolved[i])
            i += 1
        resolved = new_resolved

    return _build_circuit(circuit.num_qubits, resolved)


def simplify_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Apply a full simplification pipeline.

    Runs all optimization passes in sequence:
    1. Cancel inverse pairs
    2. Remove identity gates
    3. Fuse single-qubit gates

    Args:
        circuit: The input circuit to optimize.

    Returns:
        Optimized circuit.
    """
    result = circuit
    result = cancel_inverse_pairs(result)
    result = remove_identity_gates(result)
    result = fuse_single_qubit_gates(result)
    return result


def transpile(
    circuit: QuantumCircuit,
    optimization_level: int = 1,
) -> QuantumCircuit:
    """Transpile a circuit with configurable optimization level.

    Args:
        circuit: The input circuit.
        optimization_level: 0 = no optimization, 1 = basic, 2 = full.

    Returns:
        Optimized circuit.
    """
    if optimization_level == 0:
        return circuit
    if optimization_level == 1:
        return simplify_circuit(circuit)
    if optimization_level >= 2:
        result = simplify_circuit(circuit)
        result = fuse_single_qubit_gates(result)
        result = cancel_inverse_pairs(result)
        return result
    return circuit


def circuit_stats(circuit: QuantumCircuit) -> dict[str, int]:
    """Compute statistics about a circuit.

    Args:
        circuit: The circuit to analyze.

    Returns:
        Dictionary with gate_count, depth, and single_qubit_gates.
    """
    resolved = list(_resolve_circuit(circuit))
    total = len(resolved)
    single = sum(1 for op, _ in resolved if op.num_qubits == 1)
    return {
        "gate_count": total,
        "depth": circuit.depth,
        "single_qubit_gates": single,
        "two_qubit_gates": total - single,
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _resolve_circuit(circuit: QuantumCircuit) -> list[tuple[Operator, list[int]]]:
    """Resolve a circuit to concrete gates, handling gates property."""
    gates = circuit.gates
    if not gates and circuit.num_gates > 0:
        raise ValueError("Circuit has unbound parameters")
    return gates


def _build_circuit(
    num_qubits: int,
    gates: list[tuple[Operator, list[int]]],
) -> QuantumCircuit:
    """Build a circuit from a list of (operator, targets) pairs."""
    qc = QuantumCircuit(num_qubits)
    for op, targets in gates:
        qc.append(op, targets)
    return qc
