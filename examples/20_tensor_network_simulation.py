"""Example 20: Tensor-network simulation (MPS and tree tensor networks).

Demonstrates:
- Matrix-product-state (MPS) simulation with a bond-dimension cap
- Truncation error tracking on a 20-qubit GHZ ladder
- Tree-tensor-network (TTN) simulation beyond dense limits
- Exact agreement with the dense state vector on small circuits
- The explicit TTN sampling cap above ``_MAX_SV_QUBITS`` and the pointer
  to MPS sampling

Tensor-network simulators keep memory polynomial in the qubit count instead
of exponential, which is what makes 20+ qubit circuits practical here.
"""

import numpy as np

from microquantum import (
    MatrixProductState,
    MPSBackend,
    QuantumCircuit,
    StatevectorBackend,
    TreeTensorNetwork,
    TreeTensorNetworkBackend,
)


def ghz_ladder(num_qubits):
    """GHZ ladder: H on qubit 0, cx(i, i+1) for the rest."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def mps_with_truncation():
    """Truncate the MPS bond dimension and read the reported error.

    A GHZ ladder needs bond dimension 2, so capping at 1 forces measured
    fidelity loss rather than silent approximation.
    """
    print("=== MPS with bond-dim cap ===\n")
    qc = ghz_ladder(20)
    result = MPSBackend(max_bond_dim=1).run_circuit(
        num_qubits=20,
        gates=[(op.matrix, targets) for op, targets in qc.gates],
        shots=None,
    )
    print(f"  qubits={result.num_qubits}  shots={result.shots!r}")
    print(f"  max_bond_dim     = {result.metadata.get('max_bond_dim')}")
    print(f"  truncation_error = {result.metadata.get('truncation_error'):.3e}")
    print(f"  statevector_available = {result.metadata.get('statevector_available')}")


def ttn_agrees_with_statevector():
    """A small TTN result must match the dense state vector exactly."""
    print("=== TTN vs state-vector agreement ===\n")
    qc = QuantumCircuit(6)
    for q in range(6):
        qc.ry(0.3 * (q + 1), q)
    for q in range(5):
        qc.cnot(q, q + 1)
    sv = StatevectorBackend().run(qc, shots=None).statevector
    ttn = TreeTensorNetworkBackend().run(qc, shots=None).statevector
    print(f"  max |diff| = {np.max(np.abs(sv - ttn)):.2e}")


def ttn_many_qubit_deterministic():
    """32 entangled qubit pairs, simulated in polynomial memory."""
    print("=== TTN beyond dense limits ===\n")
    qc = QuantumCircuit(32)
    for k in range(16):
        qc.h(2 * k)
        qc.cx(2 * k, 2 * k + 1)
    gates = [(op.matrix, targets) for op, targets in qc.gates]
    result = TreeTensorNetworkBackend().run_circuit(num_qubits=32, gates=gates, shots=None)
    print(f"  qubits={result.num_qubits}  gates={len(gates)}  shots={result.shots!r}")
    print(f"  exact amplitudes retained: {result.statevector is not None} (18-qubit dense cap)")


def ttn_sampling_cap():
    """Sampling that would blow up memory is refused, pointing at MPS."""
    print("=== TTN sampling cap ===\n")
    tn = TreeTensorNetwork.from_zeros(19)
    try:
        tn.sample(16)
    except ValueError as exc:
        print(f"  TreeTensorNetwork.sample() on 19 qubits -> ValueError: {exc}")
    mps = MatrixProductState.from_zeros(19)
    print(f"  MPS.sample() works instead: shots -> {sum(mps.sample(16, seed=1).values())}")


if __name__ == "__main__":
    mps_with_truncation()
    ttn_agrees_with_statevector()
    ttn_many_qubit_deterministic()
    ttn_sampling_cap()
    print("\nExample 20 completed!")