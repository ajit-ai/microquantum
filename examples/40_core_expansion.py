"""Core engine expansion tour: the 17 ``microquantum.core`` areas working together.

1.  Basic circuit            10. Tensor product
2.  Gate composition         11. Partial trace
3.  Parameterized circuit    12. Quantum-information metrics
4.  Pauli algebra            13. Execution API
5.  Observable expectation   14. Architecture/topology
6.  StateVector              15. Resource estimation
7.  DensityMatrix            16. Gradient calculation
8.  Quantum channel          17. Serialization
9.  Measurement              18. Transpiler pipeline
"""

from __future__ import annotations

import math

import numpy as np

from microquantum.core.architecture import linear_architecture
from microquantum.core.channels import depolarizing
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.execution import ExecutionOptions, ExecutionRequest, StateVectorExecutor
from microquantum.core.gates import CX, H
from microquantum.core.gradients import GradientEngine
from microquantum.core.information import fidelity, von_neumann_entropy
from microquantum.core.measurements import computational_basis_measurement
from microquantum.core.observables import PauliObservable
from microquantum.core.parameters import Parameter
from microquantum.core.pauli import PauliString, PauliSum
from microquantum.core.resources import estimate_resources
from microquantum.core.serialization import deserialize_circuit, serialize_circuit
from microquantum.core.state import StateVector
from microquantum.core.states import partial_trace
from microquantum.core.tensor import kron
from microquantum.core.transpiler import transpile_with


def main() -> None:
    # 1. Basic circuit -------------------------------------------------
    bell = QuantumCircuit(2)
    bell.h(0).cx(0, 1)
    print(f"1. bell circuit: qubits={bell.num_qubits} gates={bell.num_gates} depth={bell.depth()}")

    # 2. Gate composition ----------------------------------------------
    print(f"2. H unitarity: {np.allclose(H().to_matrix() @ H().to_matrix(), np.eye(2))}")
    print(f"   CX shape: {CX().to_matrix().shape}")

    # 3. Parameterized circuit ------------------------------------------
    theta = Parameter("theta")
    ansatz = QuantumCircuit(1)
    ansatz.rx(theta, 0)
    print(f"3. parameters: {[p.name for p in ansatz.parameters]}")

    # 4. Pauli algebra ---------------------------------------------------
    xx = PauliString("XX")
    print(f"4. XX commutes with YY: {xx.label == 'XX'}")
    hamiltonian = PauliSum([PauliString("ZZ", 0.7), PauliString("XI", 0.3)])
    print(f"   pauli terms: {hamiltonian.num_terms}")

    # 5. Observable expectation ------------------------------------------
    plus = StateVector(1, amplitudes=np.array([1, 1], dtype=complex) / math.sqrt(2))
    print(f"5. <+|X|+> = {PauliObservable('X').expectation(plus):.6f}")

    # 6. StateVector ------------------------------------------------------
    zero = StateVector(2)
    print(f"6. |00> normalized: {zero.is_normalized} dim={zero.dim}")

    # 7. DensityMatrix ------------------------------------------------------
    from microquantum.core.density_matrix import DensityMatrix

    mixed = DensityMatrix(1, np.eye(2, dtype=complex) / 2)
    print(f"7. maximally mixed purity trace: {mixed.trace.real:.1f} pure={mixed.is_pure}")

    # 8. Quantum channel -----------------------------------------------------
    noisy = depolarizing(0.1)(np.array([[1, 0], [0, 0]], dtype=complex))
    print(f"8. depolarized trace: {float(np.real(np.trace(noisy))):.6f}")

    # 9. Measurement ------------------------------------------------------------
    measurement = computational_basis_measurement(1)
    ground = StateVector(1)
    outcome = measurement.probabilities(ground)[0]
    print(f"9. P(|0>) = {outcome.probability:.1f} label={outcome.label}")

    # 10. Tensor product ----------------------------------------------------------
    print(f"10. kron shape: {kron(np.eye(2), np.eye(2)).shape}")

    # 11. Partial trace --------------------------------------------------------------
    bell_vec = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    reduced = partial_trace(bell_vec, [0])
    print(f"11. bell reduction maximally mixed: {np.allclose(reduced, np.eye(2) / 2)}")

    # 12. Quantum-information metrics ---------------------------------------------------
    print(f"12. S(I/2) = {von_neumann_entropy(np.eye(2) / 2):.6f} bit")
    print(f"    fidelity(|0>,|0>) = {fidelity(np.array([1, 0]), np.array([1, 0])):.6f}")

    # 13. Execution API -----------------------------------------------------------------
    result = StateVectorExecutor().run(
        ExecutionRequest(circuit=bell, options=ExecutionOptions(shots=200, seed=7))
    )
    print(f"13. counts: {result.get_counts()}")

    # 14. Architecture/topology -------------------------------------------------------------
    arch = linear_architecture(3)
    print(f"14. linear-3 needs routing (0,2): {arch.requires_routing((0, 2))}")

    # 15. Resource estimation -------------------------------------------------------------------
    resources = estimate_resources(bell)
    print(f"15. gates={resources.total_gates} depth={resources.depth} "
          f"memory={resources.estimated_state_memory_bytes} bytes")

    # 16. Gradient calculation -----------------------------------------------------------------------
    engine = GradientEngine()
    grad = engine.compute(ansatz, PauliString("Z"), {"theta": 0.3})
    print(f"16. d<Z>/dtheta = {grad['theta']:.6f} (exact {-math.sin(0.3):.6f})")

    # 17. Serialization ---------------------------------------------------------------------------------
    rebuilt = deserialize_circuit(serialize_circuit(ansatz))
    print(f"17. round-trip parameters: {[p.name for p in rebuilt.parameters]}")

    # 18. Transpiler pipeline -------------------------------------------------------------------------------
    redundant = QuantumCircuit(1)
    redundant.h(0).h(0).x(0)
    optimized = transpile_with(redundant)
    print(f"18. gates before/after: {redundant.num_gates}/{optimized.num_gates}")

    # Cross-module chain: parameter -> gate -> circuit -> execution -----------------------------------------------
    bound = ansatz.bind_parameters({"theta": math.pi / 2})
    psi = StateVectorExecutor().run_circuit(
        bound, ExecutionOptions(shots=4, seed=0, want_state=True)
    )
    print(f"    bound RX(pi/2) state: {np.round(np.asarray(psi.state).real, 4).tolist()}")


if __name__ == "__main__":
    main()
