Core Expansion
==============

The expanded ``microquantum.core`` is a backend-independent quantum-computing
foundation organized into 17 focused subpackages. Domain packages (algorithms,
QML, chemistry, QEC, backends, providers) build on these Core abstractions;
Core never depends on them. NumPy remains the current numerical implementation,
isolated behind these APIs so a future backend can replace it without
redesigning the quantum model.

Subpackages
-----------

================= ============================================================
Subpackage        Responsibility
================= ============================================================
``circuit``       :class:`~microquantum.QuantumCircuit`, instructions,
                  validation, metadata
``gates``         Gate hierarchy (fixed, parameterized, controlled,
                  composite) and the standard gate library
``operators``     Linear, unitary and Hermitian operators, projectors
``pauli``         :class:`~microquantum.PauliString` /
                  :class:`~microquantum.PauliSum` algebra and commutation
``observables``   Hermitian observables with expectation/variance
``states``        Pure/mixed-state workflows: tensor products, partial
                  trace, fidelity, purity
``channels``      Quantum channels (Kraus representation, standard noise
                  channels, composition)
``registers``     Named quantum/classical registers with stable addressing
``measurements``  Projective measurements, POVMs, sampling, post-state
``parameters``    Symbolic parameters, vectors, bindings, trig expressions
``tensor``        Kronecker products, permutation, partial trace, marginals
``information``   Fidelity, trace distance, entropies, mutual information
``execution``     Backend-independent requests, results and executors
``architecture``  Hardware-independent topology and native-gate descriptions
``resources``     Deterministic circuit resource estimation
``gradients``     Parameter-shift, finite-difference and analytic gradients
``serialization`` Versioned, round-trip-safe JSON schemas
``transpiler``    Pass-based compilation pipeline (validation to scheduling)
================= ============================================================

Circuits and gates
------------------

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit, validate_circuit

   bell = QuantumCircuit(2)
   bell.h(0).cx(0, 1).measure_all()
   validate_circuit(bell)
   print(bell.num_qubits, bell.num_gates, bell.depth())

.. code-block:: python

   from microquantum.core.gates import H, RX, CX
   from microquantum.core.parameters import Parameter

   theta = Parameter("theta")
   symbolic = RX(theta)
   print(symbolic.is_parameterized, CX().num_qubits)
   print((H().to_matrix() @ H().to_matrix()).round(6).tolist())

Operators, Pauli algebra and observables
----------------------------------------

.. code-block:: python

   import numpy as np
   from microquantum.core.operators import Projector, UnitaryOperator
   from microquantum.core.gates import H

   hadamard = UnitaryOperator(H().to_matrix())
   print(hadamard.inverse().compose(hadamard) == UnitaryOperator(np.eye(2)))
   print(Projector.zero_state(1).rank)

.. code-block:: python

   import numpy as np
   from microquantum import PauliString, PauliSum, StateVector
   from microquantum.core.observables import PauliObservable
   from microquantum.core.pauli import commutes, multiply_labels

   print(multiply_labels("X", "Y"))
   print(commutes(PauliString("XX"), PauliString("YY")))
   ground = StateVector(1, amplitudes=np.array([1, 0], dtype=complex))
   energy = PauliSum([PauliString("Z", 0.7), PauliString("X", 0.3)])
   print(round(float(energy.expectation(ground)), 6))
   print(round(PauliObservable("Z").expectation(ground), 6))

States, tensor networks and measurement
---------------------------------------

.. code-block:: python

   import numpy as np
   from microquantum.core.states import partial_trace, state_fidelity
   from microquantum.core.tensor import kron, subsystem_probabilities

   bell_vec = np.array([1, 0, 0, 1], dtype=complex) / 2**0.5
   print(np.allclose(partial_trace(bell_vec, [0]), np.eye(2) / 2))
   print(round(float(state_fidelity(bell_vec, bell_vec)), 6))
   print(kron(np.eye(2), np.eye(2)).shape)
   print(subsystem_probabilities(bell_vec, [0], 2))

.. code-block:: python

   import numpy as np
   from microquantum import StateVector
   from microquantum.core.measurements import computational_basis_measurement

   qubit = StateVector(1, amplitudes=np.array([0, 1], dtype=complex))
   outcome = computational_basis_measurement(1).probabilities(qubit)[1]
   print(outcome.label, round(outcome.probability, 6))

Channels and quantum information
--------------------------------

.. code-block:: python

   import numpy as np
   from microquantum.core.channels import amplitude_damping, depolarizing
   from microquantum.core.information import purity, trace_distance, von_neumann_entropy

   ground_dm = np.array([[1, 0], [0, 0]], dtype=complex)
   mixed = depolarizing(0.5)(ground_dm)
   print(round(float(np.real(np.trace(mixed))), 6))
   print(round(von_neumann_entropy(np.eye(2, dtype=complex) / 2), 6))
   print(round(trace_distance(ground_dm, mixed), 6))
   print(round(purity(mixed), 6))
   print(amplitude_damping(0.0).is_trace_preserving())

Execution, architecture and resources
-------------------------------------

.. code-block:: python

   from microquantum.core.architecture import linear_architecture
   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.execution import ExecutionOptions, ExecutionRequest, StateVectorExecutor
   from microquantum.core.resources import estimate_resources

   circuit = QuantumCircuit(2)
   circuit.h(0).cx(0, 1)
   job = ExecutionRequest(circuit=circuit, options=ExecutionOptions(shots=64, seed=3))
   print(sorted(StateVectorExecutor().run(job).get_counts()))
   print(linear_architecture(3).requires_routing((0, 2)))
   print(estimate_resources(circuit).total_gates)

Gradients, serialization and transpilation
------------------------------------------

.. code-block:: python

   import math
   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.gradients import GradientEngine
   from microquantum.core.parameters import Parameter
   from microquantum.core.pauli import PauliString

   angle = Parameter("angle")
   ansatz = QuantumCircuit(1)
   ansatz.rx(angle, 0)
   grad = GradientEngine().compute(ansatz, PauliString("Z"), {"angle": 0.3})
   print(round(grad["angle"], 6), round(-math.sin(0.3), 6))

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.serialization import deserialize_circuit, serialize_circuit
   from microquantum.core.transpiler import default_pipeline, transpile_with

   noisy_circuit = QuantumCircuit(1)
   noisy_circuit.h(0).h(0).x(0)
   print(deserialize_circuit(serialize_circuit(noisy_circuit)).num_gates)
   print(default_pipeline().num_passes, transpile_with(noisy_circuit).num_gates)

See also ``examples/40_core_expansion.py`` for a single runnable tour of all
17 areas, and the API reference for the full symbol list.
