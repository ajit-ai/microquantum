"""MQ-07 example 07: state analysis.

:class:`StateAnalysis` inspects state-level output: statevectors
(normalization, probabilities, most probable state, diagonal-expectation
helper) and density matrices (trace, purity, measurement probabilities).
"""

import numpy as np

import microquantum as mq

circuit = mq.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

runtime = mq.ExecutionRuntime(backend=mq.LocalSimulatorBackend())
record = runtime.execute_record(circuit)
state = mq.StateAnalysis(record)

print(f"kind:                 {state.kind} (dim {state.dim})")
print(f"normalized:           {state.is_normalized()} "
      f"(||psi||^2 = {state.norm_squared():.6f})")
print(f"most probable:        {state.most_probable_bitstring()}")
print(f"Bell-state probs:     {state.probabilities()}")

# Diagonal observable expectation: <Z0> for the Bell state is 0.
bell_z0 = state.expectation([1.0, -1.0, 1.0, -1.0])
print(f"<Z0> on |Bell>:       {bell_z0:+.3f}")

rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
mixed = mq.StateAnalysis(rho)
print(f"maximally mixed:      trace={mixed.trace():.2f} "
      f"purity={mixed.purity():.2f} pure?={mixed.is_pure()}")
print(f"density probabilities {mixed.diagonal_probabilities()}")