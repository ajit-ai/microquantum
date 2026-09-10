"""MQ-07 example 05: sampling (measurement) analysis.

:class:`SamplingAnalysis` derives mathematically well-defined summaries from
measurement counts: probabilities, the most frequent outcome, entropy,
marginals and observables.  By default bitstrings are read as unsigned binary
integers (MSB first); a ``value_of`` callable overrides that mapping.
"""

import microquantum as mq

circuit = mq.QuantumCircuit(2)
circuit.h(0)
circuit.h(1)
circuit.measure_all()

runtime = mq.ExecutionRuntime(backend=mq.LocalSimulatorBackend())
record = runtime.execute_record(circuit, shots=2000, seed=11)
analysis = mq.SamplingAnalysis(record)

print(f"total shots:     {analysis.total_shots()}")
print(f"probabilities:   {analysis.probabilities()}")
print(f"most likely:     {analysis.most_likely()} "
      f"p={analysis.most_likely_probability():.3f}")
print(f"entropy:         {analysis.entropy():.4f} bits (2 outcomes -> 1.0)")
print(f"mean (as int):   {analysis.mean():.3f}")
print(f"mean (parity):   {analysis.mean(lambda b: int(b, 2) % 2):.3f}")

qubit = analysis.marginal([0])
print(f"marginal q0:     {qubit}")