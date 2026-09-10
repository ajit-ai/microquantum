"""MQ-07 example 06: expectation-value analysis.

:class:`ExpectationAnalysis` aggregates the existing
``BackendResult.expectations`` format (``{label: value}``) across repeated
executions: per-label mean / variance / standard error, plus a
parameter-to-expectation mapping for sweeps.
"""

import microquantum as mq
from microquantum.backends.base import BackendResult
from microquantum.experiments import ExecutionRecord

records = []
for theta in (0.0, 0.5, 1.0, 0.5):  # 0.5 repeated -> averaged
    circuit = mq.QuantumCircuit(1).ry(mq.Parameter("theta"), 0)
    plan = mq.ExecutionPlan.from_circuit(
        circuit, name="z", parameter_bindings={"theta": theta}
    )
    z_expectation = float(round(1 - 2 * theta, 9))
    result = BackendResult(
        num_qubits=1,
        backend_name="simulator",
        expectations={"Z": z_expectation, "X": 0.0},
    )
    runtime = mq.ExecutionRuntime(backend=mq.MockBackend())
    records.append(
        ExecutionRecord.completed(plan, runtime.default_backend, result, 0.001)
    )

analysis = mq.ExpectationAnalysis(records)
print(f"labels:            {analysis.keys}")
print(f"'Z' mean:          {analysis.mean('Z'):.4f}")
print(f"'Z' variance:      {analysis.variance('Z'):.4f}")
print(f"'Z' standard error {analysis.standard_error('Z'):.4f} (n={analysis.count('Z')})")

mapping = analysis.parameter_to_expectation("theta", "Z")
for value, expectation in mapping.items():
    print(f"  theta={value:<4} <Z>={expectation:+.3f}")