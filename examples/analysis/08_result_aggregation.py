"""MQ-07 example 08: result aggregation.

:class:`ResultAggregator` groups raw results by parameter bindings, backend,
status or any dotted-path accessor — while *always preserving the raw
records* (groups hold the original objects; nothing is replaced by a summary).
"""

import microquantum as mq
from microquantum.backends.base import BackendResult
from microquantum.experiments import ExecutionRecord

records = []
for theta, label in ((0.0, "Z"), (0.0, "Z"), (1.0, "Z")):
    circuit = mq.QuantumCircuit(1).h(0)
    plan = mq.ExecutionPlan.from_circuit(
        circuit, name="a", parameter_bindings={"theta": theta}
    )
    result = BackendResult(
        num_qubits=1, backend_name="mock", expectations={label: 1.0 - theta}
    )
    runtime = mq.ExecutionRuntime(backend=mq.MockBackend())
    records.append(
        ExecutionRecord.completed(plan, runtime.default_backend, result, 0.001)
    )

aggregator = mq.ResultAggregator(records)
print(f"records kept:        {aggregator.record_count} (original objects)")

by_theta = aggregator.group_by_parameter("theta")
print(f"groups by theta:     {list(by_theta)} with "
      f"{aggregator.group_counts(by_theta)} members")
print(f"mean <Z> per group:  {aggregator.parameter_expectations('theta', 'Z')}")
print(f"groups by backend:   {list(aggregator.group_by_backend())}")
print(f"groups by status:    {list(aggregator.group_by_status())}")

# The aggregated records are the very objects passed in (identity preserved).
assert by_theta[0.0][0] is records[0]
print("raw preservation:     identity check passed")