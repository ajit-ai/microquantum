"""MQ-07 example 04: running an experiment.

:class:`Experiment` groups fixed plans and parameter sweeps; running it
through an :class:`ExecutionRuntime` produces an :class:`ExperimentResult`
whose raw execution records are preserved verbatim.
"""

import microquantum as mq

theta = mq.Parameter("theta")
ansatz = mq.QuantumCircuit(1).ry(theta, 0)

experiment = mq.Experiment(
    "sweep-demo", description="RQ gate over three angles", shots=1024, seed=7
)
experiment.add_circuit(ansatz, name="theta=0", parameter_bindings={"theta": 0.0})
experiment.add_sweep(mq.ParameterSweep({"theta": [0.5, 1.0, 2.0]}), base=ansatz)

print(f"experiment '{experiment.name}' has {experiment.execution_count} executions")
result = experiment.run(mq.ExecutionRuntime(backend=mq.MockBackend()))

print(f"status            {result.status}")
print(f"successes/failures {result.success_count}/{result.failure_count}")
print(f"backends          {result.backend_names}")
print(f"records kept      {len(result.records)} (raw, never summarized)")

for record in result.records:
    name = record.metadata.get("sweep_name") or record.plan_name
    print(f"  {name} bound={record.parameter_bindings} counts={record.counts}")