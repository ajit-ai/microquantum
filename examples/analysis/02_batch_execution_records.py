"""MQ-07 example 02: batch execution records.

``execute_records`` runs a sequence of plans/circuits and returns one
:class:`ExecutionRecord` per input, in order.  Parameter bindings and
backend selection are preserved; failures never drop the batch.
"""

import microquantum as mq

theta = mq.Parameter("theta")
ansatz = mq.QuantumCircuit(1).ry(theta, 0)

plans = [
    mq.ExecutionPlan.from_circuit(
        ansatz, name=f"theta={value:.2f}", parameter_bindings={"theta": value}
    )
    for value in (0.0, 0.5, 1.0)
]

runtime = mq.ExecutionRuntime(backend=mq.MockBackend())
records = runtime.execute_records(plans, shots=1024, seed=3)

for record in records:
    print(
        f"{record.plan_name:<12} status={record.status.value:<9} "
        f"binding={record.parameter_bindings['theta']} "
        f"batch={record.metadata.get('batch_index')}"
    )

# Batch metadata (id + per-item index) arrives on every record.
batch_ids = {r.metadata.get("batch_id") for r in records}
print(f"batch_id uniforms  {len(batch_ids) == 1}")