"""MQ-07 example 10: structured failure handling.

Failed executions are never silently dropped: they become records in the
``failed`` state carrying an inspectable :class:`ExecutionFailure` with the
execution id, plan name, error type and message.  Batches survive partial
failure and keep every input position.
"""

import microquantum as mq

theta = mq.Parameter("theta")
circuit = mq.QuantumCircuit(1).ry(theta, 0)

runtime = mq.ExecutionRuntime(backend=mq.MockBackend())

# An unbound-but-misbound plan: bindings reference a parameter that does
# not exist on the circuit -> a deterministic ValueError.
good = mq.ExecutionPlan.from_circuit(
    circuit, name="good", parameter_bindings={"theta": 0.25}
)
bad = mq.ExecutionPlan.from_circuit(
    circuit, name="bad", parameter_bindings={"theta": 0.25, "nope": 1.0}
)

records = runtime.execute_records([good, bad, good])
print(f"batch of 3 -> statuses: {[r.status.value for r in records]}")

failed = records[1]
assert failed.error is not None
print(f"failed plan            {failed.error.plan_name}")
print(f"execution id           {failed.error.execution_id[:12]}...")
print(f"error type             {failed.error.error_type}")
print(f"message                {failed.error.message}")
print(f"serialized failure     {failed.error.to_dict()['error_type']}")

# Completed neighbours keep their full recorded result.
print(f"neighbour counts       {records[0].counts}")
print("every position recorded (no dropped batch items)")