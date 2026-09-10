"""MQ-07 example 01: execution records.

An :class:`ExecutionRecord` wraps one actual execution with portable
metadata (plan, backend, shots, seed, bindings, timing, status,
reproducibility) plus the raw :class:`BackendResult`.
"""

import microquantum as mq

circuit = mq.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runtime = mq.ExecutionRuntime(backend=mq.LocalSimulatorBackend())
record = runtime.execute_record(circuit, shots=256, seed=7, metadata={"run": 1})

print(f"execution_id       {record.execution_id[:12]}...")
print(f"status             {record.status.value}")
print(f"backend            {record.backend}")
print(f"shots / seed       {record.shots} / {record.seed}")
print(f"counts             {record.counts}")
print(f"timing             {record.timing['total_seconds']:.4f}s")
fingerprint = record.reproducibility["fingerprint"]
print(f"reproducibility    {record.reproducibility['configured_reproducibility']}")
print(f"fingerprint        {fingerprint[:16]}...")

# JSON-safe serialization of the record round-trips faithfully.
restored = mq.ExecutionRecord.from_dict(record.to_dict())
assert restored.counts == record.counts and restored.execution_id == record.execution_id
print("serialization      round trip OK")