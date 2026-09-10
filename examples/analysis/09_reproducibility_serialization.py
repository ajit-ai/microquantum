"""MQ-07 example 09: reproducibility & serialization.

Execution fingerprints (SHA-256 over the stable serialized configuration,
never memory addresses) let you reproduce and audit executions, while
records / experiment results round-trip through JSON-safe dictionaries.
"""

import json

import microquantum as mq

theta = mq.Parameter("theta")
circuit = mq.QuantumCircuit(1).ry(theta, 0)


def run(seed: int) -> mq.ExecutionRecord:
    runtime = mq.ExecutionRuntime(backend=mq.MockBackend())
    plan = mq.ExecutionPlan.from_circuit(
        circuit, name="rep", shots=512, seed=seed, parameter_bindings={"theta": 0.5}
    )
    return runtime.execute_record(plan)


a, b, c = run(3), run(3), run(4)
meta = a.reproducibility
print(f"fingerprint (seed 3):  {meta['fingerprint'][:16]}...")
print(f"same config same fp:   {a.reproducibility['fingerprint'] == b.reproducibility['fingerprint']}")
print(f"diff seed diff fp:     {a.reproducibility['fingerprint'] != c.reproducibility['fingerprint']}")
print(f"deterministic claim:   {meta['deterministic_execution']} "
      f"(hardware is never claimed bit-for-bit)")
print(f"identical counts:      {a.counts == b.counts}")

# Round trip: record -> dict -> JSON -> record, and result -> dict -> result.
payload = json.loads(a.to_json())
restored = mq.ExecutionRecord.from_dict(payload)
print(f"record round trip:     {restored.execution_id == a.execution_id} "
      f"and counts equal {restored.counts == a.counts}")

experiment = mq.Experiment("rt", shots=256, seed=1)
experiment.add_circuit(circuit, name="x", parameter_bindings={"theta": 0.2})
result = experiment.run(mq.ExecutionRuntime(backend=mq.MockBackend()))
again = mq.ExperimentResult.from_dict(json.loads(result.to_json()))
print(f"experiment round trip: {again.success_count == result.success_count}")