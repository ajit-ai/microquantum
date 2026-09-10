First Experiment
================

An :class:`~microquantum.Experiment` groups repeated, related executions —
fixed plans and/or parameter sweeps — into a single runnable unit.  Running it
through an :class:`~microquantum.ExecutionRuntime` yields an
:class:`~microquantum.ExperimentResult` whose **raw execution records are
preserved verbatim**.

.. code-block:: python

   from microquantum import (
       ExecutionRuntime,
       Experiment,
       ExpectationAnalysis,
       MockBackend,
       Parameter,
       ParameterSweep,
       QuantumCircuit,
   )

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   experiment = Experiment("rx-overview", description="RX gate sweep", shots=1024, seed=7)
   experiment.add_circuit(ansatz, name="theta=0", parameter_bindings={"theta": 0.0})
   experiment.add_sweep(ParameterSweep({"theta": [0.5, 1.0, 2.0]}), base=ansatz)

   print(f"planned executions: {experiment.execution_count}")
   result = experiment.run(ExecutionRuntime(backend=MockBackend()))

   print(f"status:      {result.status}")
   print(f"records:     {len(result.records)} (raw, never summarized)")
   print(f"failures:    {result.failure_count}")
   print(f"fingerprint: {result.records[0].configured_reproducibility}")

   for record in result.records:
       print(f"  {record.parameter_bindings} -> {record.metadata.get('backend_name')}")

Analysing the result
--------------------

Raw records feed the analysis layer:

.. code-block:: python

   analysis = ExpectationAnalysis(result)
   print(analysis.keys())
   print(analysis.mean("Z"))              # per-label mean across executions

Experiments are fully in-memory and JSON-safe:

.. code-block:: python

   data = result.to_dict()                # every record serialized
   restored = ExperimentResult.from_dict(data)

Next: :doc:`/concepts/overview`.