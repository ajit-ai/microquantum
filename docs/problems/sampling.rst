Sampling
========

A :class:`~microquantum.SamplingProblem` asks for samples from a circuit's
output distribution: ``|bitstring> -> probability``.

Usage
-----

.. code-block:: python

   from microquantum import QuantumCircuit, SamplingProblem

   qc = QuantumCircuit(2).h(0).cx(0, 1)       # Bell state
   problem = SamplingProblem(qc, num_samples=1024, name="bell")

   print(problem.circuit)
   print(problem.validate())                  # []

   data = problem.to_dict()
   restored = SamplingProblem(**data)         # JSON-safe round trip
   print(restored.num_samples)                # 1024

What it maps to
---------------

* **Solvers** usually come from the runtime helpers —
  :func:`~microquantum.execute` / :func:`~microquantum.run_parameter_sweep` /
  :class:`~microquantum.ExecutionRuntime` — which bind the circuit, dispatch
  it to a backend and return a :class:`~microquantum.BackendResult` with
  counts.
* **Analysis** — :class:`~microquantum.SamplingAnalysis` turns those counts
  into outcome probabilities, entropy and marginals (see
  :doc:`/analysis/sampling`).

Example through the runtime
---------------------------

.. code-block:: python

   from microquantum import ExecutionRuntime, StatevectorBackend

   runtime = ExecutionRuntime(backend=StatevectorBackend())
   result = runtime.execute(
       problem.circuit,
       shots=problem.num_samples,
       seed=1,
   )
   print(result.counts)