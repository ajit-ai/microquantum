Serialization
=============

Every public data type in MicroQuantum serializes to JSON-safe Python
primitives via ``to_dict()`` / ``from_dict()`` and ``to_json()`` /
``from_json()``.  This keeps circuits, problems, results and execution
metadata storable, shareable and reproducible.

What can be serialized
----------------------

* **Circuits** — ``QuantumCircuit.to_json()`` / ``from_json()``, plus
  ``save(path)`` / ``load(path)``; OpenQASM via ``qc.qasm()`` /
  ``QuantumCircuit.from_qasm(...)``.
* **Problems** — every problem (``to_dict()`` / ``to_json()``), including
  Hamiltonian-backed ones: terms serialize as labeled Pauli strings with
  complex coefficients or an explicit matrix.
* **Results** — :class:`~microquantum.BackendResult`,
  :class:`~microquantum.AlgorithmResult`, analysis results
  (:class:`~microquantum.analysis.sampling.SamplingAnalysis`,
  :class:`~microquantum.StateAnalysis`, ...) and experiment results.
* **Execution records** — :class:`~microquantum.ExecutionRecord` round-trips
  runtime objects: ``to_dict()`` stores JSON-safe data, ``from_dict()``
  decodes complex arrays back into ``complex128`` so records restorable
  without losing analysability.
* **Sweeps / experiments** — :class:`~microquantum.ParameterSweep`,
  :class:`~microquantum.Experiment` are JSON-safe by construction.
* **Plans & IR** — :class:`~microquantum.ExecutionPlan` and the IR
  (:class:`~microquantum.CompilationResult`) serialize via ``to_dict()`` /
  ``to_json()``.

Rules
-----

* Complex arrays are encoded losslessly (e.g. state-vector amplitudes,
  density matrices, operator matrices).
* Records separate *runtime objects* from *portable metadata* — only
  JSON-safe fields are emitted.
* Execution fingerprints hash the *stable, sorted, serialized configuration*,
  never object memory addresses (see :doc:`/experiments/reproducibility`).

Example
-------

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   data = qc.to_json()
   restored = QuantumCircuit.from_json(data)

   result = StatevectorBackend().run(qc, shots=256, seed=42)
   result_dict = result.to_dict()          # state array included, JSON-safe
   print(result_dict["shots"])             # 256