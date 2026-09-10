.. _execution-records:

Execution Records
=================

An :class:`~microquantum.ExecutionRecord` is the portable description of
**one actual execution**: plan context, backend/target, shots/seed, parameter
bindings, timing, status and the raw :class:`~microquantum.BackendResult` (or
a structured :class:`~microquantum.ExecutionFailure`).

The record deliberately separates **runtime objects** (the live
``BackendResult``) from **portable execution metadata**: ``to_dict()`` never
embeds live objects, and ``from_dict()`` restores the record including a
JSON-safe copy of the result (complex arrays decoded back to ``complex128``).

Recording via the runtime
-------------------------

:meth:`ExecutionRuntime.execute_records <microquantum.ExecutionRuntime.execute_records>`
runs a sequence of plans/circuits and returns **one record per input, in
order** — batches preserve parameter bindings, backend selection and per-item
``batch_id`` / ``batch_index`` metadata, and *never drop failures*:

.. code-block:: python

   from microquantum import ExecutionRuntime, MockBackend, Parameter, QuantumCircuit

   theta = Parameter("theta")
   qc = QuantumCircuit(1).ry(theta, 0)

   runtime = ExecutionRuntime(backend=MockBackend())
   records = runtime.execute_records(
       [qc], shots=512, parameter_bindings={"theta": [0.0, 0.5]}
   )

   for record in records:
       print(record.status, record.parameter_bindings, record.backend)

Record fields
-------------

* ``id`` / ``execution_id`` — stable identifier.
* ``status`` — :class:`~microquantum.ExecutionStatus` (``completed`` /
  ``failed``); ``record.is_success()``.
* ``backend`` / ``target_name`` — where it ran.
* ``shots`` / ``seed`` / ``parameter_bindings`` — how it ran.
* ``plan`` — a snapshot of the :class:`~microquantum.ExecutionPlan`.
* ``timing`` — ``total_seconds`` plus backend-provided ``queue_seconds`` /
  ``execution_seconds`` (``None`` when unknown).
* ``failure`` — an :class:`~microquantum.ExecutionFailure` on failure.
* ``result`` — the raw :class:`~microquantum.BackendResult` on success
  (accessible via ``record.counts`` / ``record.expectations`` /
  ``record.statevector``).
* ``reproducibility`` — fingerprint + ``configured_reproducibility`` vs
  ``deterministic_execution``.

Failures are first-class
------------------------

Failed executions are never silently dropped: they surface as records with
``status == "failed"`` and a structured :class:`~microquantum.ExecutionFailure`
describing the error type, message, plan name and bindings.