Execution Core
==============

MicroQuantum's execution model turns a :class:`~microquantum.QuantumCircuit`
into real, sampled measurement results through the native contract:

.. code-block:: text

   QuantumCircuit -> Backend.run(circuit, shots, seed) -> ExecutionResult
         (state)          (state evolution + measurement)   (counts/algebra)

The canonical entry point is a backend run with an optional shot count and
deterministic seed:

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   backend = StatevectorBackend()
   result = backend.run(qc, shots=1000, seed=42)
   print(result.counts)      # {'00': 503, '11': 497}   (seed=42)

The result is a :class:`~microquantum.BackendResult` exposing sampled
measurement information (``counts``, ``samples``), state information
(``state`` / ``statevector``) and execution metadata (``shots``, ``seed``,
``backend_name``, ``target_name``, ``metadata``).

Measurement
-----------

Measurements are recorded **explicitly on the circuit** as data; they never
silently alter the gate sequence or execute anything:

.. code-block:: python

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   qc.measure(0)          # measure qubit 0 only
   qc.measure_all()       # measure every qubit

* :meth:`~microquantum.QuantumCircuit.measure` marks a single qubit.
* :meth:`~microquantum.QuantumCircuit.measure_all` marks all qubits.
* ``qc.measurements`` lists the measured qubits **in order**; measured
  classical output follows that same order (left-to-right, big-endian), so
  the ordering is deterministic and documented.
* Measurement annotations survive serialization: JSON (``to_json`` /
  ``from_json``), OpenQASM 2.0 (``qasm`` / ``from_qasm``), IR and
  ``bind_parameters`` / ``__add__`` all preserve them.

When a circuit marks a *subset* of its qubits for measurement, the backend's
counts report only those qubits (e.g. measuring qubit 0 of a two-qubit
circuit yields single-bit keys like ``'0'`` / ``'1'``).  When nothing is
measured, the backend samples the full register for backward compatibility.

Shots
-----

``shots`` is the number of times the output distribution is sampled.  The
backend evolves the state once and samples ``shots`` outcomes by Born's rule;
the totals always add up exactly to the requested ``shots``:

.. code-block:: python

   result = backend.run(qc, shots=1000, seed=42)
   assert sum(result.counts.values()) == 1000

* ``shots`` must be an integer ``>= 1``; invalid values raise ``ValueError``.
* Each shot is an independent random outcome; the raw per-shot outcome
  indices are available as ``result.samples`` (big-endian).

Counts
------

``result.counts`` is a dict mapping bitstrings to shot counts.  Bitstrings
are big-endian: the leftmost character corresponds to the first measured
qubit.  ``result.get_counts()`` is an equivalent method accessor.
``result.probabilities`` normalizes the counts, and
``result.most_frequent()`` returns the top bitstring.

Seeds and reproducibility
-------------------------

Supplying the same ``seed`` reproduces the same sampled counts **on the same
backend**:

.. code-block:: python

   r1 = backend.run(qc, shots=1000, seed=42)
   r2 = backend.run(qc, shots=1000, seed=42)
   assert r1.counts == r2.counts

Reproducibility is guaranteed only for the same backend and execution
configuration; different backends are not guaranteed bit-for-bit matches.

Backends
--------

Simulation backends implement the execution model today:

* :class:`~microquantum.StatevectorBackend` — exact state-vector simulation
  with shot sampling.
* :class:`~microquantum.LocalSimulatorBackend` — the reference local
  simulator.
* :class:`~microquantum.DensityMatrixBackend` — density-matrix simulation
  (including noise).
* :class:`~microquantum.MPSBackend` /
  :class:`~microquantum.TreeTensorNetworkBackend` — tensor-network
  simulators.

Backends that cannot perform the requested operation report a clear
capability limitation through :meth:`~microquantum.Backend.validate` /
:meth:`~microquantum.Backend.supports` rather than returning fabricated
results.  See :doc:`/execution/backends` for the full backend contract.

Execution does **not** imply hardware: only simulator backends are available,
and no claim of real-device execution is made anywhere.