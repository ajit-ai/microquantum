Circuits
========

:class:`~microquantum.QuantumCircuit` is the program type.  It holds an
ordered sequence of gate instructions — an :class:`~microquantum.Operator`
applied to a list of integer target qubits — plus optional terminal
measurements.  All state is mutable and built by method chaining.

Building a circuit
------------------

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)          # Hadamard on qubit 0
   qc.cx(0, 1)      # CNOT(control=0, target=1)
   qc.rx(0.5, 0)    # RX(0.5) on qubit 0
   qc.z(1)

   print(qc.num_qubits)   # 2
   print(qc.num_gates)    # 4
   print(qc.depth())      # circuit depth
   print(qc.gates())      # [(Operator, [targets]), ...]

Appending operators
-------------------

General gate application uses :meth:`QuantumCircuit.append`:

.. code-block:: python

   from microquantum import Operator

   qc.append(Operator.CNOT(), [0, 1])
   qc.append(Operator.Ry(1.2), [1])

Shortcut methods (``h``, ``x``, ``y``, ``z``, ``s``, ``sdg``, ``t``, ``tdg``,
``rx``, ``ry``, ``rz``, ``cx``/``cnot``, ``cz``, ``swap``) cover the standard
gate set; see :doc:`gates`.

Queries
-------

* ``num_qubits`` / ``num_gates`` / ``depth()``.
* ``gate_count(gate_type=None)`` — count gates, optionally for one type.
* ``contains_gate(gate_type)`` — does the circuit use a given gate?
* ``gates()`` — raw instruction list ``(Operator, targets)``.
* ``parameters()`` — the set of free :class:`~microquantum.Parameter` s.
* ``is_parameterized()`` — ``True`` when any gate carries a symbolic angle.
* ``to_ir()`` / ``from_ir()`` — convert to/from the IR (see :doc:`/execution/runtime`).

Execution
---------

A circuit is executed through a backend, the execution runtime, or directly
against the internal NumPy engine:

* ``qc.run()`` — direct state-vector evolution, returns a
  :class:`~microquantum.StateVector`.
* ``qc.get_unitary()`` — the ``2**n x 2**n`` matrix of the whole circuit.
* ``qc.expectation_value(observable)`` — expectation of a
  :class:`~microquantum.Operator` / :class:`~microquantum.PauliSum` on the
  circuit's output state.
* ``qc.measure_all()`` + a backend/reporting runtime — sampling-based results
  (see :doc:`/getting-started/first-measurement`).

Operations on circuits
----------------------

* ``qc + other`` — concatenate circuits (creates a new circuit).
* ``qc.inverse()`` — the reverse-order, entry-wise inverse circuit.
* ``qc.bind_parameters({param: value, ...})`` — resolve a parameterized
  circuit to a concrete angle set (see :doc:`parameters`).
* ``qc.qasm()`` / ``QuantumCircuit.from_qasm(...)`` — OpenQASM interchange.
* ``qc.to_json()`` / ``QuantumCircuit.from_json(...)`` and
  ``qc.save(path)`` / ``QuantumCircuit.load(path)`` — persistence.
* ``qc.draw(...)`` — ASCII diagram.
* ``simplify_circuit(qc)`` / ``transpile(qc, basis_gates=...)`` — the circuit
  optimization helpers (see :doc:`/execution/execution-plan`).

Dynamic circuits
----------------

:class:`~microquantum.DynamicCircuit` extends the model with mid-circuit
measurement, ``reset`` and classical control
(``measure``, ``measure_all``, ``reset``, ``c_if``/``classical_if``) and is
supported by the local simulators.

Registers
---------

:class:`~microquantum.QuantumRegister` and :class:`~microquantum.ClassicalRegister`
provide the conventional named-bundle API; the integer-qubit models above are
the primary interface.