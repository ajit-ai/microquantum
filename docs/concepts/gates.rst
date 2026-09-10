Gates
=====

MicroQuantum's gate set is implemented from scratch as complex NumPy
matrices.  Every gate is an :class:`~microquantum.Operator`; 1-qubit gates are
``2x2`` unitaries, 2-qubit gates ``4x4`` unitaries, in the big-endian
convention (qubit 0 = most significant).

Single-qubit gates
------------------

* Clifford: :meth:`Operator.X <microquantum.Operator.X>`,
  :meth:`Operator.Y <microquantum.Operator.Y>`,
  :meth:`Operator.Z <microquantum.Operator.Z>`,
  :meth:`Operator.H <microquantum.Operator.H>`,
  :meth:`Operator.S <microquantum.Operator.S>` / ``Sdg``,
  :meth:`Operator.T <microquantum.Operator.T>` / ``Tdg``.
* Rotations: :meth:`Operator.Rx <microquantum.Operator.Rx>`,
  :meth:`Operator.Ry <microquantum.Operator.Ry>`,
  :meth:`Operator.Rz <microquantum.Operator.Rz>` — accept a float angle or a
  symbolic :class:`~microquantum.ParameterExpression`.

Two-qubit gates
---------------

* :meth:`Operator.CNOT <microquantum.Operator.CNOT>` (alias ``cx``) —
  control/target two-qubit ``X``.
* :meth:`Operator.CZ <microquantum.Operator.CZ>` — controlled ``Z``.
* :meth:`Operator.SWAP <microquantum.Operator.SWAP>` — swap two qubits.

Circuit shortcuts
-----------------

The circuit API mirrors these as methods; ``QuantumCircuit.append(op, targets)``
accepts any :class:`~microquantum.Operator`:

.. code-block:: python

   from microquantum import QuantumCircuit, Operator

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)                 # controlled-X
   qc.append(Operator.SWAP(), [0, 1])
   qc.append(Operator.Rz(1.5), [0])

Unitary check and application
-----------------------------

* :func:`~microquantum.apply_gate` applies a unitary to a state vector.
* :func:`~microquantum.expand_operator` embeds a small unitary onto a larger
  register (target + controls).
* :func:`~microquantum.tensor` builds Kronecker products of operators/states.
* :func:`~microquantum.expectation_value` computes ``<psi|O|psi>``.

Custom gates
------------

Any unitary NumPy matrix can be wrapped as an
:class:`~microquantum.Operator` and appended, so custom gates compose with the
built-ins without any registry registration.