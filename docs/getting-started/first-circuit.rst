First Circuit
=============

A :class:`~microquantum.QuantumCircuit` is the program you push through the
SDK.  Construct it, add gates, then run it.

Create and inspect
------------------

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)     # two qubits, |00>
   qc.h(0)                    # Hadamard on qubit 0
   qc.cx(0, 1)                # CNOT, control=0 target=1

   print(qc.num_qubits)       # 2
   print(qc.depth())          # 2
   print(qc)                  # textual circuit drawing

Unitary and state
-----------------

.. code-block:: python

   unitary = qc.get_unitary()
   print(unitary.shape)              # (4, 4)

   state = qc.run()
   print(state)                      # StateVector: ~|00> + |11>

Measurements
------------

Shots sample the output distribution.  The same circuit run through a backend
counts samples; see :doc:`first-measurement`.

Parameters
----------

Circuits can carry symbolic parameters that are bound before execution:

.. code-block:: python

   from microquantum import Parameter, QuantumCircuit

   theta = Parameter("theta")          # symbolic parameter
   pqc = QuantumCircuit(1)
   pqc.rx(theta, 0)

   bound = pqc.bind_parameters({theta: 0.5})   # new circuit, theta -> 0.5
   print(bound)

See :doc:`/concepts/parameters` for details on parameters, binding and sweeps.

Next: :doc:`first-measurement`.