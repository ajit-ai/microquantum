Examples
========

.. toctree::
   :maxdepth: 2

   bell_state

Bell State
----------

Create and measure a Bell state:

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   state = qc.run()
   print(f"Bell state: {state}")
