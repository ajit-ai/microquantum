Getting Started
===============

Installation
------------

Install microquantum using pip:

.. code-block:: bash

   pip install microquantum

Or from source:

.. code-block:: bash

   git clone https://github.com/ajit-ai/microquantum.git
   cd microquantum
   pip install -e .

Quick Start
-----------

Create your first quantum circuit:

.. code-block:: python

   from microquantum import QuantumCircuit, Operator

   # Create a 2-qubit circuit
   qc = QuantumCircuit(2)

   # Add gates
   qc.h(0)       # Hadamard on qubit 0
   qc.cx(0, 1)   # CNOT (control=0, target=1)

   # Run the circuit
   state = qc.run()
   print(f"State: {state}")

   # Get the unitary matrix
   unitary = qc.get_unitary()
   print(f"Unitary shape: {unitary.shape}")

Using the Result Contract
-------------------------

Every solver run can produce a standardized decision result:

.. code-block:: python

   from microquantum.analytics.result import Result

   result = Result(
       problem="scheduling_optimization",
       decision={"selected_route": "A->C->B", "cost": 42.0},
       confidence=0.91,
       qubit_count=8,
       runtime_ms=48.3,
   )
   print(result.to_json())

The SDK ships the contract itself; domain-specific solvers built on
microquantum may be published separately.
