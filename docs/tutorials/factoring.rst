Shor's Algorithm Tutorial
=========================

This tutorial demonstrates factoring integers using Shor's algorithm.

Basic Factoring
---------------

.. code-block:: python

   from microquantum import ShorsAlgorithm

   # Factor 15
   algo = ShorsAlgorithm(15, seed=42)
   result = algo.run()

   print(f"Factors of {result.n}: {result.factors}")
   print(f"Period found: {result.period}")
   print(f"Base used: {result.a}")
   p, q = result.factors
   assert p * q == 15

Factoring Larger Numbers
-------------------------

.. code-block:: python

   # Factor 35
   algo = ShorsAlgorithm(35, seed=42)
   result = algo.run()
   print(f"35 = {result.factors[0]} x {result.factors[1]}")

   # Factor 21
   algo = ShorsAlgorithm(21, seed=42)
   result = algo.run()
   print(f"21 = {result.factors[0]} x {result.factors[1]}")

Inspecting the Circuit
----------------------

You can also examine the quantum circuit used for order-finding:

.. code-block:: python

   algo = ShorsAlgorithm(15, seed=42)
   qc = algo.build_circuit(a=7)
   print(f"Circuit: {qc.num_qubits} qubits, {qc.num_gates} gates")

Bernstein-Vazirani Algorithm
-----------------------------

Find a hidden bitstring using a single oracle query:

.. code-block:: python

   from microquantum import BernsteinVazirani

   bv = BernsteinVazirani("1011")
   result = bv.run()
   print(f"Secret: {result.secret_string}")
   print(f"Measured: {result.measured}")
   assert result.correct

Deutsch-Jozsa Algorithm
------------------------

Determine if a function is constant or balanced:

.. code-block:: python

   from microquantum import DeutschJozsa

   # Test a balanced function
   dj = DeutschJozsa(n_qubits=3, balanced=True)
   result = dj.run()
   print(f"Function is constant: {result.is_constant}")  # False

   # Test a constant function
   dj = DeutschJozsa(n_qubits=3, balanced=False)
   result = dj.run()
   print(f"Function is constant: {result.is_constant}")  # True
