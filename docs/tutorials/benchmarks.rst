Benchmarks Tutorial
===================

This tutorial demonstrates how to run quantum benchmarks to evaluate
device performance.

Randomized Benchmarking
-----------------------

Estimate average gate error rate:

.. code-block:: python

   from microquantum import RandomizedBenchmarking

   rb = RandomizedBenchmarking(num_qubits=1, seed=42)
   result = rb.run(sequence_lengths=[1, 2, 4, 8, 16], num_samples=20)

   print(f"Average gate fidelity: {result.average_gate_fidelity:.4f}")
   print(f"Error per gate: {result.error_per_gate:.4f}")

Cross-Entropy Benchmarking
--------------------------

Measure quantum advantage potential:

.. code-block:: python

   from microquantum import CrossEntropyBenchmarking

   xeb = CrossEntropyBenchmarking(
       num_qubits=4,
       depths=[5, 10, 20],
       num_circuits=5,
       seed=42,
   )
   result = xeb.run()

   for depth, fid in zip(result.depths, result.fidelities):
       print(f"Depth {depth}: XEB fidelity = {fid:.4f}")

Quantum Volume
--------------

Measure effective quantum volume:

.. code-block:: python

   from microquantum import QuantumVolumeBenchmark

   qv = QuantumVolumeBenchmark(max_qubits=5, seed=42)
   result = qv.run()
   print(f"Quantum volume: {result.quantum_volume}")

Gate Set Tomography
-------------------

Characterize individual gate fidelity:

.. code-block:: python

   from microquantum import GateSetTomography
   from microquantum.core import Operator

   gst = GateSetTomography([Operator.H(), Operator.X(), Operator.CNOT()])
   result = gst.run()

   for name, fid in zip(result.gate_names, result.gate_fidelities):
       print(f"{name}: fidelity = {fid:.4f}")

CLOPS
-----

Measure circuit execution speed:

.. code-block:: python

   from microquantum import CLOPSBenchmark

   clops = CLOPSBenchmark(num_qubits=3, duration_seconds=2.0)
   result = clops.run()
   print(f"CLOPS: {result.clops:.0f}")
