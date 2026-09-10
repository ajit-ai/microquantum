Compilation and Routing Tutorial
=================================

This tutorial demonstrates how to use the microquantum transpiler to compile
circuits for hardware with limited qubit connectivity.

Coupling Maps
-------------

A ``CouplingMap`` defines which qubit pairs can interact directly:

.. code-block:: python

   from microquantum import CouplingMap

   # Linear chain: 0-1-2-3
   linear = CouplingMap.linear(4)

   # 2x3 grid
   grid = CouplingMap.grid(2, 3)

   # All-to-all connectivity
   full = CouplingMap.all_to_all(5)

   # Check connectivity
   print(linear.are_connected(0, 1))  # True
   print(linear.are_connected(0, 3))  # False

Routing
-------

When a circuit has two-qubit gates between non-adjacent qubits, the routing
pass automatically inserts SWAP gates:

.. code-block:: python

   from microquantum import QuantumCircuit, PassManager
   from microquantum.core.coupling import CouplingMap
   from microquantum.core.transpiler import RoutingPass

   # Create a circuit that needs routing
   qc = QuantumCircuit(4)
   qc.h(0)
   qc.cx(0, 3)  # qubits 0 and 3 are not adjacent in linear topology

   # Route for linear connectivity
   cmap = CouplingMap.linear(4)
   pm = PassManager()
   pm.append_pass(RoutingPass(cmap))
   routed = pm.run(qc)
   print(f"Original: {qc.num_gates} gates")
   print(f"Routed: {routed.num_gates} gates (includes SWAPs)")

Noise-Aware Placement
---------------------

When noise rates vary across device edges, the noise-aware pass schedules
low-noise gates first:

.. code-block:: python

   from microquantum.core.transpiler import NoiseAwarePlacementPass

   noise = {
       (0, 1): 0.01,  # low noise
       (1, 2): 0.05,  # medium noise
       (2, 3): 0.10,  # high noise
   }
   pm = PassManager()
   pm.append_pass(NoiseAwarePlacementPass(cmap, noise))
   optimized = pm.run(qc)

Optimization Levels
-------------------

The ``PassManager.from_optimization_level()`` method provides preset
compilation pipelines:

.. code-block:: python

   # Level 0: no optimization
   pm0 = PassManager.from_optimization_level(0)

   # Level 2: full optimization + routing
   pm2 = PassManager.from_optimization_level(2, coupling_map=cmap)

   # Level 3: aggressive + noise-aware
   pm3 = PassManager.from_optimization_level(
       3, coupling_map=cmap, noise_rates=noise
   )
