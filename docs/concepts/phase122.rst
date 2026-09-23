Phase 122 — Correctness Hardening & Hardware Readiness
=======================================================

Phase 122 closes correctness gaps found while building Phase 121 and
deepens hardware execution. All changes are additive; the only
behavioral corrections are a fixed amplitude-estimation pipeline
(previously inaccurate beyond trivial cases) and a single-sourced SDK
version string.

W1 — estimator correctness
--------------------------

.. code-block:: python

   import numpy as np
   from microquantum.algorithms.amplitude_estimation import AmplitudeEstimation
   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.operators import Operator

   oracle = QuantumCircuit(1)
   oracle.append(Operator.Z(), [0])
   preparation = QuantumCircuit(1)
   preparation.h(0)
   result = AmplitudeEstimation(num_evaluation_qubits=4).estimate(1, preparation, oracle)
   print(round(result.estimated_amplitude, 6), result.phases)

.. code-block:: python

   from microquantum.algorithms import QuantumCounting

   counter = QuantumCounting()
   print(counter.count(2, [0, 3]).estimated_count)
   print(counter.estimate_count(2, [0, 3]).estimated_count)

W2 — packaging, validation and replay
-------------------------------------

.. code-block:: python

   import microquantum
   from microquantum.problems import SearchProblem
   from microquantum.runtime import runtime_info

   print(microquantum.__version__ == runtime_info().version)
   print(SearchProblem(num_qubits=3, target="101").target)

.. code-block:: python

   from microquantum.providers.replay import ReplayTransport

   transport = ReplayTransport(script=[(200, {"ok": True})])
   print(transport(method="GET", url="https://x", headers=None, body=None))

W3 — execution at scale
-----------------------

.. code-block:: python

   import numpy as np
   from microquantum.backends.array_backend import asarray, eye, matmul, to_numpy

   print(to_numpy(matmul(asarray([[0, 1], [1, 0]]), asarray([[1, 0], [0, -1]]))).tolist())

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.runtime import Budget, DAGScheduler, ExecutionPlan, execute_batch
   from microquantum.runtime.errors import PlanningError

   plans = [ExecutionPlan.from_circuit(QuantumCircuit(1), shots=4) for _ in range(2)]
   print(len(execute_batch(plans, seed=0, scheduler=DAGScheduler(max_parallel=2))))
   try:
       Budget(max_shots=1).check(shots=2)
   except PlanningError as exc:
       print("budget enforced")

W4 — hardware-aware transpiler passes
-------------------------------------

.. code-block:: python

   from microquantum.core.architecture import linear_architecture
   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.transpiler import (
       CommutationAwareCancellation,
       NoiseAwareLayout,
       PassContext,
       SwapRoutingPass,
   )

   source = QuantumCircuit(3)
   source.cx(0, 2)
   context = PassContext()
   routed = SwapRoutingPass(architecture=linear_architecture(3)).transform(source, context)
   print(routed.num_gates, context.analysis["swaps_added"])
   layout_context = PassContext()
   NoiseAwareLayout(linear_architecture(3), {"q0": 0.05, "q1": 0.01, "q2": 0.03}).transform(
       source, layout_context
   )
   print(layout_context.analysis["layout"])

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.core.transpiler import CommutationAwareCancellation, PassContext

   circuit = QuantumCircuit(2)
   circuit.x(1)
   circuit.cx(0, 1)
   circuit.x(1)
   print(CommutationAwareCancellation().transform(circuit, PassContext()).num_gates)

See also ``examples/47_phase122_correctness.py`` through
``examples/49_phase122_compiler.py`` for runnable per-workstream tours.
