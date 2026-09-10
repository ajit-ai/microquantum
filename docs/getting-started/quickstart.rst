Quickstart
==========

Bell state in a few lines
-------------------------

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2)
   qc.h(0)        # Hadamard on qubit 0
   qc.cx(0, 1)    # CNOT (control=0, target=1)

   backend = StatevectorBackend()
   result = backend.run(qc, shots=1024, seed=0)

   print(result.counts)           # {'00': ~512, '11': ~512}
   print(result.most_frequent())  # '00' or '11'

Circuit -> runtime -> result
----------------------------

The :class:`~microquantum.ExecutionRuntime` is the canonical orchestrator; a
declarative :class:`~microquantum.ExecutionPlan` says *what* to run, *where*
and *how*:

.. code-block:: python

   from microquantum import ExecutionPlan, ExecutionRuntime

   plan = ExecutionPlan.from_circuit(qc, backend=backend, shots=1024, seed=0)
   result = ExecutionRuntime().execute(plan)

   print(result.counts)
   print(result.metadata["backend"])

Or use the one-line module helper:

.. code-block:: python

   from microquantum import execute

   result = execute(qc, shots=1024, seed=0)
   print(result.counts)

Problem -> algorithm -> result
------------------------------

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator, Parameter, QuantumCircuit
   from microquantum.algorithms import VQE
   from microquantum.optimizers import GradientDescent

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   vqe = VQE(ansatz, Operator.Z(), GradientDescent(learning_rate=0.3, max_iter=60))
   problem = EigenvalueProblem(Operator.Z(), k=1)

   print(vqe.validate(problem))      # []  (valid)
   result = vqe.solve(problem, initial_params={theta: 0.5})
   print(result.eigenvalue)          # approaches -1.0

Experiment + analysis
---------------------

.. code-block:: python

   from microquantum import (
       ExecutionRuntime,
       Experiment,
       ExpectationAnalysis,
       MockBackend,
       Parameter,
       ParameterSweep,
       QuantumCircuit,
   )

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   exp = Experiment("rx-sweep", shots=1024, seed=0)
   exp.add_circuit(ansatz, name="theta=0", parameter_bindings={"theta": 0.0})
   exp.add_sweep(ParameterSweep({"theta": [0.5, 1.0]}), base=ansatz)

   exp_result = exp.run(runtime=ExecutionRuntime(backend=MockBackend()))
   analysis = ExpectationAnalysis(exp_result)
   print(analysis.keys())

Next: the :doc:`/concepts/overview` for the full execution model, or jump
straight to :doc:`first-circuit`.