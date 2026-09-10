Experiments
===========

An :class:`~microquantum.Experiment` groups fixed plans *and* parameter
sweeps into a single runnable unit.  Execution is delegated entirely to an
existing :class:`~microquantum.ExecutionRuntime` — backend routing logic is
never duplicated.

Usage
-----

.. code-block:: python

   from microquantum import (
       ExecutionRuntime,
       Experiment,
       MockBackend,
       Parameter,
       ParameterSweep,
       QuantumCircuit,
   )

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   exp = Experiment(
       "rx-overview",
       description="RX gate sweep",
       backend=MockBackend(),
       shots=1024,
       seed=7,
   )
   exp.add_circuit(ansatz, name="theta=0", parameter_bindings={"theta": 0.0})
   exp.add_sweep(ParameterSweep({"theta": [0.5, 1.0, 2.0]}), base=ansatz)

   print(f"planned executions: {exp.execution_count}")   # 1 + 3 == 4
   result = exp.run(ExecutionRuntime(backend=MockBackend()))

   print(result.status)                # "completed"
   print(len(result.executions()))     # 4 records, order preserved
   print(result.success_count)         # 4
   print(result.failure_count)         # 0
   print(result.all_successful)        # True

API surface
-----------

* ``add_plan(plan)`` / ``add_plans(plans)`` — fixed
  :class:`~microquantum.ExecutionPlan` s.
* ``add_circuit(circuit, *, name=..., backend=..., shots=..., seed=...,
  parameter_bindings=...)`` — one execution honoring experiment defaults.
* ``add_sweep(sweep, base=None)`` — expand a
  :class:`~microquantum.ParameterSweep`; ``base`` may be a plan or circuit
  (defaults to the first fixed plan).
* ``execution_count`` — how many executions the current configuration will
  produce; ``records`` — accumulated after ``run``.
* ``run(runtime)`` -> :class:`~microquantum.ExperimentResult` — executes
  through ``runtime.execute_records``; throws ``ValueError`` if the
  experiment has nothing to run.

Design notes
------------

* Every expansion is a full :class:`~microquantum.ExecutionPlan`, so sweeps
  inherit backend/target/shots/seed/optimization settings.
* Raw execution records are **preserved verbatim** in the result — analysis
  never replaces them with summaries.
* :func:`~microquantum.run_experiment` wraps the same flow on
  ``default_runtime``.