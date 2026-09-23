Phase 121 — SDK Extension Surface
=================================

Phase 121 adds one additive extension layer across the whole SDK. Every
addition follows the same rules: additive-only APIs, explicit ``__all__``
exports, tests, runnable documentation and no new hard dependencies.

Workstream W1 — contracts and shared protocols
----------------------------------------------

.. code-block:: python

   import numpy as np
   from microquantum.problems import ConstrainedOptimizationProblem, LinearConstraint

   constraint = LinearConstraint(indices=(0, 1), sense="<=", rhs=1.0, penalty=5.0)
   problem = ConstrainedOptimizationProblem(
       num_variables=2, objective=lambda bits: float(bits[0]), constraints=[constraint]
   )
   print(problem.validate(), problem.penalty(np.array([1, 1])))

.. code-block:: python

   from microquantum.problems import ExcitedStateProblem, TimeEvolutionProblem

   print(TimeEvolutionProblem(time=1.5, num_steps=4).validate())
   print(ExcitedStateProblem(num_states=3, k=2).num_states)

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.runtime import Budget, ExecutionPlan

   plan = ExecutionPlan.from_circuit(QuantumCircuit(1), budget=Budget(max_shots=10))
   print(plan.cacheable, plan.to_dict()["budget"]["max_shots"])

.. code-block:: python

   from microquantum.core.parameter import Parameter
   from microquantum.optimizers import Bounds, GradientDescent, minimize_with_callbacks

   theta = Parameter("t121")
   bounds = Bounds(lower=0.0, upper=1.0)
   print(bounds.project({theta: 1.5}))
   result = minimize_with_callbacks(
       GradientDescent(learning_rate=0.5, max_iter=30),
       lambda params: float(params[theta] ** 2),
       gradient_fn=lambda params: {theta: 2.0 * float(params[theta])},
       initial_params={theta: 1.0},
   )
   print(round(result.optimal_value, 6))

.. code-block:: python

   from microquantum.mitigation import MitigationData, ZNEProtocol

   outcome = ZNEProtocol().mitigate(
       MitigationData(noisy_values=[1.0, 1.2, 1.4], noise_factors=[1.0, 3.0, 5.0])
   )
   print(round(outcome.mitigated_value, 6), outcome.method)

Workstream W2 — compilation and execution
-----------------------------------------

.. code-block:: python

   from microquantum.ir import (
       Gate,
       IRCircuit,
       Loop,
       Switch,
       assert_valid,
       loop_from_dict,
   )
   from microquantum.ir.control import Condition

   circuit = IRCircuit(num_qubits=2)
   circuit.add(Loop(body=(Gate(name="h", qubits=(0,)),), trip_count=2))
   circuit.add(Switch(condition=Condition(bit=0, value=1), cases=((1, (Gate(name="x", qubits=(1,)),)),)))
   assert_valid(circuit)
   print(circuit.num_gates, loop_from_dict(circuit[0].to_dict()).trip_count)

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.ir import AliasAnalysis, Compiler, CostModel, to_ir

   bell = QuantumCircuit(2)
   bell.h(0).cx(0, 1)
   model = CostModel(gate_costs={"h": 2.0, "cnot": 5.0})
   compiled = Compiler(cost_model=model).compile(bell)
   print(compiled.metadata["estimated_cost"], model.breakdown(to_ir(bell)))
   analysis = AliasAnalysis()
   analysis.run(to_ir(bell))
   print(analysis.interacting_pairs())

.. code-block:: python

   from microquantum.backends import AsyncJob, CalibrationData, JobStatus, RetryPolicy, with_retry

   print(CalibrationData(gate_errors={"cx": 0.01}).to_dict()["gate_errors"])
   job = AsyncJob(poll_interval_s=0.0)
   print(job.poll(lambda: JobStatus.COMPLETED))
   attempts = {"n": 0}

   def flaky():
       attempts["n"] += 1
       if attempts["n"] < 2:
           raise ConnectionError("down")
       return "up"

   print(with_retry(RetryPolicy(max_attempts=2, backoff_s=0.0), flaky))

.. code-block:: python

   from microquantum.core.circuit import QuantumCircuit
   from microquantum.runtime import DAGScheduler, ExecutionPlan, ResultCache

   plans = [ExecutionPlan.from_circuit(QuantumCircuit(1), shots=10) for _ in range(3)]
   scheduler = DAGScheduler(max_parallel=2)
   batches = scheduler.schedule(plans, dependencies={2: {0}})
   print([(b.level, b.indices) for b in batches], scheduler.depth(plans, dependencies={2: {0}}))
   cache = ResultCache(max_entries=4)
   cache.put(plans[0], {"value": 1})
   print(cache.get(plans[0]))

Workstream W3 — domain methods I
--------------------------------

.. code-block:: python

   from microquantum.algorithms import InitialPoint, QPEPhaseFilter, QuantumCounting, initial_parameters
   from microquantum.core.parameter import Parameter

   print(QuantumCounting().count(2, [0, 3]).estimated_count)
   print(QPEPhaseFilter(kappa=1.0).rotation_angles(2))
   print(initial_parameters([Parameter("a")], InitialPoint(strategy="zeros")))

.. code-block:: python

   from microquantum.analysis import HypothesisTest, bootstrap_ci

   outcome = HypothesisTest(alpha=0.05, permutations=50, seed=0).compare(
       {"00": 90, "11": 10}, {"00": 10, "11": 90}
   )
   print(outcome.significant, round(outcome.p_value, 4))
   print([round(v, 3) for v in bootstrap_ci([1.0, 2.0, 3.0, 4.0], resamples=50, seed=0)])

.. code-block:: python

   import numpy as np
   from microquantum.optimization import PUBOBuilder, QUBOBuilder, qubo_to_pauli_sum
   from microquantum.optimization.ising_pauli import pubo_to_qubo_projection

   builder = PUBOBuilder(2)
   builder.add_quadratic(0, 1, -2.0)
   builder.add_term((0, 1), 0.5)
   print(builder.build().energy(np.array([1, 1])))
   qb = QUBOBuilder(2)
   qb.add_quadratic(0, 1, 2.0)
   print(qubo_to_pauli_sum(qb.build()).num_qubits)
   cubic = PUBOBuilder(3)
   cubic.add_term((0, 1, 2), 4.0)
   print(pubo_to_qubo_projection(cubic.build()).metadata["dropped_higher_order"])

Workstream W4 — domain methods II
---------------------------------

.. code-block:: python

   import numpy as np
   from microquantum.qml import DataReuploadingClassifier, QuantumKernel, kernel_alignment

   clf = DataReuploadingClassifier(num_features=1, layers=1)
   print(clf.predict([[0.0], [1.0]]).predictions)
   matrix = QuantumKernel().evaluate([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])
   print(round(float(kernel_alignment(matrix, [0, 1])), 4))

.. code-block:: python

   from microquantum.qec import LookupDecoder, SteaneCode, Syndrome

   code = SteaneCode()
   print(code.decode_syndrome([0, 0, 1, 0, 0, 0]), code.encode_circuit().num_qubits)
   decoder = LookupDecoder(table={(0, 0, 1): [(0, "X")]}, code_name="steane")
   print(decoder.decode(Syndrome(bits=(0, 0, 1), code_name="steane")))

.. code-block:: python

   import numpy as np
   from microquantum.chemistry import ActiveSpace, FermionicOp, jordan_wigner

   print(ActiveSpace(num_core_orbitals=1, num_active_orbitals=2).select(6, 4).num_qubits)
   number = jordan_wigner(FermionicOp({(("+", 0), ("-", 0)): 1.0}), 1)
   print(np.round(number.to_operator().matrix.real, 6).tolist())

.. code-block:: python

   from microquantum.mitigation import CliffordDataRegression

   cdr = CliffordDataRegression()
   print(cdr.train([0.5, 0.7, 0.9], [0.6, 0.8, 1.0])["r_squared"])
   print(round(cdr.mitigate(0.7), 6))

Workstream W5 — experience and hardware
---------------------------------------

.. code-block:: python

   from microquantum.providers import ProviderCredentials, ProviderErrorMapper
   from microquantum.providers.base import HardwareStatus

   print(ProviderCredentials(api_token="t", proxy="http://p:8080").proxy)
   print(ProviderErrorMapper().to_status({"status": "running"}))

.. code-block:: python

   from microquantum.analytics import ReportBuilder, Result, to_records

   result = Result(problem="demo", solution={"bits": "01"}, confidence=0.9)
   print(to_records(result)[0])
   print(ReportBuilder(title="T").add_result("Outcome", result).to_markdown().splitlines()[0])

.. code-block:: python

   from microquantum.benchmarks import BenchmarkSuite, MirrorBenchmarking

   print(round(MirrorBenchmarking.polarization({"00": 90, "11": 10}, 2), 4))
   suite = BenchmarkSuite(seed=1)
   suite.add("m", MirrorBenchmarking(num_qubits=1, depths=(1,), num_circuits=1, num_shots=32, seed=0))
   print(suite.run().summary())

.. code-block:: python

   import tempfile
   from pathlib import Path
   from microquantum.experiments import AdaptiveSweep, Checkpoint, ParameterSweep

   with tempfile.TemporaryDirectory() as tmp:
       path = Path(tmp) / "checkpoint.json"
       saved = Checkpoint("e1")
       saved.mark_done("a")
       saved.save(path)
       print(Checkpoint.load(path).completed)
   adaptive = AdaptiveSweep(ParameterSweep({"theta": [0.0, 1.0, 2.0]}))
   print(len(adaptive.refine([({"theta": 0.0}, 1.0), ({"theta": 1.0}, 0.0)]).combinations()))

Workstream W6 — foundations
---------------------------

.. code-block:: python

   import math
   import numpy as np
   from microquantum.core.gates import ControlledUnitary, H
   from microquantum.core.information import concurrence, entanglement_entropy
   from microquantum.stdlib import dicke_state, graph_state, gray_code

   print(gray_code(2), round(float(abs(dicke_state(3, 1).amplitudes[1])), 6))
   print(graph_state([(0, 1)], 2).is_normalized)
   print(ControlledUnitary(H().to_matrix()).num_qubits)
   bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
   print(round(entanglement_entropy(bell, [0]), 6), round(concurrence(bell), 6))

See also ``examples/41_phase121_contracts.py`` through
``examples/46_phase121_foundations.py`` for runnable per-workstream tours.
