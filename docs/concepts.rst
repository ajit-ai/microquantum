Public Programming Model
========================

microquantum exposes a small set of public contracts that compose into a
single execution model. The same interfaces serve local NumPy simulation
today and CPU/GPU/NPU accelerators and quantum hardware in the
future.

Execution flow
--------------

The canonical path from a circuit to a measurement result::

    QuantumCircuit ──run()/submit_circuit()──> Backend ──> Job ──> BackendResult

* :class:`~microquantum.core.circuit.QuantumCircuit` describes the program
  (gates, measurements, bound parameters).
* :class:`~microquantum.backends.base.Backend` executes circuits. Use
  ``backend.run(circuit, shots, seed)`` for a direct
  :class:`~microquantum.backends.base.BackendResult`, or
  ``backend.submit_circuit(circuit)`` to receive a :class:`~microquantum.backends.base.Job`
  that wraps execution in a lifecycle envelope.
* :class:`~microquantum.backends.base.Job` tracks the lifecycle
  (``created -> queued -> running -> completed``, plus ``failed`` /
  ``cancelled``) and exposes :meth:`~microquantum.backends.base.Job.cancel`
  and :meth:`~microquantum.backends.base.Job.metadata`. Local simulators
  return already-completed jobs; hardware providers return jobs that finish
  asynchronously.
* :class:`~microquantum.backends.base.BackendResult` carries state vectors /
  density matrices, measurement counts and probabilities, and execution
  metadata, and serializes to JSON via ``to_dict()`` / ``to_json()``.

Device & target descriptors
---------------------------

:class:`~microquantum.core.device.Device` and :class:`~microquantum.core.device.Target`
describe *where* a job runs and *which* execution constraints it satisfies.
They are capability-oriented, NumPy-free plain data so hardware and
accelerator implementations are not tied to a specific numerical layer.

* ``Backend.device`` — identity, :class:`~microquantum.core.device.DeviceType`,
  qubit capacity and availability of the executing device.
* ``Backend.target`` — supported native gates, qubit count, connectivity,
  measurement/dynamic-circuit capability and shot limits. A simulator
  advertises a universal target via ``Target.universal()``; a hardware
  backend would advertise its native gate set for transpilation.

Hybrid execution runtime
------------------------

The execution runtime (:mod:`microquantum.runtime`) is the coordinator of
the canonical pipeline::

    Program -> ExecutionPlan -> Target -> Backend -> Job -> Execution -> Result

* :class:`~microquantum.runtime.ExecutionPlan` is a declarative, JSON-safe
  description of *what* to run (a circuit, an IR circuit, or an already
  compiled :class:`~microquantum.ir.CompilationResult`), *where* to run it
  (a :class:`~microquantum.core.device.Target` / backend), and *how* (shots,
  parameter bindings, initial state, seed, optimization level, options and
  user metadata).  Plans are plain data — they never execute anything.
* :class:`~microquantum.runtime.ExecutionRuntime` prepares a plan
  (``prepare``), optionally compiles it against a target (``compile``,
  reusing the MQ-03 :class:`~microquantum.ir.Compiler`), submits it to the
  selected backend (``submit``) and collects the completed job into an
  enriched :class:`~microquantum.backends.base.BackendResult` (``execute``).
  Results carry runtime metadata (``job_id``, strategy, target, shots,
  bindings, optimization level, elapsed time and a trace).  Every execution
  is recorded in the runtime's bounded :attr:`history`.
* :class:`~microquantum.runtime.ExecutionStrategy` decides whether a plan
  runs directly (``DIRECT``) or after compilation (``COMPILED``), driven by
  a pluggable handler table so execution modes can be overridden without
  forking the runtime.
* :class:`~microquantum.runtime.ExecutionTrace` records the ordered lifecycle
  of a single execution (``prepared -> bound/compiled -> validated ->
  submitted -> completed/failed``) with per-step timing — the foundation for
  observability without a heavy logging dependency.

The runtime also provides higher-level orchestration:

* ``execute_batch`` / ``submit_batch`` run many plans together; with
  ``raise_on_error=False`` a failing item is reported in place instead of
  abandoning the batch.
* ``run_parameter_sweep`` executes one circuit over many parameter bindings
  (raw floats for single-parameter circuits, or explicit binding dicts).
* ``run_hybrid`` runs a generic classical-quantum workflow from two callables
  (``build(state, step) -> work`` and ``update(state, step, result) ->
  state``).  It implements no algorithm itself — VQE/QAOA-style loops can be
  expressed on top but are intentionally not built in.

The runtime is not a backend: it never re-implements simulation or provider
logic, and a full user-provided :class:`~microquantum.backends.base.Backend`
can be dropped in through a plan's ``backend`` field.

Problems
--------

A *problem* is a plain, JSON-safe description of a computational task —
never an execution or an algorithm.  The SDK ships five concrete problem
types (all subclasses of
:class:`~microquantum.problems.base.Problem`):

* :class:`~microquantum.problems.base.SamplingProblem` — obtain samples
  ``|bitstring> -> probability`` from a circuit's output distribution.
* :class:`~microquantum.problems.optimization.OptimizationProblem` — minimize
  a binary-objective function.  It is created from a
  :class:`~microquantum.optimization.qubo.QUBOProblem`
  (``OptimizationProblem.from_qubo``) or directly from a spin-Ising
  :class:`~microquantum.core.pauli.PauliSum` (``from_ising``), and keeps both
  views in sync: ``energy(bits)`` evaluates the bitstring objective,
  ``cost_hamiltonian()`` exposes the spin Hamiltonian, ``to_qubo()`` recovers
  the binary form, and ``encode_spins`` / ``sample`` move between the two.
* :class:`~microquantum.problems.eigenvalue.HamiltonianProblem` — the
  spectrum of a Hermitian operator
  (:class:`~microquantum.core.operators.Operator` or
  :class:`~microquantum.core.pauli.PauliSum`).
* :class:`~microquantum.problems.eigenvalue.EigenvalueProblem` — like
  HamiltonianProblem, but additionally requests the lowest ``k`` eigenvalues.
* :class:`~microquantum.problems.search.SearchProblem` — find marked items in
  a ``2**num_qubits``-item database, either from an explicit target list or
  from a boolean ``predicate(index: int) -> bool``.

Every problem implements :meth:`Problem.validate`, which *returns a list of
human-readable strings* (empty = valid) instead of raising, so callers can
aggregate diagnostics.  ``to_dict()`` / ``to_json()`` are JSON-safe for every
problem — including Hamiltonian-backed ones, which serialize via
``hamiltonian_to_dict`` into labeled Pauli terms with complex coefficients
or an explicit matrix.  Problems are positional-first where it matters:
``HamiltonianProblem(H)``, ``EigenvalueProblem(H, k=2)`` and
``SamplingProblem(circuit)`` keep their payload first.

Algorithms
----------

:class:`~microquantum.algorithms.base.Algorithm` is the generic algorithm
contract.  Subclassing is optional for new algorithms but recommended:
the base establishes the uniform ``validate(problem) -> solve(problem,
runtime)`` lifecycle and a consistent
:class:`~microquantum.algorithms.base.AlgorithmResult` container.

``Algorithm.validate(problem)`` returns descriptive problems (an empty list
when valid).  ``Algorithm.solve(problem, runtime=None)`` executes the
algorithm for the given problem against the internal state-vector engine or,
when a :class:`~microquantum.runtime.ExecutionRuntime` is passed, through
the MQ-04 runtime pipeline.  Results are typed ``*Result`` dataclasses that
serialize via ``to_dict()`` / ``to_json()``; ``AlgorithmResult`` additionally
carries the algorithm name, the solved problem, the outcome, optimizer info
and free-form ``config`` / ``execution_metadata`` / ``native`` payloads
(the last is excluded from serialization).

Built-in algorithms that follow the problem-driven lifecycle:

* :class:`~microquantum.algorithms.vqe.VQE` — solve an
  :class:`~microquantum.problems.eigenvalue.EigenvalueProblem` with a
  user-provided ansatz + classical optimizer, using the parameter-shift rule
  or operator gradients depending on the Hamiltonian type.
* :class:`~microquantum.algorithms.qaoa.QAOA` — solve an
  :class:`~microquantum.problems.optimization.OptimizationProblem` by
  building a QAOA ansatz from the problem's Ising cost (CNOT parity chains
  for the ZZ terms) and a standard mixer.
* :class:`~microquantum.algorithms.grover.GroverSearch` — amplify the
  marked states of a :class:`~microquantum.problems.search.SearchProblem`,
  with an optional user-supplied oracle callable.
* :class:`~microquantum.algorithms.phase_estimation.PhaseEstimation` —
  estimate the eigenphase of a *unitary* :class:`~microquantum.core.operators.Operator`
  (``from_problem`` / ``solve`` require a unitary Hamiltonian; non-unitary
  cases should use VQE instead).

Each also exposes ``from_problem(problem, ...)`` to configure the algorithm
directly from a problem instance.  A handful of older algorithms
(``AdaptVQE``, ``VQD``, Deutsch-Jozsa, Bernstein-Vazirani, ...) still follow
the classic conventions (``compute_minimum_eigenvalue()`` / ``run()``) but
return the same serializable result shapes.

Optimizers
----------

Classical optimizers in :mod:`microquantum.optimizers` share one abstract
interface (:class:`~microquantum.optimizers.Optimizer`): subclasses implement
``_step``, ``_converged`` and ``max_iter``, and the base provides
``minimize(cost_fn, gradient_fn, initial_params) -> OptimizerResult``.
Gradients are optional — optimizers that can operate finite-differentially
fall back when ``gradient_fn`` is omitted.

* Known-good gradient-free families:
  :class:`~microquantum.optimizers.COBYLA`, :class:`~microquantum.optimizers.NelderMead`.
* Gradient-based: :class:`~microquantum.optimizers.GradientDescent`,
  :class:`~microquantum.optimizers.Adam`, plus the stochastic variants
  :class:`~microquantum.optimizers.SPSA` / :class:`~microquantum.optimizers.QNSPSA`.
* Quasi-Newton: :class:`~microquantum.optimizers.BFGS` (pure NumPy — no SciPy
  dependency); :class:`~microquantum.optimizers.L_BFGS_B` is exposed the same
  way and raises ``ImportError`` on instantiation when SciPy is unavailable.

Variational algorithms (VQE, QAOA, AdaptVQE, VQD) accept any ``Optimizer``,
so the same problem can be solved with different classical routines without
changing algorithm code (see the optimizer-comparison example).

Backends, capabilities & providers
----------------------------------

MQ-06 turns the backend slot into a layered, inspectable contract.  A backend
answers three questions before any work is dispatched:

.. code-block:: text

                         inspect                     validate                  execute
    Algorithm -> ExecutionPlan ------> BackendCapabilities -------> Backend.execute(plan)
                          |                (no work runs)               (binds + runs)
                          v
                  BackendRegistry <------ providers discover backends
                    (register,            (LocalProvider today)
                     resolve by name)

* :class:`~microquantum.backends.base.Backend` is the execution contract.
  Subclasses implement ``run_circuit`` (raw gate matrices) or ``run`` (a bound
  :class:`~microquantum.core.circuit.QuantumCircuit`); the base provides the
  plan-level surface: ``capabilities``, ``validate(plan) -> list[str]``,
  ``supports(plan) -> bool`` and ``execute(plan) -> BackendResult``.
  ``execute`` is the canonical single-call entry point — it validates, binds
  parameters via ``plan.bound()`` and runs. Every backend exposes a JSON-safe
  ``metadata()`` including its capabilities.
* :class:`~microquantum.backends.capabilities.BackendCapabilities` describe
  *what* a backend can do without running anything: a
  :class:`~microquantum.backends.capabilities.TargetClass` (simulator, CPU,
  GPU, quantum hardware, remote, custom), execution-mode tokens
  (statevector / sampling / shots / expectation values / unitary / density
  matrix), circuit-feature tokens, qubit capacity, connectivity, native gates
  and free-form metadata.  They serialize to JSON (``to_dict`` / ``to_json``
  and ``from_dict``), so a runbook or CI check can reason about a backend
  before dispatch; ``merge`` derives the capability intersection of two
  backends.
* :class:`~microquantum.backends.registry.BackendRegistry` is the discovery
  point: ``register`` (duplicates rejected unless ``replace=True``),
  ``get``/``has``/``names``, a settable ``default``, and JSON serialization.
  Plans may name a backend *by string*; the runtime
  (:class:`~microquantum.runtime.ExecutionRuntime`) resolves names through its
  attached registry (falling back to the module-level ``default_registry``).
  Backend selection precedence: ``plan.backend`` (instance or name) >
  the runtime's explicit default > the attached registry default > a lazily
  created ``statevector`` simulator.
* :class:`~microquantum.backends.provider.Provider` is discovery-only — a
  named owner of a backend family, never an executor.
  :class:`~microquantum.backends.provider.LocalProvider` exposes the built-in
  local simulators (``local_simulator``, ``statevector``,
  ``density_matrix``, plus the deterministic ``mock`` stub).
  :class:`~microquantum.backends.local.LocalSimulatorBackend` is the reference
  backend, reusing the existing state-vector engine.
* :class:`~microquantum.backends.adapter.BackendAdapter` is the boundary for
  external/private execution targets: it accepts bound circuits, translates
  them to the vendor wire format, and maps vendor results back into
  :class:`~microquantum.backends.base.BackendResult` — vendor types never leak
  through the SDK surface.  (The existing :mod:`microquantum.providers`
  hardware layer is the optional vendor-facing side of this boundary.)
* :class:`~microquantum.backends.base.BackendResult` gained MQ-06 payload
  fields alongside the classic state vector / density matrix / counts:
  raw ``samples``, labeled ``expectations``, ``eigenvalues``, a JSON-safe
  ``native`` payload, plus the ``shots``, ``seed`` and ``target_name`` of the
  run.  Everything still serializes through ``to_dict()`` / ``to_json()``.

The whole MQ-05 algorithm layer rides this path unchanged: an algorithm builds
a circuit, a plan pins a backend, and the runtime (or ``Backend.execute``
directly) validates, binds, runs and returns an enriched result.

Providers, adapters & hardware
------------------------------

* :class:`~microquantum.providers.HardwareProvider` discovers and owns a
  vendor integration (credentials, REST API); its
  :class:`~microquantum.providers.HardwareBackend` presents the job to the
  :class:`~microquantum.backends.base.Backend` contract, so hardware jobs
  look identical to local ones. Vendor-specific concepts live in
  provider-specific classes (e.g. ``IBMQuantumCredentials``) — never in the
  generic contracts.
* :class:`~microquantum.adapters.base.DomainAdapter` bridges a domain to the
  SDK: ``validate() -> encode() -> backend.run() -> decode()``, returning a
  :class:`~microquantum.adapters.base.QuantumResult`. Industry-specific
  adapters may be published separately on top of this SDK.

Intermediate representation & compilation
-----------------------------------------

Everyone who needs to introspect, optimize and retarget quantum programs uses
the internal IR plus the compilation foundation in ``microquantum.ir``:

* :func:`~microquantum.ir.to_ir` converts a
  :class:`~microquantum.core.circuit.QuantumCircuit` into an
  :class:`~microquantum.ir.IRCircuit` (an ordered list of IR nodes: ``Gate``,
  ``Measurement``, ``Reset``, ``Barrier``, ``ConditionalBlock``).
  :func:`~microquantum.ir.to_ir_dynamic` preserves mid-circuit measurement,
  reset and classically-conditioned blocks of a
  :class:`~microquantum.core.dynamic.DynamicCircuit`.
  :func:`~microquantum.ir.from_ir` rebuilds an executable
  :class:`~microquantum.core.circuit.QuantumCircuit` from compiled IR.
* The IR is MicroQuantum-owned plain data: gate names, integer qubit/classical
  indices and float or symbolic parameters, with no NumPy dependency. Gates,
  measurements, barriers and classically-controlled blocks also carry their
  source information (e.g. the originating circuit index).
* :func:`~microquantum.ir.validate` and
  :func:`~microquantum.ir.assert_valid` perform structural validation
  (qubit/classical-bit ranges, gate names, arity, parameters, conditions).
* :class:`~microquantum.ir.IRPass` is the transformation abstraction;
  :func:`~microquantum.ir.optimize` runs the standard pipeline (identity
  removal, adjacent-inverse cancellation, rotation fusion). Passes are pure —
  they return a new :class:`~microquantum.ir.IRCircuit` and never mutate
  their input.
* :class:`~microquantum.ir.Compiler` compiles a circuit or IR toward an
  MQ-02 :class:`~microquantum.core.device.Target`: it validates, optimizes,
  decomposes non-native gates into the target basis and reports compatibility
  problems. The resulting :class:`~microquantum.ir.CompilationResult` bundles
  source IR, compiled IR, applied passes and diagnostics, and serializes via
  ``to_dict()`` / ``to_json()`` alongside the other result types.

The IR is intentionally hardware-neutral: OpenQASM is treated as an optional
interchange format, never as the canonical representation. Routing, hardware
scheduling and vendor-specific execution plug in downstream of the
:class:`~microquantum.ir.Compiler` in later phases.

Execution results & analytics
-----------------------------

MQ-07 makes each execution inspectable, reproducible-by-configuration and
analysable without inventing new backend output formats.  The canonical flow
(MQ-04/MQ-06 pipeline + MQ-07 record/analysis layer) is:

.. code-block:: text

   ExecutionPlan / Circuit --[ExecutionRuntime]-> Backend -> BackendResult
                              |                       (binds, validates, runs)  |
                              v                                                 |
                      ExecutionRecord  <------------ raw result preserved ------+
                     (id, status, backend, shots, seed,
                      bindings, plan, timing, reproducibility, error)
                              |
              ExperimentalResult (raw records verbatim) -> Analysis
                                    |                  SamplingAnalysis
                                    |                  StateAnalysis
                                    +-- ResultAggregator -- ExpectationAnalysis

* :class:`~microquantum.experiments.record.ExecutionRecord` is the portable
  description of *one actual execution*: plan, backend/target, shots/seed,
  parameter bindings, timing (``total_seconds`` and backend-provided
  ``queue_seconds`` / ``execution_seconds``, ``None`` when unknown), status,
  an :class:`~microquantum.experiments.record.ExecutionFailure` on failure and
  the raw :class:`~microquantum.backends.base.BackendResult`.  The record
  deliberately separates runtime objects from portable metadata:
  ``to_dict()`` stores only JSON-safe data and ``from_dict()`` restores a
  :class:`~microquantum.backends.base.BackendResult` (complex arrays decoded
  back to ``complex128``), so records round-trip without losing analysability.
* :meth:`ExecutionRuntime.execute_records
  <microquantum.runtime.ExecutionRuntime.execute_records>` runs a sequence of
  plans/circuits and returns **one record per input, in order** — batches
  preserve parameter bindings, backend selection and per-item ``batch_id`` /
  ``batch_index`` metadata, and *never drop failures*.  Every exception is
  recorded as a structured :class:`~microquantum.experiments.record.ExecutionFailure`
  (error type, message, execution id, plan name, bindings).
* :class:`~microquantum.experiments.sweep.ParameterSweep` builds deterministic
  grids per parameter (explicit values, ``range``-style steps or
  ``linspace``-style point counts) and expands them as an ordered Cartesian
  product; ``verify``/``validate`` tie a sweep to the base work's parameters
  before any execution.
* :class:`~microquantum.experiments.experiment.Experiment` groups fixed plans
  and sweeps; running it through an
  :class:`~microquantum.runtime.ExecutionRuntime` yields an
  :class:`~microquantum.experiments.experiment.ExperimentResult` whose raw
  records are **preserved verbatim** — aggregation/analysis never destroys
  them.  Records carry a reproducibility fingerprint: a SHA-256 over the
  stable, sorted, JSON-serialized execution configuration (never object memory
  addresses), plus the ``configured_reproducibility`` vs
  ``deterministic_execution`` distinction — hardware / nondeterministic
  backends are never claimed bit-for-bit reproducible.
* Analysis is backend-independent and consensus-based, consuming existing
  result contracts rather than introducing new formats:

  * :class:`~microquantum.analysis.sampling.SamplingAnalysis` — measurement
    counts: total shots, outcome probabilities, most likely outcome, Shannon
    entropy, marginals, and observable mean/variance/std with a documented
    default (unsigned-binary integer, MSB first) or a caller-supplied
    ``value_of``.
  * :class:`~microquantum.analysis.expectation.ExpectationAnalysis` — the
    existing ``BackendResult.expectations`` ``{label: value}`` contract:
    per-label mean/variance/std/standard error and a parameter-to-expectation
    mapping.
  * :class:`~microquantum.analysis.state.StateAnalysis` — statevectors
    (normalization, probabilities, most probable state, expectation of a
    *diagonal* observable) and density matrices (trace, purity, diagonal
    measurement probabilities).
  * :class:`~microquantum.analysis.aggregation.ResultAggregator` — grouping by
    parameter bindings, backend, status or dotted-path accessors while
    preserving the original records (raw results -> aggregation -> derived
    analysis; never raw results -> replace with summary).
  * :class:`~microquantum.analysis.statistics` — population-vs-sample variance
    (``ddof``), standard error, confidence intervals (0.90/0.95/0.99 presets or
    explicit ``z``) and min/max/count, with strict numeric validation.

MQ-07 stays SDK-side and in-memory: there is no database, dashboard, web/CRUD
service, vendor/cloud integration or new quantum algorithm.  It only makes the
existing backend/runtime/algorithm outputs inspectable, serializable and
analysable.