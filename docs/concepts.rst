Public Programming Model
========================

microquantum exposes a small set of public contracts that compose into a
single execution model. The same interfaces serve local NumPy simulation
today and CPU/GPU/NPU accelerators and proprietary quantum hardware in the
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

Algorithms
----------

All public algorithms in :mod:`microquantum.algorithms` share the same
convention: instantiate with the problem definition, execute via
``run()`` (a few older algorithms expose ``solve()`` / ``estimate()`` /
``compute_minimum_eigenvalue()``) and receive a typed ``*Result`` dataclass
supporting ``to_dict()`` / ``to_json()``. :class:`~microquantum.algorithms.base.Algorithm`
documents this contract; subclassing it is optional.

Optimizers
----------

Classical optimizers in :mod:`microquantum.optimizers` share one interface:
``minimize(func, initial_params, ...) -> OptimizerResult``. Variational
algorithms (VQE, QAOA, AdaptVQE, VQD) accept any of these optimizers.

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