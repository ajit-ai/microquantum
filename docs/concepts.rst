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
  (``pending -> running -> completed``, plus ``failed``/``cancelled``) and
  exposes :meth:`~microquantum.backends.base.Job.cancel` and
  :meth:`~microquantum.backends.base.Job.metadata`. Local simulators return
  already-completed jobs; hardware providers return jobs that finish
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