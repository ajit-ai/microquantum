Backends
========

Execution backends, from the pure-Python simulator to provider adapters.
Full API details live in the generated reference; the links below jump straight
to the relevant module.

* :class:`~microquantum.backends.base.Backend` — the execution interface every
  backend implements.
* :class:`~microquantum.backends.local.LocalStateVectorBackend` — fast
  pure-Python state-vector simulator.
* :class:`~microquantum.backends.array_backend.ArrayBackend` — numpy-array based
  execution.
* :class:`~microquantum.backends.density_matrix.DensityMatrixBackend` —
  mixed-state simulation.
* :class:`~microquantum.backends.noise.NoiseModel` / :class:`~microquantum.backends.noise.NoisyBackend` —
  depolarizing and amplitude-damping noise.
* :mod:`microquantum.backends.capabilities` — declarative device capabilities.
* :mod:`microquantum.backends.executor` — shot-batching execution against the
  same circuit on many inputs.
* :class:`~microquantum.backends.adapter.BackendAdapter` — adapt a remote
  provider to the local :class:`Backend <microquantum.backends.base.Backend>`
  interface.
* :class:`~microquantum.backends.provider.BackendProvider` — provider registry
  (see :doc:`/execution/providers`).

See :doc:`/execution/backends` for the narrative guide on writing and selecting
backends.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/backends/base/index
   /api/microquantum/backends/local/index
   /api/microquantum/backends/array_backend/index
   /api/microquantum/backends/density_matrix/index
   /api/microquantum/backends/noise/index
   /api/microquantum/backends/mps/index
   /api/microquantum/backends/tensor_network/index
   /api/microquantum/backends/capabilities/index
   /api/microquantum/backends/executor/index
   /api/microquantum/backends/adapter/index
   /api/microquantum/backends/provider/index
   /api/microquantum/backends/registry/index
   /api/microquantum/backends/mock/index