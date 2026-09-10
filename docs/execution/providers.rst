Providers
=========

A :class:`~microquantum.Provider` is *discovery-only*: a named owner of a
backend family, never an executor.

* :class:`~microquantum.LocalProvider` exposes the built-in local simulators
  (``local_simulator``, ``statevector``, ``density_matrix``, plus the
  deterministic ``mock`` stub).
* :class:`~microquantum.HardwareProvider` discovers and owns a *vendor*
  integration (credentials, REST API).  Its
  :class:`~microquantum.HardwareBackend` presents the vendor job through the
  :class:`~microquantum.Backend` contract, so hardware jobs look identical to
  local ones.  Vendor-specific concepts live in provider-specific classes
  (e.g. :class:`~microquantum.IBMQuantumCredentials`,
  :class:`~microquantum.IonQCredentials`) — never in the generic contracts.

.. code-block:: python

   from microquantum import LocalProvider, available_backends

   provider = LocalProvider()
   backends = provider.available_backends()     # local simulator names
   print(backends)
   print(available_backends())

Backend registry
----------------

:class:`~microquantum.BackendRegistry` is the discovery point:
``register`` / ``get`` / ``has`` / ``names``, a settable ``default``, and
JSON serialization.  Plans may name a backend *by string*; the runtime
resolves names through its attached registry (falling back to the module-level
``default_registry``).

.. code-block:: python

   from microquantum import BackendRegistry, MockBackend

   registry = BackendRegistry()
   registry.register("mock", MockBackend())
   print(registry.has("mock"))        # True
   print(registry.names())            # ["mock"]

Hardware (optional boundary)
----------------------------

The :mod:`microquantum.providers` hardware layer is the optional
vendor-facing side of the :class:`~microquantum.BackendAdapter` boundary
(:doc:`/execution/custom-backends`); the SDK never imports vendor SDKs.