Custom Backends
===============

Two extension points let you plug execution targets into the SDK without
touching the core:

1. Subclass :class:`~microquantum.Backend` for a new *local / native*
   execution engine — implement ``run_circuit`` (raw gate matrices) or
   ``run`` (a bound :class:`~microquantum.QuantumCircuit`), and optionally
   override ``capabilities`` / ``validate`` / ``supports`` / ``execute``.
2. Subclass :class:`~microquantum.BackendAdapter` for an *external or vendor*
   system.  Implement the two abstract methods; vendor types never leak
   through the SDK surface.

Adapter example
---------------

.. code-block:: python

   from microquantum import BackendAdapter, BackendResult, QuantumCircuit

   class _FakeVendor:
       """Stand-in for a vendor SDK (never leaks through the SDK surface)."""

       def submit(self, wire, *, shots):
           return {"wire": wire, "shots": shots}          # opaque handle

       def collect(self, handle):
           return {"counts": {"00": handle["shots"]}, "shots": handle["shots"]}

   vendor = _FakeVendor()

   def to_wire(circuit):
       return circuit.qasm()

   class MyVendorBackend(BackendAdapter):
       @property
       def name(self) -> str:
           return "my-vendor"

       def submit_to_vendor(self, circuit, *, shots, seed):
           # serialize the bound circuit to the vendor's wire format
           return vendor.submit(to_wire(circuit), shots=shots)   # opaque handle

       def collect_from_vendor(self, handle, circuit) -> BackendResult:
           # map every vendor result/error back into a BackendResult
           data = vendor.collect(handle)
           return BackendResult(
               num_qubits=circuit.num_qubits,
               backend_name=self.name,
               counts=data["counts"],
               shots=int(data["shots"]),
               metadata={"vendor": "my-vendor"},
           )

   backend = MyVendorBackend()
   bound_circuit = QuantumCircuit(2).h(0).cx(0, 1)
   result = backend.run(bound_circuit, shots=1000, seed=0)
   print(result.counts)

Usage notes
-----------

* Register the backend for name-based selection
  (:class:`~microquantum.BackendRegistry`) or pass the instance straight into
  a plan's ``backend`` field — the runtime resolves either.
* Adapters take responsibility for mapping every vendor error into the SDK's
  validation/execution error style (``ValueError`` naming the backend and the
  failing capability).
* ``run_circuit`` on an adapter raises ``ValueError`` by design — external
  targets can only execute bound circuits through ``run`` / ``execute``.

Design boundary
---------------

Vendor SDKs stay inside the subclass: callers only ever see plans,
:class:`~microquantum.BackendResult` and capabilities.  This keeps
MicroQuantum dependency-free (NumPy-only runtime) and hardware-provider-free
for the SDK.