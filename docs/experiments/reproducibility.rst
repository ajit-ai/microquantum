Reproducibility
===============

MicroQuantum records a **reproducibility fingerprint** for every execution:
a SHA-256 hash over the stable, sorted, JSON-serialized execution
configuration — never over object memory addresses.

.. code-block:: python

   from microquantum import ExecutionRecord, execution_fingerprint

   fp = execution_fingerprint(
       {"plan_name": "ry", "backend": "statevector", "shots": 4096,
        "parameter_bindings": {"theta": 0.5}, "seed": 7}
   )
   print(fp)                       # 64-hex SHA-256

Configured vs deterministic
---------------------------

* ``configured_reproducibility`` — a matching configuration (bindings,
  backend, shots, seed, SDK version) yields the **same fingerprint**; the
  record stays tied to the library that produced it.
* ``deterministic_execution`` — whether the backend actually produces
  identical *output* for the same configuration.  Local simulators with a
  seed are deterministic; hardware / nondeterministic backends are **never**
  claimed bit-for-bit reproducible.

Both live under ``record.reproducibility``:

.. code-block:: python

   for record in result.executions():
       rep = record.reproducibility()
       print(rep["fingerprint"])
       print(rep["configured_reproducibility"])   # True
       print(rep["deterministic_execution"])      # True for seeded simulators

Why configuration-based?
------------------------

Because the fingerprint is derived from the serialized configuration rather
than object identity, records remain comparable across processes, machines
and runs — you can verify that “this run used the same parameters” without
executing anything.  The SDK version is part of the hash so a record stays
tied to the exact library that produced it.