Compatibility
=============

Platforms
---------

MicroQuantum is pure Python + NumPy and runs wherever NumPy runs.  CI tests on
Windows, Linux and macOS for Python 3.10 - 3.13:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Python
     - Status
   * - 3.10
     - Supported
   * - 3.11
     - Supported
   * - 3.12
     - Supported (primary development version)
   * - 3.13
     - Supported

Dependencies
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Notes
   * - ``numpy>=1.20``
     - The only hard runtime dependency.
   * - CuPy (``gpu`` extra)
     - Optional; activates a GPU-backed array engine via
       :func:`~microquantum.set_array_backend`.  Requires a local CUDA
       installation.
   * - Sphinx / furo / sphinx-autoapi (``docs`` extra)
     - Optional; only needed to rebuild the reference documentation.

Hard interop
------------

There is **no** runtime dependency on Qiskit, Cirq, PennyLane or OpenQASM.
Interchange is achieved through serialization, not imports:

* JSON (``to_dict`` / ``to_json`` on every result, plan, circuit and problem).
* OpenQASM 2.x import/export on circuits.

Because the SDK never *imports* a competing framework, it installs cleanly
alongside any other quantum stack in the same environment.

Large-state limits
------------------

Statevector / density-matrix backends allocate ``O(2**n)`` complex numbers:

.. code-block:: text

   10 qubits   ~ 16 KiB per statevector
   20 qubits   ~ 16 MiB
   30 qubits   ~ 16 GiB

For shallow, wide circuits at 20+ qubits prefer the MPS / tree-tensor-network
simulators.  Sampling backends trade memory for shots and are the most scalable
option for large ``n``.

Versioning contract
-------------------

* ``0.x`` releases do not guarantee API stability across minor versions while
  the public surface settles.
* Backward-compatible additions land as minor versions; breaking changes as
  major releases after 1.0.
* Serialized result schemas are versioned; new fields are additive.