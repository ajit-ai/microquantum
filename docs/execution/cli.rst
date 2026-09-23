Command Line
============

The ``microquantum`` console script (also available as ``python -m
microquantum``) is a thin developer-facing wrapper around the *same canonical
public Python APIs* used by programs — it has no execution logic of its own
and adds no dependencies beyond the Python standard library.

.. code-block:: console

   $ microquantum --version
   microquantum 1.1.0

   $ microquantum info
   MicroQuantum 1.1.0
   Runtime:   ExecutionRuntime
   Python:    3.12.8
   NumPy:     2.2.1
   Strategies: batch, compiled, direct, hybrid, parameter_sweep
   Backends:  local_simulator
   Default:   local_simulator

Commands
--------

``version`` / ``--version``
   Print the installed MicroQuantum version and exit.

``info``
   Print the runtime introspection snapshot produced by
   :func:`~microquantum.runtime.info.runtime_info`: implementation, interpreter
   and NumPy versions, supported execution strategies, registered backends and
   the default backend for plans that name none.

``backends [--json]``
   List registered backends with their
   :class:`~microquantum.backends.capabilities.BackendCapabilities` summaries
   (target class and execution modes).  ``--json`` emits the registry's
   ``to_dict()`` form as JSON for machine consumption.

``run FILE [--shots N] [--seed N] [--backend NAME] [--optimization-level {0,1,2}]``
   Execute an OpenQASM 2.0 circuit file through
   :func:`~microquantum.runtime.execute` and print the measurement counts:

   .. code-block:: console

      $ microquantum run bell.qasm --shots 1024 --seed 1
      Backend:     statevector
      Shots:       1024
      Distinct:    2
        |00>: 501
        |11>: 523

   ``--backend`` resolves through the registry (the default runtime falls back
   to a lazily-created ``statevector`` simulator).  ``--seed`` makes repeated
   runs reproducible.

Behaviour
---------

* Ordinary user errors (missing file, invalid QASM, unknown backend, bad
  arguments) print a single ``microquantum: error: ...`` line to stderr and
  exit non-zero — no traceback noise.
* ``--debug`` (a global flag placed before the command) restores the full
  traceback for diagnosis.
* ``microquantum`` with no command prints usage and exits ``0``; an unknown
  command exits ``2``.

The implementation lives in :mod:`microquantum._cli` — a private module that
is not part of the public API surface (see
:doc:`/developer-guide/package-ecosystem`).