Troubleshooting
===============

Common issues and their fixes.

``pip install microquantum`` fails to find a version
----------------------------------------------------

Make sure you target PyPI releases (0.4.0+).  A much older public package of
the same name existed before the SDK; installing the current release gives the
SDK, not that old stub.  Verify with:

.. code-block:: console

   python -c "import microquantum; print(microquantum.__version__)"

A circuit fails with "unbound parameters"
-----------------------------------------

Every :class:`~microquantum.Parameter` must be bound before execution.  Use
``qc.bind_parameters({...})`` or pass ``parameter_bindings`` in the
:class:`~microquantum.ExecutionPlan` (or through
:class:`~microquantum.Experiment` / :class:`~microquantum.ParameterSweep`).

``validate()`` returns diagnostics instead of raising
-----------------------------------------------------

That is by design.  Problems / plans / backends respond to ``validate()`` with
a list of human-readable strings (empty = valid).  Check the returned list
before assuming success.

QAOA/VQE give a degenerate "energy" for disconnected problems
-------------------------------------------------------------

An unconstrained objective (no constraints/penalties) can have many optimal
bitstrings — check the returned optimal state and energies rather than the
value alone.  Add penalty terms with the :class:`~microquantum.QUBOBuilder`
(for example ``add_penalty_one_hot``) to enforce feasibility.

Phase estimation reports a non-unitary Hamiltonian
--------------------------------------------------

QPE requires a *unitary* operator.  For the spectrum of a general Hermitian
operator use :doc:`/algorithms/vqe` instead.

``Sphinx`` build fails with warnings-as-errors
----------------------------------------------

The docs build uses ``-W --keep-going``; a single unresolved reference or
nitpick failure breaks the build.  Fix the ``:class:`` / ``:func:`` target (or
the API it references) rather than weakening the flag.  Run
``uv sync --group dev`` first so furo/sphinx-autoapi are present.

Results look non-deterministic
------------------------------

Sampling backends need an explicit ``seed`` for reproducible output:

.. code-block:: python

   backend.run(qc, shots=1024, seed=0)

Configured reproducibility (fingerprint) is guaranteed; *output* determinism
depends on the backend and seed.

Performance is slow for many qubits
-----------------------------------

State vectors scale as ``2**n``.  Use MPS / tensor-network simulators for
shallow circuits at larger sizes, and keep shots reasonable for sampling
backends.  GPU acceleration is opt-in via
:func:`~microquantum.set_array_backend` (requires the ``gpu`` extra).