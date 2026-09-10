Extending
=========

MicroQuantum is designed to be extended without forks.  The SDK itself stays
NumPy-only; all hardware/vendor specifics live behind boundaries.

Extension points
----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Slot
     - How to extend
   * - Custom problems
     - Subclass :class:`~microquantum.Problem` (or the concrete problem
       families) and implement ``validate()`` + serialization.
   * - Custom algorithms
     - Subclass :class:`~microquantum.Algorithm`; implement
       ``validate(problem)`` / ``solve(problem, runtime)`` returning an
       ``AlgorithmResult``-style dataclass.  The base establishes the uniform
       lifecycle.
   * - Custom backends
     - Subclass :class:`~microquantum.Backend` (implement ``run`` /
       ``run_circuit``) or :class:`~microquantum.BackendAdapter` for external
       systems (implement ``submit_to_vendor``/``collect_from_vendor``).
     - Advertise what you can do with a
       :class:`~microquantum.BackendCapabilities` and register for name-based
       selection (:class:`~microquantum.BackendRegistry`).
   * - Custom execution strategies
     - ``register_custom_strategy(name, handler)`` to teach the
       :class:`~microquantum.ExecutionRuntime` new work → execution modes.
   * - Custom IR passes
     - Subclass :class:`~microquantum.IRPass` and register with
       :class:`~microquantum.IRPassManager` / the compiler pipeline.
   * - Domain adapters
     - Subclass :class:`~microquantum.DomainAdapter` and follow
       ``validate() -> encode() -> backend.run() -> decode()`` returning a
       :class:`~microquantum.QuantumResult`.
   * - Custom optimizers
     - Subclass :class:`~microquantum.Optimizer`; implement ``_step``,
       ``_converged`` and ``max_iter``.  The base provides
       ``minimize(...) -> OptimizerResult``.

General rules
-------------

1. Vendor types never leak through the SDK surface — map them into
   :class:`~microquantum.BackendResult` / capabilities at the boundary.
2. The SDK must never ``import`` a vendor/enterprise package; extensions are
   optional and live outside ``src/microquantum``.
3. Follow the docstring conventions (Google style) so autoapi documents your
   extension automatically.
4. Keep results typed and serializable — consistency with the core result
   contracts is what makes analysis/experiments work on your extension.