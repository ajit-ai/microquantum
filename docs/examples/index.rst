Examples
========

The ``examples/`` directory ships with runnable demo scripts across ten
categories.  This page embeds the canonical snippets; run any demo directly:

.. code-block:: console

   python examples/execution/01_basic_execution.py

Consolidated notebook
---------------------

The individual ``.py`` files are the **canonical** examples.  As a convenience
layer, the repository also ships a single **consolidated Jupyter notebook**
that walks through the entire corpus:

.. code-block:: console

   examples/MicroQuantum_Examples.ipynb

Every notebook cell simply executes the corresponding authoritative example
file (through a small :code:`run_example` helper defined in the notebook's
setup cell), so the notebook can never drift from the examples.  Open it with
any Jupyter client (for example ``jupyter lab`` or VS Code) from the
repository root:

.. code-block:: console

   jupyter lab examples/MicroQuantum_Examples.ipynb

The notebook is validated by the conformance suite (JSON structure, full
example coverage, no missing file references, and in-order execution of every
code cell) with no Jupyter runtime dependency on the SDK itself.

.. toctree::
   :maxdepth: 2

   bell_state

.. note::

   The examples are exercised in CI like tests: every demo in ``examples/``
   is executed against the *installed* package to keep the SDK honest.

Example tree (all current examples)
-----------------------------------

.. code-block:: text

   examples/
   ├── 01_basic_circuits.py
   ├── 02_grover_search.py
   ├── 03_vqe_h2.py
   ├── 04_qaoa_maxcut.py
   ├── 05_noise_simulation.py
   ├── 12_parameterized_circuit.py
   ├── 13_multiple_parameters.py
   ├── 14_parameter_sweep.py
   ├── 15_parameter_shift_gradient.py
   ├── 16_gradient_descent_loop.py
   ├── 17_statevector_simulation.py
   ├── 18_density_matrix_simulation.py
   ├── 19_noise_through_executor.py
   ├── 20_tensor_network_simulation.py
   ├── 21_cross_simulator_comparison.py
   ├── 22_ir_roundtrip.py
   ├── 23_ir_to_circuit.py
   ├── 24_basic_compilation.py
   ├── 25_optimization_before_after.py
   ├── 26_parameterized_compilation.py
   ├── 27_measurement_preserving_optimization.py
   ├── 28_target_validation.py
   ├── 29_execute_compiled_circuit.py
   ├── rsa_vulnerability_assessment.py
   ├── algorithms/            VQE, QAOA, Grover, phase estimation, ...
   │   ├── 01_problem_hierarchy.py
   │   ├── 02_algorithm_lifecycle.py
   │   ├── 03_algorithm_result_serialization.py
   │   └── 04_runtime_separation.py
   ├── analysis/              expectation & sampling analysis
   │   ├── 01_execution_record.py
   │   ├── 02_batch_execution_records.py
   │   ├── 03_parameter_sweep.py
   │   ├── 04_experiment_run.py
   │   ├── 05_sampling_analysis.py
   │   ├── 06_expectation_analysis.py
   │   ├── 07_state_analysis.py
   │   ├── 08_result_aggregation.py
   │   ├── 09_reproducibility_serialization.py
   │   └── 10_failure_handling.py
   ├── backend/               local backends, capabilities, custom backends
   │   ├── 01_backend_basic.py
   │   ├── 02_backend_capabilities.py
   │   ├── 03_backend_registry.py
   │   ├── 04_backend_selection.py
   │   ├── 05_runtime_backend_execution.py
   │   └── 06_custom_backend.py
   ├── execution/             runtime-plan-backend execution end to end
   │   ├── 01_basic_execution.py
   │   ├── 02_explicit_plan.py
   │   ├── 03_job_lifecycle.py
   │   ├── 04_batch_execution.py
   │   ├── 05_parameter_sweep.py
   │   ├── 06_hybrid_loop.py
   │   ├── 07_custom_backend.py
   │   ├── 08_target_and_backend.py
   │   ├── 09_failure_handling.py
   │   ├── 10_end_to_end.py
   │   ├── 11_single_qubit_measurement.py
   │   ├── 12_bell_state_measurement.py
   │   └── 13_seeded_execution.py
   ├── extension/             custom problem / algorithm / optimizer / runtime
   │   ├── 01_custom_problem.py
   │   ├── 02_custom_algorithm.py
   │   ├── 03_custom_grover_diffusion.py
   │   ├── 04_custom_optimizer.py
   │   └── 05_custom_runtime.py
   ├── ir/                    IR, compilation, transpiler passes
   │   ├── 01_circuit_to_ir.py
   │   ├── 02_inspect_ir.py
   │   ├── 03_optimization_passes.py
   │   ├── 04_parameterized_binding.py
   │   ├── 05_dynamic_readiness.py
   │   ├── 06_target_aware_compilation.py
   │   └── 07_custom_pass.py
   ├── math/                  QFT, phase estimation, expectation values
   │   ├── 01_qft_circuit.py
   │   ├── 02_phase_estimation.py
   │   └── 03_hamiltonian_expectation.py
   ├── optimization/          QUBO builders, Ising conversion, optimizers
   │   ├── 01_qaoa_maxcut_problem.py
   │   ├── 02_qaoa_ising_cost.py
   │   ├── 03_qubo_workflow.py
   │   └── 04_bfgs_basics.py
   ├── search/                Grover search / oracle / predicate / runtime
   │   ├── 01_grover_target.py
   │   ├── 02_grover_oracle.py
   │   ├── 03_grover_predicate.py
   │   └── 04_grover_runtime.py
   └── variational/           VQE workflows
       ├── 01_vqe_ground_state.py
       ├── 02_vqe_runtime.py
       └── 03_optimizer_comparison.py

Bell State
----------

Create and measure a Bell state:

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   state = qc.run()
   print(f"Bell state: {state}")