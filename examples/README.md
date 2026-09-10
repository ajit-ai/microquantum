# Examples

This folder contains runnable demonstrations of the microquantum SDK. Run any
example with `uv run python examples/<file>.py` from the repository root.

## Core programming model

- `01_basic_circuits.py` — circuit construction, simulation, drawing, QASM
  and JSON I/O.
- `02_grover_search.py` — Grover's search algorithm end to end.
- `03_vqe_h2.py` — VQE on the H2 Hamiltonian.
- `04_qaoa_maxcut.py` — QAOA on a MaxCut instance.
- `05_noise_simulation.py` — noisy simulation with error channels.

## Intermediate representation & compilation

- `ir/01_circuit_to_ir.py` — converting a `QuantumCircuit` into IR.
- `ir/02_inspect_ir.py` — IR inspection and JSON serialization
  (qubits, depth, gate counts, measurements, conditions).
- `ir/03_optimization_passes.py` — the pure IRPass pipeline
  (identity removal, inverse cancellation, rotation fusion).
- `ir/04_parameterized_binding.py` — carrying symbolic parameters into the
  IR, binding them, and rebuilding an executable circuit.
- `ir/05_dynamic_readiness.py` — mid-circuit measurement, `reset` and
  classically conditioned gates as first-class IR nodes.
- `ir/06_target_aware_compilation.py` — compiling a circuit toward an
  MQ-02 `Target` (decomposition into the native basis, diagnostics).
- `ir/07_custom_pass.py` — extending the compiler with a user-defined
  `IRPass`.

## Hybrid execution runtime

- `execution/01_basic_execution.py` — one-shot execution with the module
  helper `microquantum.execute()`.
- `execution/02_explicit_plan.py` — an explicit serializable `ExecutionPlan`.
- `execution/03_job_lifecycle.py` — `Job` lifecycle metadata and cancellation.
- `execution/04_batch_execution.py` — batch execution with per-item results
  and tolerant `raise_on_error=False` mode.
- `execution/05_parameter_sweep.py` — sweeping a circuit over parameter
  bindings (raw floats and explicit binding dicts).
- `execution/06_hybrid_loop.py` — a generic classical -> quantum ->
  classical loop with `run_hybrid()`.
- `execution/07_custom_backend.py` — plugging a user backend into the
  runtime.
- `execution/08_target_and_backend.py` — target-aware compilation plus
  backend capability checks.
- `execution/09_failure_handling.py` — deterministic plan/backend/batch
  failure reporting.
- `execution/10_end_to_end.py` — full pipeline with IR, plan, target,
  backend, enriched result and JSON exports.

## Algorithms on problems (MQ-05)

The generic `Algorithm`/`Problem` layer: problems are JSON-safe data,
algorithms consume them through the uniform `validate` + `solve` lifecycle,
and the same algorithm can ride the internal engine or the MQ-04 runtime.

- `algorithms/01_problem_hierarchy.py` — the five problem abstractions
  (Sampling, Optimization, Hamiltonian, Eigenvalue, Search), validation and
  JSON serialization.
- `algorithms/02_algorithm_lifecycle.py` — `validate(problem)` /
  `solve(problem, runtime)` lifecycle, reuse of one configured algorithm
  across problems, descriptive validation errors.
- `algorithms/03_algorithm_result_serialization.py` — `AlgorithmResult`
  `to_dict`/`to_json`, excluding the `native` payload.
- `algorithms/04_runtime_separation.py` — problem/algorithm separation and
  runtime-integrated solves.

## Optimization

- `optimization/01_qaoa_maxcut_problem.py` — MaxCut as QUBO ->
  OptimizationProblem -> QAOA, compared against brute force.
- `optimization/02_qaoa_ising_cost.py` — QAOA with a direct Ising
  Hamiltonian cost function.
- `optimization/03_qubo_workflow.py` — QUBO / OptimizationProblem /
  Ising energies agree exactly (round-trip).
- `optimization/04_bfgs_basics.py` — the BFGS quasi-Newton optimizer
  (gradient and gradient-free modes).

## Variational algorithms

- `variational/01_vqe_ground_state.py` — VQE on a 2-qubit ZZ+X Hamiltonian
  using the generic `EigenvalueProblem`.
- `variational/02_vqe_runtime.py` — VQE driving the ExecutionRuntime.
- `variational/03_optimizer_comparison.py` — GradientDescent, Adam, BFGS,
  COBYLA and NelderMead as drop-in optimizers.

## Search

- `search/01_grover_target.py` — Grover over a `SearchProblem` marker.
- `search/02_grover_oracle.py` — a hand-rolled phase-flip oracle circuit.
- `search/03_grover_predicate.py` — `SearchProblem.is_marked` predicate view.
- `search/04_grover_runtime.py` — engine vs runtime execution of the same
  Grover instance.

## Mathematical primitives

- `math/01_qft_circuit.py` — QFT (forward + inverse) fundamentals.
- `math/02_phase_estimation.py` — QPE on engineered unitaries.
- `math/03_hamiltonian_expectation.py` — PauliSum expectations and energy
  landscapes.

## Extending the SDK

- `extension/01_custom_problem.py` — subclassing `Problem` with custom fields
  and validation.
- `extension/02_custom_algorithm.py` — a custom `Algorithm` through the
  generic lifecycle.
- `extension/03_custom_grover_diffusion.py` — custom Grover oracle +
  custom diffusion shape.
- `extension/04_custom_optimizer.py` — a custom `Optimizer` inside VQE.
- `extension/05_custom_runtime.py` — routing a solve through a custom
  `ExecutionRuntime` subclass.

## Backends & providers (MQ-06)

- `backend/01_backend_basic.py` — the canonical plan-level backend contract
  (`validate` / `supports` / `execute`) and runtime agreement.
- `backend/02_backend_capabilities.py` — inspecting and merging
  `BackendCapabilities`, discovering backends through `LocalProvider`.
- `backend/03_backend_registry.py` — `BackendRegistry` registration,
  duplicate/unknown-name handling and JSON serialization.
- `backend/04_backend_selection.py` — backend selection precedence
  (plan > runtime default > registry default > lazy default).
- `backend/05_runtime_backend_execution.py` — a name-selected plan through the
  runtime with the enriched result payload.
- `backend/06_custom_backend.py` — building a custom `Backend` with
  capabilities and the full plan contract.