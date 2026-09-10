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