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