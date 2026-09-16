"""Conformance levels B and C - expected results of official examples.

Every official example must print meaningful output that never leaks the
repr of a bound method (a method used as a property).  A targeted registry
asserts semantic results (level C) for the deterministic and
mathematically-justified outputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import conformance_helpers as h
import pytest

_EXAMPLES: list[tuple[str, Path]] = [
    (str(p.relative_to(h.REPO_ROOT)).replace("\\", "/"), p)
    for p in h.all_example_files()
]


def _ids() -> list[str]:
    return [rel for rel, _ in _EXAMPLES]


# Registry: rel-path -> list of callables/stdout substring assertions.
# Callables receive the completed-process object; substrings must appear.
REGISTRY: dict[str, list] = {
    "examples/01_basic_circuits.py": [
        "All examples completed!",
        "|00>: 0.5000",
        "|11>: 0.5000",
        "|1111>: 0.5000",
        "OPENQASM 2.0",
        "Imported back: 2 qubits, 3 gates",
        "Loaded from JSON: 2 qubits, 3 gates",
        "depth 3",  # method-contract form qc.depth()
        "Depth: 6",  # method-contract form qc.depth()
    ],
    "examples/02_grover_search.py": [
        "Target state: |101>",
        "Amplitudes after 2 Grover iterations:",
        "|101>: 0.9453 <-- TARGET",
        "Target probability: 0.9453",
    ],
    "examples/03_vqe_h2.py": [
        lambda p: "Ground state energy:" in p.stdout,
        lambda p: "Exact value:" in p.stdout or "Error:" in p.stdout,
    ],
    "examples/04_qaoa_maxcut.py": [
        "Ground state energy: 2.5000",
        "Graph: 5 vertices, 5 edges",
        "Parameters: 4",
    ],
    "examples/05_noise_simulation.py": [
        lambda p: re.search(r"fidelity|Fidelity|survival|noise", p.stdout, re.I)
        is not None,
    ],
    "examples/12_parameterized_circuit.py": [
        "unbound parameters: (Parameter('theta'),)",
        "is_parameterized: True",
        "bound.is_parameterized: False",
        "original.is_parameterized: True  (unchanged)",
        lambda p: re.search(r"counts=\{\'[01]\': \d+, \'[01]\': \d+\}", p.stdout)
        is not None,
    ],
    "examples/13_multiple_parameters.py": [
        lambda p: "TypeError" in (p.stdout + p.stderr) or "raise" in p.stdout,
    ],
    "examples/14_parameter_sweep.py": [
        lambda p: re.search(r"P\(1\)", p.stdout) is not None,
        lambda p: "re-running with the same seed" in p.stdout,
    ],
    "examples/15_parameter_shift_gradient.py": [
        "theta.gradient()            = 1.0",
        "(2 * theta).gradient()      = 2.0",
        "d<Z>/dtheta at 0.7854 = -0.707107 (exact -0.707107)",
        "d<H>/dtheta = 0.582061 (exact 0.582061)",
        "shift -0.78332691 | finite-diff -0.78332691",
    ],
    "examples/16_gradient_descent_loop.py": [
        "== energy landscape sanity check ==",
        lambda p: re.search(r"E\(\+0\.0000, \+0\.0000\) = \+1\.000000", p.stdout)
        is not None,
        lambda p: re.search(r"optimal energy: -1\.0\d+", p.stdout) is not None,
    ],
    "examples/algorithms/01_problem_hierarchy.py": [
        lambda p: "SamplingProblem" in p.stdout or "QUBO" in p.stdout or "Problem" in p.stdout,
    ],
    "examples/algorithms/02_algorithm_lifecycle.py": [
        lambda p: re.search(r"validate|valid", p.stdout, re.I) is not None,
    ],
    "examples/algorithms/03_algorithm_result_serialization.py": [
        lambda p: "json" in p.stdout.lower() or "serialize" in p.stdout.lower(),
    ],
    "examples/algorithms/04_runtime_separation.py": [
        lambda p: "runtime" in p.stdout.lower() or "plan" in p.stdout.lower(),
    ],
    "examples/analysis/01_execution_record.py": [
        lambda p: "execution_id" in p.stdout or "fingerprint" in p.stdout,
    ],
    "examples/analysis/05_sampling_analysis.py": [
        lambda p: re.search(r"mean|variance|std", p.stdout.lower()) is not None,
    ],
    "examples/analysis/06_expectation_analysis.py": [
        lambda p: "expectation" in p.stdout.lower() or "mean" in p.stdout.lower(),
    ],
    "examples/analysis/07_state_analysis.py": [
        lambda p: "state" in p.stdout.lower(),
    ],
    "examples/analysis/08_result_aggregation.py": [
        lambda p: "aggregate" in p.stdout.lower() or "mean" in p.stdout.lower(),
    ],
    "examples/analysis/09_reproducibility_serialization.py": [
        lambda p: "same" in p.stdout.lower() or "seed" in p.stdout.lower(),
    ],
    "examples/analysis/10_failure_handling.py": [
        lambda p: "fail" in p.stdout.lower() or "error" in p.stdout.lower(),
    ],
    "examples/backend/01_backend_basic.py": [
        lambda p: "counts" in p.stdout.lower() or "backend" in p.stdout.lower(),
    ],
    "examples/backend/02_backend_capabilities.py": [
        lambda p: "target_class" in p.stdout or "circuit_features" in p.stdout,
    ],
    "examples/backend/03_backend_registry.py": [
        lambda p: "backend" in p.stdout.lower(),
    ],
    "examples/backend/04_backend_selection.py": [
        lambda p: "selected" in p.stdout.lower() or "backend" in p.stdout.lower(),
    ],
    "examples/execution/01_basic_execution.py": [
        lambda p: "counts" in p.stdout.lower() or "result" in p.stdout.lower(),
    ],
    "examples/execution/02_explicit_plan.py": [
        lambda p: "plan" in p.stdout.lower(),
    ],
    "examples/execution/03_job_lifecycle.py": [
        lambda p: "job" in p.stdout.lower() or "status" in p.stdout.lower(),
    ],
    "examples/execution/04_batch_execution.py": [
        lambda p: "batch" in p.stdout.lower(),
    ],
    "examples/execution/05_parameter_sweep.py": [
        lambda p: "sweep" in p.stdout.lower() or "theta" in p.stdout.lower(),
    ],
    "examples/execution/06_hybrid_loop.py": [
        lambda p: "step" in p.stdout.lower() or "round" in p.stdout.lower(),
    ],
    "examples/execution/08_target_and_backend.py": [
        lambda p: "target" in p.stdout.lower(),
    ],
    "examples/execution/10_end_to_end.py": [
        lambda p: "depth" in p.stdout.lower(),
    ],
    "examples/execution/11_single_qubit_measurement.py": [
        "counts       : {'1': 1000}",
        "most frequent: 1",
        lambda p: "shots        : 1000 (sum of counts = 1000)" in p.stdout,
    ],
    "examples/execution/12_bell_state_measurement.py": [
        "total shots : 1000",
        "only '00'/'11' observed: True",
        lambda p: re.search(r"\|00>:\s+\d+", p.stdout) is not None,
        lambda p: re.search(r"\|11>:\s+\d+", p.stdout) is not None,
    ],
    "examples/execution/13_seeded_execution.py": [
        "same seed reproducible : True",
        "different seed differs : True",
        "seed recorded on result: 42",
    ],
    "examples/extension/01_custom_problem.py": [
        lambda p: "CustomProblem" in p.stdout or "custom" in p.stdout.lower(),
    ],
    "examples/extension/02_custom_algorithm.py": [
        lambda p: "CustomAlgorithm" in p.stdout or "custom" in p.stdout.lower(),
    ],
    "examples/extension/03_custom_grover_diffusion.py": [
        lambda p: "grover" in p.stdout.lower() or "diffusion" in p.stdout.lower(),
    ],
    "examples/extension/04_custom_optimizer.py": [
        lambda p: "optimizer" in p.stdout.lower() or "iter" in p.stdout.lower(),
    ],
    "examples/extension/05_custom_runtime.py": [
        lambda p: "runtime" in p.stdout.lower() or "custom" in p.stdout.lower(),
    ],
    "examples/ir/01_circuit_to_ir.py": [
        lambda p: "IRCircuit" in p.stdout or "ir" in p.stdout.lower(),
    ],
    "examples/ir/02_inspect_ir.py": [
        lambda p: re.search(r"depth:\s+\d+", p.stdout) is not None,
        lambda p: re.search(r"gates:\s+\d+", p.stdout) is not None,
    ],
    "examples/ir/03_optimization_passes.py": [
        lambda p: "depth: " in p.stdout,
        lambda p: "gates: " in p.stdout,
    ],
    "examples/ir/04_parameterized_binding.py": [
        lambda p: "bind" in p.stdout.lower(),
    ],
    "examples/ir/05_dynamic_readiness.py": [
        lambda p: re.search(r"dynamic|conditional|mid.circuit", p.stdout.lower())
        is not None,
    ],
    "examples/ir/06_target_aware_compilation.py": [
        lambda p: "target" in p.stdout.lower(),
    ],
    "examples/ir/07_custom_pass.py": [
        lambda p: "pass" in p.stdout.lower() or "custom" in p.stdout.lower(),
    ],
    "examples/math/01_qft_circuit.py": [
        lambda p: re.search(r"\|000> = 0\.3\d*", p.stdout) is not None,
    ],
    "examples/math/02_phase_estimation.py": [
        "estimated phase 0.375 (expected 0.375)",
        lambda p: "estimated phase" in p.stdout,
    ],
    "examples/math/03_hamiltonian_expectation.py": [
        lambda p: re.search(r"-?\d\.\d+", p.stdout) is not None,
    ],
    "examples/optimization/01_qaoa_maxcut_problem.py": [
        "Brute-force optimum: -2.000  (a cut of 2)",
        lambda p: re.search(r"cut = 2", p.stdout) is not None,
        lambda p: "QAOA energy:" in p.stdout,
    ],
    "examples/optimization/02_qaoa_ising_cost.py": [
        lambda p: "ising" in p.stdout.lower() or "cost" in p.stdout.lower(),
    ],
    "examples/optimization/03_qubo_workflow.py": [
        lambda p: "qubo" in p.stdout.lower(),
    ],
    "examples/optimization/04_bfgs_basics.py": [
        lambda p: re.search(r"iter|energy|-?\d\.\d", p.stdout, re.I) is not None,
    ],
    "examples/rsa_vulnerability_assessment.py": [
        lambda p: re.search(r"vulnerab|modulus|factor|bit", p.stdout, re.I) is not None,
    ],
    "examples/search/01_grover_target.py": [
        lambda p: "101" in p.stdout or "target" in p.stdout.lower(),
    ],
    "examples/search/02_grover_oracle.py": [
        lambda p: "oracle" in p.stdout.lower(),
    ],
    "examples/search/03_grover_predicate.py": [
        lambda p: "predicate" in p.stdout.lower() or "mark" in p.stdout.lower(),
    ],
    "examples/search/04_grover_runtime.py": [
        lambda p: "grover" in p.stdout.lower(),
    ],
    "examples/variational/01_vqe_ground_state.py": [
        "validated:       True",
        lambda p: re.search(r"VQE energy:\s+-0\.\d+", p.stdout) is not None,
        lambda p: re.search(r"Exact ground energy:\s+-1\.0000", p.stdout) is not None,
    ],
    "examples/variational/02_vqe_runtime.py": [
        lambda p: "runtime" in p.stdout.lower() or "energy" in p.stdout.lower(),
    ],
    "examples/variational/03_optimizer_comparison.py": [
        lambda p: re.search(r"\b(BFGS|Adam|COBYLA|SPSA|GradientDescent)\b", p.stdout) is not None,
    ],
}


@pytest.mark.parametrize("example", _EXAMPLES, ids=_ids())
def test_official_example_output_contract(example) -> None:
    rel, path = example
    proc = h.run_example(path)
    assert proc.returncode == 0, f"{rel} failed: {proc.stderr[-3000:]}"
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert combined.strip(), f"{rel} produced no output"
    # Universal level-B check: a documented method must never leak its
    # bound-method repr (a method typed as a property).
    assert "<bound method" not in combined, (
        f"{rel} prints a bound-method repr (method used as a property): "
        f"{combined.strip()[-2000:]}"
    )
    checks = REGISTRY.get(rel)
    if not checks:
        return
    for check in checks:
        if callable(check):
            assert check(proc), f"{rel} failed semantic check -> output:\n{combined[-4000:]}"
        else:
            assert check in combined, (
                f"{rel} expected {check!r} in output:\n{combined[-4000:]}"
            )