"""Shared helpers for the MQ-16 compiler <-> runtime integration suite.

These helpers turn the semantic-equivalence contract into plain functions
so both the regression suite (``tests/test_compiler_runtime_integration.py``)
and the permanent conformance suite
(``tests/conformance/test_compiler_runtime_integration.py``) exercise the
exact same logic:

* compile a circuit with the :class:`~microquantum.Compiler`,
* rebuild the executable circuit with ``CompilationResult.circuit()``,
* run original and compiled through the actual backends / runtime,
* compare *observable* results: exact state where available, principled
  statistical agreement where sampling applies.

Numerical tolerances are centralized here.  Deterministic ``shots=None``
runs are compared with ``np.allclose`` at float precision; sampled runs use
a per-outcome probability bound derived from the seeded shot count.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from microquantum import (
    CompilationResult,
    Compiler,
    DensityMatrixBackend,
    MPSBackend,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    TreeTensorNetworkBackend,
    execute,
)
from microquantum.runtime.plan import ExecutionPlan

#: Deterministic (``shots=None``) probability/state tolerance.
TOL_EXACT = 1e-9
#: State-vector fidelity floor for exact preservation.
TOL_FIDELITY = 1e-9
#: Default seeded shot count for stochastic comparisons.
SHOTS_SAMPLED = 4000
#: Per-outcome probability bound for seeded sampling agreement.
TOL_SAMPLING = 0.05

#: The real local simulator backends that MicroQuantum officially supports.
#: ``MockBackend`` is intentionally excluded: it is a deterministic routing
#: stub whose counts are a pure function of gate labels, not a simulator.
SIMULATORS: list[tuple[str, Callable[[], object]]] = [
    ("statevector", lambda: StatevectorBackend()),
    ("density_matrix", lambda: DensityMatrixBackend()),
    ("mps", lambda: MPSBackend()),
    ("ttn", lambda: TreeTensorNetworkBackend()),
]


# ---------------------------------------------------------------------------
# Deterministic corpus
# ---------------------------------------------------------------------------


def bell() -> QuantumCircuit:
    """Bell state: H(q0); CX(q0, q1)."""
    return QuantumCircuit(2).h(0).cnot(0, 1)


def ghz(size: int = 3) -> QuantumCircuit:
    """GHZ state: H(q0); CX(q0,q1); CX(q1,q2); ..."""
    qc = QuantumCircuit(size)
    qc.h(0)
    for i in range(1, size):
        qc.cnot(i - 1, i)
    return qc


def rotation_circuit() -> QuantumCircuit:
    """Mixed-axis rotations with a negative angle (sign-sensitive)."""
    return QuantumCircuit(2).rx(0.7, 0).ry(-1.1, 1).h(0)


def cz_circuit() -> QuantumCircuit:
    """CZ surrounded by Hadamards: H; CZ; H."""
    return QuantumCircuit(2).h(0).cz(0, 1).h(0)


def swap_circuit() -> QuantumCircuit:
    """X(q0); SWAP(q0, q1)."""
    return QuantumCircuit(2).x(0).swap(0, 1)


def parameterized_circuit() -> QuantumCircuit:
    """Symbolic RX/RZ around a CNOT (MQ-16 parameter pipeline)."""
    theta, phi = Parameter("theta"), Parameter("phi")
    return QuantumCircuit(2).rx(theta, 0).cnot(0, 1).rz(phi, 1)


#: Angles spanning both signs, all quadrants and the +/-pi boundaries.
ROTATION_ANGLES = [
    -3.141592653589793,
    -2.5,
    -1.5707963267948966,
    -1.1,
    -0.5,
    -0.17,
    0.0,
    0.17,
    0.5,
    1.1,
    1.5707963267948966,
    2.5,
    3.141592653589793,
]

#: Deterministic corpus used throughout the differential contract.  The
#: parameterized circuit is pre-bound so the generic helpers treat it like
#: any ordinary circuit; the *symbolic* compile+bind pipeline is covered by
#: dedicated parameter tests in the suite files.
EQUIVALENCE_CORPUS: list[tuple[str, QuantumCircuit]] = [
    ("bell", bell()),
    ("ghz", ghz()),
    ("rotation", rotation_circuit()),
    ("cz", cz_circuit()),
    ("swap", swap_circuit()),
    ("parameterized", parameterized_circuit().bind_parameters({"theta": 0.9, "phi": -1.2})),
]


# ---------------------------------------------------------------------------
# Basic observables
# ---------------------------------------------------------------------------


def statevector(circuit: QuantumCircuit) -> np.ndarray:
    """Exact state vector of *circuit* via the state-vector backend."""
    result = StatevectorBackend().run(circuit, shots=None)
    assert result.statevector is not None, "statevector missing on shots=None"
    return np.asarray(result.statevector, dtype=np.complex128)


def density_matrix(circuit: QuantumCircuit) -> np.ndarray:
    """Exact density matrix of *circuit* via the density-matrix backend."""
    result = DensityMatrixBackend().run(circuit, shots=None)
    assert result.density_matrix is not None, "density matrix missing on shots=None"
    return np.asarray(result.density_matrix, dtype=np.complex128)


def fidelity(source: QuantumCircuit, other: QuantumCircuit) -> float:
    """|⟨source|other⟩|^2 between two circuits' exact state vectors."""
    return float(abs(np.vdot(statevector(source), statevector(other))) ** 2)


def probability_vector(result) -> Optional[np.ndarray]:
    """Measurement probabilities from an exact ``shots=None`` result.

    Uses the state vector when present, otherwise the density-matrix
    diagonal.  Returns ``None`` when the backend exposes neither (reserved
    for backends that ship no exact representation).
    """
    if result.statevector is not None:
        return np.abs(np.asarray(result.statevector)) ** 2
    if result.density_matrix is not None:
        return np.real(np.diag(result.density_matrix))
    return None


def exact_probabilities(circuit: QuantumCircuit, backend) -> Optional[np.ndarray]:
    """Probability vector of *circuit* on *backend* with ``shots=None``."""
    return probability_vector(backend.run(circuit, shots=None))


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalize a counts histogram to a probability mapping."""
    total = sum(counts.values())
    assert total > 0, "cannot normalize empty counts"
    return {bitstring: count / total for bitstring, count in counts.items()}


def sampled_probability_map(
    circuit: QuantumCircuit, backend, shots: int = SHOTS_SAMPLED, seed: int = 7
) -> dict[str, float]:
    """Seeded sampled probabilities of *circuit* on *backend*."""
    counts = backend.run(circuit, shots=shots, seed=seed).counts
    return counts_to_probabilities(counts)


# ---------------------------------------------------------------------------
# Compile + execute pipeline
# ---------------------------------------------------------------------------


def compile_for(circuit: QuantumCircuit, level: int, target: Optional[Target] = None) -> CompilationResult:
    """Compile a circuit at *level* (optionally toward *target*)."""
    return Compiler(optimization_level=level).compile(circuit, target=target)


def assert_execution_equivalent(
    circuit: QuantumCircuit,
    *,
    level: int,
    backend_factory: Callable[[], object],
    shots: Optional[int] = None,
    seed: int = 7,
    tol: Optional[float] = None,
    target: Optional[Target] = None,
) -> CompilationResult:
    """Prove original and compiled circuits are observably equivalent.

    Compiles *circuit* at *level*, rebuilds the executable circuit, and runs
    both original and compiled on a fresh *backend_factory* instance.

    * ``shots=None`` -> compare exact probability vectors (float equality
      within *tol*, default :data:`TOL_EXACT`).
    * ``shots > 0``  -> compare seeded probability estimates (per-outcome
      bound :data:`TOL_SAMPLING`).

    Returns the :class:`CompilationResult` so callers can also inspect the
    transformation and its metadata.
    """
    tolerance = tol if tol is not None else TOL_EXACT
    cr = compile_for(circuit, level, target=target)
    compiled = cr.circuit()

    backend = backend_factory()
    if shots is None:
        original_probs = exact_probabilities(circuit, backend)
        compiled_probs = exact_probabilities(compiled, backend)
        if original_probs is not None and compiled_probs is not None:
            assert np.allclose(original_probs, compiled_probs, atol=tolerance), (
                f"exact probabilities diverged for level {level}: "
                f"max |delta| = "
                f"{np.max(np.abs(original_probs - compiled_probs)):.2e}"
            )
        else:
            assert backend.run(circuit, shots=None).counts == {}
            assert backend.run(compiled, shots=None).counts == {}
    else:
        original = sampled_probability_map(circuit, backend, shots=shots, seed=seed)
        compiled_map = sampled_probability_map(compiled, backend, shots=shots, seed=seed)
        for outcome in set(original) | set(compiled_map):
            delta = abs(original.get(outcome, 0.0) - compiled_map.get(outcome, 0.0))
            assert delta <= TOL_SAMPLING, (
                f"sampled probabilities diverged for level {level}, "
                f"outcome {outcome!r}: |delta| = {delta:.3f}"
            )
    return cr


def assert_statevector_preserved(
    circuit: QuantumCircuit, level: int, tol: float = TOL_FIDELITY
) -> CompilationResult:
    """Compile at *level* and require exact state-vector preservation."""
    cr = compile_for(circuit, level)
    f = fidelity(circuit, cr.circuit())
    assert f > 1.0 - tol, f"level {level} changed the state (fidelity {f:.12f})"
    return cr


def assert_runtime_equivalent(
    circuit: QuantumCircuit,
    *,
    level: int,
    backend,
    shots: int,
    seed: int = 7,
    tolerance: float = TOL_SAMPLING,
) -> CompilationResult:
    """Compare all three execution routes for a compiled program.

    Route A: explicit :class:`Compiler` -> ``backend.run(compiled)``.
    Route B: reuse the :class:`CompilationResult` through the existing
    runtime (``ExecutionPlan(compiled=...)``).
    Route C: let the existing runtime compile at run time
    (``execute(circuit, optimization_level=level)``).

    All three must produce the same seeded probability estimates.
    """
    cr = compile_for(circuit, level)

    route_a = sampled_probability_map(cr.circuit(), backend, shots=shots, seed=seed)

    plan = ExecutionPlan(compiled=cr, backend=backend, shots=shots, seed=seed)
    route_b = counts_to_probabilities(execute(plan=plan).counts)

    route_c_result = execute(
        circuit,
        backend=backend,
        shots=shots,
        seed=seed,
        optimization_level=level,
    )
    route_c = counts_to_probabilities(route_c_result.counts)

    for outcome in set(route_a) | set(route_b) | set(route_c):
        for label, probs in (("A", route_a), ("B", route_b), ("C", route_c)):
            delta = abs(probs.get(outcome, 0.0) - route_a.get(outcome, 0.0))
            assert delta <= tolerance, (
                f"route {label} diverged from route A for outcome {outcome!r}: "
                f"|delta| = {delta:.3f}"
            )
    return cr