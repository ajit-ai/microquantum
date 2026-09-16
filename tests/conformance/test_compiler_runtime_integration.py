"""Conformance - compiler <-> runtime integration (MQ-16).

Permanent public-face checks for the semantic-equivalence contract between
the :class:`~microquantum.Compiler` and the existing execution runtime:

* differential equivalence of compiled programs across optimization levels
  0/1/2 and every supported local simulator (exact where available,
  principled bounds where sampling applies),
* runtime-route equivalence (``Compiler`` + ``backend.run`` vs
  ``ExecutionPlan(compiled=...)`` reuse vs ``execute(optimization_level)``),
* directed two-qubit gate order consistency across backends (guards the
  ``expand_operator`` k == n ordering contract behind ``GateDecomposition``
  of ``swap``/``cz`` toward native bases),
* negative rotation-angle sign preservation, parameter binding after
  compilation, measurement preservation, empty circuits, and the error
  contract of ``ExecutionPlan.compiled``.

The assertions reuse the exact helpers and tolerances exercised by the full
regression suite ``tests/test_compiler_runtime_integration.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from microquantum import (
    Compiler,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    execute,
)
from microquantum.runtime.plan import ExecutionPlan

# The shared MQ-16 helpers live next to the regression suite (tests/).
_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from compiler_runtime_helpers import (  # noqa: E402
    EQUIVALENCE_CORPUS,
    ROTATION_ANGLES,
    SHOTS_SAMPLED,
    SIMULATORS,
    TOL_EXACT,
    TOL_FIDELITY,
    TOL_SAMPLING,
    assert_execution_equivalent,
    assert_runtime_equivalent,
    bell,
    cz_circuit,
    fidelity,
    swap_circuit,
)

LEVELS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Differential equivalence (deterministic; cheap enough for daily runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus_name,qc", EQUIVALENCE_CORPUS, ids=[n for n, _ in EQUIVALENCE_CORPUS])
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
def test_compiled_keeps_exact_probabilities(corpus_name, qc, level, backend_name, factory):
    """Compiled and original produce the same exact probabilities."""
    assert_execution_equivalent(
        qc, level=level, backend_factory=factory, shots=None, tol=TOL_EXACT
    )


@pytest.mark.parametrize("level", LEVELS)
def test_statevector_fidelity_preserved(level):
    for _name, qc in EQUIVALENCE_CORPUS:
        cr = _compile(qc, level)
        assert fidelity(qc, cr.circuit()) > 1.0 - TOL_FIDELITY


# ---------------------------------------------------------------------------
# Statistical agreement + runtime routes (seeded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
def test_seeded_sampling_agrees(backend_name, factory):
    assert_execution_equivalent(
        bell(),
        level=2,
        backend_factory=factory,
        shots=SHOTS_SAMPLED,
        seed=11,
        tol=TOL_SAMPLING,
    )


@pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
def test_runtime_routes_agree(backend_name, factory):
    assert_runtime_equivalent(
        bell(), level=1, backend=factory(), shots=2048, seed=3
    )


# ---------------------------------------------------------------------------
# Directed two-qubit gate order (expand_operator k == n contract)
# ---------------------------------------------------------------------------


def test_directed_cnot_consistent_across_backends():
    qc = QuantumCircuit(2).x(0).cnot(1, 0)
    expected = np.zeros(4)
    expected[2] = 1.0
    for _name, factory in SIMULATORS:
        result = factory().run(qc, shots=None)
        probs = (
            np.abs(np.asarray(result.statevector)) ** 2
            if result.statevector is not None
            else np.real(np.diag(result.density_matrix))
        )
        np.testing.assert_allclose(probs, expected, atol=TOL_EXACT)


def test_basis_decomposition_equivalent():
    target = Target(name="conformance-basis", num_qubits=2, native_gates=("h", "x", "cx"))
    for _name, factory in SIMULATORS:
        assert_execution_equivalent(
            swap_circuit(), level=1, backend_factory=factory, shots=None, target=target
        )
        assert_execution_equivalent(
            cz_circuit(), level=1, backend_factory=factory, shots=None, target=target
        )


# ---------------------------------------------------------------------------
# Parameters, measurements, empty circuits
# ---------------------------------------------------------------------------


def test_symbolic_compile_then_bind_then_execute():
    from microquantum import StatevectorBackend

    theta, phi = Parameter("theta"), Parameter("phi")
    qc = QuantumCircuit(2).rx(theta, 0).cnot(0, 1).rz(phi, 1)
    bindings = {"theta": 0.6, "phi": -0.9}
    for level in LEVELS:
        compiled = _compile(qc, level).circuit().bind_parameters(bindings)
        assert fidelity(qc.bind_parameters(bindings), compiled) > 1.0 - TOL_FIDELITY
        StatevectorBackend().run(compiled, shots=64, seed=1)


def test_unmeasured_source_gains_terminal_measurements():
    cr = _compile(bell(), 1)
    assert cr.circuit().measurements == [0, 1]


def test_empty_circuit_survives_compilation():
    qc = QuantumCircuit(1)
    for level in LEVELS:
        cr = _compile(qc, level)
        probs = np.abs(np.asarray(
            StatevectorBackend().run(cr.circuit(), shots=None).statevector
        )) ** 2
        np.testing.assert_allclose(probs, [1.0, 0.0], atol=TOL_EXACT)


# ---------------------------------------------------------------------------
# Rotation-sign regression and error contracts
# ---------------------------------------------------------------------------


def test_negative_rotation_angles_preserved():
    for axis in ("rx", "ry", "rz"):
        for level in (1, 2):
            for angle in ROTATION_ANGLES:
                qc = QuantumCircuit(1)
                getattr(qc, axis)(angle, 0)
                cr = _compile(qc, level)
                assert fidelity(qc, cr.circuit()) > 1.0 - TOL_FIDELITY


def test_execution_plan_compiled_type_validated():
    with pytest.raises(TypeError, match="CompilationResult"):
        ExecutionPlan(compiled="not a compilation result")
    with pytest.raises(TypeError):
        ExecutionPlan(compiled=object())


def test_incompatible_plan_rejected_by_default():
    target = Target(name="no-measure", num_qubits=2, supports_measurement=False, native_gates=("h", "cx"))
    cr = _compile(bell(), 1, target=target)
    assert not cr.is_compatible
    plan = ExecutionPlan(compiled=cr, backend=StatevectorBackend(), shots=16, seed=1)
    with pytest.raises(ValueError, match="incompatible"):
        execute(plan=plan)


def test_reset_ir_not_representable_as_static_circuit():
    from microquantum import DynamicCircuit

    cr = Compiler(optimization_level=0).compile(DynamicCircuit(1).reset(0).to_ir())
    with pytest.raises(ValueError, match="not representable"):
        cr.circuit()


def _compile(circuit: QuantumCircuit, level: int, target=None):
    return Compiler(optimization_level=level).compile(circuit, target=target)