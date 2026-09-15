"""MQ-13 tests: analytical gradients via the parameter-shift rule.

Covers symbolic ``Parameter.gradient`` / ``ParameterExpression.gradient``,
the chain rule for linear angle expressions, multi-occurrence product
rule, Operator/PauliString/PauliSum observables with ``targets``,
backend-integrated evaluation (``backend=``/``seed``/``shots``), strict
validation, and end-to-end gradient-descent consumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from microquantum import (
    Operator,
    Parameter,
    PauliString,
    PauliSum,
    QuantumCircuit,
    StatevectorBackend,
    gradient,
    parameter_shift_gradient,
    tensor,
)
from microquantum.backends.base import BackendResult
from microquantum.backends.local import LocalSimulatorBackend
from microquantum.core.measurement import expectation_value


def _energy(qc: QuantumCircuit, observable, params, targets=None) -> float:
    """Exact observable expectation for bound parameters (independent path)."""
    state = qc.bind_parameters(dict(params)).run()
    if isinstance(observable, Operator):
        return expectation_value(state, observable, targets=targets)
    if isinstance(observable, PauliString):
        term = observable
        if targets is None:
            return term.expectation(state)
        chars = ["I"] * state.num_qubits
        for char, t in zip(term.label, targets, strict=False):
            chars[t] = char
        return PauliString("".join(chars), term.coefficient).expectation(state)
    if isinstance(observable, PauliSum):
        return sum(_energy(qc, t, params, targets) for t in observable.terms)
    raise TypeError(type(observable))


def _finite_diff(qc, observable, param, params, targets=None, h: float = 1e-6) -> float:
    """Central-difference reference for an exact expectation function."""
    plus = dict(params)
    minus = dict(params)
    plus[param] = params[param] + h
    minus[param] = params[param] - h
    return (_energy(qc, observable, plus, targets) - _energy(qc, observable, minus, targets)) / (2 * h)


# ---------------------------------------------------------------------------


class TestSymbolicGradients:
    """``Parameter.gradient`` and ``ParameterExpression.gradient``."""

    def test_parameter_self_gradient(self) -> None:
        theta = Parameter("theta")
        assert theta.gradient() == 1.0
        assert theta.gradient(theta) == 1.0
        assert theta.gradient("theta") == 1.0

    def test_parameter_name_based(self) -> None:
        theta = Parameter("theta")
        twin = Parameter("theta")
        assert theta.gradient(twin) == 1.0

    def test_parameter_unrelated_zero(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        assert theta.gradient(phi) == 0.0
        assert theta.gradient("phi") == 0.0

    def test_expression_gradient_is_coefficient(self) -> None:
        theta = Parameter("theta")
        assert (2 * theta).gradient() == 2.0
        assert (-theta).gradient() == -1.0
        assert (theta * 2).gradient() == 2.0
        assert (theta + 0.5).gradient() == 1.0

    def test_expression_gradient_matches_and_not(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        expr = 3 * theta + 0.25
        assert expr.gradient(theta) == 3.0
        assert expr.gradient("theta") == 3.0
        assert expr.gradient(phi) == 0.0
        assert expr.gradient("phi") == 0.0

    def test_zero_coefficient_expression(self) -> None:
        theta = Parameter("theta")
        expr = 0 * theta  # type: ignore[assignment]
        assert expr.gradient() == 0.0

    def test_constant_offset_irrelevant(self) -> None:
        theta = Parameter("theta")
        assert (5 - theta).gradient() == -1.0
        assert (theta + 1.5).gradient(theta) == 1.0


class TestChainRuleExpressions:
    """Analytic gradients respect d(a*theta + c)/dtheta = a."""

    def test_scaled_ry(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(2 * theta, 0)
        val = np.pi / 6
        grad = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        assert grad == pytest.approx(-2 * np.sin(2 * val), abs=1e-10)

    def test_offset_ry_constant_cancels(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta + 0.7, 0)
        val = np.pi / 4
        grad = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        assert grad == pytest.approx(-np.sin(val + 0.7), abs=1e-10)

    def test_negated_ry(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(-theta, 0)
        val = np.pi / 5
        grad = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        assert grad == pytest.approx(-np.sin(val), abs=1e-10)

    @pytest.mark.parametrize("val", [0.0, 0.3, np.pi / 4, np.pi / 2])
    def test_full_vector_with_expressions(self, val: float) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(2 * theta, 0).rx(-phi, 1)
        obs = PauliString("Z", 1.0)
        grads = gradient(qc, obs, {theta: val, phi: val}, targets=[0])
        assert grads[theta] == pytest.approx(-2 * np.sin(2 * val), abs=1e-10)
        assert grads[phi] == pytest.approx(0.0, abs=1e-10)

    def test_multi_occurrence_product_rule(self) -> None:
        theta = Parameter("theta")
        # Ry(theta).Ry(2*theta) on same qubit compose: total angle = 3*theta.
        # <Z> = cos(3*theta) => d/dtheta = -3*sin(3*theta).
        qc = QuantumCircuit(1).ry(theta, 0).ry(2 * theta, 0)
        val = 0.4
        grad = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        expected = -3 * np.sin(3 * val)
        assert grad == pytest.approx(expected, abs=1e-10)

    def test_multi_occurrence_different_qubits(self) -> None:
        theta = Parameter("theta")
        # Ry(theta,0) Ry(theta,1) with ZZ observable:
        # <ZZ> = cos(theta)*cos(theta) = cos^2(theta)
        # d/dtheta = -2*sin(theta)*cos(theta) = -sin(2*theta)
        qc = QuantumCircuit(2).ry(theta, 0).ry(theta, 1)
        grad = parameter_shift_gradient(
            qc, PauliSum.from_label("ZZ"), theta, {theta: 0.3}
        )
        assert grad == pytest.approx(-np.sin(2 * 0.3), abs=1e-10)


class TestPauliObservables:
    """Operator / PauliString / PauliSum agree and support targets."""

    def test_pauli_string_matches_operator(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        val = np.pi / 4
        op_grad = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        ps_grad = parameter_shift_gradient(
            qc, PauliString("Z"), theta, {theta: val}
        )
        assert op_grad == pytest.approx(ps_grad, abs=1e-10)
        assert ps_grad == pytest.approx(-np.sin(val), abs=1e-10)

    def test_pauli_sum_energy_gradient(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        obs = PauliSum([PauliString("X", 0.5), PauliString("Z", -0.3)])
        val = 0.6
        grad = parameter_shift_gradient(qc, obs, theta, {theta: val})
        expected = 0.5 * np.cos(val) + 0.3 * np.sin(val)
        assert grad == pytest.approx(expected, abs=1e-10)

    def test_pauli_string_targets_subset(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).ry(phi, 1)
        val_t, val_p = 0.4, 0.8
        # X on qubit 1 only (Ry gate): <X> = sin(phi), d<X>/d phi = cos(phi).
        grad_phi = parameter_shift_gradient(
            qc, PauliString("X"), phi, {theta: val_t, phi: val_p}, targets=[1]
        )
        assert grad_phi == pytest.approx(np.cos(val_p), abs=1e-10)
        # X on qubit 1 only: gradient w.r.t. theta (Ry on qubit 0) = 0.
        grad_theta = parameter_shift_gradient(
            qc, PauliString("X"), theta, {theta: val_t, phi: val_p}, targets=[1]
        )
        assert grad_theta == pytest.approx(0.0, abs=1e-10)

    def test_zz_coupling_gradient(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).rz(theta, 0).ry(phi, 1)
        obs = PauliSum.from_label("ZZ")
        vals = {theta: 0.5, phi: -0.3}
        grads = gradient(qc, obs, vals)
        # Compare independently against the Operator path.
        for param, g in grads.items():
            via_op = parameter_shift_gradient(
                qc, obs.to_operator(), param, vals
            )
            assert g == pytest.approx(via_op, abs=1e-10)
            assert g == pytest.approx(_finite_diff(qc, obs, param, vals), abs=1e-7)

    def test_pauli_string_needs_targets(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0)
        with pytest.raises(ValueError, match="targets="):
            parameter_shift_gradient(qc, PauliString("Z"), theta, {theta: 0.0})

    def test_targets_length_mismatch(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0)
        with pytest.raises(ValueError, match="target"):
            parameter_shift_gradient(
                qc, PauliString("Z"), theta, {theta: 0.0}, targets=[0, 1]
            )


class TestBackendIntegration:
    """Evaluation through the MQ-11/12 execution core."""

    @pytest.mark.parametrize("backend_cls", [StatevectorBackend, LocalSimulatorBackend])
    @pytest.mark.parametrize("val", [0.0, np.pi / 4, np.pi / 2])
    def test_backend_gradient_matches_engine(
        self, backend_cls, val: float
    ) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        expected = -np.sin(val)
        engine = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
        via_backend = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}, backend=backend_cls()
        )
        assert via_backend == pytest.approx(expected, abs=1e-10)
        assert via_backend == pytest.approx(engine, abs=1e-10)

    def test_backend_exact_and_reproducible(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        params = {theta: 0.3, phi: 0.9}
        obs = PauliSum.from_label("ZI")
        b1 = gradient(qc, obs, params, backend=StatevectorBackend(), seed=7)
        b2 = gradient(qc, obs, params, backend=StatevectorBackend(), seed=7)
        b3 = gradient(qc, obs, params, backend=StatevectorBackend(), seed=99)
        # Exact statevector expectations: identical regardless of seed/shots.
        for p in params:
            assert b1[p] == pytest.approx(b2[p], abs=1e-12)
            assert b1[p] == pytest.approx(b3[p], abs=1e-12)

    def test_backend_returning_no_statevector_raises(self) -> None:
        class _NoStateBackend:
            name = "no_state"

            def run(self, circuit, **kwargs) -> BackendResult:
                return BackendResult(num_qubits=circuit.num_qubits, backend_name=self.name)

        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="statevector"):
            parameter_shift_gradient(
                qc, Operator.Z(), theta, {theta: 0.0}, backend=_NoStateBackend()
            )

    def test_backend_without_run_raises(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(TypeError, match="run"):
            parameter_shift_gradient(
                qc, Operator.Z(), theta, {theta: 0.0}, backend="not-a-backend"
            )


class TestValidation:
    """Strict validation of the gradient contract."""

    def test_param_values_must_be_mapping(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(TypeError, match="Mapping"):
            parameter_shift_gradient(qc, Operator.Z(), theta, [("theta", 0.0)])
        with pytest.raises(TypeError, match="Mapping"):
            gradient(qc, Operator.Z(), [("theta", 0.0)])

    def test_bad_shift_raises(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="sin"):
            parameter_shift_gradient(qc, Operator.Z(), theta, {theta: 0.0}, shift=0.0)
        with pytest.raises(ValueError, match="sin"):
            parameter_shift_gradient(qc, Operator.Z(), theta, {theta: 0.0}, shift=np.pi)

    def test_invalid_observable_type(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(TypeError, match="observable"):
            parameter_shift_gradient(qc, "Z", theta, {theta: 0.0})  # type: ignore[arg-type]

    def test_param_not_in_circuit(self) -> None:
        theta = Parameter("theta")
        phantom = Parameter("beta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(KeyError, match="not found in any gate"):
            parameter_shift_gradient(qc, Operator.Z(), phantom, {phantom: 0.0})

    def test_missing_parameter_value(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        with pytest.raises(KeyError, match="not found in param_values"):
            parameter_shift_gradient(qc, Operator.Z(), theta, {theta: 0.0})

    def test_non_parameterized_circuit_empty_gradient(self) -> None:
        qc = QuantumCircuit(1).h(0)
        grads = gradient(qc, Operator.Z(), {})
        assert grads == {}


class TestComposition:
    """Gradients over composed (concatenated) circuits."""

    def test_composed_circuit_gradient(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        left = QuantumCircuit(2).ry(theta, 0)
        right = QuantumCircuit(2).rx(phi, 1)
        combo = left + right
        params = {theta: 0.2, phi: 0.3}
        grads = gradient(combo, Operator.Z(), params, targets=[0])
        assert grads[theta] == pytest.approx(-np.sin(0.2), abs=1e-10)
        assert grads[phi] == pytest.approx(0.0, abs=1e-10)


class TestFiniteDifferenceAgreement:
    """Parameter-shift matches numerical derivatives on entangled circuits."""

    def test_multi_parameter_entangled(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        alpha = Parameter("alpha")
        qc = QuantumCircuit(3)
        qc.ry(theta, 0).rz(-phi, 1).cx(0, 1).rx(2 * alpha, 2).cx(1, 2)
        obs = tensor(tensor(Operator.Z(), Operator.Z()), Operator.Z())
        params = {theta: 0.6, phi: 1.1, alpha: 0.9}
        grads = gradient(qc, obs, params)
        assert set(grads) == set(params)
        for p in params:
            via_shift = grads[p]
            via_fd = _finite_diff(qc, obs, p, params)
            assert via_shift == pytest.approx(via_fd, abs=1e-7)

    def test_gradient_keys_are_circuit_parameters(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        grads = gradient(qc, Operator.Z(), {theta: 0.1, phi: 0.2}, targets=[0])
        for key in grads:
            assert key in qc.parameters


class TestOptimizerEndToEnd:
    """``gradient`` plugs into gradient-mode optimizers (VQE-style)."""

    def test_gradient_descent_minimizes_ry_energy(self) -> None:
        from microquantum.optimizers import GradientDescent

        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        obs = PauliSum.from_label("Z")  # E(theta) = cos(theta), min -1 at pi.

        def cost(params) -> float:
            return _energy(qc, obs, params)

        def grad_fn(params) -> dict[Parameter, float]:
            return gradient(qc, obs, params)

        result = GradientDescent(learning_rate=0.2, max_iter=500).minimize(
            cost, gradient_fn=grad_fn, initial_params={theta: 0.5}
        )
        assert result.converged
        assert result.history[0] > result.history[-1]
        assert result.optimal_value == pytest.approx(-1.0, abs=1e-3)
        (best,) = result.optimal_parameters.values()
        assert best == pytest.approx(np.pi, abs=1e-2)

    def test_adam_with_pauli_sum_observable(self) -> None:
        from microquantum.optimizers import Adam

        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1).cx(0, 1)
        obs = PauliSum([PauliString("ZZ", 0.5), PauliString("ZI", 0.25)])
        params0 = {theta: 0.6, phi: 0.6}

        def cost(params) -> float:
            return _energy(qc, obs, params)

        def grad_fn(params) -> dict[Parameter, float]:
            return gradient(qc, obs, params)

        result = Adam(learning_rate=0.1, max_iter=300).minimize(
            cost, gradient_fn=grad_fn, initial_params=params0
        )
        assert result.history[-1] < result.history[0]