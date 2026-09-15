"""MQ-12 tests: parameterized circuits & parameter execution.

Covers first-class ``Parameter`` support: deterministic parameter
introspection, strict binding validation, parameter-preserving JSON
serialization, the OpenQASM 2.0 export policy, ``parameter_values=``
execution and per-value sweep execution across all local backends.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from microquantum import (
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    StatevectorBackend,
)
from microquantum.backends.density_matrix import DensityMatrixBackend
from microquantum.backends.local import LocalSimulatorBackend
from microquantum.backends.mock import MockBackend

_BIND_HELPERS = ["Parameter instances or parameter names"]


def _twist(theta: Parameter) -> QuantumCircuit:
    """Ry(theta)|0> on one qubit, measured."""
    return QuantumCircuit(1).ry(theta, 0).measure_all()


# ---------------------------------------------------------------------------
# Parameter expression arithmetic
# ---------------------------------------------------------------------------


class TestParameterExpressionMQ12:
    def test_parameter_arithmetic_produces_expression(self) -> None:
        p = Parameter("theta")
        for expr in (2 * p, p * 2, -p, p + 0.5, 0.5 + p, p - 0.5):
            assert isinstance(expr, ParameterExpression)

    def test_subtract_expression(self) -> None:
        p = Parameter("theta")
        expr = p - 0.5
        assert expr.evaluate({p: 1.5}) == pytest.approx(1.0)

    def test_scale_expression(self) -> None:
        p = Parameter("theta")
        expr = 2 * p
        assert expr.evaluate({p: 1.25}) == pytest.approx(2.5)

    def test_parameter_identity_is_name_based(self) -> None:
        assert Parameter("theta") == Parameter("theta")
        assert hash(Parameter("theta")) == hash(Parameter("theta"))
        assert {Parameter("theta"), Parameter("theta")} == {Parameter("theta")}


# ---------------------------------------------------------------------------
# Parameter discovery
# ---------------------------------------------------------------------------


class TestParameterIntrospectionMQ12:
    def test_dedup_same_name_expression_and_parameter(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0).rz(2 * theta, 0)
        assert qc.parameters == (theta,)
        assert len(qc.parameters) == 1

    def test_parameters_preserved_through_measurement(self) -> None:
        theta = Parameter("theta")
        qc = _twist(theta)
        assert qc.parameters == (theta,)
        assert qc.measurements == [0]

    def test_parameters_sorted_by_name(self) -> None:
        qc = (
            QuantumCircuit(3)
            .ry(Parameter("z"), 0)
            .rx(Parameter("alpha"), 1)
            .rz(Parameter("m"), 2)
        )
        assert [p.name for p in qc.parameters] == ["alpha", "m", "z"]

    def test_parameters_are_read_only_view(self) -> None:
        qc = QuantumCircuit(1).ry(Parameter("theta"), 0)
        params = qc.parameters
        with pytest.raises(TypeError):
            params[0] = Parameter("other")  # type: ignore[index]


# ---------------------------------------------------------------------------
# Strict binding validation
# ---------------------------------------------------------------------------


class TestBindingValidationMQ12:
    def _circuit(self) -> QuantumCircuit:
        return (
            QuantumCircuit(2)
            .ry(Parameter("theta"), 0)
            .rx(Parameter("phi"), 1)
        )

    def test_unknown_parameter_by_object_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            self._circuit().bind_parameters({Parameter("nope"): 1.0})

    def test_unknown_parameter_by_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            self._circuit().bind_parameters({"nope": 1.0})

    def test_mixed_known_and_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            self._circuit().bind_parameters({"theta": 0.5, "nope": 1.0})

    def test_error_lists_available_and_unknown(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._circuit().bind_parameters({"sigma": 1.0})
        message = str(excinfo.value)
        assert "'sigma'" in message
        assert "'phi'" in message and "'theta'" in message

    def test_non_numeric_value_raises(self) -> None:
        for bad in ("abc", None, [1.0], Parameter("theta")):
            with pytest.raises(TypeError, match="must be numeric"):
                self._circuit().bind_parameters({"theta": bad})

    def test_bool_value_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be numeric"):
            self._circuit().bind_parameters({"theta": True})

    def test_complex_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be real"):
            self._circuit().bind_parameters({"theta": 1 + 1j})

    def test_ambiguous_object_and_name_raises(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            self._circuit().bind_parameters({"theta": 0.5, Parameter("theta"): 1.0})

    def test_non_mapping_input_raises(self) -> None:
        with pytest.raises(TypeError, match="expects a Mapping"):
            self._circuit().bind_parameters([("theta", 0.5)])

    def test_invalid_key_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Parameter instances"):
            self._circuit().bind_parameters({1: 0.5})

    def test_numpy_scalar_values_accepted(self) -> None:
        bound = self._circuit().bind_parameters(
            {"theta": np.float64(1.0), "phi": np.float32(2.0)}
        )
        assert not bound.is_parameterized

    def test_int_and_float_values_accepted(self) -> None:
        bound = self._circuit().bind_parameters({"theta": 1, "phi": 2.5})
        assert not bound.is_parameterized


# ---------------------------------------------------------------------------
# Binding semantics & immutability
# ---------------------------------------------------------------------------


class TestBindingSemanticsMQ12:
    def test_partial_binding_keeps_remaining_parameters(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        partial = qc.bind_parameters({theta: 0.5})
        assert partial.is_parameterized
        assert partial.parameters == (phi,)

    def test_chained_partial_binding_fully_binds(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        state = qc.bind_parameters({theta: np.pi}).bind_parameters({phi: 0.0}).run()
        assert abs(state.amplitudes[2]) == pytest.approx(1.0)

    def test_original_circuit_is_never_modified(self) -> None:
        theta = Parameter("theta")
        qc = _twist(theta)
        before = list(qc._gate_instructions)
        qc.bind_parameters({theta: 0.5})
        assert list(qc._gate_instructions) == before
        assert qc.is_parameterized
        assert qc.parameters == (theta,)

    def test_empty_binding_returns_parameterized_copy(self) -> None:
        theta = Parameter("theta")
        qc = _twist(theta)
        copy = qc.bind_parameters({})
        assert copy.is_parameterized
        assert copy._gate_instructions == qc._gate_instructions

    def test_numeric_equivalence_bound_equals_numeric_gate(self) -> None:
        theta = Parameter("theta")
        parametrized = QuantumCircuit(1).ry(theta, 0)
        numeric = QuantumCircuit(1).ry(0.7, 0)
        bound_state = parametrized.bind_parameters({theta: 0.7}).run()
        numeric_state = numeric.run()
        assert np.allclose(bound_state.amplitudes, numeric_state.amplitudes)

    def test_expression_angle_binds(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(2 * theta + 0.5, 0)
        bound = qc.bind_parameters({theta: 0.25})
        numeric = QuantumCircuit(1).ry(1.0, 0)
        assert np.allclose(bound.run().amplitudes, numeric.run().amplitudes)


# ---------------------------------------------------------------------------
# Execution integration
# ---------------------------------------------------------------------------


class TestParameterExecutionMQ12:
    @pytest.fixture(params=["statevector", "local", "density_matrix", "mock"])
    def backend(self, request: pytest.FixtureRequest):
        mapping = {
            "statevector": StatevectorBackend,
            "local": LocalSimulatorBackend,
            "density_matrix": DensityMatrixBackend,
            "mock": MockBackend,
        }
        return mapping[request.param]()

    def test_bind_then_execute(self, backend) -> None:
        theta = Parameter("theta")
        bound = _twist(theta).bind_parameters({theta: 1.7})
        result = backend.run(bound, shots=128, seed=3)
        assert sum(result.counts.values()) == 128

    def test_parameter_values_convenience(self, backend) -> None:
        theta = Parameter("theta")
        qc = _twist(theta)
        direct = backend.run(qc, shots=128, seed=3, parameter_values={theta: 1.7})
        bound = backend.run(
            qc.bind_parameters({theta: 1.7}), shots=128, seed=3
        )
        assert sum(direct.counts.values()) == sum(bound.counts.values()) == 128

    def test_unbound_execution_raises(self, backend) -> None:
        theta = Parameter("theta")
        with pytest.raises(ValueError, match="unbound parameters"):
            backend.run(_twist(theta), shots=32)

    def test_unbound_parameter_values_with_unknown_raises(self, backend) -> None:
        theta = Parameter("theta")
        with pytest.raises(ValueError, match="unknown parameter"):
            backend.run(
                _twist(theta), shots=32, parameter_values={"nope": 1.0}
            )

    def test_seed_reproducibility(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(3).ry(theta, 0).cx(0, 1).cx(1, 2).measure_all()
        counts1 = StatevectorBackend().run(
            qc, shots=2000, seed=11, parameter_values={theta: 2.1}
        ).counts
        counts2 = StatevectorBackend().run(
            qc, shots=2000, seed=11, parameter_values={theta: 2.1}
        ).counts
        assert counts1 == counts2
        assert sum(counts1.values()) == 2000

    def test_measurement_subset_with_parameters(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0).cx(0, 1).measure(1)
        result = StatevectorBackend().run(
            qc, shots=1000, seed=5, parameter_values={theta: 1.7}
        )
        assert all(len(k) == 1 for k in result.counts)
        assert sum(result.counts.values()) == 1000

    def test_sweep_loop_bind_execute(self) -> None:
        theta = Parameter("theta")
        qc = _twist(theta)
        counts_for_value: dict[float, dict[str, int]] = {}
        for value in (0.0, 0.5, 1.0, np.pi):
            bound = qc.bind_parameters({theta: float(value)})
            result = StatevectorBackend().run(bound, shots=50, seed=1)
            counts_for_value[float(value)] = result.counts
        assert sum(counts_for_value[0.0].values()) == 50
        assert any(v > 0 for v in counts_for_value[np.pi].values())


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestParameterSerializationMQ12:
    def test_json_round_trip_parameter_gate(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).h(0).ry(theta, 1).cx(0, 1).measure_all()
        restored = QuantumCircuit.from_json(qc.to_json())
        assert restored.num_gates == qc.num_gates
        assert [p.name for p in restored.parameters] == ["theta"]
        assert restored.measurements == qc.measurements

    def test_json_round_trip_expression_gate(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(2 * theta + 0.5, 0)
        restored = QuantumCircuit.from_json(qc.to_json())
        bound = restored.bind_parameters({theta: 0.25})
        numeric = QuantumCircuit(1).ry(1.0, 0)
        assert np.allclose(bound.run().amplitudes, numeric.run().amplitudes)

    def test_param_spec_in_mixed_circuit(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(3).ry(theta, 0).h(1).rx(phi, 2).swap(1, 2)
        data = json.loads(qc.to_json())
        parameterized = [g for g in data["gates"] if g.get("parameterized")]
        assert len(parameterized) == 2
        assert data["num_gates"] == 4

    def test_reloaded_parameters_bind_by_original_name(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        restored = QuantumCircuit.from_json(qc.to_json())
        bound = restored.bind_parameters({theta: np.pi, phi: 0.0})
        state = bound.run()
        assert abs(state.amplitudes[2]) == pytest.approx(1.0)

    def test_bound_state_equivalence_after_round_trip(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0).cx(0, 1)
        original_state = qc.bind_parameters({theta: 1.7}).run()
        restored = QuantumCircuit.from_json(qc.to_json())
        restored_state = restored.bind_parameters({theta: 1.7}).run()
        assert np.allclose(
            original_state.amplitudes, restored_state.amplitudes
        )

    def test_save_load_round_trip(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "param_circuit.json"
            qc.save(path)
            loaded = QuantumCircuit.load(path)
            assert [p.name for p in loaded.parameters] == ["theta"]


# ---------------------------------------------------------------------------
# OpenQASM export policy
# ---------------------------------------------------------------------------


class TestParameterQasmMQ12:
    def test_exporting_parameterized_circuit_raises(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="bind_parameters"):
            qc.qasm()

    def test_core_to_qasm_raises(self) -> None:
        from microquantum.core.qasm import to_qasm

        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(2 * theta, 0)
        with pytest.raises(ValueError, match="no symbolic parameters"):
            to_qasm(qc)

    def test_bound_circuit_exports_and_round_trips(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0).cx(0, 1)
        bound = qc.bind_parameters({theta: 1.7})
        qasm_str = bound.qasm()
        assert "ry(" in qasm_str
        rebuilt = QuantumCircuit.from_qasm(qasm_str)
        assert rebuilt.num_gates == 2


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestParameterCompositionMQ12:
    def test_same_name_parameters_merge_on_addition(self) -> None:
        # Parameter identity is name-based, so theta in both circuits is the
        # same logical parameter after concatenation.
        theta = Parameter("theta")
        left = QuantumCircuit(1).ry(theta, 0)
        right = QuantumCircuit(1).rz(theta, 0)
        combined = left + right
        assert combined.parameters == (theta,)
        assert len(combined.parameters) == 1

    def test_composition_bind_numeric_equivalence(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        combined = (
            QuantumCircuit(2).ry(theta, 0)
            + QuantumCircuit(2).rx(phi, 1).cx(0, 1)
        )
        bound = combined.bind_parameters({theta: 0.3, phi: 0.9})
        numeric = (
            QuantumCircuit(2).ry(0.3, 0)
            + QuantumCircuit(2).rx(0.9, 1).cx(0, 1)
        )
        assert np.allclose(bound.run().amplitudes, numeric.run().amplitudes)

    def test_measurements_preserved_through_composition_and_binding(self) -> None:
        theta = Parameter("theta")
        qc1 = QuantumCircuit(2).ry(theta, 0).measure(0)
        qc2 = QuantumCircuit(2).cx(0, 1)
        combined = qc1 + qc2
        bound = combined.bind_parameters({theta: 0.5})
        assert bound.measurements == [0]