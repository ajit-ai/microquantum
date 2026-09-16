"""MQ-14 simulation expansion: behavioral contract tests.

Covers the expanded simulator family contract end to end:

* deterministic ``shots=None`` execution (exact state, empty counts),
* consistent shots validation everywhere (never silently accepted),
* state-vector / density-matrix / MPS / TTN / mock cross-simulator parity,
* parameter binding, measurement annotations, seeds, and result API,
* noise integration through the density-matrix backend and the Executor,
* TB resource boundaries for dense simulators and the TTN sampling cap,
* explicit failure for silently-ignored options (noise models).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from microquantum import (
    Backend,
    BackendResult,
    DensityMatrix,
    DensityMatrixBackend,
    ExecutionPlan,
    Executor,
    MatrixProductState,
    MockBackend,
    MPSBackend,
    NoiseModel,
    Operator,
    Parameter,
    QuantumCircuit,
    StateVector,
    StatevectorBackend,
    TreeTensorNetwork,
    TreeTensorNetworkBackend,
    execute,
)
from microquantum.backends.capabilities import (
    EXECUTION_DENSITY_MATRIX,
    EXECUTION_STATEVECTOR,
)

AllBackends = (
    StatevectorBackend,
    DensityMatrixBackend,
    MPSBackend,
    TreeTensorNetworkBackend,
    MockBackend,
)

DenseBackends = (
    StatevectorBackend,
    DensityMatrixBackend,
    MPSBackend,
    TreeTensorNetworkBackend,
)
"""Backends exposing a real simulation engine (not the mock stub)."""


def bell() -> QuantumCircuit:
    """|00> + |11> (unnormalized)."""
    return QuantumCircuit(2).h(0).cnot(0, 1)


def x_single() -> QuantumCircuit:
    """X acting on a single qubit: |0> -> |1>."""
    return QuantumCircuit(1).x(0)


def run_and_assert_deterministic(backend: Backend) -> BackendResult:
    result = backend.run(bell(), shots=None)
    assert result.counts == {}
    assert result.samples is None
    assert result.shots is None
    return result


# ----------------------------------------------------------------------
# Uniform simulator contract
# ----------------------------------------------------------------------


class TestSimulatorContract:
    @pytest.mark.parametrize("factory", AllBackends)
    def test_deterministic_shots_none(self, factory: Callable[[], Backend]) -> None:
        """shots=None means deterministic: no sampling, no counts."""
        run_and_assert_deterministic(factory())

    @pytest.mark.parametrize("factory", AllBackends)
    def test_default_shots_unchanged(self, factory: Callable[[], Backend]) -> None:
        result = factory().run(bell())
        assert result.shots == 1024
        total = sum(result.counts.values())
        assert total == 1024

    @pytest.mark.parametrize("factory", AllBackends)
    def test_results_carry_backend_name_and_qubits(
        self, factory: Callable[[], Backend]
    ) -> None:
        backend = factory()
        result = backend.run(bell(), shots=64)
        assert result.backend_name == backend.name
        assert result.num_qubits == 2

    @pytest.mark.parametrize("factory", AllBackends)
    def test_invalid_shots_rejected(self, factory: Callable[[], Backend]) -> None:
        backend = factory()
        with pytest.raises(ValueError, match="shots"):
            backend.run(bell(), shots=0)
        with pytest.raises(ValueError, match="shots"):
            backend.run(bell(), shots=-5)
        with pytest.raises(ValueError, match="shots"):
            backend.run(bell(), shots="many")

    @pytest.mark.parametrize("factory", AllBackends)
    def test_seeded_sampling_is_reproducible(
        self, factory: Callable[[], Backend]
    ) -> None:
        backend = factory()
        first = backend.run(bell(), shots=1000, seed=42)
        second = backend.run(bell(), shots=1000, seed=42)
        assert first.counts == second.counts
        assert first.samples == second.samples
        assert first.seed == 42

    @pytest.mark.parametrize("factory", DenseBackends)
    def test_capability_and_target_advertisement(
        self, factory: Callable[[], Backend]
    ) -> None:
        backend = factory()
        assert backend.capabilities.is_simulator
        assert backend.target.name == f"{backend.name}_simulator"
        assert backend.target.supports_gate("cx")
        assert "engine" in backend.capabilities.metadata
        assert backend.capabilities.metadata["device_type"] == "cpu"

    def test_density_matrix_execution_advertised(self) -> None:
        dm = DensityMatrixBackend()
        assert dm.capabilities.supports_execution(EXECUTION_DENSITY_MATRIX)
        assert dm.capabilities.metadata["engine"] == "density_matrix"

    def test_statevector_mode_does_not_advertise_density(self) -> None:
        sv = StatevectorBackend()
        assert sv.capabilities.supports_execution(EXECUTION_STATEVECTOR)
        assert not sv.capabilities.supports_execution(EXECUTION_DENSITY_MATRIX)
        assert sv.capabilities.metadata["engine"] == "statevector"

    def test_measurement_subset_restricted(self) -> None:
        qc = bell()
        qc.measure(0)
        result = StatevectorBackend().run(qc, shots=300, seed=5)
        assert set(result.counts) <= {"0", "1"}
        assert result.metadata.get("measured_qubits") == [0]

    def test_no_measurement_annotation_still_runs(self) -> None:
        result = StatevectorBackend().run(bell(), shots=16)
        assert sum(result.counts.values()) == 16


def gate_matrix(qc: QuantumCircuit) -> np.ndarray:
    """Concrete matrix of the first gate, for ``run_circuit`` calls."""
    return qc.gates[0][0].matrix


# ----------------------------------------------------------------------
# State-vector numerical correctness
# ----------------------------------------------------------------------


class TestStatevectorKnownStates:
    def test_hadamard_plus_state(self) -> None:
        result = StatevectorBackend().run(QuantumCircuit(1).h(0), shots=None)
        sv = result.statevector
        assert abs(abs(sv[0]) ** 2 - 0.5) < 1e-12
        assert abs(abs(sv[1]) ** 2 - 0.5) < 1e-12

    def test_x_flip_is_deterministic(self) -> None:
        result = StatevectorBackend().run(x_single(), shots=500)
        assert result.counts == {"1": 500}
        sv = result.statevector
        assert np.allclose(sv, [0.0, 1.0])

    def test_bell_state_amplitudes(self) -> None:
        result = StatevectorBackend().run(bell(), shots=None)
        sv = result.statevector
        assert abs(abs(sv[0]) ** 2 - 0.5) < 1e-12
        assert abs(abs(sv[1]) ** 2) < 1e-12
        assert abs(abs(sv[2]) ** 2) < 1e-12
        assert abs(abs(sv[3]) ** 2 - 0.5) < 1e-12

    def test_bell_counts_balanced(self) -> None:
        result = StatevectorBackend().run(bell(), shots=2000, seed=3)
        p00 = result.counts.get("00", 0) / 2000
        p11 = result.counts.get("11", 0) / 2000
        assert 0.42 < p00 < 0.58
        assert 0.42 < p11 < 0.58

    def test_empty_circuit_stays_in_zero(self) -> None:
        result = StatevectorBackend().run(QuantumCircuit(3), shots=200)
        assert result.counts == {"000": 200}

    def test_toffoli_truth_table(self) -> None:
        # Build the 3-qubit Toffoli (CCX): flips the target (q2, index bit 0)
        # when both controls (q0 => bit 2, q1 => bit 1) are set.
        toffoli = np.zeros((8, 8), dtype=np.complex128)
        for i in range(8):
            out = i ^ (1 if (i & 4) and (i & 2) else 0)
            toffoli[out, i] = 1.0
        amplitudes = np.zeros(8, dtype=np.complex128)
        amplitudes[7] = 1.0  # |111>
        initial = StateVector(num_qubits=3, amplitudes=amplitudes)
        result = StatevectorBackend().run_circuit(
            num_qubits=3,
            gates=[(toffoli, [0, 1, 2])],
            shots=None,
            initial_state=initial,
        )
        expected = np.zeros(8, dtype=np.complex128)
        expected[6] = 1.0  # |110>
        assert np.allclose(result.statevector, expected)


# ----------------------------------------------------------------------
# Parameterized circuits
# ----------------------------------------------------------------------


class TestStatevectorParameters:
    def test_bound_parameter_executes(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        result = StatevectorBackend().run(qc, parameter_values={theta: np.pi}, shots=500)
        assert result.counts == {"1": 500}

    def test_unbound_parameter_rejected(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        with pytest.raises(ValueError):
            StatevectorBackend().run(qc)

    def test_parameter_insertion_after_execution(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        bound = qc.bind_parameters({theta: 0.0})
        result = StatevectorBackend().run(bound, shots=400)
        assert result.counts == {"0": 400}

    def test_same_circuit_different_parameters(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        zero = StatevectorBackend().run(qc, parameter_values={theta: 0.0}, shots=200)
        one = StatevectorBackend().run(qc, parameter_values={theta: np.pi}, shots=200)
        assert zero.counts == {"0": 200}
        assert one.counts == {"1": 200}


# ----------------------------------------------------------------------
# Density-matrix simulation
# ----------------------------------------------------------------------


class TestDensityMatrixEquivalent:
    def test_pure_state_matches_statevector(self) -> None:
        qc = QuantumCircuit(2).h(0).cnot(0, 1)
        sv = StatevectorBackend().run(qc, shots=None).statevector
        dm = DensityMatrixBackend().run(qc, shots=None).density_matrix
        assert np.allclose(np.abs(sv) ** 2, np.diag(np.real(dm)), atol=1e-12)

    def test_result_carries_density_matrix(self) -> None:
        result = DensityMatrixBackend().run(bell(), shots=None)
        assert result.density_matrix.shape == (4, 4)
        assert np.allclose(np.trace(result.density_matrix).real, 1.0, atol=1e-12)

    def test_density_sampling_follows_diagonal(self) -> None:
        result = DensityMatrixBackend().run(bell(), shots=2000, seed=6)
        p00 = result.counts.get("00", 0) / 2000
        assert 0.42 < p00 < 0.58

    def test_mixed_state_expectation(self) -> None:
        # X then depolarizing p=3/4 leaves P(1) = 1 - 2p/3 = 0.5 on |1>.
        model = NoiseModel().depolarizing(0.75)
        result = DensityMatrixBackend().run_circuit(
            num_qubits=1,
            gates=[(gate_matrix(x_single()), [0])],
            shots=4000,
            seed=11,
            noise_model=model,
        )
        p1 = result.counts.get("1", 0) / 4000
        assert 0.44 < p1 < 0.56


# ----------------------------------------------------------------------
# Noise integration
# ----------------------------------------------------------------------


class TestNoiseIntegration:
    def test_invalid_noise_model_type_raises(self) -> None:
        backend = DensityMatrixBackend()
        with pytest.raises(TypeError, match="NoiseModel"):
            backend.run_circuit(num_qubits=2, gates=[], shots=16, noise_model="junk")
        with pytest.raises(TypeError, match="NoiseModel"):
            backend.run_circuit(
                num_qubits=2, gates=[], shots=16, noise_model=42
            )

    def test_zero_noise_is_noise(self) -> None:
        matrix = gate_matrix(x_single())
        clean = DensityMatrixBackend().run_circuit(
            num_qubits=1, gates=[(matrix, [0])], shots=None
        ).density_matrix
        model = NoiseModel().depolarizing(0.0)
        noisy = DensityMatrixBackend().run_circuit(
            num_qubits=1,
            gates=[(matrix, [0])],
            shots=None,
            noise_model=model,
        ).density_matrix
        assert np.allclose(clean, noisy)

    def test_depolarizing_probability_calibrated(self) -> None:
        model = NoiseModel().depolarizing(0.6)
        result = DensityMatrixBackend().run_circuit(
            num_qubits=1,
            gates=[(gate_matrix(x_single()), [0])],
            shots=3000,
            seed=4,
            noise_model=model,
        )
        # X then depolarizing p=0.6 leaves P(0) = 2p/3 = 0.4.
        p0 = result.counts.get("0", 0) / 3000
        assert 0.34 < p0 < 0.46

    def test_executor_noise_deterministic_probabilities(self) -> None:
        model = NoiseModel().amplitude_damping(0.5)
        result = Executor(noise_model=model).run(x_single(), shots=None)
        assert result.shots is None
        assert result.counts == {}
        assert result.statevector is None
        assert result.probabilities.get("0", 0.0) == pytest.approx(0.5, abs=1e-9)
        assert result.probabilities.get("1", 0.0) == pytest.approx(0.5, abs=1e-9)

    def test_executor_noise_sampling_consistent(self) -> None:
        model = NoiseModel().bit_flip(0.2)
        result = Executor(noise_model=model).run(x_single(), shots=4000, seed=8)
        # X then bit-flip p=0.2 -> P(0) = p = 0.2.
        p0 = result.probabilities.get("0", 0.0)
        assert abs(p0 - 0.2) < 0.03

    def test_executor_rejects_unknown_noise(self) -> None:
        with pytest.raises(TypeError, match="NoiseModel"):
            Executor(noise_model="junk")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Tensor-network simulators
# ----------------------------------------------------------------------


class TestTensorNetworkSmallCircuits:
    def test_mps_matches_statevector(self) -> None:
        qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
        sv = StatevectorBackend().run(qc, shots=None).statevector
        mps = MPSBackend().run(qc, shots=None).statevector
        assert sv is not None and mps is not None
        assert abs(abs(np.vdot(sv, mps)) ** 2 - 1.0) < 1e-9

    def test_mps_truncation_tracked(self) -> None:
        result = MPSBackend(max_bond_dim=1).run(bell(), shots=None)
        assert result.metadata["max_bond_dim"] == 1
        assert result.metadata["truncation_error"] >= 0.0
        assert result.metadata["statevector_available"] is True

    def test_ttn_matches_statevector(self) -> None:
        qc = QuantumCircuit(2).h(0).cnot(0, 1)
        sv = StatevectorBackend().run(qc, shots=None).statevector
        ttn = TreeTensorNetworkBackend().run(qc, shots=None).statevector
        assert sv is not None and ttn is not None
        assert np.allclose(sv, ttn, atol=1e-10)

    def test_ttn_sampling_guard(self) -> None:
        tn = TreeTensorNetwork.from_zeros(19)
        with pytest.raises(ValueError, match="MatrixProductState"):
            tn.sample(16)

    def test_ttn_backend_large_n_deterministic(self) -> None:
        result = TreeTensorNetworkBackend().run_circuit(
            num_qubits=20, gates=[], shots=None
        )
        assert result.counts == {}
        assert result.statevector is None

    def test_ttn_backend_large_n_sampling_fails_explicitly(self) -> None:
        with pytest.raises(ValueError, match="MatrixProductState"):
            TreeTensorNetworkBackend().run_circuit(
                num_qubits=20, gates=[], shots=8
            )

    def test_mps_large_n_sampling_via_peeling(self) -> None:
        result = MPSBackend().run_circuit(
            num_qubits=19, gates=[], shots=32, seed=1
        )
        assert sum(result.counts.values()) == 32
        assert result.statevector is None

    def test_mps_large_n_deterministic(self) -> None:
        result = MPSBackend().run_circuit(num_qubits=19, gates=[], shots=None)
        assert result.counts == {}
        assert result.statevector is None


# ----------------------------------------------------------------------
# Cross-simulator consistency
# ----------------------------------------------------------------------


class TestCrossSimulatorConsistency:
    SIMS = {
        "statevector": lambda: StatevectorBackend(),
        "density_matrix": lambda: DensityMatrixBackend(),
        "mps": lambda: MPSBackend(),
        "ttn": lambda: TreeTensorNetworkBackend(),
    }

    @staticmethod
    def _state_probs(result: BackendResult) -> np.ndarray:
        """Probability vector from whichever exact representation is present."""
        if result.statevector is not None:
            return np.abs(result.statevector) ** 2
        if result.density_matrix is not None:
            return np.real(np.diag(result.density_matrix))
        raise AssertionError("expected an exact (deterministic) result")

    def test_deterministic_states_agree(self) -> None:
        qc = QuantumCircuit(3).ry(1.1, 0).ry(0.4, 1).cnot(0, 1).cnot(1, 2)
        probs: dict[str, np.ndarray] = {}
        for name, factory in self.SIMS.items():
            result = factory().run(qc, shots=None)
            probs[name] = self._state_probs(result)
        for name, prob in probs.items():
            assert np.allclose(probs["statevector"], prob, atol=1e-8), name

    def test_seeded_sampling_statistically_consistent(self) -> None:
        qc = bell()
        reference = StatevectorBackend().run(qc, shots=4000, seed=21).counts
        for name, factory in self.SIMS.items():
            counts = factory().run(qc, shots=4000, seed=21).counts
            assert set(counts) == set(reference), name
            for key in reference:
                diff = abs(counts[key] / 4000 - reference[key] / 4000)
                assert diff < 0.05, (name, key)

    def test_statevector_and_ttn_share_sampling_path(self) -> None:
        # Both draw from the same default_rng + choice stream on the same p.
        qc = QuantumCircuit(1).ry(2 * np.pi / 3, 0)
        sv = StatevectorBackend().run(qc, shots=1200, seed=13).counts
        ttn = TreeTensorNetworkBackend().run(qc, shots=1200, seed=13).counts
        for key in set(sv) | set(ttn):
            assert abs(sv.get(key, 0) - ttn.get(key, 0)) <= 2, key


# ----------------------------------------------------------------------
# Result API
# ----------------------------------------------------------------------


class TestResultApi:
    def test_backend_result_serialization_roundtrip(self) -> None:
        result = StatevectorBackend().run(bell(), shots=None)
        data = result.to_dict()
        rebuilt = BackendResult.from_dict(data)
        assert rebuilt.shots is None
        assert rebuilt.counts == {}
        assert np.allclose(rebuilt.statevector, result.statevector)

    def test_backend_result_sampled_roundtrip(self) -> None:
        result = DensityMatrixBackend().run(bell(), shots=250, seed=31)
        rebuilt = BackendResult.from_dict(result.to_dict())
        assert rebuilt.counts == result.counts
        assert np.allclose(rebuilt.density_matrix, result.density_matrix)

    def test_backend_result_probabilities_property(self) -> None:
        result = StatevectorBackend().run(bell(), shots=200, seed=17)
        probs = result.probabilities
        assert sum(probs.values()) == pytest.approx(1.0)
        assert set(probs) <= {"00", "11"}

    def test_backend_result_most_frequent(self) -> None:
        result = StatevectorBackend().run(x_single(), shots=50)
        assert result.most_frequent() == "1"
        deterministic = StatevectorBackend().run(bell(), shots=None)
        with pytest.raises(ValueError):
            deterministic.most_frequent()

    def test_executor_result_json_safe(self) -> None:
        import json

        result = Executor(noise_model=NoiseModel().depolarizing(0.1)).run(
            QuantumCircuit(1).x(0), shots=100, seed=2
        )
        payload = json.loads(result.to_json())
        assert payload["shots"] == 100
        assert payload["metadata"]["backend"] == "density_matrix"
        assert result.to_dict()["shots"] == 100

    def test_executor_expectation_from_probabilities(self) -> None:
        z = Operator(np.diag([1.0, -1.0]))
        result = Executor().run(x_single(), shots=1000, seed=15)
        assert result.expectation(z) == pytest.approx(-1.0, abs=0.001)


# ----------------------------------------------------------------------
# Resource boundaries
# ----------------------------------------------------------------------


class TestResourceBoundaries:
    def test_statevector_budget_enforced(self) -> None:
        with pytest.raises(ValueError, match="state vector"):
            StateVector(28)  # ~4 GiB, above the 2 GiB budget -> refused
        with pytest.raises(ValueError):
            StateVector(65)

    def test_density_matrix_budget_enforced(self) -> None:
        with pytest.raises(ValueError, match="density matrix"):
            DensityMatrix(14)  # 2^28 elements, above the budget
        with pytest.raises(ValueError):
            DensityMatrix(65)
        small = DensityMatrix(1, matrix=np.eye(2, dtype=np.complex128))
        assert small.num_qubits == 1

    def test_caller_supplied_arrays_not_blocked(self) -> None:
        amplitudes = np.zeros(4, dtype=np.complex128)
        amplitudes[2] = 1.0
        state = StateVector(num_qubits=2, amplitudes=amplitudes)
        assert np.allclose(state.amplitudes, amplitudes)

    def test_invalid_qubit_counts_rejected(self) -> None:
        with pytest.raises(ValueError):
            StateVector(0)
        with pytest.raises(ValueError):
            StateVector(-1)
        with pytest.raises(ValueError):
            StateVector(2.5)
        with pytest.raises(ValueError):
            DensityMatrix(0)

    def test_tuple_tensor_networks_unaffected(self) -> None:
        assert MatrixProductState.from_zeros(30).num_qubits == 30
        assert TreeTensorNetwork.from_zeros(30).num_qubits == 30


# ----------------------------------------------------------------------
# Public API consistency
# ----------------------------------------------------------------------


class TestPublicApiConsistency:
    def test_executor_conflict_raises(self) -> None:
        with pytest.raises(ValueError, match="both"):
            Executor(backend=StatevectorBackend(), noise_model=NoiseModel().bit_flip(0.1))

    def test_executor_via_backend_deterministic(self) -> None:
        result = Executor(backend=TreeTensorNetworkBackend()).run(
            QuantumCircuit(2).h(0).cnot(0, 1), shots=None
        )
        assert result.shots is None
        assert result.counts == {}
        assert result.statevector is not None

    def test_execution_plan_deterministic(self) -> None:
        plan = ExecutionPlan.from_circuit(bell(), shots=None)
        assert plan.shots is None
        result = execute(plan)
        assert result.shots is None
        assert result.metadata.get("shots") is None

    def test_runtime_deterministic(self) -> None:
        result = execute(bell(), shots=None)
        assert result.shots is None
        assert result.counts == {}

    def test_runtime_invalid_shots_rejected(self) -> None:
        with pytest.raises(ValueError):
            execute(bell(), shots=0)

    def test_submit_deterministic(self) -> None:
        from microquantum import submit

        job = submit(bell(), shots=None)
        assert job.result is not None
        assert job.result.shots is None
        assert job.result.counts == {}

    def test_density_backend_noise_is_valid_option(self) -> None:
        model = NoiseModel().depolarizing(0.25)
        result = DensityMatrixBackend().run_circuit(
            num_qubits=1,
            gates=[(gate_matrix(QuantumCircuit(1).h(0)), [0])],
            shots=2000,
            seed=19,
            noise_model=model,
        )
        # Depolarizing p=0.25 on |+> -> P(0) = P(1) = 0.5.
        p0 = result.counts.get("0", 0) / 2000
        assert 0.44 < p0 < 0.56