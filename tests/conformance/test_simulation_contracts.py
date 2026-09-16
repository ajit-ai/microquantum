"""Conformance - simulation contracts (MQ-14).

Permanent public-face checks for the expanded simulator family:

* deterministic ``shots=None`` execution across every simulator backend,
* strict shots validation (never silently accepted),
* capability / target advertisement (incl. the density-matrix mode),
* dense-simulator memory boundaries and the TTN sampling cap,
* Executor conflict rules and noise-model typing,
* cross-simulator parity of exact results,
* BackendResult serialization round trips.

Every assertion matches behavior documented in
``docs/execution/simulation.rst``.
"""

from __future__ import annotations

import numpy as np
import pytest

from microquantum import (
    BackendResult,
    DensityMatrix,
    DensityMatrixBackend,
    ExecutionPlan,
    Executor,
    MockBackend,
    MPSBackend,
    NoiseModel,
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

ALL_SIMULATORS = [
    StatevectorBackend(),
    DensityMatrixBackend(),
    MPSBackend(),
    TreeTensorNetworkBackend(),
    MockBackend(),
]

EXACT_SIMULATORS = [
    StatevectorBackend(),
    DensityMatrixBackend(),
    MPSBackend(),
    TreeTensorNetworkBackend(),
]


def bell() -> QuantumCircuit:
    return QuantumCircuit(2).h(0).cnot(0, 1)


# ---------------------------------------------------------------------------
# Deterministic execution (shots=None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ALL_SIMULATORS, ids=lambda b: b.name)
def test_shots_none_means_deterministic(backend) -> None:
    result = backend.run(bell(), shots=None)
    assert result.shots is None
    assert result.counts == {}
    assert result.samples is None
    # An exact representation must always be present for real simulators.
    if backend.name != "mock":
        assert result.statevector is not None or result.density_matrix is not None


@pytest.mark.parametrize("backend", ALL_SIMULATORS, ids=lambda b: b.name)
def test_default_shots_is_1024(backend) -> None:
    result = backend.run(bell())
    assert result.shots == 1024
    assert sum(result.counts.values()) == 1024


@pytest.mark.parametrize("backend", ALL_SIMULATORS, ids=lambda b: b.name)
def test_invalid_shots_rejected_everywhere(backend) -> None:
    for bad in (0, -1, -10):
        with pytest.raises(ValueError):
            backend.run(bell(), shots=bad)
    with pytest.raises(ValueError):
        backend.run(bell(), shots="many")


@pytest.mark.parametrize("backend", ALL_SIMULATORS, ids=lambda b: b.name)
def test_seeded_sampling_reproducible(backend) -> None:
    a = backend.run(bell(), shots=1000, seed=7)
    b = backend.run(bell(), shots=1000, seed=7)
    assert a.counts == b.counts
    assert a.samples == b.samples
    assert a.seed == 7


def test_runtime_deterministic_and_validated() -> None:
    assert execute(bell(), shots=None).shots is None
    with pytest.raises(ValueError):
        execute(bell(), shots=0)
    plan = ExecutionPlan.from_circuit(bell(), shots=None)
    assert plan.shots is None
    assert execute(plan).shots is None


# ---------------------------------------------------------------------------
# Capability and target advertisement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", EXACT_SIMULATORS, ids=lambda b: b.name)
def test_simulator_capabilities_and_target(backend) -> None:
    assert backend.capabilities.is_simulator
    assert backend.target.name == f"{backend.name}_simulator"
    assert backend.target.supports_gate("cx")
    assert backend.target.supports_gate("rx")
    assert backend.capabilities.metadata["engine"] == backend.name
    assert backend.capabilities.metadata["device_type"] == "cpu"


def test_density_matrix_mode_advertised() -> None:
    dm = DensityMatrixBackend()
    assert dm.capabilities.supports_execution(EXECUTION_DENSITY_MATRIX)
    assert dm.capabilities.supports_execution(EXECUTION_STATEVECTOR)
    sv = StatevectorBackend()
    assert sv.capabilities.supports_execution(EXECUTION_STATEVECTOR)
    assert not sv.capabilities.supports_execution(EXECUTION_DENSITY_MATRIX)


# ---------------------------------------------------------------------------
# Executor contract
# ---------------------------------------------------------------------------


def test_executor_conflicting_noise_and_backend_rejected() -> None:
    with pytest.raises(ValueError):
        Executor(backend=StatevectorBackend(), noise_model=NoiseModel().bit_flip(0.1))


def test_executor_rejects_non_noise_model() -> None:
    with pytest.raises(TypeError):
        Executor(noise_model="junk")  # type: ignore[arg-type]


def test_executor_deterministic_noise_probabilities() -> None:
    model = NoiseModel().amplitude_damping(0.5)
    result = Executor(noise_model=model).run(QuantumCircuit(1).x(0), shots=None)
    assert result.shots is None
    assert result.counts == {}
    assert result.probabilities["0"] == pytest.approx(0.5, abs=1e-9)
    assert result.probabilities["1"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Memory boundaries and tensor-network limits
# ---------------------------------------------------------------------------


def test_dense_statevector_budget() -> None:
    # 2^27 * 16 bytes == 2 GiB exactly (allowed); the next size is refused.
    with pytest.raises(ValueError):
        StateVector(28)
    with pytest.raises(ValueError):
        StateVector(65)


def test_dense_density_matrix_budget() -> None:
    with pytest.raises(ValueError):
        DensityMatrix(14)
    with pytest.raises(ValueError):
        DensityMatrix(65)


def test_ttn_sampling_cap_fails_explicitly() -> None:
    with pytest.raises(ValueError):
        TreeTensorNetwork.from_zeros(19).sample(16)
    # Additionally, the backend refuses through a nominal sampling path.
    with pytest.raises(ValueError):
        TreeTensorNetworkBackend().run_circuit(num_qubits=20, gates=[], shots=8)


# ---------------------------------------------------------------------------
# Cross-simulator parity
# ---------------------------------------------------------------------------


def _probability_vector(result: BackendResult) -> np.ndarray:
    if result.statevector is not None:
        return np.abs(result.statevector) ** 2
    if result.density_matrix is not None:
        return np.real(np.diag(result.density_matrix))
    raise AssertionError("expected an exact result")


def test_exact_states_agree_across_simulators() -> None:
    qc = QuantumCircuit(3).ry(1.1, 0).ry(0.4, 1).cnot(0, 1).cnot(1, 2)
    vectors = {
        backend.name: _probability_vector(backend.run(qc, shots=None))
        for backend in EXACT_SIMULATORS
    }
    reference = vectors["statevector"]
    for name, prob in vectors.items():
        assert np.allclose(reference, prob, atol=1e-8), name


# ---------------------------------------------------------------------------
# Noise through the density-matrix backend
# ---------------------------------------------------------------------------


def test_density_noise_calibration() -> None:
    x = QuantumCircuit(1).x(0).gates[0][0].matrix
    model = NoiseModel().depolarizing(0.6)
    result = DensityMatrixBackend().run_circuit(
        num_qubits=1,
        gates=[(x, [0])],
        shots=3000,
        seed=4,
        noise_model=model,
    )
    # X then depolarizing p=0.6 -> P(0) = 2p/3 = 0.4.
    p0 = result.counts.get("0", 0) / 3000
    assert 0.34 < p0 < 0.46


def test_unsupported_noise_model_type_fails() -> None:
    with pytest.raises(TypeError):
        DensityMatrixBackend().run_circuit(
            num_qubits=1, gates=[], shots=16, noise_model=object()
        )


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def test_deterministic_result_serialization_roundtrip() -> None:
    result = StatevectorBackend().run(bell(), shots=None)
    rebuilt = BackendResult.from_dict(result.to_dict())
    assert rebuilt.shots is None
    assert rebuilt.counts == {}
    assert np.allclose(rebuilt.statevector, result.statevector)


def test_sampled_result_serialization_roundtrip() -> None:
    result = MPSBackend().run(bell(), shots=250, seed=3)
    rebuilt = BackendResult.from_dict(result.to_dict())
    assert rebuilt.counts == result.counts
    assert rebuilt.shots == 250
    assert rebuilt.samples == result.samples


def test_result_probabilities_and_most_frequent() -> None:
    result = StatevectorBackend().run(QuantumCircuit(1).x(0), shots=64)
    assert result.most_frequent() == "1"
    assert result.probabilities == {"1": 1.0}