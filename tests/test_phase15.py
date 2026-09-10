"""Tests for Phase 15: GPU-aware tensor-network simulation.

Covers the pluggable NumPy/CuPy array backend, the matrix-product-state
simulator (exact + truncated), the balanced tree-tensor-network
simulator, and Executor integration.  All tests run on the NumPy array
backend and require no GPU or network access.
"""
import numpy as np
import pytest

from microquantum import (
    Executor,
    MPSBackend,
    MatrixProductState,
    QuantumCircuit,
    StateVector,
    StatevectorBackend,
    TreeTensorNetwork,
    TreeTensorNetworkBackend,
    available_backends,
    get_array_backend,
    gpu_available,
    is_gpu,
    set_array_backend,
)
from microquantum.backends.array_backend import xp
from microquantum.backends.base import BackendResult


def gates_of(circ: QuantumCircuit) -> list[tuple[np.ndarray, list[int]]]:
    """Extract bound gate matrices and target lists from a circuit."""
    out = []
    for ins in circ._gate_instructions:
        if QuantumCircuit._is_parameterized_gate(ins):
            continue
        g, t = ins
        out.append((np.asarray(g.matrix, dtype=np.complex128), list(t)))
    return out


def reference_statevector(circ: QuantumCircuit) -> np.ndarray:
    """Dense reference state via the statevector backend."""
    res = StatevectorBackend().run_circuit(
        circ.num_qubits, gates_of(circ), shots=1, seed=0
    )
    return np.asarray(res.statevector, dtype=np.complex128)


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def random_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Circuit with entangled RY rotations and arbitrary CNOT pairs."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        if rng.random() < 0.5:
            qc.ry(rng.random() * np.pi, int(rng.integers(n)))
        else:
            a, b = rng.choice(n, 2, replace=False)
            qc.cx(int(a), int(b))
    return qc


# =============================================================================
# Pluggable array backend
# =============================================================================
class TestArrayBackend:
    def test_default_is_numpy(self) -> None:
        assert get_array_backend() == "numpy"
        assert "numpy" in available_backends()
        assert not is_gpu()

    def test_set_numpy_is_noop(self) -> None:
        assert set_array_backend("numpy") == "numpy"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            set_array_backend("torch")

    def test_cupy_fallback_with_warning(self) -> None:
        # CuPy is not a hard dependency: requesting it without an install
        # must degrade gracefully to NumPy.
        if gpu_available():
            pytest.skip("CuPy installed; fallback not testable")
        with pytest.warns(RuntimeWarning):
            assert set_array_backend("cupy") == "numpy"
        assert get_array_backend() == "numpy"

    def test_dispatch_helpers_match_numpy(self) -> None:
        mod = xp()
        z = mod.zeros((2, 2), dtype=complex)
        assert z.shape == (2, 2)
        eye = mod.eye(3)
        assert np.allclose(np.asarray(eye), np.eye(3))
        a = mod.zeros((4, 4), dtype=complex)
        a[0, 1] = 1 + 2j
        assert np.asarray(a)[0, 1] == 1 + 2j


# =============================================================================
# MatrixProductState simulator
# =============================================================================
class TestMatrixProductState:
    def test_bell_exact(self) -> None:
        qc = bell_circuit()
        mps = MatrixProductState.from_zeros(2)
        for g, t in gates_of(qc):
            mps.apply_single(g, t[0]) if len(t) == 1 else mps.apply_two(g, list(t))
        assert mps.fidelity_to(reference_statevector(qc)) > 1 - 1e-12
        assert mps.bond_dims == [2]

    def test_ghz_exact(self) -> None:
        qc = QuantumCircuit(6)
        qc.h(0)
        for k in range(1, 6):
            qc.cx(0, k)
        mps = MatrixProductState.from_zeros(6)
        for g, t in gates_of(qc):
            if len(t) == 1:
                mps.apply_single(g, t[0])
            else:
                mps.apply_two(g, list(t))
        assert mps.fidelity_to(reference_statevector(qc)) > 1 - 1e-12

    def test_ghz_with_bond_cap_stays_ghz(self) -> None:
        """GHZ only needs bond 2, so capping at 2 stays exact."""
        qc = QuantumCircuit(4)
        qc.h(0)
        for k in range(1, 4):
            qc.cx(0, k)
        mps = MatrixProductState.from_zeros(4)
        for g, t in gates_of(qc):
            if len(t) == 1:
                mps.apply_single(g, t[0])
            else:
                mps.apply_two(g, list(t), max_bond_dim=2, truncation_threshold=0.0)
        assert mps.truncation_error == 0.0
        assert mps.fidelity_to(reference_statevector(qc)) > 1 - 1e-12

    def test_random_circuit_exact(self) -> None:
        qc = random_circuit(5, 30, seed=7)
        mps = MatrixProductState.from_zeros(5)
        for g, t in gates_of(qc):
            mps.apply_single(g, t[0]) if len(t) == 1 else mps.apply_two(g, list(t))
        assert mps.fidelity_to(reference_statevector(qc)) > 1 - 1e-9

    def test_capped_circuit_trades_fidelity_for_speed(self) -> None:
        qc = random_circuit(6, 40, seed=7)
        mps = MatrixProductState.from_zeros(6)
        for g, t in gates_of(qc):
            if len(t) == 1:
                mps.apply_single(g, t[0])
            else:
                mps.apply_two(g, list(t), max_bond_dim=3, truncation_threshold=0.0)
        assert mps.truncation_error > 0
        assert mps.truncation_error < 0.15
        assert mps.fidelity_to(reference_statevector(qc)) > 0.85

    def test_sampling_bell_distribution(self) -> None:
        qc = bell_circuit()
        mps = MatrixProductState.from_zeros(2)
        for g, t in gates_of(qc):
            if len(t) == 1:
                mps.apply_single(g, t[0])
            else:
                mps.apply_two(g, list(t))
        counts = mps.sample(4000, seed=1)
        assert abs(counts.get("00", 0) - 2000) < 300
        assert abs(counts.get("11", 0) - 2000) < 300

    def test_from_statevector_roundtrip(self) -> None:
        v = np.zeros(8, dtype=complex)
        for i, val in [(0, 0.5), (3, 0.5j), (7, 0.5), (4, -0.5)]:
            v[i] = val
        v /= np.linalg.norm(v)
        mps = MatrixProductState.from_statevector(v, 3)
        assert mps.fidelity_to(v) > 1 - 1e-9

    def test_statevector_guard(self) -> None:
        mps = MatrixProductState.from_zeros(19)
        assert not mps.valid_statevector()


# =============================================================================
# MatrixProductStateBackend
# =============================================================================
class TestMPSBackend:
    def test_basic_run(self) -> None:
        bk = MPSBackend(seed=3)
        qc = bell_circuit()
        res = bk.run_circuit(2, gates_of(qc), shots=1024, seed=3)
        assert isinstance(res, BackendResult)
        assert res.backend_name == "mps"
        assert res.statevector is not None

    def test_state_matches_statevector_backend(self) -> None:
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(1, 3)
        qc.ry(0.7, 2)
        res = MPSBackend().run_circuit(4, gates_of(qc), shots=1, seed=0)
        fid = float(abs(np.vdot(res.statevector, reference_statevector(qc))) ** 2)
        assert fid > 1 - 1e-9

    def test_initial_state(self) -> None:
        init = StateVector(2, np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]))
        res = MPSBackend().run_circuit(2, [], shots=1, seed=0, initial_state=init)
        assert np.allclose(np.abs(res.statevector), [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

    def test_large_system_no_statevector(self) -> None:
        qc = QuantumCircuit(19)
        qc.h(0)
        res = MPSBackend().run_circuit(19, gates_of(qc), shots=1, seed=0)
        assert res.statevector is None
        assert res.metadata["statevector_available"] is False


# =============================================================================
# TreeTensorNetwork simulator
# =============================================================================
class TestTreeTensorNetwork:
    def test_bell_exact(self) -> None:
        qc = bell_circuit()
        ttn = TreeTensorNetwork.from_zeros(2)
        for g, t in gates_of(qc):
            ttn.apply_single(g, t[0]) if len(t) == 1 else ttn.apply_two(g, list(t))
        assert ttn.fidelity_to(reference_statevector(qc)) > 1 - 1e-12

    def test_ghz_exact(self) -> None:
        qc = QuantumCircuit(6)
        qc.h(0)
        for k in range(1, 6):
            qc.cx(0, k)
        ttn = TreeTensorNetwork.from_zeros(6)
        for g, t in gates_of(qc):
            if len(t) == 1:
                ttn.apply_single(g, t[0])
            else:
                ttn.apply_two(g, list(t))
        assert ttn.fidelity_to(reference_statevector(qc)) > 1 - 1e-9

    def test_random_circuit_exact_with_reversed_cnots(self) -> None:
        qc = random_circuit(5, 30, seed=7)
        ttn = TreeTensorNetwork.from_zeros(5)
        for g, t in gates_of(qc):
            ttn.apply_single(g, t[0]) if len(t) == 1 else ttn.apply_two(g, list(t))
        assert ttn.fidelity_to(reference_statevector(qc)) > 1 - 1e-9

    def test_reversed_control_order(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(2)
        qc.cx(2, 0)
        ttn = TreeTensorNetwork.from_zeros(3)
        for g, t in gates_of(qc):
            ttn.apply_single(g, t[0]) if len(t) == 1 else ttn.apply_two(g, list(t))
        assert ttn.fidelity_to(reference_statevector(qc)) > 1 - 1e-9

    def test_sampling_matches_statevector_backend(self) -> None:
        qc = bell_circuit()
        ttn = TreeTensorNetwork.from_zeros(2)
        for g, t in gates_of(qc):
            if len(t) == 1:
                ttn.apply_single(g, t[0])
            else:
                ttn.apply_two(g, list(t))
        counts = ttn.sample(4000, seed=1)
        assert abs(counts.get("00", 0) - 2000) < 300
        assert abs(counts.get("11", 0) - 2000) < 300

    def test_from_statevector_roundtrip(self) -> None:
        v = np.zeros(8, dtype=complex)
        for i, val in [(0, 0.5), (3, 0.5j), (7, 0.5), (4, -0.5)]:
            v[i] = val
        v /= np.linalg.norm(v)
        ttn = TreeTensorNetwork.from_statevector(v, 3)
        assert ttn.root.tensor.shape == (1, 2, 2)
        assert ttn.fidelity_to(v) > 1 - 1e-9

    def test_truncation_bound(self) -> None:
        qc = random_circuit(6, 40, seed=7)
        ttn = TreeTensorNetwork.from_zeros(6)
        for g, t in gates_of(qc):
            if len(t) == 1:
                ttn.apply_single(g, t[0])
            else:
                ttn.apply_two(g, list(t), max_bond_dim=3, truncation_threshold=0.0)
        assert 0 < ttn.truncation_error < 1.0
        assert ttn.fidelity_to(reference_statevector(qc)) > 0.3


# =============================================================================
# TreeTensorNetworkBackend
# =============================================================================
class TestTreeTensorNetworkBackend:
    def test_basic_run(self) -> None:
        bk = TreeTensorNetworkBackend(seed=3)
        qc = bell_circuit()
        res = bk.run_circuit(2, gates_of(qc), shots=1024, seed=3)
        assert isinstance(res, BackendResult)
        assert res.backend_name == "ttn"
        assert res.statevector is not None
        assert res.metadata["statevector_available"] is True

    def test_counts_match_statevector_backend(self) -> None:
        qc = bell_circuit()
        t = TreeTensorNetworkBackend(seed=3)
        s = StatevectorBackend()
        rc = t.run_circuit(2, gates_of(qc), shots=1024, seed=3)
        sv = s.run_circuit(2, gates_of(qc), shots=1024, seed=3)
        assert rc.counts == sv.counts

    def test_probabilities(self) -> None:
        qc = bell_circuit()
        res = TreeTensorNetworkBackend().run_circuit(2, gates_of(qc), shots=1000, seed=0)
        probs = res.probabilities
        assert abs(probs.get("00", 0.0) - 0.5) < 0.1
        assert abs(probs.get("11", 0.0) - 0.5) < 0.1


# =============================================================================
# Executor integration
# =============================================================================
class TestExecutorIntegration:
    @pytest.mark.parametrize("backend", [MPSBackend(), TreeTensorNetworkBackend()])
    def test_both_backends_via_executor(self, backend) -> None:
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 3)
        qc.ry(0.7, 2)
        res = Executor(backend).run(qc, shots=1000, seed=7)
        assert res.metadata["backend"] == backend.name
        assert res.statevector is not None
        fid = float(abs(np.vdot(res.statevector, reference_statevector(qc))) ** 2)
        assert fid > 1 - 1e-9
        assert all(isinstance(v, int) for v in res.counts.values())