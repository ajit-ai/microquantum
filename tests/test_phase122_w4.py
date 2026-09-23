"""Phase 122 W4: hardware-aware transpiler passes."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core.architecture import linear_architecture
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.transpiler import (
    CommutationAwareCancellation,
    NoiseAwareLayout,
    PassContext,
    SwapRoutingPass,
)


def _unitary(circuit: QuantumCircuit) -> np.ndarray:
    """Dense unitary of a measurement-free circuit."""
    return np.asarray(circuit.get_unitary().matrix, dtype=np.complex128)


def _assert_routed_equivalent(
    routed: QuantumCircuit, original: QuantumCircuit, layout: dict[int, int]
) -> None:
    """Assert ``U_routed = M · U_orig`` for the logical-to-slot map."""
    num_qubits = original.num_qubits
    dim = 2**num_qubits
    perm = np.zeros((dim, dim), dtype=np.complex128)
    for logical_index in range(dim):
        slot_index = 0
        for qubit in range(num_qubits):
            bit = (logical_index >> (num_qubits - 1 - qubit)) & 1
            slot_index |= bit << (num_qubits - 1 - layout[qubit])
        perm[slot_index, logical_index] = 1.0
    assert np.allclose(_unitary(routed), perm @ _unitary(original), atol=1e-9)


class TestSwapRoutingPass:
    def test_routes_distant_pair_with_swaps(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        context = PassContext()
        routed = SwapRoutingPass(architecture=linear_architecture(3)).transform(circuit, context)
        assert context.analysis["swaps_added"] == 1
        _assert_routed_equivalent(routed, circuit, context.analysis["final_layout"])
        assert routed.num_qubits == 3

    def test_connected_pair_untouched(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.h(2)
        context = PassContext()
        routed = SwapRoutingPass(architecture=linear_architecture(3)).transform(circuit, context)
        assert context.analysis["swaps_added"] == 0
        assert routed.num_gates == 2
        _assert_routed_equivalent(routed, circuit, context.analysis["final_layout"])

    def test_explicit_coupling_and_layout_record(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        circuit.cx(2, 0)
        context = PassContext()
        routed = SwapRoutingPass(coupling=[(0, 1), (1, 2)]).transform(circuit, context)
        _assert_routed_equivalent(routed, circuit, context.analysis["final_layout"])
        layout = context.analysis["final_layout"]
        assert sorted(layout.values()) == [0, 1, 2]

    def test_no_connectivity_is_passthrough(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        context = PassContext()
        routed = SwapRoutingPass().transform(circuit, context)
        assert context.analysis["swaps_added"] == 0
        assert routed.num_gates == 1

    def test_disconnected_graph_rejected(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        with pytest.raises(ValueError, match="No routing path"):
            SwapRoutingPass(coupling=[(0, 1)]).transform(circuit, PassContext())
        with pytest.raises(ValueError, match="qubits"):
            SwapRoutingPass(architecture=linear_architecture(2)).transform(circuit, PassContext())
        assert SwapRoutingPass().name == "swap-routing"


class TestNoiseAwareLayout:
    def test_places_measured_qubits_on_best_hardware(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 2)
        circuit.measure(2)
        context = PassContext()
        arch = linear_architecture(3)
        calibration = {"q0": 0.05, "q1": 0.01, "q2": 0.03}
        result = NoiseAwareLayout(arch, calibration).transform(circuit, context)
        assert result is circuit
        layout = context.analysis["layout"]
        assert layout[2] == 1
        assert context.analysis["estimated_readout_error"] == pytest.approx(0.01)

    def test_uniform_calibration_identity(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        context = PassContext()
        NoiseAwareLayout(linear_architecture(2)).transform(circuit, context)
        assert context.analysis["layout"] == {0: 0, 1: 1}

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            NoiseAwareLayout(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="qubits"):
            NoiseAwareLayout(linear_architecture(1)).transform(QuantumCircuit(2), PassContext())
        assert NoiseAwareLayout(linear_architecture(2)).name == "noise-aware-layout"


class TestCommutationAwareCancellation:
    def test_x_slides_through_cx_target(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)
        context = PassContext()
        out = CommutationAwareCancellation().transform(circuit, context)
        assert out.num_gates == 1
        assert context.analysis["cancelled_pairs"] == 1
        assert context.analysis["slid"] >= 1
        assert np.allclose(_unitary(out), _unitary(circuit), atol=1e-9)

    def test_z_slides_through_cx_control(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.z(0)
        circuit.cx(0, 1)
        circuit.z(0)
        out = CommutationAwareCancellation().transform(circuit, PassContext())
        assert out.num_gates == 1
        assert np.allclose(_unitary(out), _unitary(circuit), atol=1e-9)

    def test_z_on_target_does_not_cancel(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.z(1)
        circuit.cx(0, 1)
        circuit.z(1)
        out = CommutationAwareCancellation().transform(circuit, PassContext())
        assert out.num_gates == 3
        assert np.allclose(_unitary(out), _unitary(circuit), atol=1e-9)

    def test_z_slides_through_cz(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.z(1)
        circuit.cz(0, 1)
        circuit.z(1)
        out = CommutationAwareCancellation().transform(circuit, PassContext())
        assert out.num_gates == 1

    def test_equivalence_on_mixed_circuit(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)
        circuit.z(0)
        circuit.cx(0, 2)
        circuit.z(0)
        out = CommutationAwareCancellation().transform(circuit, PassContext())
        assert out.num_gates <= circuit.num_gates - 2
        assert np.allclose(_unitary(out), _unitary(circuit), atol=1e-9)
        assert CommutationAwareCancellation().name == "commutation-aware-cancellation"
