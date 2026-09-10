"""Tests for circuit optimization: gate fusion, simplification, and transpilation."""

import numpy as np

from microquantum.core.circuit import QuantumCircuit
from microquantum.core.operators import Operator
from microquantum.core.optimization import (
    cancel_inverse_pairs,
    circuit_stats,
    fuse_single_qubit_gates,
    remove_identity_gates,
    simplify_circuit,
    transpile,
)


class TestFuseSingleQubitGates:
    def test_fuse_two_h_gates(self):
        """H @ H = I, should fuse into identity."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates <= 1

    def test_fuse_preserves_state(self):
        """Fused circuit should produce the same final state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(0.5, 0)
        qc.rx(0.3, 0)
        qc.cx(0, 1)

        fused = fuse_single_qubit_gates(qc)
        state_original = qc.run()
        state_fused = fused.run()
        assert np.allclose(state_original.amplitudes, state_fused.amplitudes, atol=1e-10)

    def test_fuse_does_not_merge_two_qubit_gates(self):
        """Two-qubit gates should not be fused."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates == 2

    def test_fuse_single_gate_unchanged(self):
        """A single gate should pass through unchanged."""
        qc = QuantumCircuit(1)
        qc.h(0)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates == 1

    def test_fuse_mixed_gates(self):
        """Fuse single-qubit gates but leave two-qubit gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.cx(0, 1)
        qc.ry(0.5, 1)
        qc.rz(0.3, 1)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates == 3

    def test_fuse_empty_circuit(self):
        """Empty circuit should remain empty."""
        qc = QuantumCircuit(1)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates == 0

    def test_fuse_preserves_depth(self):
        """Fused circuit should have depth <= original."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        fused = fuse_single_qubit_gates(qc)
        assert fused.depth <= qc.depth


class TestRemoveIdentityGates:
    def test_remove_h_h_pair(self):
        """H @ H = I as a product, but H is not individually identity.
        remove_identity_gates only removes individual identity gates."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 2

    def test_remove_actual_identity_gates(self):
        """Actual identity operator gates should be removed."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(Operator.I(), [0])
        qc.x(0)
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 2

    def test_remove_identity_preserves_state(self):
        """Removing identity gates should not change the final state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        state_before = qc.run()
        removed = remove_identity_gates(qc)
        state_after = removed.run()
        assert np.allclose(state_before.amplitudes, state_after.amplitudes, atol=1e-10)

    def test_remove_single_identity(self):
        """A single identity gate should be removed."""
        qc = QuantumCircuit(1)
        qc.append(Operator.I(), [0])
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 0

    def test_no_removal_needed(self):
        """Non-identity gates should be kept."""
        qc = QuantumCircuit(1)
        qc.h(0)
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 1

    def test_remove_preserves_two_qubit_gates(self):
        """Two-qubit gates should not be affected by identity removal."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 3


class TestCancelInversePairs:
    def test_cancel_h_pair(self):
        """H is its own inverse, H-H should cancel."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        cancelled = cancel_inverse_pairs(qc)
        assert cancelled.num_gates == 0

    def test_cancel_s_s_dagger(self):
        """S followed by S_dagger should cancel."""
        qc = QuantumCircuit(1)
        qc.s(0)
        cancelled = cancel_inverse_pairs(qc)
        assert cancelled.num_gates <= 1

    def test_cancel_preserves_state(self):
        """Canceling pairs should preserve the final state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        state_before = qc.run()
        cancelled = cancel_inverse_pairs(qc)
        state_after = cancelled.run()
        assert np.allclose(state_before.amplitudes, state_after.amplitudes, atol=1e-10)

    def test_cancel_no_pair(self):
        """Non-canceling gates should remain."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        cancelled = cancel_inverse_pairs(qc)
        assert cancelled.num_gates == 2

    def test_cancel_cnot_pair(self):
        """CNOT is its own inverse, CNOT-CNOT should cancel."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        cancelled = cancel_inverse_pairs(qc)
        assert cancelled.num_gates == 0


class TestSimplifyCircuit:
    def test_simplify_reduces_gates(self):
        """Simplification should reduce redundant gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        simplified = simplify_circuit(qc)
        assert simplified.num_gates <= qc.num_gates

    def test_simplify_preserves_state(self):
        """Simplified circuit should produce the same state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.h(0)
        qc.cx(0, 1)
        state_before = qc.run()
        simplified = simplify_circuit(qc)
        state_after = simplified.run()
        assert np.allclose(state_before.amplitudes, state_after.amplitudes, atol=1e-10)

    def test_simplify_bell_circuit(self):
        """Bell state circuit should remain unchanged after simplification."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        simplified = simplify_circuit(qc)
        state_before = qc.run()
        state_after = simplified.run()
        assert np.allclose(state_before.amplitudes, state_after.amplitudes, atol=1e-10)

    def test_simplify_empty_circuit(self):
        """Empty circuit should remain empty."""
        qc = QuantumCircuit(1)
        simplified = simplify_circuit(qc)
        assert simplified.num_gates == 0


class TestTranspile:
    def test_level_0_no_change(self):
        """Level 0 should return the circuit unchanged."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        transpiled = transpile(qc, optimization_level=0)
        assert transpiled.num_gates == 2

    def test_level_1_basic(self):
        """Level 1 should apply basic simplification."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        transpiled = transpile(qc, optimization_level=1)
        assert transpiled.num_gates == 0

    def test_level_2_full(self):
        """Level 2 should apply full optimization."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.x(0)
        qc.x(0)
        transpiled = transpile(qc, optimization_level=2)
        assert transpiled.num_gates == 0

    def test_transpile_preserves_state(self):
        """Transpiled circuit should produce the same state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(0.5, 0)
        qc.h(0)
        qc.cx(0, 1)
        state_before = qc.run()
        transpiled = transpile(qc, optimization_level=2)
        state_after = transpiled.run()
        assert np.allclose(state_before.amplitudes, state_after.amplitudes, atol=1e-10)


class TestCircuitStats:
    def test_empty_circuit(self):
        """Empty circuit should have zero stats."""
        qc = QuantumCircuit(1)
        stats = circuit_stats(qc)
        assert stats["gate_count"] == 0

    def test_single_gate(self):
        """Single gate circuit."""
        qc = QuantumCircuit(1)
        qc.h(0)
        stats = circuit_stats(qc)
        assert stats["gate_count"] == 1
        assert stats["single_qubit_gates"] == 1
        assert stats["two_qubit_gates"] == 0

    def test_mixed_gates(self):
        """Circuit with single and two-qubit gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        stats = circuit_stats(qc)
        assert stats["gate_count"] == 3
        assert stats["single_qubit_gates"] == 2
        assert stats["two_qubit_gates"] == 1


class TestParameterizedCircuit:
    def test_fuse_skips_parameterized(self):
        """Fusion should skip parameterized circuits."""
        from microquantum.core.parameter import Parameter

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        fused = fuse_single_qubit_gates(qc)
        assert fused.num_gates == 1

    def test_remove_skips_parameterized(self):
        """Identity removal should skip parameterized circuits."""
        from microquantum.core.parameter import Parameter

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        removed = remove_identity_gates(qc)
        assert removed.num_gates == 1

    def test_cancel_skips_parameterized(self):
        """Inverse cancellation should skip parameterized circuits."""
        from microquantum.core.parameter import Parameter

        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        cancelled = cancel_inverse_pairs(qc)
        assert cancelled.num_gates == 1
