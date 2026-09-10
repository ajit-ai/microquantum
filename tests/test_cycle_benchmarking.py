"""Tests for cycle benchmarking, layer fidelity, and Pauli twirling."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core import Operator, QuantumCircuit
from microquantum.benchmarks.cycle_benchmarking import (
    CycleBenchmarking,
    CycleBenchmarkingResult,
    LayerFidelity,
    LayerFidelityResult,
    PauliTwirling,
    TwirlingResult,
)


class TestCycleBenchmarking:
    """Cycle benchmarking tests."""

    def test_single_qubit(self) -> None:
        """CB on 1 qubit should run without error."""
        cb = CycleBenchmarking(
            num_qubits=1,
            cycle_lengths=[1, 2, 4],
            num_samples=5,
            seed=42,
        )
        result = cb.run()
        assert isinstance(result, CycleBenchmarkingResult)
        assert result.num_qubits == 1
        assert len(result.cycle_lengths) == 3
        assert len(result.fidelities) == 3

    def test_two_qubits(self) -> None:
        """CB on 2 qubits should run without error."""
        cb = CycleBenchmarking(
            num_qubits=2,
            cycle_lengths=[1, 2, 3],
            num_samples=5,
            seed=42,
        )
        result = cb.run()
        assert result.num_qubits == 2
        assert all(0 <= f <= 1 for f in result.fidelities)

    def test_fidelity_decay(self) -> None:
        """Fidelities should be valid probabilities between 0 and 1."""
        cb = CycleBenchmarking(
            num_qubits=2,
            cycle_lengths=[1, 4, 8],
            num_samples=10,
            gates_per_cycle=2,
            seed=42,
        )
        result = cb.run()
        for f in result.fidelities:
            assert 0.0 <= f <= 1.0

    def test_error_non_negative(self) -> None:
        """Error per cycle should be non-negative."""
        cb = CycleBenchmarking(
            num_qubits=2,
            cycle_lengths=[1, 2, 4],
            num_samples=5,
            seed=42,
        )
        result = cb.run()
        assert result.error_per_cycle >= 0.0
        assert result.fidelity_per_gate >= 0.0

    def test_num_qubits_property(self) -> None:
        cb = CycleBenchmarking(num_qubits=3, seed=42)
        assert cb.num_qubits == 3

    def test_invalid_qubits(self) -> None:
        with pytest.raises(ValueError, match="num_qubits"):
            CycleBenchmarking(num_qubits=0)

    def test_repr(self) -> None:
        cb = CycleBenchmarking(num_qubits=2, cycle_lengths=[1, 2], seed=42)
        assert "CycleBenchmarking" in repr(cb)


class TestLayerFidelity:
    """Layer fidelity tests."""

    def test_default_layers(self) -> None:
        """Default layers should include identity, h_all, x_all."""
        lf = LayerFidelity(num_qubits=2, seed=42)
        result = lf.run()
        assert isinstance(result, LayerFidelityResult)
        assert "identity" in result.layer_names
        assert len(result.fidelities) == len(result.layer_names)

    def test_worst_layer(self) -> None:
        """worst_layer should be the layer with minimum fidelity."""
        lf = LayerFidelity(num_qubits=1, seed=42)
        result = lf.run()
        worst_idx = int(np.argmin(result.fidelities))
        assert result.worst_layer == result.layer_names[worst_idx]

    def test_average_fidelity(self) -> None:
        """average_fidelity should be mean of all fidelities."""
        lf = LayerFidelity(num_qubits=1, seed=42)
        result = lf.run()
        expected = float(np.mean(result.fidelities))
        assert result.average_fidelity == pytest.approx(expected, rel=1e-6)

    def test_custom_layers(self) -> None:
        """Custom layers should be used instead of defaults."""
        custom = {
            "h_only": [(Operator.H(), [0])],
            "x_only": [(Operator.X(), [0])],
        }
        lf = LayerFidelity(num_qubits=1, layers=custom, seed=42)
        result = lf.run()
        assert result.layer_names == ["h_only", "x_only"]
        assert len(result.fidelities) == 2

    def test_repr(self) -> None:
        lf = LayerFidelity(num_qubits=2, seed=42)
        assert "LayerFidelity" in repr(lf)


class TestPauliTwirling:
    """Pauli twirling tests."""

    def test_twirl_no_two_qubit_gates(self) -> None:
        """Circuit with no 2-qubit gates should pass through unchanged."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)

        twirler = PauliTwirling(seed=42)
        result = twirler.twirl(qc)
        assert isinstance(result, TwirlingResult)
        assert result.num_twirled_gates == 0
        assert result.twirled_circuit.num_qubits == 1

    def test_twirl_cnot_circuit(self) -> None:
        """Circuit with CNOT should have twirling gates added."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        twirler = PauliTwirling(seed=42)
        result = twirler.twirl(qc)
        assert result.num_twirled_gates == 1
        assert result.twirled_circuit.num_qubits == 2
        # Twirled circuit should have more gates than original
        assert result.twirled_depth >= result.original_depth

    def test_reproducible(self) -> None:
        """Same seed should produce same result."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(1, 0)

        t1 = PauliTwirling(seed=42).twirl(qc)
        t2 = PauliTwirling(seed=42).twirl(qc)
        assert t1.num_twirled_gates == t2.num_twirled_gates
        assert t1.twirled_depth == t2.twirled_depth

    def test_multiple_cnots(self) -> None:
        """Multiple CNOTs should all be twirled."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)

        result = PauliTwirling(seed=42).twirl(qc)
        assert result.num_twirled_gates == 3

    def test_cz_twirled(self) -> None:
        """CZ gates should also be twirled."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        result = PauliTwirling(seed=42).twirl(qc)
        assert result.num_twirled_gates == 1

    def test_repr(self) -> None:
        twirler = PauliTwirling()
        assert "PauliTwirling" in repr(twirler)
