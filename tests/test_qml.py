"""Tests for quantum machine learning module."""
import numpy as np
import pytest

from microquantum.core.circuit import QuantumCircuit
from microquantum.qml.classifier import ClassifierResult, VariationalClassifier
from microquantum.qml.encoding import (
    AmplitudeEncoding,
    AngleEncoding,
    IQPEncoding,
    ZFeatureMap,
)
from microquantum.qml.kernels import QuantumKernel


class TestAngleEncoding:
    def test_creation(self):
        enc = AngleEncoding(3)
        assert enc.num_qubits == 3

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="Need >= 1"):
            AngleEncoding(0)

    def test_invalid_gate(self):
        with pytest.raises(ValueError, match="gate must be"):
            AngleEncoding(2, gate="invalid")

    def test_encode_correct_length(self):
        enc = AngleEncoding(3)
        qc = enc.encode([0.1, 0.2, 0.3])
        assert qc.num_qubits == 3

    def test_encode_wrong_length(self):
        enc = AngleEncoding(3)
        with pytest.raises(ValueError, match="Expected 3 features"):
            enc.encode([0.1, 0.2])

    def test_encode_produces_state(self):
        enc = AngleEncoding(2)
        qc = enc.encode([np.pi / 2, 0])
        state = qc.run()
        assert len(state.amplitudes) == 4

    def test_rx_encoding(self):
        enc = AngleEncoding(2, gate="rx")
        qc = enc.encode([np.pi / 2, np.pi / 4])
        assert qc.num_qubits == 2

    def test_rz_encoding(self):
        enc = AngleEncoding(2, gate="rz")
        qc = enc.encode([np.pi / 2, np.pi / 4])
        assert qc.num_qubits == 2

    def test_repr(self):
        enc = AngleEncoding(3, gate="ry")
        assert "AngleEncoding" in repr(enc)
        assert "3" in repr(enc)


class TestAmplitudeEncoding:
    def test_creation(self):
        enc = AmplitudeEncoding(3)
        assert enc.num_qubits == 2  # ceil(log2(3)) = 2

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="Need >= 1"):
            AmplitudeEncoding(0)

    def test_encode_normalizes(self):
        enc = AmplitudeEncoding(4)
        qc = enc.encode([1.0, 2.0, 3.0, 4.0])
        state = qc.run()
        norm = np.linalg.norm(state.amplitudes)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_encode_wrong_length(self):
        enc = AmplitudeEncoding(4)
        with pytest.raises(ValueError, match="Expected 4 features"):
            enc.encode([1.0, 2.0])

    def test_encode_zero_vector(self):
        enc = AmplitudeEncoding(2)
        with pytest.raises(ValueError, match="zero"):
            enc.encode([0.0, 0.0])

    def test_encode_padding(self):
        enc = AmplitudeEncoding(3)
        qc = enc.encode([1.0, 0.0, 0.0])
        state = qc.run()
        assert len(state.amplitudes) == 4

    def test_repr(self):
        enc = AmplitudeEncoding(5)
        assert "AmplitudeEncoding" in repr(enc)


class TestIQPEncoding:
    def test_creation(self):
        enc = IQPEncoding(4, reps=3)
        assert enc.num_qubits == 4

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="Need >= 1"):
            IQPEncoding(0)

    def test_invalid_reps(self):
        with pytest.raises(ValueError, match="Need >= 1 rep"):
            IQPEncoding(3, reps=0)

    def test_encode(self):
        enc = IQPEncoding(3)
        qc = enc.encode([0.1, 0.2, 0.3])
        assert qc.num_qubits == 3

    def test_encode_different_reps(self):
        enc1 = IQPEncoding(3, reps=1)
        enc2 = IQPEncoding(3, reps=3)
        qc1 = enc1.encode([0.1, 0.2, 0.3])
        qc2 = enc2.encode([0.1, 0.2, 0.3])
        assert len(qc2._gate_instructions) > len(qc1._gate_instructions)

    def test_repr(self):
        enc = IQPEncoding(4, reps=2)
        assert "IQPEncoding" in repr(enc)


class TestZFeatureMap:
    def test_creation(self):
        fm = ZFeatureMap(4, reps=2)
        assert fm.num_qubits == 4

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="Need >= 1"):
            ZFeatureMap(0)

    def test_encode(self):
        fm = ZFeatureMap(3)
        qc = fm.encode([0.1, 0.2, 0.3])
        assert qc.num_qubits == 3

    def test_encode_has_h_gates(self):
        fm = ZFeatureMap(2, reps=1)
        qc = fm.encode([0.5, 1.0])
        names = [instr[0].name for instr in qc._gate_instructions
                 if not QuantumCircuit._is_parameterized_gate(instr)]
        assert names.count("h") == 2

    def test_encode_has_cnots(self):
        fm = ZFeatureMap(3, reps=1)
        qc = fm.encode([0.1, 0.2, 0.3])
        names = [instr[0].name for instr in qc._gate_instructions
                 if not QuantumCircuit._is_parameterized_gate(instr)]
        assert "cnot" in names

    def test_repr(self):
        fm = ZFeatureMap(3, reps=2)
        assert "ZFeatureMap" in repr(fm)


class TestQuantumKernel:
    def test_creation(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        assert k.num_qubits == 2

    def test_default_encoder(self):
        k = QuantumKernel()
        assert k.num_qubits == 2

    def test_build_kernel_circuit(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        qc = k.build_kernel_circuit([0.1, 0.2], [0.3, 0.4])
        # 1 ancilla + 2 * 2 data qubits = 5
        assert qc.num_qubits == 5

    def test_evaluate_single_same_point(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        val = k.evaluate_single([0.0, 0.0], [0.0, 0.0])
        # Same point should give high kernel value
        assert val > 0.5

    def test_evaluate_single_different_points(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        val = k.evaluate_single([0.0, 0.0], [np.pi, np.pi])
        assert 0.0 <= val <= 1.0

    def test_evaluate_matrix(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        X1 = [[0.0, 0.0], [1.0, 1.0]]
        X2 = [[0.0, 0.0], [2.0, 2.0]]
        K = k.evaluate(X1, X2)
        assert K.shape == (2, 2)
        assert K[0, 0] > K[0, 1]  # Similar points have higher kernel

    def test_repr(self):
        enc = AngleEncoding(2)
        k = QuantumKernel(enc)
        assert "QuantumKernel" in repr(k)


class TestVariationalClassifier:
    def test_creation(self):
        clf = VariationalClassifier(num_features=3, num_classes=2)
        assert clf.num_features == 3
        assert clf.num_classes == 2

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="Need >= 1 feature"):
            VariationalClassifier(num_features=0)

    def test_invalid_classes(self):
        with pytest.raises(ValueError, match="Need >= 2 classes"):
            VariationalClassifier(num_features=3, num_classes=1)

    def test_parameters_exist(self):
        clf = VariationalClassifier(num_features=2, ansatz_depth=2)
        assert len(clf.parameters) > 0

    def test_predict(self):
        clf = VariationalClassifier(num_features=2, ansatz_depth=1)
        X = [[0.0, 0.0], [1.0, 1.0]]
        result = clf.predict(X)
        assert isinstance(result, ClassifierResult)
        assert len(result.predictions) == 2
        assert all(p in [0, 1] for p in result.predictions)

    def test_predict_probabilities(self):
        clf = VariationalClassifier(num_features=2, ansatz_depth=1)
        X = [[0.0, 0.0]]
        result = clf.predict(X)
        assert len(result.probabilities) == 1
        assert sum(result.probabilities[0]) == pytest.approx(1.0, abs=1e-6)

    def test_score(self):
        clf = VariationalClassifier(num_features=2, ansatz_depth=1)
        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [0, 1]
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_multi_class(self):
        clf = VariationalClassifier(num_features=3, num_classes=3, ansatz_depth=1)
        X = [[0.0, 0.0, 0.0]]
        result = clf.predict(X)
        assert len(result.probabilities[0]) == 3

    def test_repr(self):
        clf = VariationalClassifier(num_features=3, num_classes=2)
        assert "VariationalClassifier" in repr(clf)


class TestClassifierResult:
    def test_default(self):
        r = ClassifierResult()
        assert r.predictions == []
        assert r.probabilities == []
        assert r.accuracy == 0.0
