"""Phase 121 W4: domain methods II."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.chemistry import (
    ActiveSpace,
    FermionicOp,
    H2Hamiltonian,
    jordan_wigner,
)
from microquantum.core.parameter import Parameter
from microquantum.core.pauli import PauliString, PauliSum
from microquantum.mitigation import CDRTrainingPoint, CliffordDataRegression
from microquantum.qec import (
    LookupDecoder,
    SteaneCode,
    Syndrome,
)
from microquantum.qml import (
    DataReuploadingClassifier,
    QuantumKernel,
    kernel_alignment,
)


class TestReuploadingClassifier:
    def test_predict_and_score(self) -> None:
        clf = DataReuploadingClassifier(num_features=1, layers=1)
        assert clf.num_qubits == 2
        assert len(clf.parameters) == 2
        result = clf.predict([[0.0], [1.0]])
        assert len(result.predictions) == 2
        assert 0.0 <= clf.score([[0.0]], [0]) <= 1.0
        circuit = clf.build_circuit([0.5], clf.parameters)
        assert circuit.num_qubits == 2
        with pytest.raises(ValueError):
            DataReuploadingClassifier(num_features=0)
        with pytest.raises(ValueError):
            DataReuploadingClassifier(num_features=1, layers=0)
        with pytest.raises(ValueError):
            clf.build_circuit([0.1, 0.2], clf.parameters)

    def test_fit_trains(self) -> None:
        clf = DataReuploadingClassifier(num_features=1, layers=1)
        result = clf.fit([[0.0], [0.0], [3.0], [3.0]], [0, 0, 1, 1])
        assert result.optimizer_result is not None
        assert 0.0 <= result.accuracy <= 1.0
        assert clf.score([[0.0], [3.0]], [0, 1]) >= 0.0
        with pytest.raises(ValueError):
            clf.fit([[0.0]], [])
        with pytest.raises(ValueError):
            clf.fit([], [])
        with pytest.raises(ValueError):
            clf.fit([[0.0]], [5])

    def test_kernel_alignment(self) -> None:
        kernel = QuantumKernel()
        matrix = kernel.evaluate([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])
        score = kernel_alignment(matrix, [0, 1])
        assert -1.0 <= score <= 1.0
        perfect = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert kernel_alignment(perfect, [0, 1]) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            kernel_alignment(np.zeros((2, 2)), [0, 1])
        with pytest.raises(ValueError):
            kernel_alignment(np.eye(2), [0, 2])
        with pytest.raises(ValueError):
            kernel_alignment(np.eye(3), [0, 1])


class TestSteaneCode:
    def test_properties(self) -> None:
        code = SteaneCode()
        assert code.num_data_qubits == 7
        assert code.num_syndrome_qubits == 6
        assert code.total_qubits == 13
        assert code.distance == 3
        assert len(code.x_stabilizers) == 3
        assert "[[7,1,3]]" in repr(code)

    def test_logical_zero_is_stabilized(self) -> None:
        from microquantum.core.circuit import QuantumCircuit

        code = SteaneCode()
        encoded = code.encode_circuit()
        assert encoded.num_qubits == 7
        state = encoded.run()
        assert state.is_normalized
        for support in code.x_stabilizers:
            circuit = QuantumCircuit(7)
            for qubit in support:
                circuit.x(qubit)
            probe = encoded + circuit
            overlap = abs(state.inner_product(probe.run()))
            assert overlap == pytest.approx(1.0, abs=1e-9)

    def test_syndrome_and_decoding(self) -> None:
        code = SteaneCode()
        assert code.syndrome_circuit().num_qubits == 13
        assert code.decode_syndrome([0, 0, 0, 0, 0, 0]) is None
        assert code.decode_syndrome([0, 0, 1, 0, 0, 0]) == (0, "X")
        assert code.decode_syndrome([1, 0, 0, 0, 0, 0]) == (3, "X")
        assert code.decode_syndrome([0, 0, 0, 1, 1, 1]) == (6, "Z")
        assert code.decode_syndrome([0, 0, 1, 0, 0, 1]) == (0, "Y")
        assert code.error_locations([0, 0, 0, 0, 0, 0]) == []
        assert code.error_locations([0, 0, 1, 0, 0, 0]) == [(0, "X")]
        with pytest.raises(ValueError):
            code.decode_syndrome([0, 0])
        with pytest.raises(ValueError):
            code.decode_syndrome([0, 0, 0, 0, 0, 2])

    def test_all_single_qubit_errors_decode(self) -> None:
        code = SteaneCode()
        for qubit in range(7):
            position = qubit + 1
            bits = [(position >> 2) & 1, (position >> 1) & 1, position & 1]
            assert code.decode_syndrome(bits + [0, 0, 0]) == (qubit, "X")
            assert code.decode_syndrome([0, 0, 0] + bits) == (qubit, "Z")


class TestDecoder:
    def test_lookup_decoder(self) -> None:
        decoder = LookupDecoder(
            table={(0, 0, 1): [(0, "X")], (0, 0, 0): []},
            code_name="steane",
        )
        assert decoder.decode(Syndrome(bits=(0, 0, 1), code_name="steane")) == [(0, "X")]
        assert decoder.decode(Syndrome(bits=(1, 1, 1), code_name="steane")) == []
        with pytest.raises(ValueError):
            decoder.decode(Syndrome(bits=(0, 0, 1), code_name="shor"))
        with pytest.raises(ValueError):
            Syndrome(bits=(0, 0, 2))
        round_tripped = Syndrome.from_dict(
            Syndrome(bits=(1, 0, 1), code_name="s", round=2).to_dict()
        )
        assert round_tripped.round == 2


class TestChemistryTransforms:
    def test_active_space_selection(self) -> None:
        space = ActiveSpace(num_core_orbitals=1, num_active_orbitals=2)
        selection = space.select(num_electrons=6, num_orbitals=4)
        assert selection.num_active_electrons == 4
        assert selection.num_qubits == 4
        assert selection.to_dict()["num_core_orbitals"] == 1
        with pytest.raises(ValueError):
            ActiveSpace(num_core_orbitals=-1, num_active_orbitals=2)
        with pytest.raises(ValueError):
            ActiveSpace(num_core_orbitals=0, num_active_orbitals=0)
        with pytest.raises(ValueError):
            space.select(num_electrons=6, num_orbitals=2)
        with pytest.raises(ValueError):
            ActiveSpace(num_core_orbitals=4, num_active_orbitals=1).select(
                num_electrons=2, num_orbitals=6
            )

    def test_jordan_wigner_number_operator(self) -> None:
        operator = FermionicOp({(("+", 0), ("-", 0)): 1.0})
        result = jordan_wigner(operator, 1)
        assert isinstance(result, PauliSum)
        matrix = result.to_operator().matrix
        assert np.allclose(matrix, [[0, 0], [0, 1]], atol=1e-12)

    def test_jordan_wigner_hopping(self) -> None:
        operator = FermionicOp({(("+", 0), ("-", 1)): 1.0, (("+", 1), ("-", 0)): 1.0})
        result = jordan_wigner(operator, 2)
        matrix = result.to_operator().matrix
        expected = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=complex
        )
        assert np.allclose(matrix, expected, atol=1e-12)

    def test_jordan_wigner_validation(self) -> None:
        with pytest.raises(ValueError):
            FermionicOp({(("x", 0),): 1.0})
        with pytest.raises(ValueError):
            FermionicOp({(("+", -1),): 1.0})
        with pytest.raises(ValueError):
            jordan_wigner(FermionicOp({}), 0)
        assert jordan_wigner(FermionicOp({(): 2.0}), 1).num_terms == 1
        assert FermionicOp({}).num_orbitals == 0

    def test_h2_still_works(self) -> None:
        h2 = H2Hamiltonian()
        assert h2.num_terms > 0
        assert h2.num_qubits == 2


class TestCDR:
    def test_train_and_mitigate(self) -> None:
        cdr = CliffordDataRegression()
        assert not cdr.is_trained
        summary = cdr.train([0.5, 0.7, 0.9], [0.6, 0.8, 1.0])
        assert cdr.is_trained
        assert summary["r_squared"] == pytest.approx(1.0)
        assert cdr.mitigate(0.7) == pytest.approx(0.8, abs=1e-9)
        assert "slope=" in repr(cdr)
        assert CliffordDataRegression.from_dict(cdr.to_dict()).slope == pytest.approx(
            cdr.slope
        )
        point = CDRTrainingPoint(noisy=0.5, exact=0.6)
        assert point.noisy == pytest.approx(0.5)

    def test_cdr_validation(self) -> None:
        cdr = CliffordDataRegression()
        with pytest.raises(RuntimeError):
            _ = cdr.slope
        with pytest.raises(RuntimeError):
            cdr.mitigate(0.5)
        with pytest.raises(ValueError):
            cdr.train([0.5], [0.5, 0.6])
        with pytest.raises(ValueError):
            cdr.train([0.5], [0.5])
        with pytest.raises(ValueError):
            cdr.train([float("nan"), 0.5], [0.5, 0.6])
        with pytest.raises(ValueError):
            cdr.mitigate(float("inf"))
        assert "untrained" in repr(cdr)

    def test_pauli_imports_used(self) -> None:
        assert PauliString("Z").label == "Z"
        assert PauliSum([]).num_terms == 0
        assert isinstance(Parameter("q"), Parameter)
