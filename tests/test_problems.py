"""Tests for the generic problem abstractions (MQ-05)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from microquantum import (
    EigenvalueProblem,
    HamiltonianProblem,
    OptimizationProblem,
    Problem,
    SamplingProblem,
    SearchProblem,
)
from microquantum.core import Operator, PauliString, PauliSum, QuantumCircuit, tensor


class TestProblemBase:
    def test_problem_to_dict_has_type_discriminator(self) -> None:
        p = Problem(name="demo", num_qubits=2, metadata={"domain": "test"})
        data = p.to_dict()
        assert data["type"] == "Problem"
        assert data["name"] == "demo"
        assert data["num_qubits"] == 2
        assert data["metadata"] == {"domain": "test"}

    def test_problem_validate(self) -> None:
        assert Problem(name="ok").validate() == []
        assert (Problem(name="").validate() != []) and (
            any("name" in m for m in Problem(name="").validate())
        )

    def test_problem_json_roundtrip(self) -> None:
        p = Problem(name="demo", num_qubits=1)
        loaded = json.loads(p.to_json())
        assert loaded["name"] == "demo"

    def test_problem_rejects_bad_qubit_count(self) -> None:
        with pytest.raises(ValueError):
            Problem(num_qubits=0)


class TestSamplingProblem:
    def test_from_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        sp = SamplingProblem(qc, num_samples=100)
        assert sp.num_qubits == 2
        assert sp.validate() == []

    def test_from_num_qubits(self) -> None:
        sp = SamplingProblem(num_qubits=3, num_samples=50)
        assert sp.num_qubits == 3
        assert sp.validate() == []

    def test_requires_circuit_or_qubits(self) -> None:
        with pytest.raises(ValueError):
            SamplingProblem(circuit=None, num_qubits=None)

    def test_to_dict_type(self) -> None:
        sp = SamplingProblem(num_qubits=1)
        assert sp.to_dict()["type"] == "Sampling"


class TestOptimizationProblem:
    def test_basic_energy(self) -> None:
        p = OptimizationProblem(
            num_variables=2,
            objective=lambda bits: float(bits[0] * bits[1] + bits[1]),
            name="simple",
        )
        assert p.energy(np.array([1.0, 1.0])) == pytest.approx(2.0)
        assert p.energy(np.array([0.0, 0.0])) == pytest.approx(0.0)
        assert p.validate() == []

    def test_requires_objective_or_ising(self) -> None:
        p = OptimizationProblem(num_variables=2)
        assert p.validate() != []

    def test_rejects_num_qubits(self) -> None:
        with pytest.raises(ValueError):
            OptimizationProblem(num_variables=2, num_qubits=2)

    def test_from_qubo_roundtrip(self) -> None:
        from microquantum.optimization.qubo import QUBOBuilder

        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_linear(0, -1.0)
        builder.add_linear(1, -1.0)
        qubo = builder.build("maxcut2")

        p = OptimizationProblem.from_qubo(qubo)
        assert p.num_variables == 2
        assert p.validate() == []
        # energies preserved through the Ising view
        for a in (0, 1):
            for b in (0, 1):
                spin = p.encode_spins(np.array([float(a), float(b)]))
                assert p.energy(np.array([float(a), float(b)])) == pytest.approx(
                    p._ising_spin_energy(spin)
                )
        # QUBO -> Ising -> QUBO roundtrip keeps energies
        qubo2 = p.to_qubo()
        for a in (0, 1):
            for b in (0, 1):
                assert qubo.energy(np.array([a, b])) == pytest.approx(
                    qubo2.energy(np.array([a, b]))
                )

    def test_from_ising(self) -> None:
        ising = PauliSum([PauliString("ZZ", 1.0), PauliString("Z", -0.5)])
        p = OptimizationProblem.from_ising(ising, name="spin")
        assert p.num_variables == 2
        assert p.cost_hamiltonian() is ising

    def test_encoding_and_sampling(self) -> None:
        p = OptimizationProblem(num_variables=3, objective=lambda b: 0.0)
        assert p.qubits_needed == 3
        spin = p.encode_spins(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(spin, [1.0, -1.0, 1.0])
        sample = p.sample(seed=123)
        assert sample.shape == (3,)

    def test_to_dict_is_json_safe(self) -> None:
        ising = PauliSum([PauliString("ZZ", 1.0)])
        p = OptimizationProblem.from_ising(ising, name="spin")
        data = p.to_dict()
        assert data["type"] == "Optimization"
        assert data["num_variables"] == 2
        json.dumps(data)

    def test_standard_binary_encoding(self) -> None:
        from microquantum.problems.optimization import standard_binary_encoding

        np.testing.assert_allclose(standard_binary_encoding(5, 4), [0, 1, 0, 1])
        with pytest.raises(ValueError):
            standard_binary_encoding(4, 2)


class TestHamiltonianProblems:
    def test_hamiltonian_problem_positional(self) -> None:
        p = HamiltonianProblem(Operator.Z(), name="ham")
        assert p.num_qubits == 1
        assert p.validate() == []

    def test_hamiltonian_problem_qubit_mismatch(self) -> None:
        with pytest.raises(ValueError):
            HamiltonianProblem(tensor(Operator.Z(), Operator.Z()), num_qubits=1)

    def test_hamiltonian_expectation(self) -> None:
        p = HamiltonianProblem(Operator.Z())
        from microquantum.core import QuantumCircuit

        qc = QuantumCircuit(1)
        state = qc.run()
        assert p.expectation(state) == pytest.approx(1.0)

    def test_eigenvalue_problem_k(self) -> None:
        p = EigenvalueProblem(Operator.Z(), k=2)
        assert p.k == 2
        assert p.validate() == []
        with pytest.raises(ValueError):
            EigenvalueProblem(Operator.Z(), k=0)

    def test_pauli_sum_as_hamiltonian(self) -> None:
        ising = PauliSum([PauliString("Z", 1.0)])
        p = HamiltonianProblem(ising, name="pauliham")
        assert p.num_qubits == 1
        assert p.validate() == []

    def test_to_dict_type_discriminators(self) -> None:
        assert HamiltonianProblem(Operator.Z()).to_dict()["type"] == "Hamiltonian"
        assert EigenvalueProblem(Operator.Z(), k=1).to_dict()["type"] == "Eigenvalue"


class TestSearchProblem:
    def test_target_based(self) -> None:
        p = SearchProblem(num_qubits=3, target=[6, 7], name="find")
        assert p.target_indices() == [6, 7]
        assert p.num_solutions() == 2
        assert p.is_marked("110")
        assert not p.is_marked("000")
        assert p.validate() == []

    def test_predicate_based(self) -> None:
        p = SearchProblem(num_qubits=3, predicate=lambda i: i % 4 == 3)
        assert p.target_indices() == [3, 7]
        assert p.is_marked("011")
        assert p.num_solutions() == 2

    def test_requires_one_provider(self) -> None:
        assert SearchProblem(num_qubits=2).validate() != []
        both = SearchProblem(num_qubits=2, target=0, predicate=lambda i: True)
        assert both.validate() != []

    def test_target_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            SearchProblem(num_qubits=2, target=4)

    def test_to_dict_type(self) -> None:
        p = SearchProblem(num_qubits=2, target=1)
        assert p.to_dict()["type"] == "Search"
        json.dumps(p.to_dict())