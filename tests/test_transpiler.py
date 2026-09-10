"""Tests for the transpiler pipeline."""

import numpy as np

from microquantum.core import Operator, PassManager, QuantumCircuit, TargetGateSet
from microquantum.core.transpiler import (
    CancellationPass,
    FusionPass,
    GateDecompositionPass,
    IdentityRemovalPass,
    LayoutMappingPass,
)


class TestPassManager:
    def test_empty_pass_manager(self):
        pm = PassManager()
        assert pm.num_passes == 0

    def test_append_pass(self):
        pm = PassManager()
        pm.append_pass(CancellationPass())
        assert pm.num_passes == 1

    def test_append_passes(self):
        pm = PassManager()
        pm.append_passes([CancellationPass(), FusionPass(), IdentityRemovalPass()])
        assert pm.num_passes == 3

    def test_clear(self):
        pm = PassManager()
        pm.append_passes([CancellationPass(), FusionPass()])
        pm.clear()
        assert pm.num_passes == 0

    def test_run_empty_circuit(self):
        pm = PassManager()
        pm.append_pass(CancellationPass())
        qc = QuantumCircuit(2)
        result = pm.run(qc)
        assert result.num_qubits == 2
        assert result.num_gates == 0

    def test_run_preserves_state(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state_before = qc.run()

        pm = PassManager()
        pm.append_pass(CancellationPass())
        result = pm.run(qc)
        state_after = result.run()

        assert np.allclose(state_before.amplitudes, state_after.amplitudes)

    def test_run_with_stats(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        pm = PassManager()
        pm.append_pass(CancellationPass())
        result, stats = pm.run_with_stats(qc)
        assert isinstance(stats, list)
        assert len(stats) == 1
        assert stats[0][0] == "cancellation"

    def test_from_optimization_level_0(self):
        pm = PassManager.from_optimization_level(0)
        assert pm.num_passes == 0

    def test_from_optimization_level_1(self):
        pm = PassManager.from_optimization_level(1)
        assert pm.num_passes > 0

    def test_from_optimization_level_3(self):
        pm3 = PassManager.from_optimization_level(3)
        pm1 = PassManager.from_optimization_level(1)
        assert pm3.num_passes >= pm1.num_passes


class TestPasses:
    def test_cancellation_removes_inverse_pairs(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        pm = PassManager()
        pm.append_pass(CancellationPass())
        result = pm.run(qc)
        assert result.num_gates == 0

    def test_fusion_fuses_single_qubit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        pm = PassManager()
        pm.append_pass(FusionPass())
        result = pm.run(qc)
        assert result.num_gates == 1

    def test_identity_removal(self):
        qc = QuantumCircuit(1)
        qc.append(Operator.I(), [0])
        pm = PassManager()
        pm.append_pass(IdentityRemovalPass())
        result = pm.run(qc)
        assert result.num_gates == 0

    def test_layout_mapping(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 2)
        layout = {0: 2, 1: 1, 2: 0}
        pm = PassManager()
        pm.append_pass(LayoutMappingPass(layout=layout))
        result = pm.run(qc)
        assert result.num_qubits == 3

    def test_parameterized_circuit_passthrough(self):
        from microquantum.core import Parameter
        qc = QuantumCircuit(1)
        p = Parameter("theta")
        qc.rx(p, 0)
        pm = PassManager()
        pm.append_pass(CancellationPass())
        result = pm.run(qc)
        assert result.is_parameterized


class TestTargetGateSet:
    def test_default(self):
        tgs = TargetGateSet()
        assert "h" in tgs.basis_gates
        assert "cx" in tgs.basis_gates

    def test_custom(self):
        tgs = TargetGateSet(single_qubit_gates={"rx", "rz"}, two_qubit_gates={"cz"})
        assert tgs.basis_gates == {"rx", "rz", "cz"}


class TestGateDecomposition:
    def test_identity_preserved(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        pm = PassManager()
        pm.append_pass(GateDecompositionPass())
        result = pm.run(qc)
        state_orig = qc.run()
        state_new = result.run()
        assert np.allclose(state_orig.amplitudes, state_new.amplitudes)

    def test_cnot_in_circuit(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        pm = PassManager()
        pm.append_pass(GateDecompositionPass())
        result = pm.run(qc)
        assert result.num_gates >= 1
