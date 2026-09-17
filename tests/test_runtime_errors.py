"""Phase 119 tests: stage-tagged runtime errors.

Every stage error subclasses :class:`ValueError` so pre-existing
``except ValueError`` handlers keep working; the stages are:

    planning -- compilation -- runtime dispatch -- backend execution
"""

import pytest

import microquantum
from microquantum import ExecutionRuntime, Parameter, QuantumCircuit
from microquantum.backends import MockBackend
from microquantum.core.device import Target
from microquantum.runtime import (
    BackendExecutionError,
    CompilationError,
    ExecutionError,
    PlanningError,
    RuntimeDispatchError,
)


class _FailingBackend(MockBackend):
    """Backend whose jobs always fail at the execution stage."""

    def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
        raise AssertionError("simulation exploded on purpose")

    def validate(self, plan):
        if "fail" in plan.name:
            return ["backend cannot run this plan"]
        return super().validate(plan)


class TestErrorHierarchy:
    def test_stage_errors_subclass_valueerror(self):
        for error in (
            ExecutionError,
            PlanningError,
            CompilationError,
            RuntimeDispatchError,
            BackendExecutionError,
        ):
            assert issubclass(error, ValueError)

    def test_stage_errors_subclass_common_base(self):
        for error in (
            PlanningError,
            CompilationError,
            RuntimeDispatchError,
            BackendExecutionError,
        ):
            assert issubclass(error, ExecutionError)

    def test_stage_errors_are_distinct(self):
        names = {
            PlanningError,
            CompilationError,
            RuntimeDispatchError,
            BackendExecutionError,
        }
        assert len(names) == 4

    def test_existing_valueerror_handlers_still_catch(self):
        rt = ExecutionRuntime()
        with pytest.raises(ValueError):
            rt.execute(QuantumCircuit(1).h(0), shots=-1)
        with pytest.raises(ValueError):
            rt.execute(QuantumCircuit(1).h(0), shots=0)

    @pytest.mark.parametrize(
        "error",
        [PlanningError, CompilationError, RuntimeDispatchError, BackendExecutionError],
    )
    def test_canonical_module(self, error):
        import microquantum.runtime.errors as errors

        assert getattr(errors, error.__name__) is error


class TestPlanningStage:
    @staticmethod
    def _parameterized_circuit():
        return QuantumCircuit(1).rx(Parameter("theta"), 0)

    def test_unbound_parameters_raise_planning_error(self):
        rt = ExecutionRuntime()
        with pytest.raises(PlanningError, match="unbound parameters"):
            rt.execute(self._parameterized_circuit())

    def test_prepare_invalid_plan_raises_planning_error(self):
        rt = ExecutionRuntime()
        plan = microquantum.ExecutionPlan.from_circuit(self._parameterized_circuit())
        with pytest.raises(PlanningError, match="problems"):
            rt.prepare(plan)

    def test_plan_type_error_stays_type_error(self):
        rt = ExecutionRuntime()
        with pytest.raises(TypeError):
            rt.prepare(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            rt.execute("not a circuit")  # type: ignore[arg-type]


class TestCompilationStage:
    def test_compile_raises_compilation_error_for_bad_target(self):
        rt = ExecutionRuntime()
        small = Target(name="tiny", num_qubits=1)
        circuit = QuantumCircuit(4).h(0).cx(0, 1)
        with pytest.raises(CompilationError, match="target problems"):
            rt.compile(circuit, target=small)

    def test_execute_raises_compilation_error_for_bad_target(self):
        rt = ExecutionRuntime()
        small = Target(name="tiny", num_qubits=1)
        plan = microquantum.ExecutionPlan.from_circuit(
            QuantumCircuit(4).h(0).cx(0, 1), target=small
        )
        with pytest.raises(CompilationError):
            rt.execute(plan)


class TestDispatchStage:
    def test_backend_validation_failure_raises_dispatch_error(self):
        rt = ExecutionRuntime(backend=_FailingBackend())
        plan = microquantum.ExecutionPlan.from_circuit(
            QuantumCircuit(1), name="fail-plan"
        )
        with pytest.raises(RuntimeDispatchError, match="cannot run on backend"):
            rt.execute(plan)

    def test_unknown_backend_stays_keyerror(self):
        rt = ExecutionRuntime()
        with pytest.raises(KeyError):
            rt.resolve_backend("does_not_exist")


class TestBackendStage:
    def test_failed_job_raises_backend_error(self):
        rt = ExecutionRuntime(backend=_FailingBackend())
        plan = microquantum.ExecutionPlan.from_circuit(QuantumCircuit(1))
        with pytest.raises(BackendExecutionError):
            rt.execute(plan)

    def test_failed_batch_item_recorded_not_raised(self):
        rt = ExecutionRuntime(backend=_FailingBackend())
        records = rt.execute_records([QuantumCircuit(1)])
        assert records[0].error is not None
        assert records[0].status.value == "failed"


class TestPublicSurface:
    def test_runtime_package_exports_errors(self):
        import microquantum.runtime as runtime_mod

        for name in (
            "ExecutionError",
            "PlanningError",
            "CompilationError",
            "RuntimeDispatchError",
            "BackendExecutionError",
        ):
            assert name in runtime_mod.__all__
            assert hasattr(runtime_mod, name)

    def test_top_level_re_export(self):
        assert microquantum.RuntimeConfig is not None
        assert microquantum.runtime_info is not None