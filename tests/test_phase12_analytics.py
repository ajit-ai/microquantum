"""Tests for Phase 12: Analytics toolkit core (CSV loading, base analytics)."""
import json
import os
import tempfile

import numpy as np
import pytest

from microquantum.analytics import AnalysisResult, BaseAnalytics
from microquantum.analytics.csv_loader import load_csv, load_distance_matrix


# =============================================================================
# CSV Loader Tests
# =============================================================================
class TestCSVLoader:
    """Test CSV loading utilities."""

    def test_load_csv(self) -> None:
        """Test basic CSV loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            f.flush()
            filepath = f.name

        try:
            headers, data = load_csv(filepath)
            assert headers == ["a", "b", "c"]
            assert data.shape == (2, 3)
            assert data[0, 0] == 1.0
            assert data[1, 2] == 6.0
        finally:
            os.unlink(filepath)

    def test_load_csv_no_header(self) -> None:
        """Test CSV loading without header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1,2,3\n4,5,6\n")
            f.flush()
            filepath = f.name

        try:
            headers, data = load_csv(filepath, has_header=False)
            assert headers == ["col_0", "col_1", "col_2"]
            assert data.shape == (2, 3)
        finally:
            os.unlink(filepath)

    def test_load_csv_not_found(self) -> None:
        """Test CSV loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")

    def test_load_distance_matrix(self) -> None:
        """Test distance matrix loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1,2,3\n4,5,6\n7,8,9\n")
            f.flush()
            filepath = f.name

        try:
            matrix = load_distance_matrix(filepath)
            assert matrix.shape == (3, 3)
            assert matrix[0, 0] == 1.0
            assert matrix[0, 1] == 2.0
            assert matrix[1, 0] == 4.0
        finally:
            os.unlink(filepath)


# =============================================================================
# Result Contract Tests
# =============================================================================
class TestResultContract:
    """Test the standardized generic Result schema."""

    def test_result_defaults(self) -> None:
        """Test Result default construction."""
        from microquantum.analytics.result import Result

        result = Result(problem="optimization", solution={"value": 0.9})
        assert result.problem == "optimization"
        assert result.solution == {"value": 0.9}
        assert result.confidence == 1.0
        assert result.qubit_count == 0
        assert result.runtime_ms == 0.0

    def test_result_to_dict_recursive_numpy(self) -> None:
        """Test Result serialization handles nested numpy values."""
        from microquantum.analytics.result import Result

        result = Result(
            problem="routing",
            solution={"tour": np.array([0, 2, 1])},
            confidence=0.93,
            fidelity=0.99,
            qubit_count=4,
            runtime_ms=12.5,
            quantum_trace={"metric": {"quantum_value": 10.0, "classical_value": 12.0}},
        )
        data = result.to_dict()
        assert data["solution"]["tour"] == [0, 2, 1]
        assert data["fidelity"] == 0.99
        assert data["quantum_trace"]["metric"]["quantum_value"] == 10.0
        json.dumps(data)  # must be JSON-safe

    def test_result_to_json(self) -> None:
        """Test Result JSON string serialization."""
        from microquantum.analytics.result import Result

        result = Result(problem="t", solution=0.9)
        payload = result.to_json()
        assert isinstance(payload, str)
        assert json.loads(payload)["solution"] == 0.9

    def test_result_improved_over_baseline(self) -> None:
        """Test improved-over-baseline property."""
        from microquantum.analytics.result import Result

        r1 = Result(problem="t", solution=0.9, baseline=0.7)
        assert r1.improved_over_baseline is True

        r2 = Result(problem="t", solution=0.5, baseline=0.7)
        assert r2.improved_over_baseline is False

        r3 = Result(problem="t", solution="pass")
        assert r3.improved_over_baseline is None

    def test_legacy_kwargs_deprecated(self) -> None:
        """Test deprecated decision/classical_baseline kwargs still work."""
        import warnings

        from microquantum.analytics.result import Result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Result(
                problem="t",
                decision=0.9,
                classical_baseline=0.7,
            )

        assert result.solution == 0.9
        assert result.baseline == 0.7
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_legacy_property_aliases(self) -> None:
        """Test deprecated decision/classical_baseline properties."""
        import warnings

        from microquantum.analytics.result import Result

        result = Result(problem="t", solution=0.9, baseline=0.7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.decision == 0.9
            assert result.classical_baseline == 0.7
            assert result.improved_over_classical is True

        assert len(caught) >= 3
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)


# =============================================================================
# Base Analytics Tests
# =============================================================================
class TestBaseAnalytics:
    """Test base analytics class."""

    def test_analysis_result(self) -> None:
        """Test AnalysisResult creation."""
        result = AnalysisResult(
            solution={"key": "value"},
            quantum_metrics={"metric": 1.0},
            classical_metrics={"metric": 0.8},
            comparison={"improvement": 25.0},
            success=True,
        )

        assert result.success is True
        assert result.solution["key"] == "value"
        assert result.quantum_metrics["metric"] == 1.0

    def test_analysis_result_to_json(self) -> None:
        """Test AnalysisResult JSON serialization."""
        result = AnalysisResult(
            solution={"energy": -1.857},
            success=True,
        )
        payload = result.to_json()
        assert isinstance(payload, str)
        json_data = json.loads(payload)
        assert "solution" in json_data
        assert json_data["solution"]["energy"] == -1.857
        assert json_data["success"] is True

    def test_compute_comparison(self) -> None:
        """Test comparison computation."""
        class _Probe(BaseAnalytics):
            @property
            def name(self) -> str:
                return "probe"

            @property
            def description(self) -> str:
                return "probe"

            @property
            def required_inputs(self) -> list[str]:
                return []

            @property
            def algorithm(self) -> str:
                return "probe"

            def analyze(self, **kwargs):  # type: ignore[no-untyped-def]
                return AnalysisResult()

        comp = _Probe()._compute_comparison(
            quantum_value=1.0,
            classical_value=0.8,
            metric_name="test",
            higher_is_better=True,
        )
        assert comp["improvement_pct"] == pytest.approx(25.0, abs=0.1)
        assert comp["quantum_wins"] is True