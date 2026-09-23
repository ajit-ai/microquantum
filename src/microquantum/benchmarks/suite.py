"""Volumetric benchmark suites with deterministic seeding and reports.

:class:`BenchmarkSuite` collects no-argument ``run()`` benchmarks
(quantum volume, CLOPS, mirror fidelity, ...) under one roof and
produces a JSON-safe :class:`SuiteReport` with per-benchmark entries
and a metric summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from .._json import JSONSerializable, json_string

__all__ = [
    "BenchmarkCase",
    "SuiteReport",
    "BenchmarkSuite",
]


class BenchmarkCase(Protocol):
    """Structural protocol for suite-compatible benchmarks."""

    def run(self) -> Any:
        """Run the benchmark and return its result object."""
        ...  # pragma: no cover - protocol stub


@dataclass
class SuiteReport(JSONSerializable):
    """Report of one suite run."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def succeeded(self) -> list[str]:
        """Names of benchmarks that ran cleanly."""
        return [entry["name"] for entry in self.entries if entry.get("ok")]

    @property
    def failed(self) -> list[str]:
        """Names of benchmarks that raised."""
        return [entry["name"] for entry in self.entries if not entry.get("ok")]

    def summary(self) -> dict[str, float]:
        """Mean value per metric across successful entries."""
        grouped: dict[str, list[float]] = {}
        for entry in self.entries:
            if not entry.get("ok"):
                continue
            metric = str(entry.get("metric", entry["name"]))
            grouped.setdefault(metric, []).append(float(entry.get("value", 0.0)))
        return {metric: sum(values) / len(values) for metric, values in grouped.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {"entries": [dict(entry) for entry in self.entries], "seed": self.seed}

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SuiteReport:
        """Rebuild a report from :meth:`to_dict` output."""
        return cls(
            entries=[dict(entry) for entry in data.get("entries", [])],
            seed=data.get("seed"),
        )


class BenchmarkSuite:
    """Named collection of benchmarks run as one suite.

    Args:
        seed: Suite seed recorded in the report (benchmarks keep
            their own seeds unless configured otherwise).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._cases: list[tuple[str, BenchmarkCase, dict[str, Any]]] = []

    def add(
        self, name: str, benchmark: BenchmarkCase, config: Optional[dict[str, Any]] = None
    ) -> BenchmarkSuite:
        """Register *benchmark* under *name*; returns self for chaining."""
        if not name:
            raise ValueError("Benchmark name must be non-empty")
        if any(existing == name for existing, _, _ in self._cases):
            raise ValueError(f"Duplicate benchmark name '{name}'")
        self._cases.append((name, benchmark, dict(config or {})))
        return self

    @property
    def names(self) -> list[str]:
        """Registered benchmark names in order."""
        return [name for name, _, _ in self._cases]

    def run(self) -> SuiteReport:
        """Run every benchmark, recording errors instead of raising."""
        entries: list[dict[str, Any]] = []
        for name, benchmark, config in self._cases:
            try:
                result = benchmark.run()
                to_dict = getattr(result, "to_dict", None)
                payload = to_dict() if callable(to_dict) else {"value": result}
                entries.append(
                    {
                        "name": name,
                        "ok": True,
                        "metric": payload.get("metric_name", name)
                        if isinstance(payload, dict)
                        else name,
                        "value": payload.get("value", 0.0)
                        if isinstance(payload, dict)
                        else 0.0,
                        "config": dict(config),
                        "result": payload if isinstance(payload, dict) else {"value": payload},
                    }
                )
            except Exception as exc:
                entries.append({"name": name, "ok": False, "error": str(exc), "config": dict(config)})
        return SuiteReport(entries=entries, seed=self._seed)

    def __len__(self) -> int:
        return len(self._cases)

    def __repr__(self) -> str:
        return f"BenchmarkSuite(benchmarks={self.names})"
