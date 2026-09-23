"""Programmatic report builder over analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .result import Result

__all__ = [
    "ReportBuilder",
    "to_records",
]


def _is_scalar(value: Any) -> bool:
    """True for plain scalar report values."""
    return value is None or isinstance(value, (bool, int, float, str))


def to_records(result: Result) -> list[dict[str, Any]]:
    """Flatten a :class:`Result` into ``{field, value}`` rows.

    Scalar attributes become one row each; ``solution``,
    ``quantum_trace``, ``baseline``, ``classical_baseline`` and
    ``decision`` mappings expand one level.  Non-scalar leftovers are
    kept by reference for downstream JSON-safe handling.
    """
    rows: list[dict[str, Any]] = [
        {"field": "problem", "value": result.problem},
        {"field": "confidence", "value": result.confidence},
        {"field": "fidelity", "value": result.fidelity},
        {"field": "qubit_count", "value": result.qubit_count},
        {"field": "runtime_ms", "value": result.runtime_ms},
        {"field": "duration", "value": result.duration},
    ]
    for section in ("solution", "baseline", "quantum_trace"):
        payload = getattr(result, section, None)
        if isinstance(payload, dict):
            for key, value in payload.items():
                rows.append({"field": f"{section}.{key}", "value": value})
        elif payload is not None and _is_scalar(payload):
            rows.append({"field": section, "value": payload})
    return rows


@dataclass
class ReportBuilder:
    """Compose markdown reports from sections, tables and records."""

    title: str = "MicroQuantum Report"
    _sections: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def add_section(self, title: str, body: str) -> ReportBuilder:
        """Append a markdown body section; returns self for chaining."""
        if not title:
            raise ValueError("Section title must be non-empty")
        self._sections.append((title, body))
        return self

    def add_table(
        self, title: str, rows: list[dict[str, Any]], columns: Union[list[str], None] = None
    ) -> ReportBuilder:
        """Append a markdown table built from row dictionaries."""
        if not title:
            raise ValueError("Table title must be non-empty")
        if not rows:
            return self.add_section(title, "_No rows._")
        headers = list(columns) if columns is not None else sorted(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        return self.add_section(title, "\n".join(lines))

    def add_result(self, title: str, result: Result) -> ReportBuilder:
        """Append a :class:`Result` rendered via :func:`to_records`."""
        return self.add_table(title, to_records(result), columns=["field", "value"])

    def to_markdown(self) -> str:
        """Render the full markdown document."""
        lines = [f"# {self.title}", ""]
        for title, body in self._sections:
            lines.extend([f"## {title}", "", body, ""])
        return "\n".join(lines).rstrip("\n") + "\n"

    def __len__(self) -> int:
        return len(self._sections)
