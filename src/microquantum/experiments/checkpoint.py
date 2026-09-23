"""Checkpointing for resumable experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "Checkpoint",
]


class Checkpoint:
    """Resume record for a long-running experiment or sweep.

    Tracks completed plan names so a re-run can skip them via
    :meth:`pending`.  Persisted as JSON through :meth:`save` /
    :meth:`load`.
    """

    def __init__(self, experiment_id: str) -> None:
        if not experiment_id:
            raise ValueError("experiment_id must be non-empty")
        self._experiment_id = experiment_id
        self._completed: list[str] = []

    @property
    def experiment_id(self) -> str:
        """Experiment identifier."""
        return self._experiment_id

    @property
    def completed(self) -> list[str]:
        """Completed plan names in completion order."""
        return list(self._completed)

    def mark_done(self, plan_name: str) -> None:
        """Record *plan_name* as completed (idempotent)."""
        if plan_name not in self._completed:
            self._completed.append(plan_name)

    def pending(self, expected: Sequence[str]) -> list[str]:
        """Expected plan names not yet completed, in given order."""
        done = set(self._completed)
        return [name for name in expected if name not in done]

    @property
    def is_complete(self) -> bool:
        """Whether anything is recorded as complete (non-empty)."""
        return bool(self._completed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {"experiment_id": self._experiment_id, "completed": list(self._completed)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Rebuild from :meth:`to_dict` output."""
        checkpoint = cls(str(data["experiment_id"]))
        for name in data.get("completed", []):
            checkpoint.mark_done(str(name))
        return checkpoint

    def save(self, path: str | Path) -> None:
        """Write the checkpoint to a JSON file (creating parents)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        """Read a checkpoint written by :meth:`save`.

        Raises:
            ValueError: If the file is missing or not a JSON object.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read checkpoint file '{path}': {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint file '{path}' is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Checkpoint file must contain a JSON object")
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self._completed)

    def __repr__(self) -> str:
        return f"Checkpoint(experiment_id='{self._experiment_id}', completed={len(self)})"
