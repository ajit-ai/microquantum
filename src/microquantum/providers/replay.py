"""Record/replay transports for offline provider testing.

:class:`ReplayTransport` is an :data:`HttpResponder` that replays a
scripted ``(status, payload)`` sequence while recording every request,
so provider submit/status/result flows run deterministically with no
network.  Scripts and recordings serialize to JSON fixtures for
version-controlled regression suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import HttpResponder

__all__ = [
    "ReplayTransport",
]


def _decode_body(body: Optional[bytes]) -> Optional[str]:
    """Decode a request body for JSON-safe recording."""
    if body is None:
        return None
    return bytes(body).decode("utf-8", errors="replace")


class ReplayTransport:
    """Scripted HTTP transport with request recording.

    Args:
        script: Queued ``(status, payload)`` responses, consumed in
            order — one per call.
        passthrough: Optional live transport used when the script is
            exhausted; its responses are appended to the script.
        record: When True (default), every request is logged for
            :meth:`save` / inspection.
    """

    def __init__(
        self,
        script: Sequence[tuple[int, Mapping[str, Any]]] = (),
        *,
        passthrough: Optional[HttpResponder] = None,
        record: bool = True,
    ) -> None:
        self._script: list[tuple[int, dict[str, Any]]] = [
            (int(status), dict(payload)) for status, payload in script
        ]
        self._passthrough = passthrough
        self._record = record
        self._requests: list[dict[str, Any]] = []

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Recorded requests in call order."""
        return [dict(entry) for entry in self._requests]

    @property
    def remaining(self) -> int:
        """Queued responses left in the script."""
        return len(self._script)

    def __call__(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Serve the next scripted response (or delegate upstream)."""
        if self._record:
            self._requests.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": _decode_body(body),
                }
            )
        if self._script:
            return self._script.pop(0)
        if self._passthrough is not None:
            status, payload = self._passthrough(method, url, dict(headers or {}), body)
            self._script.append((status, dict(payload)))
            return self._script.pop(0)
        raise LookupError(f"No recorded response left for {method} {url}")

    def save(self, path: str | Path) -> None:
        """Write requests + remaining script to a JSON fixture."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"requests": self._requests, "script": self._script},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayTransport:
        """Rebuild a transport from :meth:`save` output (script restored).

        Raises:
            ValueError: If the fixture is missing or malformed.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read replay fixture '{path}': {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Replay fixture '{path}' is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Replay fixture must be a JSON object")
        transport = cls(
            script=[
                (int(status), dict(payload)) for status, payload in data.get("script", [])
            ]
        )
        transport._requests = [dict(entry) for entry in data.get("requests", [])]
        return transport

    def __len__(self) -> int:
        return len(self._requests)

    def __repr__(self) -> str:
        return f"ReplayTransport(requests={len(self._requests)}, remaining={len(self._script)})"
