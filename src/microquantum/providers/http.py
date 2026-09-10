"""Minimal HTTP transport for hardware providers.

Uses only the standard library (``urllib.request``) and exposes a
swappable callable that tests can replace with a fake responder.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from .base import HttpResponder


class HttpError(RuntimeError):
    """Raised when a provider API request fails.

    Attributes:
        status: HTTP status code of the failed response.
    """

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def http_request(
    method: str,
    url: str,
    headers: Optional[dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> tuple[int, dict[str, Any]]:
    """Perform an HTTP request and return (status, parsed JSON).

    Args:
        method: HTTP method (GET, POST, DELETE, ...).
        url: Fully-qualified URL.
        headers: Optional request headers.
        body: Optional raw request body bytes.

    Returns:
        Tuple of (status_code, decoded JSON response).

    Raises:
        HttpError: If the response has a non-2xx status or bad JSON.
    """
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)

    if not raw:
        return status, {}

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}

    if status >= 400:
        detail = payload.get("error", payload.get("message", raw.decode("utf-8", "replace")))
        raise HttpError(f"Provider API error ({status}): {detail}", status)

    return status, payload


def make_responder(transport: HttpResponder) -> HttpResponder:
    """Return a transport wrapper so tests can inject a fake responder."""
    return transport