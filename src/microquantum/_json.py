"""Internal JSON-serialization helpers shared by the public result types.

Provides the canonical output contract used across the SDK:

* ``to_dict()`` returns a JSON-safe ``dict``.  Numpy arrays, complex
  numbers, parameter keys, quantum circuits and state vectors are all
  reduced to JSON-native types.
* ``to_json()`` returns a pretty-printed JSON ``str``.

Complex values are encoded element-wise as ``{"real": ..., "imag": ...}``
dictionaries so that single scalars and whole arrays follow the same rule.

These helpers are implementation details of the public result classes and
are not part of the public API.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import numpy as np

# NOTE: ``core`` imports are intentionally performed lazily inside
# ``json_safe`` to avoid a circular import: ``core`` submodules (e.g.
# ``core.dynamic``) import ``JSONSerializable`` from this module during
# package initialization.


def json_safe(value: Any) -> Any:
    """Recursively reduce a value to JSON-serializable Python types."""
    from .core.circuit import QuantumCircuit
    from .core.serialization import to_dict as circuit_to_dict
    from .core.state import StateVector

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, StateVector):
        return {
            "num_qubits": value.num_qubits,
            "amplitudes": json_safe(value.amplitudes),
        }
    if isinstance(value, QuantumCircuit):
        return circuit_to_dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(
            {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        )
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def json_string(data: Any) -> str:
    """Serialize JSON-safe data to a pretty-printed JSON string.

    Unknown objects are stringified as a final fallback so that
    ``to_json()`` never raises on unusual field values.
    """
    return json.dumps(data, indent=2, default=str)


class JSONSerializable:
    """Mixin providing a uniform JSON-safe output contract.

    ``to_dict()`` serializes all instance fields, including numpy arrays,
    complex numbers, nested result objects, quantum circuits and state
    vectors.  ``to_json()`` returns the pretty-printed JSON string.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return cast(dict[str, Any], json_safe(vars(self)))

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())