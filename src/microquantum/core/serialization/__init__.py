"""Stable, versioned serialization for Core objects.

JSON is used as the wire format, but the contract is the documented
schema (``type``/``version``/``payload``), never private Python
attributes.  All helpers are deterministic, validated and round-trip
safe.
"""

from __future__ import annotations

import json
from typing import Any

from ._model import (
    FORMAT_VERSION,
    from_dict,
    from_json,
    load,
    save,
    to_dict,
    to_json,
)

__all__ = [
    "SCHEMA_VERSION",
    "FORMAT_VERSION",
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "save",
    "load",
    "serialize_circuit",
    "deserialize_circuit",
    "serialize_gate",
    "deserialize_gate",
    "serialize_parameters",
    "deserialize_parameters",
    "serialize_observable",
    "deserialize_observable",
    "serialize_state",
    "deserialize_state",
    "serialize_result",
    "dumps_canonical",
]

SCHEMA_VERSION = 1


def dumps_canonical(payload: Any) -> str:
    """Serialize *payload* to canonical JSON (sorted keys, compact)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _envelope(kind: str, payload: Any, version: int = SCHEMA_VERSION) -> dict[str, Any]:
    return {"type": kind, "version": version, "payload": payload}


def _check_envelope(data: Any, kind: str) -> Any:
    if not isinstance(data, dict) or data.get("type") != kind:
        raise ValueError(f"Expected serialized '{kind}', got {type(data).__name__}")
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported {kind} schema version {data.get('version')}")
    return data["payload"]


# -- parameters --------------------------------------------------------


def serialize_parameters(binding: Any) -> dict[str, Any]:
    """Serialize a parameter binding (mapping or ParameterBinding)."""
    raw = getattr(binding, "values", None)
    values = raw if isinstance(raw, dict) else dict(binding)
    payload = {str(k): [float(complex(v).real), float(complex(v).imag)] for k, v in values.items()}
    return _envelope("parameters", payload)


def deserialize_parameters(data: Any) -> dict[str, complex]:
    """Deserialize to a plain ``{name: value}`` mapping."""
    payload = _check_envelope(data, "parameters")
    if not isinstance(payload, dict):
        raise ValueError("Invalid parameters payload")
    return {str(k): complex(v[0], v[1]) for k, v in payload.items()}


# -- gates -------------------------------------------------------------


def serialize_gate(gate: Any) -> dict[str, Any]:
    """Serialize a Gate/Operator-like object to a schema dictionary."""
    name = str(getattr(gate, "name", "custom")).lower()
    nq = int(getattr(gate, "num_qubits", 1))
    raw_params = getattr(gate, "parameters", ()) or ()
    params: list[Any] = []
    param_iter = raw_params if isinstance(raw_params, (list, tuple)) else [raw_params]
    for p in param_iter:
        pname = getattr(p, "name", None)
        if pname is not None and not isinstance(p, (int, float, complex)):
            params.append({"parameter": str(pname)})
        elif isinstance(p, (int, float)):
            params.append(float(p))
        elif isinstance(p, complex):
            params.append([p.real, p.imag])
        else:
            params.append(str(p))
    matrix = getattr(gate, "matrix", None)
    mat_payload = None
    if matrix is not None and not getattr(gate, "is_parameterized", False):
        import numpy as _np  # noqa: PLC0415

        arr = _np.asarray(matrix, dtype=_np.complex128)
        mat_payload = {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    return _envelope("gate", {"name": name, "num_qubits": nq, "params": params, "matrix": mat_payload})


def deserialize_gate(data: Any) -> Any:
    """Deserialize a gate produced by :func:`serialize_gate`."""
    payload = _check_envelope(data, "gate")
    from microquantum.core.gates import make_gate  # noqa: PLC0415
    from microquantum.core.parameters import Parameter  # noqa: PLC0415

    params: list[Any] = []
    for p in payload.get("params", []):
        if isinstance(p, dict) and "parameter" in p:
            params.append(Parameter(str(p["parameter"])))
        elif isinstance(p, list):
            params.append(complex(p[0], p[1]))
        else:
            params.append(p)
    try:
        return make_gate(str(payload["name"]), tuple(params))
    except ValueError:
        # Unknown/custom gate: fall back to a dense UnitaryGate when a
        # matrix payload is present.
        mat = payload.get("matrix")
        if mat is None:
            raise
        import numpy as _np  # noqa: PLC0415

        arr = _np.asarray(mat["real"]) + 1j * _np.asarray(mat["imag"])
        from microquantum.core.gates import UnitaryGate  # noqa: PLC0415

        return UnitaryGate(arr, name=str(payload["name"]))


# -- circuits ----------------------------------------------------------


def serialize_circuit(circuit: Any) -> dict[str, Any]:
    """Serialize a :class:`QuantumCircuit` to a schema dictionary."""
    from microquantum.core.circuit import (  # noqa: PLC0415
        QuantumCircuit,
        _narrow_concrete,
        _narrow_parameterized,
    )

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Expected QuantumCircuit, got {type(circuit).__name__}")
    instructions: list[dict[str, Any]] = []
    for instr in circuit._gate_instructions:  # noqa: SLF001
        if circuit._is_parameterized_gate(instr):  # noqa: SLF001
            name, param, target = _narrow_parameterized(instr)
            pname = getattr(param, "name", None)
            payload_param: Any
            if pname is not None and not isinstance(param, (int, float, complex)):
                payload_param = {"parameter": str(pname)}
                # Preserve affine expression structure when available.
                coef = getattr(param, "coefficient", None)
                const = getattr(param, "constant", None)
                if coef is not None or const is not None:
                    payload_param = {
                        "parameter": str(pname),
                        "coefficient": [complex(coef).real, complex(coef).imag]
                        if coef is not None
                        else [1.0, 0.0],
                        "constant": [complex(const).real, complex(const).imag]
                        if const is not None
                        else [0.0, 0.0],
                    }
            elif isinstance(param, complex):
                payload_param = [param.real, param.imag]
            else:
                payload_param = float(param) if isinstance(param, (int, float)) else str(param)
            instructions.append(
                {"gate": str(name), "qubits": [int(target)], "param": payload_param}
            )
        else:
            op, targets = _narrow_concrete(instr)
            instructions.append({"gate": op.name, "qubits": [int(t) for t in targets]})
    payload = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": int(getattr(circuit, "num_clbits", 0) or 0),
        "instructions": instructions,
        "measurements": [int(m) for m in circuit._measurements],  # noqa: SLF001
    }
    return _envelope("circuit", payload)


def deserialize_circuit(data: Any) -> Any:
    """Deserialize a circuit produced by :func:`serialize_circuit`."""
    from microquantum.core.circuit import QuantumCircuit  # noqa: PLC0415
    from microquantum.core.parameters import Parameter, ParameterExpression  # noqa: PLC0415

    payload = _check_envelope(data, "circuit")
    circuit = QuantumCircuit(int(payload["num_qubits"]))
    for instr in payload.get("instructions", []):
        gate_name = str(instr["gate"]).lower()
        qubits = [int(q) for q in instr["qubits"]]
        method = getattr(circuit, gate_name, None)
        param = instr.get("param", None)
        if param is not None and method is not None and gate_name in ("rx", "ry", "rz"):
            if isinstance(param, dict) and "parameter" in param:
                base = Parameter(str(param["parameter"]))
                coef = param.get("coefficient", [1.0, 0.0])
                const = param.get("constant", [0.0, 0.0])
                expression: Any = base
                if abs(complex(coef[0], coef[1]) - 1.0) > 1e-12 or abs(complex(const[0], const[1])) > 1e-12:
                    expression = ParameterExpression(
                        base,
                        coefficient=complex(coef[0], coef[1]),
                        constant=complex(const[0], const[1]),
                    )
                method(expression, qubits[0])
            else:
                value = complex(param[0], param[1]) if isinstance(param, list) else float(param)
                method(value.real if isinstance(value, complex) and value.imag == 0 else value, qubits[0])
        elif method is not None:
            try:
                method(*qubits)
            except TypeError:
                method(qubits[0] if len(qubits) == 1 else qubits)
        else:  # pragma: no cover - unknown gate names
            raise ValueError(f"Unknown gate '{gate_name}' in serialized circuit")
    for m in payload.get("measurements", []):
        circuit.measure(int(m))
    return circuit


# -- observables / states / results ------------------------------------


def serialize_observable(observable: Any) -> dict[str, Any]:
    """Serialize a Pauli/Matrix/Sum observable (or PauliSum)."""
    label = getattr(observable, "label", None)
    if label is not None:
        coef = complex(getattr(observable, "coefficient", 1.0))
        return _envelope("observable", {"kind": "pauli", "label": str(label), "coefficient": [coef.real, coef.imag]})
    to_matrix = getattr(observable, "to_matrix", None)
    if callable(to_matrix):
        import numpy as _np  # noqa: PLC0415

        mat = _np.asarray(to_matrix(), dtype=_np.complex128)
        return _envelope(
            "observable",
            {"kind": "matrix", "real": mat.real.tolist(), "imag": mat.imag.tolist()},
        )
    terms = getattr(observable, "terms", None)
    iterable = terms() if callable(terms) else terms
    if iterable is not None:
        out = []
        for t in iterable:
            out.append(
                {"label": str(t.label), "coefficient": [complex(getattr(t, "coefficient", 1.0)).real, complex(getattr(t, "coefficient", 1.0)).imag]}
            )
        return _envelope("observable", {"kind": "pauli-sum", "terms": out})
    raise TypeError(f"Cannot serialize observable of type {type(observable).__name__}")


def deserialize_observable(data: Any) -> Any:
    """Deserialize an observable produced by :func:`serialize_observable`."""
    from microquantum.core.observables import (  # noqa: PLC0415
        MatrixObservable,
        PauliObservable,
        SumObservable,
    )

    payload = _check_envelope(data, "observable")
    kind = payload.get("kind")
    if kind == "pauli":
        coef = payload.get("coefficient", [1.0, 0.0])
        return PauliObservable(str(payload["label"]), complex(coef[0], coef[1]))
    if kind == "matrix":
        import numpy as _np  # noqa: PLC0415

        return MatrixObservable(_np.asarray(payload["real"]) + 1j * _np.asarray(payload["imag"]))
    if kind == "pauli-sum":
        terms = [PauliObservable(t["label"]) for t in payload["terms"]]
        weights = [complex(c[0], c[1]).real for c in [t["coefficient"] for t in payload["terms"]]]
        return SumObservable(terms, weights)
    raise ValueError(f"Unknown observable kind '{kind}'")


def serialize_state(state: Any) -> dict[str, Any]:
    """Serialize a state vector or density matrix."""
    import numpy as _np  # noqa: PLC0415

    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        arr = _np.asarray(amps, dtype=_np.complex128)
        nq = int(_np.log2(arr.shape[0]))
        return _envelope("state", {"kind": "statevector", "num_qubits": nq, "real": arr.real.tolist(), "imag": arr.imag.tolist()})
    for attr in ("matrix", "rho"):
        mat = getattr(state, attr, None)
        if isinstance(mat, _np.ndarray):
            arr = _np.asarray(mat, dtype=_np.complex128)
            return _envelope("state", {"kind": "density-matrix", "real": arr.real.tolist(), "imag": arr.imag.tolist()})
    arr = _np.asarray(state, dtype=_np.complex128)
    kind = "statevector" if arr.ndim == 1 else "density-matrix"
    return _envelope("state", {"kind": kind, "real": arr.real.tolist(), "imag": arr.imag.tolist()})


def deserialize_state(data: Any) -> Any:
    """Deserialize a state produced by :func:`serialize_state`."""
    import numpy as _np  # noqa: PLC0415

    payload = _check_envelope(data, "state")
    real = _np.asarray(payload["real"], dtype=float)
    imag = _np.asarray(payload["imag"], dtype=float)
    arr = (real + 1j * imag).astype(_np.complex128)
    if payload.get("kind") == "statevector":
        from microquantum.core.state import StateVector  # noqa: PLC0415

        nq = int(payload.get("num_qubits", int(_np.log2(arr.shape[0]))))
        return StateVector(nq, amplitudes=arr)
    import math as _math  # noqa: PLC0415

    from microquantum.core.density_matrix import DensityMatrix  # noqa: PLC0415

    nq = int(_math.log2(arr.shape[0]))
    return DensityMatrix(nq, arr)


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize an execution/measurement result to a schema dictionary."""
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            if "type" not in payload:
                return _envelope("result", payload)
            if payload.get("type") == "result":
                return {"type": "result", "version": payload.get("version", 1), "payload": payload.get("payload", {})}
        raise ValueError(f"Unsupported result envelope: {payload!r}")
    counts = getattr(result, "counts", None)
    if counts is not None:
        data = counts() if callable(counts) else counts
        if not isinstance(data, dict):
            raise TypeError(f"Cannot serialize counts of type {type(data).__name__}")
        return _envelope("result", {"counts": dict(data)})
    raise TypeError(f"Cannot serialize result of type {type(result).__name__}")
