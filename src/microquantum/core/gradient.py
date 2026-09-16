"""Analytical gradients via the Parameter-Shift Rule (MQ-13).

The parameter-shift rule computes the exact derivative of an expectation
value :math:`\\langle O\\rangle(\\theta)` for circuits built from
single-qubit rotation gates.  For :math:`U(\\theta)=e^{-i\\theta P/2}`
with :math:`P\\in\\{X,Y,Z\\}` the derivative is

.. math::

    \\frac{d\\langle O\\rangle}{d\\theta}
        = \\frac{\\langle O\\rangle(\\theta+s)
                - \\langle O\\rangle(\\theta-s)}{2\\,\\sin(s)}\\;,

for any shift :math:`s` that is not an integer multiple of
:math:`\\pi` (the default :math:`s=\\pi/2`).

Policies (MQ-13):

* **Identity**: parameters are matched by name (consistent with the
  MQ-12 parameter model), so a ``Parameter`` works transparently even
  when another instance with the same name appears in the circuit.
* **Chain rule**: when a gate angle is a
  :class:`~microquantum.ParameterExpression` ``a*theta + c``, each
  occurrence contributes ``a`` times the parameter-shift difference;
  a parameter used in several gates contributes the sum of its
  per-occurrence gradients (product rule).
* **Observables**: ``Operator``, ``PauliString`` and ``PauliSum`` are
  all supported; ``PauliString``/``PauliSum`` terms are evaluated
  without building dense matrices, optionally over a subset of qubits
  via ``targets``.
* **Execution**: by default shifted circuits are simulated with the
  built-in state-vector engine.  Pass ``backend=`` to route evaluation
  through the MQ-11/12 execution core (``backend.run`` with
  ``seed``/``shots`` for reproducibility).
* **Strictness**: ``param_values`` must be a ``Mapping`` and must
  provide values for every circuit parameter; observables must be one of
  the supported types; a shift with :math:`\\sin(s)\\approx 0` is
  rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

import numpy as np

from .circuit import QuantumCircuit, _narrow_parameterized
from .operators import Operator
from .parameter import Parameter, ParameterExpression
from .pauli import PauliString, PauliSum
from .state import StateVector

_GATE_MAP = {"rx": Operator.Rx, "ry": Operator.Ry, "rz": Operator.Rz}


# ---------------------------------------------------------------------------
# Observable evaluation helpers
# ---------------------------------------------------------------------------


def _pad_pauli_label(label: str, targets: list[int], num_qubits: int) -> str:
    """Place a Pauli ``label`` on the given ``targets`` of an n-qubit system."""
    if len(label) != len(targets):
        raise ValueError(
            f"Pauli observable '{label}' acts on {len(label)} qubit(s) "
            f"but {len(targets)} target(s) were given"
        )
    chars = ["I"] * num_qubits
    for char, target in zip(label, targets, strict=False):
        if target < 0 or target >= num_qubits:
            raise ValueError(
                f"Target {target} out of range for a "
                f"{num_qubits}-qubit system (valid: 0..{num_qubits - 1})"
            )
        chars[target] = char
    return "".join(chars)


def _pauli_term_expectation(
    term: PauliString,
    state: StateVector,
    targets: Optional[list[int]],
) -> float:
    """Expectation of a single Pauli term, padding it to the state width."""
    if targets is None:
        if term.num_qubits != state.num_qubits:
            raise ValueError(
                f"Pauli observable '{term.label}' acts on "
                f"{term.num_qubits} qubit(s) but the circuit has "
                f"{state.num_qubits}; pass targets= to place it on a subset"
            )
        padded = term.label
    else:
        padded = _pad_pauli_label(term.label, targets, state.num_qubits)
    return PauliString(padded, term.coefficient).expectation(state)


def _observable_expectation(
    state: StateVector,
    observable: Union[Operator, PauliString, PauliSum],
    targets: Optional[list[int]],
) -> float:
    """Compute <state|H|state> for Operator, PauliString or PauliSum."""
    if isinstance(observable, PauliString):
        return _pauli_term_expectation(observable, state, targets)
    if isinstance(observable, PauliSum):
        return float(
            sum(
                _pauli_term_expectation(term, state, targets)
                for term in observable.terms
            )
        )
    if isinstance(observable, Operator):
        from .measurement import expectation_value as _exp

        return float(_exp(state, observable, targets=targets))
    raise TypeError(
        f"observable must be an Operator, PauliString or PauliSum, "
        f"got {type(observable).__name__}"
    )


# ---------------------------------------------------------------------------
# Circuit evaluation helpers
# ---------------------------------------------------------------------------


def _name_of(gate_param: Union[Parameter, ParameterExpression]) -> str:
    """Name of the parameter a gate instruction is parameterized by."""
    if isinstance(gate_param, ParameterExpression):
        return gate_param.parameter.name
    return gate_param.name


def _resolve_angle(
    gate_param: Union[Parameter, ParameterExpression],
    param_values: Mapping[Union[str, Parameter], float],
) -> float:
    """Resolve a gate parameter to a float given binding values."""
    if isinstance(gate_param, Parameter):
        return float(_lookup_value(gate_param, param_values))
    if isinstance(gate_param, ParameterExpression):
        return gate_param.evaluate(dict(param_values))
    raise TypeError(
        f"unexpected gate parameter type: {type(gate_param).__name__}"
    )


def _lookup_value(
    param: Parameter,
    param_values: Mapping[Union[str, Parameter], float],
) -> float:
    """Look up a parameter value by object, then by name."""
    if param in param_values:
        return float(param_values[param])
    if param.name in param_values:
        return float(param_values[param.name])
    raise KeyError(
        f"Parameter '{param.name}' not found in param_values; "
        f"provide values for every parameter of the circuit"
    )


def _occurrence_gradient_factor(
    gate_param: Union[Parameter, ParameterExpression],
    target_param: Parameter,
) -> float:
    """Chain-rule factor of a gate occurrence w.r.t. ``target_param``.

    A bare parameter contributes ``1``; an expression ``a*p + c``
    contributes ``a`` (the derivative of the angle with respect to the
    parameter).  Unrelated parameters contribute ``0``.
    """
    if isinstance(gate_param, Parameter):
        return 1.0 if gate_param.name == target_param.name else 0.0
    if isinstance(gate_param, ParameterExpression):
        if gate_param.parameter.name != target_param.name:
            return 0.0
        return float(np.real(gate_param.coefficient))
    raise TypeError(
        f"unexpected gate parameter type: {type(gate_param).__name__}"
    )


def _find_occurrences(
    circuit: "QuantumCircuit",
    target_param: Parameter,
) -> list[tuple[int, float]]:
    """Return ``(gate_idx, chain_factor)`` for every occurrence of target."""
    occurrences: list[tuple[int, float]] = []
    for index, instr in enumerate(circuit._gate_instructions):
        if not QuantumCircuit._is_parameterized_gate(instr):
            continue
        p_instr = _narrow_parameterized(instr)
        factor = _occurrence_gradient_factor(p_instr[1], target_param)
        if factor != 0.0:
            occurrences.append((index, factor))
    return occurrences


def _build_shifted_circuit(
    circuit: "QuantumCircuit",
    gate_idx: int,
    shift_amount: float,
    param_values: Mapping[Union[str, Parameter], float],
) -> "QuantumCircuit":
    """Build a bound circuit with only the ``gate_idx`` angle shifted.

    Every parameterized gate is resolved to a concrete rotation; the gate
    at ``gate_idx`` is evaluated at its resolved angle plus
    ``shift_amount``.  Concrete gates and measurement annotations are
    carried over unchanged.
    """
    resolved = QuantumCircuit(circuit.num_qubits)
    for index, instr in enumerate(circuit._gate_instructions):
        if not QuantumCircuit._is_parameterized_gate(instr):
            resolved._gate_instructions.append(instr)
            continue
        p_instr = _narrow_parameterized(instr)
        gate_type = p_instr[0]
        gate_param = p_instr[1]
        target = p_instr[2]
        angle = _resolve_angle(gate_param, param_values)
        if index == gate_idx:
            angle += shift_amount
        op = _GATE_MAP[gate_type](angle)
        resolved._gate_instructions.append((op, [target]))
    resolved._measurements = list(circuit._measurements)
    return resolved


def _circuit_expectation(
    circuit: "QuantumCircuit",
    observable: Union[Operator, PauliString, PauliSum],
    targets: Optional[list[int]],
    backend: object,
    seed: Optional[int],
    shots: Optional[int],
) -> float:
    """Exact expectation of ``observable`` on a bound circuit.

    Uses the built-in state-vector engine by default; with ``backend``
    the circuit is executed through ``backend.run`` and the resulting
    statevector is used for the exact expectation.
    """
    if backend is None:
        state = circuit.run()
    else:
        run = getattr(backend, "run", None)
        if not callable(run):
            raise TypeError(
                f"backend must expose a run(circuit, ...) method, "
                f"got {type(backend).__name__}"
            )
        result = run(circuit, shots=1024 if shots is None else shots, seed=seed)
        amplitudes = getattr(result, "statevector", None)
        if amplitudes is None:
            name = getattr(backend, "name", None) or repr(backend)
            raise ValueError(
                f"backend '{name}' returned no statevector; exact "
                f"parameter-shift gradients require a statevector-"
                f"simulating backend"
            )
        state = StateVector(
            num_qubits=circuit.num_qubits,
            amplitudes=np.asarray(amplitudes, dtype=np.complex128),
        )
    return _observable_expectation(state, observable, targets)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parameter_shift_gradient(
    circuit: "QuantumCircuit",
    observable: Union[Operator, PauliString, PauliSum],
    param: Parameter,
    param_values: Mapping[Union[str, Parameter], float],
    shift: float = np.pi / 2,
    targets: Optional[list[int]] = None,
    *,
    backend: object = None,
    seed: Optional[int] = None,
    shots: Optional[int] = None,
) -> float:
    """Compute ``d<O>/dparam`` with the Parameter-Shift Rule.

    For a rotation :math:`U(\\theta)=e^{-i\\theta G/2}` with Pauli
    generator :math:`G`, the derivative of an expectation value is

    .. math::

        \\frac{d\\langle O\\rangle}{d\\theta}
            = \\frac{\\langle O\\rangle(\\theta+s)
                    - \\langle O\\rangle(\\theta-s)}{2\\,\\sin(s)}\\;.

    When the parameter appears in several gates (or inside a
    :class:`~microquantum.ParameterExpression` with a coefficient), each
    occurrence is shifted independently, scaled by the chain-rule factor
    ``a`` of its angle expression w.r.t. ``param``, and the results are
    summed (product rule).

    Args:
        circuit: Parameterized quantum circuit.
        observable: ``Operator``, ``PauliString`` or ``PauliSum``.
        param: The parameter to differentiate with respect to (matched by
            name).
        param_values: Mapping of parameter names/objects to numeric values
            for the differentiation point.  Must cover every parameter in
            the circuit.
        shift: Shift angle for the rule (default ``pi/2``).  Must not be
            an integer multiple of ``pi``.
        targets: Optional qubit indices a Pauli/operator observable acts
            on.  Needed when an ``Operator``/``PauliString``/``PauliSum``
            covers a strict subset of the circuit's qubits.
        backend: Optional execution backend used to evaluate the shifted
            circuits.  Defaults to the built-in state-vector engine.
        seed: Optional seed passed to ``backend.run`` for reproducible
            evaluation.
        shots: Optional shot count passed to ``backend.run``.  Expectation
            values use the exact statevector, so ``shots`` only affects
            the (ignored) sampled counts.

    Returns:
        The analytical partial derivative ``d<O>/d(param)``.

    Raises:
        ValueError: If ``shift`` is an integer multiple of pi, an
            observable is malformed, or the backend returns no
            statevector.
        TypeError: If ``param_values`` is not a ``Mapping`` or the
            observable is not an ``Operator``/``PauliString``/``PauliSum``.
        KeyError: If ``param`` (or any other circuit parameter) has no
            value in ``param_values``, or ``param`` does not occur in any
            gate.
    """
    if not isinstance(param_values, Mapping):
        raise TypeError(
            f"param_values must be a Mapping of parameters to values, "
            f"got {type(param_values).__name__}"
        )
    sin_shift = np.sin(shift)
    if abs(sin_shift) < 1e-12:
        raise ValueError(
            f"shift value {shift} gives sin(s) ~= 0; use a shift that "
            f"is not an integer multiple of pi"
        )
    # Validate the observable type eagerly for a clean error message.
    if not isinstance(observable, (Operator, PauliString, PauliSum)):
        raise TypeError(
            f"observable must be an Operator, PauliString or PauliSum, "
            f"got {type(observable).__name__}"
        )

    occurrences = _find_occurrences(circuit, param)
    if not occurrences:
        raise KeyError(
            f"Parameter '{param.name}' not found in any gate in circuit"
        )

    total = 0.0
    for gate_idx, factor in occurrences:
        circuit_plus = _build_shifted_circuit(
            circuit, gate_idx, shift, param_values
        )
        circuit_minus = _build_shifted_circuit(
            circuit, gate_idx, -shift, param_values
        )
        exp_plus = _circuit_expectation(
            circuit_plus, observable, targets, backend, seed, shots
        )
        exp_minus = _circuit_expectation(
            circuit_minus, observable, targets, backend, seed, shots
        )
        total += factor * (exp_plus - exp_minus) / (2.0 * sin_shift)

    return float(total)


def gradient(
    circuit: "QuantumCircuit",
    observable: Union[Operator, PauliString, PauliSum],
    param_values: Mapping[Union[str, Parameter], float],
    shift: float = np.pi / 2,
    targets: Optional[list[int]] = None,
    *,
    backend: object = None,
    seed: Optional[int] = None,
    shots: Optional[int] = None,
) -> dict[Parameter, float]:
    """Compute the full analytic gradient vector for all parameters.

    Evaluates ``d<O>/d(theta_i)`` for every :class:`Parameter` of
    ``circuit`` (in the deterministic :attr:`QuantumCircuit.parameters`
    order), delegating each entry to
    :func:`parameter_shift_gradient`.  The result uses the circuit's
    :class:`Parameter` objects as keys, so it plugs directly into the
    gradient-aware optimizers (``gradient_fn=``).

    Args:
        circuit: Parameterized quantum circuit.
        observable: ``Operator``, ``PauliString`` or ``PauliSum``.
        param_values: Mapping of parameter names/objects to numeric values
            for the differentiation point.  Must cover every parameter in
            the circuit.
        shift: Shift angle for the rule (default ``pi/2``).
        targets: Optional qubit indices the observable acts on.
        backend: Optional execution backend (see
            :func:`parameter_shift_gradient`).
        seed: Optional seed for backend evaluation.
        shots: Optional shot count for backend evaluation (expectations
            use the statevector, so ``shots`` does not change the result).

    Returns:
        A ``{Parameter: float}`` mapping with one analytic gradient entry
        per circuit parameter.

    Raises:
        TypeError: If ``param_values`` is not a ``Mapping``.
        KeyError: If a circuit parameter has no value in ``param_values``.
    """
    if not isinstance(param_values, Mapping):
        raise TypeError(
            f"param_values must be a Mapping of parameters to values, "
            f"got {type(param_values).__name__}"
        )
    result: dict[Parameter, float] = {}
    for param in circuit.parameters:
        result[param] = parameter_shift_gradient(
            circuit,
            observable,
            param,
            param_values,
            shift=shift,
            targets=targets,
            backend=backend,
            seed=seed,
            shots=shots,
        )
    return result


__all__ = ["parameter_shift_gradient", "gradient"]