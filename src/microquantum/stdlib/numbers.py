"""Standard library: numeric and angle utilities (Phase 117).

Small, dependency-light helpers for the angle arithmetic that shows up
everywhere in quantum programming (rotation gates, phase accumulation,
gradient shifts):

* :func:`mod_2pi` — reduce an angle to its canonical representative in
  ``[0, 2*pi)``.
* :func:`wrap_angle` — symmetric principal value in ``[-pi, pi]``.
* :func:`is_angle_close` — equality *modulo* ``2*pi`` within a tolerance
  (two rotations are equivalent when their angles differ by a full turn).
* :func:`is_identity_angle` — whether a rotation by the angle is the
  identity (its reduced angle is ``0`` modulo ``2*pi``).

Numpy scalar types are accepted and coerced to ``float``; NaN and infinities
are rejected.  Complex numbers and strings are rejected.
"""

from __future__ import annotations

import math
from typing import Any

_TAU = 2.0 * math.pi


def _as_real(value: Any, *, name: str) -> float:
    """Coerce a real value to float, rejecting complex/non-finite input."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a real number, got a {type(value).__name__}")
    if isinstance(value, complex):
        raise TypeError(f"{name} must be a real number, got complex {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def mod_2pi(angle: Any) -> float:
    """Reduce an angle to its canonical representative in ``[0, 2*pi)``.

    Args:
        angle: Any real-valued angle (int, float or numpy scalar).

    Returns:
        ``angle`` modulo ``2*pi`` in ``[0, 2*pi)``.  ``0.0`` stays ``0.0``.

    Raises:
        TypeError: If ``angle`` is complex or not a real number.
        ValueError: If ``angle`` is NaN or infinite.
    """
    value = _as_real(angle, name="angle")
    remainder = math.remainder(value, _TAU)
    return remainder if remainder >= 0.0 else remainder + _TAU


def wrap_angle(angle: Any) -> float:
    """Reduce an angle to its symmetric principal value in ``[-pi, pi]``.

    Uses :func:`math.remainder`, so the result is the numerically closest
    representative modulo ``2*pi``: ``0.0`` maps to ``0.0``, angles just below
    ``pi`` stay positive and angles just above ``pi`` become slightly
    negative.

    Args:
        angle: Any real-valued angle (int, float or numpy scalar).

    Returns:
        The principal value in ``[-pi, pi]``.

    Raises:
        TypeError: If ``angle`` is complex or not a real number.
        ValueError: If ``angle`` is NaN or infinite.
    """
    value = _as_real(angle, name="angle")
    return math.remainder(value, _TAU)


def is_angle_close(left: Any, right: Any, *, tol: float = 1e-10) -> bool:
    """Check whether two angles are equal modulo ``2*pi`` within ``tol``.

    Two angles are close when rotating either by any whole number of full
    turns brings them within ``tol`` of each other — i.e. the rotations they
    describe are the same.

    Args:
        left: First real-valued angle.
        right: Second real-valued angle.
        tol: Absolute tolerance on the modulo-``2*pi`` difference.

    Returns:
        ``True`` when the angles describe equivalent rotations.

    Raises:
        TypeError: If an angle is complex or not a real number.
        ValueError: If an angle is NaN/infinite, or ``tol`` is not a real
            non-negative number.
    """
    if _as_real(tol, name="tol") < 0:
        raise ValueError(f"tol must be non-negative, got {tol!r}")
    diff = (_as_real(left, name="left") - _as_real(right, name="right")) % _TAU
    return diff < tol or (_TAU - diff) < tol


def is_identity_angle(angle: Any, *, tol: float = 1e-9) -> bool:
    """Check whether a rotation by ``angle`` equals the identity.

    A rotation is the identity when its reduced angle is ``0`` modulo
    ``2*pi`` (including a full turn), up to tolerance ``tol``.

    Args:
        angle: Any real-valued angle.
        tol: Absolute tolerance applied against the reduced angle.

    Returns:
        ``True`` when rotating by ``angle`` is (approximately) the identity.

    Raises:
        TypeError: If ``angle`` is complex or not a real number.
        ValueError: If ``angle`` is NaN/infinite, or ``tol`` is not a real
            non-negative number.
    """
    if _as_real(tol, name="tol") < 0:
        raise ValueError(f"tol must be non-negative, got {tol!r}")
    reduced = mod_2pi(angle)
    return reduced < tol or reduced > _TAU - tol


__all__ = ["is_angle_close", "is_identity_angle", "mod_2pi", "wrap_angle"]