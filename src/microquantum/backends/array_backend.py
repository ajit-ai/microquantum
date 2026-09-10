"""Pluggable array backend for tensor-network simulation.

NumPy is the default execution backend.  When CuPy is installed the user
may switch to GPU execution with :func:`set_array_backend`.  All Phase-15
simulators dispatch their tensor operations through this module, so they
run unchanged on either backend.

Usage::

    from microquantum.backends.array_backend import set_array_backend, xp

    set_array_backend("cupy")   # GPU when available; falls back to numpy
    assert gpu_available()

    A = xp().zeros((4, 4), dtype=complex)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

_DEFAULT_BACKEND = "numpy"

_current: str = _DEFAULT_BACKEND


def available_backends() -> list[str]:
    """Return the list of array backends installed on this machine."""
    names = [_DEFAULT_BACKEND]
    try:
        import cupy  # noqa: F401
    except ImportError:
        return names
    names.append("cupy")
    return names


def gpu_available() -> bool:
    """True when CuPy is importable (independent of the active backend)."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


def set_array_backend(name: str) -> str:
    """Activate an array backend by name ('numpy' or 'cupy').

    If 'cupy' is requested but not installed, the backend falls back to
    NumPy with a warning.  Returns the backend that ended up active.
    """
    global _current

    name = name.lower()
    if name == _DEFAULT_BACKEND:
        _current = _DEFAULT_BACKEND
        return _current

    if name not in ("cupy",):
        raise ValueError(
            f"Unknown array backend '{name}'. "
            f"Available: {available_backends()}"
        )

    try:
        import cupy  # noqa: F401
    except ImportError:
        warnings.warn(
            "CuPy not installed; falling back to the NumPy array backend.",
            RuntimeWarning,
            stacklevel=2,
        )
        _current = _DEFAULT_BACKEND
        return _current

    _current = "cupy"
    return _current


def get_array_backend() -> str:
    """Return the currently active array backend name."""
    return _current


def is_gpu() -> bool:
    """True when the active backend is CuPy."""
    return _current == "cupy"


def xp() -> Any:
    """Return the active array module (numpy or cupy).

    The returned module exposes the familiar numpy namespace
    (einsum, tensordot, transpose, ...).
    """
    if _current == "cupy":
        import cupy

        return cupy
    import numpy

    return numpy


def to_numpy(array: Any) -> Any:
    """Convert an array to a NumPy ndarray (no-op for NumPy arrays)."""
    if _current == "cupy" and hasattr(array, "get"):
        return array.get()
    if array is None:
        return None
    mod = xp()
    if hasattr(mod, "asnumpy") and hasattr(array, "device"):
        return mod.asnumpy(array)
    return array


# ---------------------------------------------------------------------------
# Forwarding helpers (defined here so simulator code reads cleanly)
# ---------------------------------------------------------------------------

def eye(dim: int, dtype: Any = complex) -> Any:
    """Identity matrix on the active backend."""
    return xp().eye(dim, dtype=dtype)


def zeros(shape: Any, dtype: Any = complex) -> Any:
    """Zeros array on the active backend."""
    return xp().zeros(shape, dtype=dtype)


def ones(shape: Any, dtype: Any = complex) -> Any:
    """Ones array on the active backend."""
    return xp().ones(shape, dtype=dtype)


def asarray(x: Any, dtype: Any = None) -> Any:
    """Converter to the active backend."""
    return xp().asarray(x, dtype=dtype)


def reshape(a: Any, shape: Any) -> Any:
    """Reshape an array."""
    return xp().reshape(a, shape)


def transpose(a: Any, axes: Optional[Any] = None) -> Any:
    """Transpose an array."""
    return xp().transpose(a, axes)


def tensordot(a: Any, b: Any, axes: Any = 0) -> Any:
    """Tensor contraction on the active backend."""
    return xp().tensordot(a, b, axes)


def einsum(subscripts: str, *operands: Any) -> Any:
    """Einstein summation on the active backend."""
    return xp().einsum(subscripts, *operands)


def conj(a: Any) -> Any:
    """Complex conjugate."""
    return xp().conj(a)


def abs(a: Any) -> Any:
    """Elementwise absolute value."""
    return xp().abs(a)


def real(a: Any) -> Any:
    """Real part of an array."""
    return xp().real(a)


def imag(a: Any) -> Any:
    """Imaginary part of an array."""
    return xp().imag(a)


def sum(a: Any, axis: Optional[int] = None) -> Any:
    """Sum over an axis."""
    return xp().sum(a, axis=axis)


def sqrt(a: Any) -> Any:
    """Elementwise square root."""
    return xp().sqrt(a)


def clip(a: Any, a_min: Any, a_max: Any) -> Any:
    """Elementwise clipping."""
    return xp().clip(a, a_min, a_max)


def maximum(a: Any, b: Any) -> Any:
    """Elementwise maximum."""
    return xp().maximum(a, b)


def stack(arrays: list[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along a new axis."""
    return xp().stack(arrays, axis=axis)


def concatenate(arrays: list[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along an existing axis."""
    return xp().concatenate(arrays, axis=axis)


def matmul(a: Any, b: Any) -> Any:
    """Matrix product."""
    return xp().matmul(a, b)


def svd(a: Any, full_matrices: bool = False) -> tuple[Any, Any, Any]:
    """Singular value decomposition (u, s, vh)."""
    return xp().linalg.svd(a, full_matrices=full_matrices)  # type: ignore[no-any-return]


def qr(a: Any) -> tuple[Any, Any]:
    """QR decomposition (q, r)."""
    return xp().linalg.qr(a)  # type: ignore[no-any-return]


def real_if_close(a: Any) -> Any:
    """Return real values when the imaginary part is negligible."""
    from numpy import real_if_close as _ric

    return _ric(to_numpy(a))