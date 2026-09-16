"""Dense-simulation resource boundaries (MQ-14).

State-vector simulation needs ``2^N`` complex amplitudes (16 bytes each) and
density-matrix simulation needs ``2^(2N)``; both grow so quickly that an
obviously impossible allocation must be refused up front with a descriptive
error instead of exhausting process memory.

These are *resource* boundaries, not feature limits: they only guard dense
allocations, and only for allocations the class itself would perform (caller
supplied arrays are already allocated by the caller).  Tensor-network
simulators (MPS / TTN) bypass dense memory by design and are unaffected.
"""

from __future__ import annotations

import numpy as np

#: A 64-qubit state vector is astronomically beyond any feasible dense
#: allocation; capping there also keeps the byte estimate a cheap integer op.
MAX_DENSE_QUBITS = 64

#: Estimated dense allocation budget (bytes) above which an allocation is
#: refused up front.  2 GiB keeps ordinary simulations comfortable while
#: still failing *before* an allocation the machine could not honour.
MAX_DENSE_BYTES = 2**31

_BYTES_PER_COMPLEX128 = 16


def dense_allocation_bytes(num_qubits: int, *, density: bool) -> int:
    """Estimated bytes a dense simulation would need (``0`` if infeasible)."""
    if num_qubits < 0 or num_qubits > MAX_DENSE_QUBITS:
        return 0
    exponent = 2 * num_qubits if density else num_qubits
    return (1 << exponent) * _BYTES_PER_COMPLEX128


def check_dense_allocation(
    num_qubits: int,
    *,
    density: bool,
    kind: str,
) -> None:
    """Validate a dense simulation qubit count before allocating.

    Raises:
        ValueError: If ``num_qubits`` is not a valid qubit count or the
            estimated allocation exceeds :data:`MAX_DENSE_BYTES`.
    """
    if not isinstance(num_qubits, (int, np.integer)):
        raise ValueError(
            f"num_qubits must be an integer, got {type(num_qubits).__name__}"
        )
    num_qubits = int(num_qubits)
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

    estimated = dense_allocation_bytes(num_qubits, density=density)
    if estimated == 0 or estimated > MAX_DENSE_BYTES:
        total = (
            f"2^{2 * num_qubits} complex elements"
            if density
            else f"2^{num_qubits} complex amplitudes"
        )
        estimate_txt = f" (~{_fmt(estimated)})" if estimated else ""
        raise ValueError(
            f"{kind} of {num_qubits} qubits needs {total}{estimate_txt} "
            f"(max {_fmt(MAX_DENSE_BYTES)}), which exceeds the dense-simulation "
            f"budget. Reduce num_qubits or use an exponential-safe simulator such "
            f"as MatrixProductState / TreeTensorNetwork."
        )


def _fmt(value: int) -> str:
    if value <= 0:
        return ">2 GiB"
    if value >= 2**30:
        return f"{value / 2**30:.3g} GiB"
    return f"{value / 2**20:.3g} MiB"