"""QPE phase-filter gadgets shared by HHL and phase estimation.

:class:`QPEPhaseFilter` builds the eigenvalue-inversion rotation block
(CX / Ry / CX ladders with geometrically decreasing angles) reused by
HHL-style post-processing, and exposes the rotation-angle schedule for
inspection and testing.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.circuit import QuantumCircuit

__all__ = [
    "QPEPhaseFilter",
]


class QPEPhaseFilter:
    """Eigenvalue-inversion rotation filter.

    Args:
        kappa: Condition-number scale; rotation angles are
            ``kappa / 2**i`` per counting qubit ``i``.  Defaults to
            ``pi / 2`` matching the HHL convention.
    """

    def __init__(self, kappa: Optional[float] = None) -> None:
        resolved = math.pi / 2 if kappa is None else float(kappa)
        if resolved <= 0:
            raise ValueError("kappa must be positive")
        self._kappa = resolved

    @property
    def kappa(self) -> float:
        """Condition-number scale."""
        return self._kappa

    def rotation_angles(self, num_counting: int) -> list[float]:
        """Angle schedule ``[kappa, kappa/2, ...]`` of length *num_counting*."""
        if num_counting < 1:
            raise ValueError("num_counting must be >= 1")
        return [self._kappa / (2**i) for i in range(num_counting)]

    def filter_circuit(self, num_counting: int) -> QuantumCircuit:
        """Build the inversion block on ``num_counting + 1`` qubits.

        Each counting qubit controls an Ry rotation of the ancilla
        (the last qubit) with its scheduled angle.
        """
        angles = self.rotation_angles(num_counting)
        circuit = QuantumCircuit(num_counting + 1)
        for control, angle in enumerate(angles):
            circuit.cx(control, num_counting)
            circuit.ry(angle, num_counting)
            circuit.cx(control, num_counting)
        return circuit

    def __repr__(self) -> str:
        return f"QPEPhaseFilter(kappa={self._kappa})"
