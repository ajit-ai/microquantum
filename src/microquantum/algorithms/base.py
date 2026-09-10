"""Algorithm contract.

The built-in public algorithms follow a single convention: instantiate
with the problem definition, then execute it via ``run()`` (a few older
algorithms expose the legacy entry points ``solve()`` / ``estimate()`` or
``compute_minimum_eigenvalue()``), returning a typed ``*Result`` dataclass
that supports ``to_dict()`` / ``to_json()``.

Algorithms build and execute a :class:`~microquantum.core.circuit.QuantumCircuit`.
Where a circuit can be produced independently it is exposed via
``build_circuit()``.  Execution uses the SDK's backends / executor or the
internal state-vector engine; variational algorithms accept a classical
:class:`~microquantum.optimizers.base.Optimizer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Lightweight contract shared by all public quantum algorithms.

    Subclassing is optional — this class documents the convention used by
    the built-in algorithms rather than forcing a uniform implementation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm identifier."""
