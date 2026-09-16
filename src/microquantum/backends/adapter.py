"""Backend adapter boundary for external/private providers (MQ-06).

A :class:`BackendAdapter` is a :class:`Backend` whose execution is
delegated to an external system (a vendor SDK, a REST API, or a private
enterprise engine) hidden behind the adapter.  Vendor-specific types must
never leak through the adapter's public surface: callers only ever see
plans, :class:`BackendResult` and capabilities.

Subclasses implement the adapter bottom: convert a bound
:class:`~microquantum.core.circuit.QuantumCircuit` to the vendor's wire
format, submit, and translate the vendor result back into a
:class:`BackendResult`.  The adapter takes responsibility for mapping every
vendor error into the validation/execution error style used by the SDK.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.parameter import Parameter
from ..core.state import StateVector
from .base import Backend, BackendResult

if TYPE_CHECKING:
    pass


class BackendAdapter(Backend):
    """Base class bridging the SDK backend contract to an external system.

    The base implementation binds plans, validates them, and hands the bound
    circuit to :meth:`submit_to_vendor` + :meth:`collect_from_vendor`; the
    adapter boundary keeps vendor objects inside the subclass.
    """

    @abstractmethod
    def submit_to_vendor(
        self, circuit: QuantumCircuit, *, shots: Optional[int], seed: Optional[int]
    ) -> Any:
        """Serialize a bound circuit to the vendor and submit it.

        Returns:
            A vendor handle (opaque to the SDK — never surfaced publicly).
        """

    @abstractmethod
    def collect_from_vendor(self, vendor_handle: Any, circuit: QuantumCircuit) -> BackendResult:
        """Translate a vendor result into a :class:`BackendResult`.

        Mapping failures must raise ``ValueError`` with a message naming the
        backend and the capability that failed.
        """

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: Optional[int] = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        raise ValueError(
            f"adapter backend '{self.name}' cannot execute raw gate matrices; "
            "execute a bound QuantumCircuit instead (run/execute on a plan)"
        )

    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
        parameter_values: Optional[Mapping[Union[str, Parameter], Union[int, float, complex]]] = None,
    ) -> BackendResult:
        if parameter_values is not None:
            circuit = circuit.bind_parameters(parameter_values)
        if shots is not None:
            handle = self.submit_to_vendor(circuit, shots=shots, seed=seed)
            return self.collect_from_vendor(handle, circuit)
        raise ValueError(
            f"adapter backend '{self.name}' requires an explicit positive "
            "shots count; deterministic (shots=None) execution is only "
            "supported for simulators."
        )


__all__ = ["BackendAdapter"]