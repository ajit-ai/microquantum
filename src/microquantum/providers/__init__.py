"""Real hardware provider clients for microquantum.

Provides raw HTTP clients for IBM Quantum and IonQ plus a Backend adapter
so existing Executor code can target real hardware. Only standard-library
HTTP is used - no Qiskit/Cirq/OpenQASM dependencies.
"""
from .backend import HardwareBackend
from .base import (
    HardwareJob,
    HardwareProvider,
    HardwareStatus,
    ProviderCredentials,
)
from .ibm import IBMQuantumCredentials, IBMQuantumProvider
from .ionq import IonQCredentials, IonQProvider
from .jobs import TERMINAL_STATUSES, PollingJob, ProviderErrorMapper
from .replay import ReplayTransport
from .serializer import CircuitSerializer, UnsupportedGateError

__all__ = [
    "HardwareProvider",
    "HardwareJob",
    "HardwareStatus",
    "ProviderCredentials",
    "PollingJob",
    "ProviderErrorMapper",
    "TERMINAL_STATUSES",
    "ReplayTransport",
    "HardwareBackend",
    "IBMQuantumCredentials",
    "IBMQuantumProvider",
    "IonQCredentials",
    "IonQProvider",
    "CircuitSerializer",
    "UnsupportedGateError",
]