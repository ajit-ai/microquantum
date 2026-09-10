"""Domain adapter framework for microquantum.

Provides the abstraction layer between domain-specific physics problems
and quantum computing backends. Each adapter encodes a physical problem
into quantum circuits and decodes results back to domain language.
"""

from .base import DomainAdapter, QuantumProblem, QuantumResult
from .caching import ResultCache

__all__ = [
    "DomainAdapter",
    "QuantumProblem",
    "QuantumResult",
    "ResultCache",
]
