"""Quantum Error Correction codes.

Implements basic QEC codes with encoding, syndrome measurement,
and error correction/detection capabilities.

Supported codes:
- RepetitionCode: classical repetition over quantum channels
- BitFlipCode: 3-qubit bit-flip correction
- PhaseFlipCode: 3-qubit phase-flip correction
- ShorCode: 9-qubit code (concatenated bit+phase flip)
"""

from .repetition import BitFlipCode, PhaseFlipCode, RepetitionCode
from .shor import ShorCode

__all__ = [
    "RepetitionCode",
    "BitFlipCode",
    "PhaseFlipCode",
    "ShorCode",
]
