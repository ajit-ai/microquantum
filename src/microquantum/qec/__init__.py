"""Quantum Error Correction codes.

Implements basic QEC codes with encoding, syndrome measurement,
and error correction/detection capabilities.

Supported codes:
- RepetitionCode: classical repetition over quantum channels
- BitFlipCode: 3-qubit bit-flip correction
- PhaseFlipCode: 3-qubit phase-flip correction
- ShorCode: 9-qubit code (concatenated bit+phase flip)
- SteaneCode: 7-qubit CSS code (Hamming-based)
"""

from .decoder import Decoder, LookupDecoder, Syndrome
from .repetition import BitFlipCode, PhaseFlipCode, RepetitionCode
from .shor import ShorCode
from .steane import STEANE_X_STABILIZERS, STEANE_Z_STABILIZERS, SteaneCode

__all__ = [
    "RepetitionCode",
    "BitFlipCode",
    "PhaseFlipCode",
    "ShorCode",
    "SteaneCode",
    "STEANE_X_STABILIZERS",
    "STEANE_Z_STABILIZERS",
    "Syndrome",
    "Decoder",
    "LookupDecoder",
]
