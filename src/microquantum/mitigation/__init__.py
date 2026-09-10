"""Error mitigation techniques for NISQ quantum computing.

Provides post-processing methods to reduce the effect of noise in
quantum computations without requiring full quantum error correction.

Techniques:
- ZeroNoiseExtrapolation: Extrapolate to zero noise from scaled results
- ProbabilisticErrorCancellation: Invert noise via quasi-probability
- MeasurementErrorMitigation: Correct readout errors via calibration
"""
from .mem import MeasurementErrorMitigation, MitigationMatrix
from .pec import ProbabilisticErrorCancellation
from .zne import ExtrapolationResult, ZeroNoiseExtrapolation

__all__ = [
    "ZeroNoiseExtrapolation",
    "ExtrapolationResult",
    "ProbabilisticErrorCancellation",
    "MeasurementErrorMitigation",
    "MitigationMatrix",
]
