"""System Standard Library (Phase 117).

Foundational, reusable building blocks for MicroQuantum programs, the runtime
and the compiler:

* :mod:`microquantum.stdlib.bits` — MSB-first bitstring/integer conversions
  and Hamming distances.
* :mod:`microquantum.stdlib.numbers` — angle normalization and
  modulo-``2*pi`` angle comparisons for rotations.
* :mod:`microquantum.stdlib.states` — common state-vector factories built on
  :class:`microquantum.StateVector`.

The standard library is a stable public layer: it stays dependency-light,
depends only on :mod:`microquantum.core` where needed, adds no application
behaviour, and never duplicates APIs defined elsewhere in the SDK.
"""

from .bits import (
    bits_to_int,
    bitstring_to_int,
    gray_code,
    hamming_distance,
    hamming_weight,
    int_to_bits,
    int_to_bitstring,
)
from .numbers import is_angle_close, is_identity_angle, mod_2pi, wrap_angle
from .states import (
    basis_state,
    bell_state,
    dicke_state,
    ghz_state,
    graph_state,
    uniform_superposition,
    w_state,
)

__all__ = [
    "bits_to_int",
    "bitstring_to_int",
    "gray_code",
    "hamming_distance",
    "hamming_weight",
    "int_to_bits",
    "int_to_bitstring",
    "is_angle_close",
    "is_identity_angle",
    "mod_2pi",
    "wrap_angle",
    "basis_state",
    "bell_state",
    "dicke_state",
    "ghz_state",
    "graph_state",
    "uniform_superposition",
    "w_state",
]