"""Standard library: bitstring and integer utilities (Phase 117).

Bitstring conventions match the rest of the SDK: measurement outcomes are
MSB-first strings (``"101"``) whose unsigned-binary integer value is
``int(bitstring, 2)``.  Every helper here adopts the same left-to-right,
most-significant-bit-first order, so ``int_to_bits`` / ``bits_to_int`` and
``int_to_bitstring`` / ``bitstring_to_int`` are exact inverses of one
another.

These are dependency-light, application-independent building blocks intended
for user programs, the runtime and the compiler alike.
"""

from __future__ import annotations

from typing import Iterable


def _validate_int(value: int, *, name: str) -> int:
    """Reject bools and non-ints; return a non-negative integer as-is."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_bitstring(bitstring: str, *, name: str) -> None:
    if not isinstance(bitstring, str):
        raise TypeError(f"{name} must be a bitstring str, got {type(bitstring).__name__}")
    if not bitstring:
        raise ValueError(f"{name} must not be empty")
    if any(char not in "01" for char in bitstring):
        raise ValueError(
            f"{name} must contain only '0'/'1' characters, got {bitstring!r}"
        )


def _validate_width(value: int, width: int, *, name: str) -> None:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{name} must be an int, got {type(width).__name__}")
    if width < 1:
        raise ValueError(f"{name} must be >= 1, got {width}")
    if value.bit_length() > width:
        raise ValueError(
            f"{value} needs {value.bit_length()} bits, but {name} is only {width}"
        )


def int_to_bitstring(value: int, width: int | None = None) -> str:
    """Convert a non-negative integer to an MSB-first binary bitstring.

    Args:
        value: Non-negative integer to convert.
        width: Optional fixed width (zero-padded).  Must be >= 1 and large
            enough to represent ``value``.

    Returns:
        A ``0``/``1`` string with no leading zeros (unless ``width`` pads it).

    Raises:
        TypeError: If ``value``/``width`` is not an ``int``.
        ValueError: If ``value`` is negative, or ``width`` is too small.
    """
    value = _validate_int(value, name="value")
    bits = format(value, "b")
    if width is not None:
        _validate_width(value, width, name="width")
        bits = bits.zfill(width)
    return bits


def bitstring_to_int(bitstring: str) -> int:
    """Convert an MSB-first bitstring to its unsigned integer value.

    Args:
        bitstring: Non-empty string of ``0``/``1`` characters.

    Returns:
        The integer value (MSB first), i.e. ``int(bitstring, 2)``.

    Raises:
        TypeError: If ``bitstring`` is not a ``str``.
        ValueError: If ``bitstring`` is empty or contains non-binary chars.
    """
    _validate_bitstring(bitstring, name="bitstring")
    return int(bitstring, 2)


def int_to_bits(value: int, width: int | None = None) -> tuple[int, ...]:
    """Convert a non-negative integer to an MSB-first tuple of bits.

    Args:
        value: Non-negative integer to convert.
        width: Optional fixed length (most-significant side zero-padded).
            Must be >= 1 and large enough to represent ``value``.

    Returns:
        A tuple of ``0``/``1`` entries in MSB-first order (no leading zeros
        unless ``width`` pads them).

    Raises:
        TypeError: If ``value``/``width`` is not an ``int``.
        ValueError: If ``value`` is negative, or ``width`` is too small.
    """
    return tuple(int(char) for char in int_to_bitstring(value, width=width))


def bits_to_int(bits: Iterable[int]) -> int:
    """Convert an MSB-first iterable of bits to its integer value.

    Args:
        bits: A non-empty iterable of integer bits (``0`` or ``1``), read in
            MSB-first order.

    Returns:
        The unsigned integer value the bits represent.

    Raises:
        TypeError: If any element is not an ``int``.
        ValueError: If ``bits`` is empty or contains values other than 0/1.
    """
    result = 0
    seen = False
    for bit in bits:
        seen = True
        if isinstance(bit, bool) or not isinstance(bit, int):
            raise TypeError(
                f"bits must contain integers, got {type(bit).__name__}"
            )
        if bit not in (0, 1):
            raise ValueError(f"bits must contain only 0/1, got {bit}")
        result = (result << 1) | bit
    if not seen:
        raise ValueError("bits must not be empty")
    return result


def hamming_weight(value: int | str) -> int:
    """Number of set bits in an integer or a ``0``/``1`` bitstring.

    Args:
        value: Either a non-negative integer (its set bits counted) or an
            MSB-first bitstring (its ``'1'`` characters counted).

    Returns:
        The Hamming weight.

    Raises:
        TypeError: If ``value`` is neither an ``int`` nor a ``str``.
        ValueError: If the integer is negative, or the bitstring is empty or
            non-binary.
    """
    if isinstance(value, str):
        _validate_bitstring(value, name="value")
        return value.count("1")
    value = _validate_int(value, name="value")
    return value.bit_count()


def hamming_distance(left: int | str, right: int | str) -> int:
    """Number of positions where two integers or bitstrings differ.

    Args:
        left: Non-negative integer or MSB-first bitstring.
        right: Non-negative integer or MSB-first bitstring of the same kind.
            Two bitstrings must have equal length.

    Returns:
        The Hamming distance.

    Raises:
        TypeError: If the arguments have different kinds, or either is not an
            ``int``/``str``.
        ValueError: If an integer is negative, or a bitstring is empty,
            non-binary, or the two bitstrings have different lengths.
    """
    if isinstance(left, str) or isinstance(right, str):
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError(
                "hamming_distance requires both arguments to be ints or both "
                "to be bitstrings"
            )
        _validate_bitstring(left, name="left")
        _validate_bitstring(right, name="right")
        if len(left) != len(right):
            raise ValueError(
                f"bitstrings must have equal length, got {len(left)} and {len(right)}"
            )
        return sum(a != b for a, b in zip(left, right, strict=True))
    left_int = _validate_int(left, name="left")
    right_int = _validate_int(right, name="right")
    return (left_int ^ right_int).bit_count()


__all__ = [
    "bits_to_int",
    "bitstring_to_int",
    "hamming_distance",
    "hamming_weight",
    "int_to_bits",
    "int_to_bitstring",
]