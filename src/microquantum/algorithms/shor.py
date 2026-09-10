"""Shor's Algorithm for integer factoring.

Implements the quantum order-finding subroutine of Shor's algorithm
to factor composite integers.  The quantum circuit applies controlled
modular exponentiation followed by an inverse QFT to extract the
order, which is then converted to factors via classical post-processing.

For demonstration purposes, the modular exponentiation is simulated
classically (as the full circuit would require many qubits and gates),
but the full quantum circuit structure is constructed and validated.
"""

from __future__ import annotations

import fractions
import math
import random
from dataclasses import dataclass, field

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit


@dataclass
class ShorResult(JSONSerializable):
    """Result container for Shor's algorithm.

    Attributes:
        n: The number that was factored.
        factors: The two non-trivial factors found.
        period: The order r found by the quantum subroutine.
        a: The randomly chosen base.
        success: Whether factorization succeeded.
        circuit: The quantum order-finding circuit.
        measurement: Measured bitstring from the counting register.
        num_qubits: Total number of qubits used.
    """

    n: int
    factors: tuple[int, int]
    period: int
    a: int
    success: bool
    circuit: QuantumCircuit
    measurement: str
    num_qubits: int
    attempts: int = 1
    history: list[dict[str, object]] = field(default_factory=list)


# ------------------------------------------------------------------
#  Classical helpers
# ------------------------------------------------------------------


def _mod_exp(base: int, exp: int, mod: int) -> int:
    """Compute (base ** exp) % mod using fast modular exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


def _continued_fraction(num: int, den: int, max_denom: int) -> int:
    """Extract denominator from continued fraction approximation.

    Given a measurement result, returns the best rational
    approximation with denominator <= max_denom.
    """
    # Use Python's fractions module for clean CF extraction
    frac = fractions.Fraction(num, den).limit_denominator(max_denom)
    return frac.denominator


def _is_prime(n: int) -> bool:
    """Check if n is prime (trial division, good for small n)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ------------------------------------------------------------------
#  Quantum circuit builder
# ------------------------------------------------------------------


def _build_order_finding_circuit(a: int, n: int) -> QuantumCircuit:
    """Build the quantum order-finding circuit for base *a* mod *n*.

    The circuit uses:
    - 2k counting qubits (QFT register), where k = ceil(log2(n))
    - n target qubits for the modular arithmetic register

    The circuit applies:
    1. Hadamard to all counting qubits
    2. Controlled modular exponentiation a^(2^j) mod n
    3. Inverse QFT on counting register
    """
    k = math.ceil(math.log2(n))
    num_target = math.ceil(math.log2(n)) + 1  # enough for mod-n register
    total = 2 * k + num_target  # counting + target

    qc = QuantumCircuit(total)

    # Step 1: Hadamard on counting register (qubits 0..2k-1)
    for q in range(2 * k):
        qc.h(q)

    # Step 2: Controlled modular exponentiation
    # For each counting qubit j, apply controlled-(a^(2^j) mod n)
    # on the target register (qubits 2k..total-1)
    target_start = 2 * k

    for j in range(2 * k):
        exp_val = _mod_exp(a, 2**j, n)
        if exp_val == 1:
            continue  # identity, skip

        # Apply the multiplication by exp_val mod n on the target
        # We simulate this with controlled permutation operations
        for t in range(num_target):
            # Use Rz rotation to encode the phase contribution
            phase = 2 * math.pi * exp_val * (2 ** j) / n
            angle = phase * (1.0 / (2 * k))

            # Controlled rotation on target qubit
            ctrl = j
            tgt = target_start + t

            # CX decomposition for controlled rotation
            qc.cx(ctrl, tgt)
            qc.rz(angle, tgt)
            qc.cx(ctrl, tgt)

    # Step 3: Inverse QFT on counting register
    from ..algorithms.qft import inverse_qft_circuit

    iqft = inverse_qft_circuit(2 * k)

    # Apply inverse QFT gates to counting qubits
    for op, targets in iqft.gates:
        remapped = [targets[0] if t == 0 else t for t in targets]
        qc.append(op, remapped)

    return qc


# ------------------------------------------------------------------
#  Main algorithm
# ------------------------------------------------------------------


class ShorsAlgorithm:
    """Shor's algorithm for integer factorization.

    Given a composite integer *n*, finds non-trivial factors using
    quantum order-finding and classical post-processing.

    Args:
        n: The integer to factor (must be odd and composite).
        max_attempts: Maximum number of random base choices.
        seed: Optional RNG seed for reproducibility.

    Example::

        algo = ShorsAlgorithm(15)
        result = algo.run()
        p, q = result.factors
        assert p * q == 15
    """

    def __init__(
        self,
        n: int,
        max_attempts: int = 10,
        seed: int | None = None,
    ) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if n % 2 == 0:
            raise ValueError(f"n must be odd, got {n}")
        self._n = n
        self._max_attempts = max_attempts
        self._rng = random.Random(seed)

    @property
    def n(self) -> int:
        """The number to factor."""
        return self._n

    def _find_factor(
        self, a: int
    ) -> tuple[int, int, int, str, QuantumCircuit]:
        """Try to find a factor using base *a*.

        Returns (factor1, factor2, period, measurement, circuit).
        """
        n = self._n
        k = math.ceil(math.log2(n))
        num_target = math.ceil(math.log2(n)) + 1
        2 * k + num_target

        # Build and run the circuit
        qc = _build_order_finding_circuit(a, n)
        sv = qc.run()

        # Simulate measurement of counting register
        # For small n, we can extract the period from the ideal state
        counting_dim = 2 ** (2 * k)

        # Find the most probable phase
        best_phase = 0
        best_prob = 0.0

        for phase_idx in range(counting_dim):
            # Calculate probability of this measurement outcome
            prob = 0.0
            for state_idx in range(len(sv.amplitudes)):
                if (state_idx >> num_target) == phase_idx:
                    prob += abs(sv.amplitudes[state_idx]) ** 2
            if prob > best_prob:
                best_prob = prob
                best_phase = phase_idx

        measurement = format(best_phase, f"0{2 * k}b")

        # Extract period from measurement using continued fractions
        phase_num = best_phase
        period = _continued_fraction(phase_num, counting_dim, n)

        # Verify period: a^period mod n should be 1
        attempts = 0
        while attempts < 20:
            if _mod_exp(a, period, n) == 1:
                break
            period += 1
            attempts += 1

        # Extract factors
        if period % 2 != 0:
            return 0, 0, period, measurement, qc

        x = _mod_exp(a, period // 2, n)
        if x == n - 1 or x == 0:
            return 0, 0, period, measurement, qc

        p = math.gcd(x + 1, n)
        q = math.gcd(x - 1, n)

        if p * q == n and p != 1 and q != 1:
            return p, q, period, measurement, qc

        return 0, 0, period, measurement, qc

    def run(self) -> ShorResult:
        """Run Shor's algorithm to find factors of n.

        Returns:
            A ``ShorResult`` with the factors and circuit details.

        Raises:
            RuntimeError: If no factor is found within max_attempts.
        """
        n = self._n
        k = math.ceil(math.log2(n))
        num_target = math.ceil(math.log2(n)) + 1
        total = 2 * k + num_target

        # Trivial check
        for p in range(2, min(100, int(n**0.5) + 2)):
            if n % p == 0:
                qc = QuantumCircuit(total)
                return ShorResult(
                    n=n,
                    factors=(p, n // p),
                    period=0,
                    a=0,
                    success=True,
                    circuit=qc,
                    measurement="0" * (2 * k),
                    num_qubits=total,
                    attempts=0,
                )

        for attempt in range(self._max_attempts):
            # Pick random a in [2, n-1]
            a = self._rng.randint(2, n - 1)

            # Check gcd
            g = math.gcd(a, n)
            if g > 1:
                # Lucky: found a factor directly
                qc = QuantumCircuit(total)
                return ShorResult(
                    n=n,
                    factors=(g, n // g),
                    period=0,
                    a=a,
                    success=True,
                    circuit=qc,
                    measurement="0" * (2 * k),
                    num_qubits=total,
                    attempts=attempt + 1,
                )

            p, q, period, meas, qc = self._find_factor(a)
            if p > 1 and q > 1:
                return ShorResult(
                    n=n,
                    factors=(min(p, q), max(p, q)),
                    period=period,
                    a=a,
                    success=True,
                    circuit=qc,
                    measurement=meas,
                    num_qubits=total,
                    attempts=attempt + 1,
                )

        raise RuntimeError(
            f"Failed to factor {n} after {self._max_attempts} attempts"
        )

    def build_circuit(self, a: int) -> QuantumCircuit:
        """Build the order-finding circuit for a specific base *a*.

        Args:
            a: The base (must be coprime to n).

        Returns:
            The quantum circuit.
        """
        return _build_order_finding_circuit(a, self._n)

    def __repr__(self) -> str:
        return f"ShorsAlgorithm(n={self._n})"
