"""High-level circuit executor with shot noise and batch support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._json import json_safe, json_string
from ..core.circuit import QuantumCircuit, _narrow_concrete
from ..core.measurement import sample_state
from ..core.operators import Operator
from ..core.state import StateVector
from .base import Backend, _normalize_shots
from .noise import NoiseModel


@dataclass
class ExecutorResult:
    """Result container returned by :class:`Executor`.

    Attributes:
        counts: Measured bitstring counts.
        probabilities: Probability distribution derived from counts.
        shots: Number of measurement shots.
        num_qubits: Number of qubits in the circuit.
        statevector: Final state vector when available (no noise, no backend).
        metadata: Backend name, noise model info, seed, etc.
    """

    counts: dict[str, int]
    probabilities: dict[str, float]
    shots: Optional[int]
    num_qubits: int
    statevector: Optional[NDArray[np.complex128]] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def most_frequent(self) -> str:
        """Return the bitstring with the highest count."""
        if not self.counts:
            return ""
        return max(self.counts, key=lambda k: self.counts[k])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Complex-valued ``statevector`` entries are encoded element-wise
        as ``{"real": ..., "imag": ...}`` pairs.
        """
        return {
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "shots": int(self.shots) if self.shots is not None else None,
            "num_qubits": int(self.num_qubits),
            "statevector": json_safe(self.statevector),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def expectation(self, observable: Operator) -> float:
        """Compute the expectation value of an observable from probabilities.

        Args:
            observable: An :class:`Operator` whose eigenvalues correspond to
                computational-basis measurement outcomes.  The operator must
                be diagonal in the computational basis (e.g. a Pauli-Z string
                or any diagonal observable).  For a full matrix the trace
                ``sum_i p_i * <i|O|i>`` is computed.

        Returns:
            Real expectation value ``<O> = sum_i p_i * o_i``.
        """
        from ..core.operators import Operator

        if not isinstance(observable, Operator):
            raise TypeError("observable must be an Operator instance")

        n = self.num_qubits
        if observable.num_qubits != n:
            raise ValueError(
                f"Observable acts on {observable.num_qubits} qubit(s) "
                f"but result has {n} qubit(s)"
            )

        mat = observable.matrix
        dim = mat.shape[0]
        diag = np.real(np.diag(mat))

        expectation = 0.0
        for bitstring, prob in self.probabilities.items():
            idx = int(bitstring, 2)
            if idx < dim:
                expectation += prob * diag[idx]
        return float(expectation)

    def fidelity(self, state: StateVector) -> float:
        """Compute fidelity with a reference state vector.

        Requires the statevector to be available (noiseless, no backend).

        Args:
            state: Reference state vector to compare against.

        Returns:
            Fidelity in [0, 1].

        Raises:
            ValueError: If no statevector is stored in this result.
        """
        if self.statevector is None:
            raise ValueError(
                "Statevector not available in this result. "
                "Fidelity can only be computed for noiseless "
                "simulations without a backend."
            )
        sv = StateVector(num_qubits=self.num_qubits, amplitudes=self.statevector)
        return sv.fidelity(state)

    def __repr__(self) -> str:
        shots = "deterministic" if self.shots is None else f"shots={self.shots}"
        return (
            f"ExecutorResult(num_qubits={self.num_qubits}, "
            f"{shots}, "
            f"outcomes={len(self.counts)})"
        )

    def __str__(self) -> str:
        shots = "deterministic" if self.shots is None else f"{self.shots} shots"
        lines = [
            f"ExecutorResult ({self.num_qubits} qubits, {shots})",
        ]
        for bitstring in sorted(self.counts):
            lines.append(f"  |{bitstring}>: {self.counts[bitstring]}")
        return "\n".join(lines)


class Executor:
    """Orchestrates quantum circuit execution with optional noise.

    The executor handles the full execution pipeline: circuit validation,
    optional backend delegation, noise model application, and shot-based
    sampling.  When no backend is provided it falls back to native state
    vector simulation via :mod:`microquantum.core.engine`.

    Args:
        backend: Optional backend for execution delegation.
        noise_model: Optional noise model applied after each gate when
            using internal density-matrix simulation.  Providing both a
            backend and a noise model is an error — the noise model would
            otherwise be silently ignored.

    Raises:
        ValueError: If both ``backend`` and ``noise_model`` are provided.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> None:
        if backend is not None and noise_model is not None:
            raise ValueError(
                "Executor accepts either a backend or a noise_model, not both; "
                "a noise model would be silently ignored by backend delegation."
            )
        if noise_model is not None and not isinstance(noise_model, NoiseModel):
            raise TypeError(
                "noise_model must be a NoiseModel or None, "
                f"got {type(noise_model).__name__}"
            )
        self._backend = backend
        self._noise_model = noise_model
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = 1024,
        seed: Optional[int] = None,
    ) -> ExecutorResult:
        """Execute a single circuit.

        Args:
            circuit: The quantum circuit to execute (must be bound).
            shots: Number of measurement shots.  ``None`` requests a
                deterministic (un-sampled) run with no counts.
            seed: RNG seed.  Falls back to the instance seed.

        Returns:
            :class:`ExecutorResult` with counts, probabilities, and metadata.
        """
        effective_seed = seed if seed is not None else self._seed
        _normalize_shots(shots)

        if self._backend is not None:
            return self._run_via_backend(circuit, shots, effective_seed)

        if self._noise_model is not None:
            return self._run_with_noise(circuit, shots, effective_seed)

        return self._run_statevector(circuit, shots, effective_seed)

    def run_batch(
        self,
        circuits: list[QuantumCircuit],
        shots: Optional[int] = 1024,
        seed: Optional[int] = None,
    ) -> list[ExecutorResult]:
        """Execute a list of circuits independently.

        Args:
            circuits: List of quantum circuits.
            shots: Number of measurement shots per circuit.  ``None``
                requests a deterministic (un-sampled) run per circuit.
            seed: Base RNG seed.  Each circuit receives an offset seed
                to keep results independent.

        Returns:
            List of :class:`ExecutorResult`, one per circuit.
        """
        effective_seed = seed if seed is not None else self._seed
        results: list[ExecutorResult] = []
        for idx, circuit in enumerate(circuits):
            sub_seed = effective_seed + idx if effective_seed is not None else None
            results.append(self.run(circuit, shots=shots, seed=sub_seed))
        return results

    def run_and_average(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1024,
    ) -> ExecutorResult:
        """Run multiple circuits and average their probability distributions.

        This is useful for techniques like Zero-Noise Extrapolation (ZNE)
        where several noise-scaled circuits are executed and their results
        are combined.

        Args:
            circuits: List of quantum circuits to average over.
            shots: Total shots *per circuit* (not divided among circuits).

        Returns:
            Averaged :class:`ExecutorResult` with combined counts.
        """
        if not circuits:
            raise ValueError("circuits list must not be empty")

        merged_counts: dict[str, int] = {}
        total_shots = 0
        last_num_qubits = 0
        sv_accum: Optional[NDArray[np.complex128]] = None
        has_sv = True

        for circuit in circuits:
            result = self.run(circuit, shots=shots, seed=None)
            last_num_qubits = result.num_qubits
            if result.statevector is None:
                has_sv = False

            for bs, count in result.counts.items():
                merged_counts[bs] = merged_counts.get(bs, 0) + count
            total_shots += result.shots or 0

            if has_sv and result.statevector is not None:
                if sv_accum is None:
                    sv_accum = result.statevector.copy()
                else:
                    sv_accum = sv_accum + result.statevector

        probabilities = {
            k: v / total_shots for k, v in merged_counts.items()
        } if total_shots > 0 else {}

        avg_sv: Optional[NDArray[np.complex128]] = None
        if has_sv and sv_accum is not None:
            avg_sv = sv_accum / len(circuits)

        return ExecutorResult(
            counts=merged_counts,
            probabilities=probabilities,
            shots=total_shots,
            num_qubits=last_num_qubits,
            statevector=avg_sv,
            metadata={
                "backend": self._backend.name if self._backend else "statevector",
                "num_circuits": len(circuits),
                "averaged": True,
            },
        )

    # ------------------------------------------------------------------
    # Internal execution strategies
    # ------------------------------------------------------------------

    def _run_statevector(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int],
        seed: Optional[int],
    ) -> ExecutorResult:
        """Execute via native state-vector simulation."""
        state = circuit.run()

        if shots is None:
            counts: dict[str, int] = {}
            probabilities: dict[str, float] = {}
            result_shots = None
        else:
            measurement = sample_state(state, shots=shots, seed=seed)
            counts = measurement.counts
            probabilities = measurement.get_probabilities()
            result_shots = measurement.shots

        return ExecutorResult(
            counts=counts,
            probabilities=probabilities,
            shots=result_shots,
            num_qubits=circuit.num_qubits,
            statevector=state.amplitudes.copy(),
            metadata={
                "backend": "statevector",
                "seed": seed,
                "deterministic": shots is None,
            },
        )

    def _run_with_noise(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int],
        seed: Optional[int],
    ) -> ExecutorResult:
        """Execute via density-matrix simulation with noise after each gate."""
        from ..core.density_matrix import DensityMatrix

        circuit._ensure_bound()
        n = circuit.num_qubits

        rho = DensityMatrix(n)

        from ..core.circuit import _narrow_concrete

        for instr in circuit._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(instr):
                continue
            c_instr = _narrow_concrete(instr)
            op = c_instr[0]
            targets = list(c_instr[1])
            full_gate = self._expand_gate(op.matrix, targets, n)
            rho = rho.apply_unitary(full_gate)
            assert self._noise_model is not None
            rho = self._noise_model.apply(rho)

        probs = np.real(np.diag(rho.matrix))
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total > 0:
            probs = probs / total

        if shots is None:
            counts: dict[str, int] = {}
            probabilities: dict[str, float] = {
                format(i, f"0{n}b"): float(probs[i]) for i in range(rho.dim)
            }
            probabilities = {k: v for k, v in probabilities.items() if v > 0}
            result_shots: Optional[int] = None
        else:
            rng = np.random.default_rng(seed)
            outcomes = rng.choice(rho.dim, size=shots, p=probs)

            counts = {}
            for outcome in outcomes:
                bs = format(int(outcome), f"0{n}b")
                counts[bs] = counts.get(bs, 0) + 1

            probabilities = {k: v / shots for k, v in counts.items()}
            result_shots = shots

        return ExecutorResult(
            counts=counts,
            probabilities=probabilities,
            shots=result_shots,
            num_qubits=n,
            statevector=None,
            metadata={
                "backend": "density_matrix",
                "noise_model": str(self._noise_model),
                "seed": seed,
                "deterministic": shots is None,
            },
        )

    def _run_via_backend(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int],
        seed: Optional[int],
    ) -> ExecutorResult:
        """Delegate execution to a Backend instance."""
        circuit._ensure_bound()

        # Real hardware needs the circuit (gate names), not raw matrices.
        from ..providers.backend import HardwareBackend

        if isinstance(self._backend, HardwareBackend):
            if shots is None:
                raise ValueError(
                    "Hardware backends require an explicit positive shots "
                    "count; deterministic (shots=None) execution is only "
                    "supported for simulators."
                )
            backend_result = self._backend.run(circuit, shots=shots)
            probabilities = backend_result.probabilities
            return ExecutorResult(
                counts=backend_result.counts,
                probabilities=probabilities,
                shots=shots,
                num_qubits=circuit.num_qubits,
                statevector=None,
                metadata={
                    "backend": backend_result.backend_name,
                    **backend_result.metadata,
                },
            )

        gates = []
        for instr in circuit._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(instr):
                continue
            op, targets = _narrow_concrete(instr)
            gates.append((op.matrix, list(targets)))

        assert self._backend is not None
        backend_result = self._backend.run_circuit(
            num_qubits=circuit.num_qubits,
            gates=gates,
            shots=shots,
            seed=seed,
        )
        probabilities = {} if shots is None else backend_result.probabilities

        sv: Optional[NDArray[np.complex128]] = (
            backend_result.statevector
            if backend_result.statevector is not None
            else None
        )

        return ExecutorResult(
            counts=backend_result.counts,
            probabilities=probabilities,
            shots=backend_result.shots,
            num_qubits=circuit.num_qubits,
            statevector=sv,
            metadata={
                "backend": backend_result.backend_name,
                "seed": seed,
                **backend_result.metadata,
            },
        )

    @staticmethod
    def _expand_gate(
        gate_matrix: NDArray[np.complex128],
        targets: list[int],
        num_qubits: int,
    ) -> NDArray[np.complex128]:
        """Expand a gate to act on the full Hilbert space."""
        from ..core.operators import Operator
        from ..core.tensor import expand_operator

        op = Operator(np.asarray(gate_matrix, dtype=np.complex128))
        return expand_operator(op, targets, num_qubits).matrix

    def __repr__(self) -> str:
        backend_name = self._backend.name if self._backend else "statevector"
        noise_info = f", noise={self._noise_model}" if self._noise_model else ""
        return f"Executor(backend='{backend_name}'{noise_info})"
