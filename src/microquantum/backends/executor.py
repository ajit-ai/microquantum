"""High-level circuit executor with shot noise and batch support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._json import json_safe, json_string
from ..core.circuit import QuantumCircuit
from ..core.measurement import sample_state
from ..core.operators import Operator
from ..core.state import StateVector
from .base import Backend
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
    shots: int
    num_qubits: int
    statevector: Optional[NDArray[np.complex128]] = None
    metadata: dict = field(default_factory=dict)

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
            "shots": int(self.shots),
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
        return (
            f"ExecutorResult(num_qubits={self.num_qubits}, "
            f"shots={self.shots}, "
            f"outcomes={len(self.counts)})"
        )

    def __str__(self) -> str:
        lines = [
            f"ExecutorResult ({self.num_qubits} qubits, {self.shots} shots)",
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
            using internal density-matrix simulation (ignored when a
            backend is supplied).
        seed: Global RNG seed.  Per-call *seed* arguments take precedence.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._backend = backend
        self._noise_model = noise_model
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> ExecutorResult:
        """Execute a single circuit.

        Args:
            circuit: The quantum circuit to execute (must be bound).
            shots: Number of measurement shots.
            seed: RNG seed.  Falls back to the instance seed.

        Returns:
            :class:`ExecutorResult` with counts, probabilities, and metadata.
        """
        effective_seed = seed if seed is not None else self._seed

        if self._backend is not None:
            return self._run_via_backend(circuit, shots, effective_seed)

        if self._noise_model is not None:
            return self._run_with_noise(circuit, shots, effective_seed)

        return self._run_statevector(circuit, shots, effective_seed)

    def run_batch(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> list[ExecutorResult]:
        """Execute a list of circuits independently.

        Args:
            circuits: List of quantum circuits.
            shots: Number of measurement shots per circuit.
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
            total_shots += result.shots

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
        shots: int,
        seed: Optional[int],
    ) -> ExecutorResult:
        """Execute via native state-vector simulation."""
        state = circuit.run()
        measurement = sample_state(state, shots=shots, seed=seed)

        return ExecutorResult(
            counts=measurement.counts,
            probabilities=measurement.get_probabilities(),
            shots=measurement.shots,
            num_qubits=circuit.num_qubits,
            statevector=state.amplitudes.copy(),
            metadata={
                "backend": "statevector",
                "seed": seed,
            },
        )

    def _run_with_noise(
        self,
        circuit: QuantumCircuit,
        shots: int,
        seed: Optional[int],
    ) -> ExecutorResult:
        """Execute via density-matrix simulation with noise after each gate."""
        from ..core.density_matrix import DensityMatrix

        circuit._ensure_bound()
        n = circuit.num_qubits

        rho = DensityMatrix(n)

        for instr in circuit._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(instr):
                continue
            op = instr[0]  # type: ignore[assignment]
            targets = list(instr[1])  # type: ignore[arg-type]
            full_gate = self._expand_gate(op.matrix, targets, n)  # type: ignore[union-attr]
            rho = rho.apply_unitary(full_gate)
            rho = self._noise_model.apply(rho)  # type: ignore[union-attr]

        probs = np.real(np.diag(rho.matrix))
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total > 0:
            probs = probs / total

        rng = np.random.default_rng(seed)
        outcomes = rng.choice(rho.dim, size=shots, p=probs)

        counts: dict[str, int] = {}
        for outcome in outcomes:
            bs = format(int(outcome), f"0{n}b")
            counts[bs] = counts.get(bs, 0) + 1

        probabilities = {k: v / shots for k, v in counts.items()}

        return ExecutorResult(
            counts=counts,
            probabilities=probabilities,
            shots=shots,
            num_qubits=n,
            statevector=None,
            metadata={
                "backend": "density_matrix",
                "noise_model": str(self._noise_model),
                "seed": seed,
            },
        )

    def _run_via_backend(
        self,
        circuit: QuantumCircuit,
        shots: int,
        seed: Optional[int],
    ) -> ExecutorResult:
        """Delegate execution to a Backend instance."""
        circuit._ensure_bound()

        # Real hardware needs the circuit (gate names), not raw matrices.
        from ..providers.backend import HardwareBackend

        if isinstance(self._backend, HardwareBackend):
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
            gates.append((instr[0].matrix, list(instr[1])))  # type: ignore[union-attr,arg-type]

        backend_result = self._backend.run_circuit(  # type: ignore[union-attr]
            num_qubits=circuit.num_qubits,
            gates=gates,
            shots=shots,
            seed=seed,
        )

        probabilities = backend_result.probabilities
        sv: Optional[NDArray[np.complex128]] = (
            backend_result.statevector
            if backend_result.statevector is not None
            else None
        )

        return ExecutorResult(
            counts=backend_result.counts,
            probabilities=probabilities,
            shots=shots,
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
