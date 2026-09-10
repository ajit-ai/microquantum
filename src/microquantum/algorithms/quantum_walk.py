"""Quantum walk on graphs.

Implements discrete-time and continuous-time quantum walks on graphs.
Quantum walks are the quantum analogue of classical random walks and
provide quadratic speedups for graph search, element distinctness,
and detecting graph properties.

Discrete-time walk uses a coin operator (Grover diffusion) and a
conditional shift operator. The walk operator W = S · (C ⊗ I) is
applied repeatedly for a specified number of steps.

Continuous-time walk uses the adjacency matrix as the Hamiltonian
and simulates e^{-iHt} via first-order Trotter-Suzuki decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.operators import Operator


@dataclass
class QuantumWalkResult(JSONSerializable):
    """Result from a quantum walk execution.

    Attributes:
        circuit: The quantum circuit used for the walk.
        num_nodes: Number of nodes in the graph.
        num_steps: Number of walk steps (or Trotter steps for continuous).
        probabilities: Mapping from node index to measurement probability.
        most_likely_node: Node with the highest probability.
    """

    circuit: QuantumCircuit
    num_nodes: int
    num_steps: int
    probabilities: dict[int, float]
    most_likely_node: int


class DiscreteQuantumWalk:
    """Discrete-time quantum walk on an undirected graph.

    Uses a Grover-like coin operator on the coin register and a
    conditional shift operator that moves the walker to neighboring
    nodes. The walk starts at node 0 with the coin in uniform
    superposition.

    The circuit uses ceil(log2(num_nodes)) coin qubits and
    ceil(log2(num_nodes)) position qubits. The walk operator is
    constructed as a matrix and applied directly via the circuit.

    Args:
        adjacency: Adjacency list. adjacency[i] lists neighbors of node i.
        num_steps: Number of walk steps. If None, uses floor(pi/4 * sqrt(N)).

    Raises:
        ValueError: If adjacency is empty, has inconsistent edges,
            or num_steps < 1.
    """

    def __init__(
        self,
        adjacency: list[list[int]],
        num_steps: int | None = None,
    ) -> None:
        if not adjacency:
            raise ValueError("Adjacency list must be non-empty")
        for i, neighbors in enumerate(adjacency):
            for j in neighbors:
                if j < 0 or j >= len(adjacency):
                    raise ValueError(
                        f"Neighbor {j} of node {i} is out of range "
                        f"(valid: 0..{len(adjacency) - 1})"
                    )
                if i not in adjacency[j]:
                    raise ValueError(
                        f"Graph must be undirected: {j} is in "
                        f"adjacency[{i}] but {i} is not in adjacency[{j}]"
                    )

        self._adjacency = [list(n) for n in adjacency]
        self._num_nodes = len(adjacency)

        if num_steps is None:
            self._num_steps = max(
                1, int(math.floor(math.pi / 4 * math.sqrt(self._num_nodes)))
            )
        else:
            if num_steps < 1:
                raise ValueError(f"num_steps must be >= 1, got {num_steps}")
            self._num_steps = num_steps

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes

    @property
    def adjacency(self) -> list[list[int]]:
        """Adjacency list of the graph."""
        return [list(n) for n in self._adjacency]

    @property
    def num_steps(self) -> int:
        """Number of walk steps."""
        return self._num_steps

    @property
    def _coin_qubits(self) -> int:
        """Number of qubits needed for the coin/position register."""
        return max(1, math.ceil(math.log2(self._num_nodes)))

    def _build_coin_operator(self) -> NDArray[np.complex128]:
        """Grover diffusion operator on the coin register.

        D = 2|s><s| - I where |s> is uniform superposition.
        """
        coin_dim = 2 ** self._coin_qubits
        s = np.ones(coin_dim, dtype=np.complex128) / np.sqrt(coin_dim)
        result = 2.0 * np.outer(s, s) - np.eye(coin_dim, dtype=np.complex128)
        return np.asarray(result, dtype=np.complex128)

    def _build_shift_operator(self) -> NDArray[np.complex128]:
        """Conditional shift operator.

        Maps |coin, node> to |coin', neighbor(node, coin)> where
        coin' = 1 - coin (coin flip after shift).
        """
        coin_q = self._coin_qubits
        coin_dim = 2 ** coin_q
        total_dim = 2 ** (2 * coin_q)

        shift = np.zeros((total_dim, total_dim), dtype=np.complex128)

        for node in range(self._num_nodes):
            deg = len(self._adjacency[node])
            if deg == 0:
                continue
            for c in range(coin_dim):
                src = (c << coin_q) | node
                neighbor = self._adjacency[node][c % deg]
                dst = (c << coin_q) | neighbor
                shift[dst, src] = 1.0

        for node in range(self._num_nodes, coin_dim):
            for c in range(coin_dim):
                idx = (c << coin_q) | node
                shift[idx, idx] = 1.0

        return shift

    def build_circuit(self) -> QuantumCircuit:
        """Build the quantum walk circuit.

        Layout: coin register (qubits 0..coin_q-1), position register
        (qubits coin_q..2*coin_q-1). The walk operator is applied
        num_steps times to the full coin+position register.

        Returns:
            QuantumCircuit implementing the discrete quantum walk.
        """
        coin_q = self._coin_qubits
        total_qubits = 2 * coin_q

        coin_op = self._build_coin_operator()
        shift_op = self._build_shift_operator()

        pos_dim = 2 ** coin_q
        full_coin = np.kron(coin_op, np.eye(pos_dim, dtype=np.complex128))
        walk_matrix = shift_op @ full_coin
        walk_power = np.linalg.matrix_power(walk_matrix, self._num_steps)
        walk_op = Operator(np.asarray(walk_power, dtype=np.complex128))

        qc = QuantumCircuit(total_qubits)
        target_qubits = list(range(total_qubits))

        for q in range(coin_q):
            qc.h(q)

        for _ in range(self._num_steps):
            qc.append(walk_op, target_qubits)

        return qc

    def run(self) -> QuantumWalkResult:
        """Execute the discrete quantum walk.

        Returns:
            QuantumWalkResult with the walk outcome.
        """
        qc = self.build_circuit()
        state = qc.run()

        coin_q = self._coin_qubits
        probs: dict[int, float] = {}
        amps = state.amplitudes

        for idx in range(len(amps)):
            node = idx & ((1 << coin_q) - 1)
            if node < self._num_nodes:
                probs[node] = probs.get(node, 0.0) + float(
                    abs(amps[idx]) ** 2
                )

        most_likely = max(probs, key=probs.get)  # type: ignore[arg-type]

        return QuantumWalkResult(
            circuit=qc,
            num_nodes=self._num_nodes,
            num_steps=self._num_steps,
            probabilities=probs,
            most_likely_node=most_likely,
        )

    def __repr__(self) -> str:
        return (
            f"DiscreteQuantumWalk(nodes={self._num_nodes}, "
            f"steps={self._num_steps}, "
            f"qubits={2 * self._coin_qubits})"
        )


class ContinuousQuantumWalk:
    """Continuous-time quantum walk on an undirected graph.

    Simulates the Schrodinger equation with Hamiltonian H equal to
    the adjacency matrix. Uses first-order Trotter-Suzuki decomposition
    to approximate e^{-iHt} as a product of two-qubit gates, one
    per edge.

    The circuit uses ceil(log2(num_nodes)) position qubits. Each
    edge (u,v) contributes a term exp(-i*t*(|u><v| + |v><u|))
    which is implemented as a conditional phase gate.

    Args:
        adjacency: Adjacency list. adjacency[i] lists neighbors of node i.
        evolution_time: Evolution time t. If None, uses
            pi / (2 * sqrt(max_degree)) for search.

    Raises:
        ValueError: If adjacency is empty or evolution_time <= 0.
    """

    def __init__(
        self,
        adjacency: list[list[int]],
        evolution_time: float | None = None,
    ) -> None:
        if not adjacency:
            raise ValueError("Adjacency list must be non-empty")

        self._adjacency = [list(n) for n in adjacency]
        self._num_nodes = len(adjacency)

        if evolution_time is None:
            max_deg = max(len(n) for n in adjacency) if adjacency else 1
            self._evolution_time = math.pi / (2.0 * math.sqrt(max(max_deg, 1)))
        else:
            if evolution_time <= 0:
                raise ValueError(
                    f"evolution_time must be > 0, got {evolution_time}"
                )
            self._evolution_time = float(evolution_time)

        self._num_qubits = max(
            1, math.ceil(math.log2(self._num_nodes))
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes

    @property
    def adjacency(self) -> list[list[int]]:
        """Adjacency list of the graph."""
        return [list(n) for n in self._adjacency]

    @property
    def evolution_time(self) -> float:
        """Evolution time for the walk."""
        return self._evolution_time

    def _trotter_step_edge(
        self, u: int, v: int
    ) -> NDArray[np.complex128]:
        """First-order Trotter step for edge (u, v).

        Computes exp(-i*t*(|u><v| + |v><u|)) in the full 2^n space.
        Only entries (u,u), (v,v), (u,v), (v,u) are non-trivial.
        """
        n = self._num_qubits
        dim = 2 ** n
        theta = self._evolution_time

        gate = np.eye(dim, dtype=np.complex128)
        gate[u, u] = math.cos(theta)
        gate[v, v] = math.cos(theta)
        gate[u, v] = -1j * math.sin(theta)
        gate[v, u] = -1j * math.sin(theta)

        return gate

    @staticmethod
    def _apply_two_qubit_gate(
        gate: NDArray[np.complex128],
        qubit_i: int,
        qubit_j: int,
        num_qubits: int,
    ) -> NDArray[np.complex128]:
        """Expand a 2-qubit gate to the full Hilbert space.

        Places the gate on qubits (qubit_i, qubit_j) with identity
        on all other qubits. Qubit ordering is MSB-first.
        """
        k = min(qubit_i, qubit_j)
        hi = max(qubit_i, qubit_j)

        n_before = k
        n_between = hi - k - 1
        n_after = num_qubits - hi - 1

        parts: list[NDArray[np.complex128]] = []
        if n_before > 0:
            parts.append(np.eye(2**n_before, dtype=np.complex128))
        parts.append(gate)
        if n_between > 0:
            parts.append(np.eye(2**n_between, dtype=np.complex128))
        if n_after > 0:
            parts.append(np.eye(2**n_after, dtype=np.complex128))

        result = np.asarray(parts[0], dtype=np.complex128)
        for part in parts[1:]:
            result = np.asarray(np.kron(result, part), dtype=np.complex128)

        return result

    def build_circuit(self) -> QuantumCircuit:
        """Build the continuous-time quantum walk circuit.

        Applies Hadamard gates to create uniform superposition, then
        applies the Trotterized time evolution operator as a sequence
        of full-dimension gates (one per edge).

        Returns:
            QuantumCircuit implementing the continuous quantum walk.
        """
        n = self._num_qubits
        qc = QuantumCircuit(n)

        for q in range(n):
            qc.h(q)

        seen: set[tuple[int, int]] = set()
        for u in range(self._num_nodes):
            for v in self._adjacency[u]:
                edge = (min(u, v), max(u, v))
                if edge in seen:
                    continue
                seen.add(edge)

                gate_matrix = self._trotter_step_edge(u, v)
                op = Operator(gate_matrix)
                qc.append(op, list(range(n)))

        return qc

    def run(self) -> QuantumWalkResult:
        """Execute the continuous-time quantum walk.

        Returns:
            QuantumWalkResult with the walk outcome.
        """
        qc = self.build_circuit()
        state = qc.run()

        n = self._num_qubits
        node_mask = (1 << n) - 1
        probs: dict[int, float] = {}
        amps = state.amplitudes

        for idx in range(len(amps)):
            node = idx & node_mask
            if node < self._num_nodes:
                probs[node] = probs.get(node, 0.0) + float(
                    abs(amps[idx]) ** 2
                )

        most_likely = max(probs, key=probs.get)  # type: ignore[arg-type]

        seen_edges: set[tuple[int, int]] = set()
        for u in range(self._num_nodes):
            for v in self._adjacency[u]:
                seen_edges.add((min(u, v), max(u, v)))

        return QuantumWalkResult(
            circuit=qc,
            num_nodes=self._num_nodes,
            num_steps=len(seen_edges),
            probabilities=probs,
            most_likely_node=most_likely,
        )

    def __repr__(self) -> str:
        return (
            f"ContinuousQuantumWalk(nodes={self._num_nodes}, "
            f"time={self._evolution_time:.4f}, "
            f"qubits={self._num_qubits})"
        )
