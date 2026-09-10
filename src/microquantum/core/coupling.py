"""Coupling maps for quantum device connectivity.

Provides a ``CouplingMap`` class representing the connectivity graph
of a quantum processor, along with factory methods for common
topologies (linear, grid, heavy-hex, all-to-all).
"""

from __future__ import annotations

from collections import deque


class CouplingMap:
    """Undirected graph of qubit connectivity.

    Each edge represents a pair of qubits that can interact directly
    via a two-qubit gate.  When two qubits are not connected, SWAP
    gates must be inserted to route the interaction.

    Args:
        edges: List of ``(i, j)`` qubit pairs that are connected.

    Raises:
        ValueError: If edges reference qubit indices outside the range
            ``[0, num_qubits)``, or if ``num_qubits`` is inconsistent.

    Example::

        cmap = CouplingMap([(0, 1), (1, 2), (2, 3)])
        assert cmap.are_connected(0, 1)
        assert not cmap.are_connected(0, 3)
    """

    def __init__(
        self,
        edges: list[tuple[int, int]],
        num_qubits: int | None = None,
    ) -> None:
        if not edges:
            raise ValueError("CouplingMap requires at least one edge")

        self._edges = [(min(a, b), max(a, b)) for a, b in edges]

        if num_qubits is not None:
            self._num_qubits = num_qubits
        else:
            all_qubits = set()
            for i, j in self._edges:
                all_qubits.add(i)
                all_qubits.add(j)
            self._num_qubits = max(all_qubits) + 1

        # Validate edges
        for i, j in self._edges:
            if i < 0 or i >= self._num_qubits:
                raise ValueError(
                    f"Qubit {i} out of range [0, {self._num_qubits})"
                )
            if j < 0 or j >= self._num_qubits:
                raise ValueError(
                    f"Qubit {j} out of range [0, {self._num_qubits})"
                )

        # Build adjacency
        self._adj: dict[int, set[int]] = {
            i: set() for i in range(self._num_qubits)
        }
        for i, j in self._edges:
            self._adj[i].add(j)
            self._adj[j].add(i)

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the coupling map."""
        return self._num_qubits

    @property
    def edges(self) -> list[tuple[int, int]]:
        """List of connected qubit pairs (sorted)."""
        return list(self._edges)

    @property
    def size(self) -> int:
        """Number of edges."""
        return len(self._edges)

    def neighbors(self, qubit: int) -> set[int]:
        """Return qubits directly connected to *qubit*."""
        return set(self._adj[qubit])

    def degree(self, qubit: int) -> int:
        """Number of connections for *qubit*."""
        return len(self._adj[qubit])

    def are_connected(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if two qubits can interact directly."""
        return qubit_b in self._adj[qubit_a]

    def shortest_path(self, source: int, target: int) -> list[int]:
        """Find shortest path between two qubits via BFS.

        Args:
            source: Starting qubit.
            target: Destination qubit.

        Returns:
            List of qubit indices from source to target (inclusive).
            Returns ``[source]`` if source == target.

        Raises:
            ValueError: If no path exists between the qubits.
        """
        if source == target:
            return [source]

        visited: set[int] = {source}
        queue: deque[tuple[int, list[int]]] = deque()
        queue.append((source, [source]))

        while queue:
            current, path = queue.popleft()
            for neighbor in self._adj[current]:
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        raise ValueError(
            f"No path between qubits {source} and {target}"
        )

    def distance(self, qubit_a: int, qubit_b: int) -> int:
        """Number of SWAPs needed to bring two qubits adjacent.

        Returns the length of the shortest path minus one (the number
        of SWAPs).  Returns 0 if already connected.
        """
        path = self.shortest_path(qubit_a, qubit_b)
        return max(0, len(path) - 2)

    def is_connected_graph(self) -> bool:
        """Check if the coupling map forms a single connected component."""
        if self._num_qubits <= 1:
            return True
        visited: set[int] = set()
        queue: deque[int] = deque()
        queue.append(0)
        visited.add(0)
        while queue:
            current = queue.popleft()
            for neighbor in self._adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == self._num_qubits

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def linear(cls, n: int) -> CouplingMap:
        """Create a linear chain: 0-1-2-...-(n-1).

        Args:
            n: Number of qubits (>= 2).
        """
        if n < 2:
            raise ValueError("Linear coupling requires at least 2 qubits")
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls(edges, num_qubits=n)

    @classmethod
    def grid(cls, rows: int, cols: int) -> CouplingMap:
        """Create a rectangular grid topology.

        Qubit ``(r, c)`` maps to index ``r * cols + c``.

        Args:
            rows: Number of rows (>= 1).
            cols: Number of columns (>= 1).
        """
        if rows < 1 or cols < 1:
            raise ValueError("Grid dimensions must be >= 1")
        n = rows * cols
        edges: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    edges.append((q, q + 1))
                if r + 1 < rows:
                    edges.append((q, q + cols))
        return cls(edges, num_qubits=n)

    @classmethod
    def heavy_hex(cls, rows: int, cols: int) -> CouplingMap:
        """Create a heavy-hex topology similar to IBM Eagle processors.

        This is a simplified heavy-hex with alternating rows of
        different connectivity.

        Args:
            rows: Number of rows (>= 2, must be even for best layout).
            cols: Number of columns (>= 2).
        """
        if rows < 2 or cols < 2:
            raise ValueError("Heavy-hex requires rows >= 2 and cols >= 2")
        edges: list[tuple[int, int]] = []
        n = rows * cols

        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    edges.append((q, q + 1))
                if r + 1 < rows:
                    edges.append((q, q + cols))

        return cls(edges, num_qubits=n)

    @classmethod
    def all_to_all(cls, n: int) -> CouplingMap:
        """Create a fully-connected topology.

        Every qubit pair is directly connected.

        Args:
            n: Number of qubits (>= 2).
        """
        if n < 2:
            raise ValueError(
                "All-to-all coupling requires at least 2 qubits"
            )
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(edges, num_qubits=n)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_adjacency_list(self) -> list[list[int]]:
        """Return adjacency list representation."""
        return [sorted(self._adj[i]) for i in range(self._num_qubits)]

    @classmethod
    def from_adjacency_list(cls, adj: list[list[int]]) -> CouplingMap:
        """Create from an adjacency list.

        Args:
            adj: ``adj[i]`` lists neighbors of node i.
        """
        edges: list[tuple[int, int]] = []
        for i, neighbors in enumerate(adj):
            for j in neighbors:
                if j > i:
                    edges.append((i, j))
        return cls(edges, num_qubits=len(adj))

    def __repr__(self) -> str:
        return (
            f"CouplingMap(num_qubits={self._num_qubits}, "
            f"size={self.size})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CouplingMap):
            return NotImplemented
        return (
            self._num_qubits == other._num_qubits
            and self._edges == other._edges
        )

    def __hash__(self) -> int:
        return hash((self._num_qubits, tuple(self._edges)))
