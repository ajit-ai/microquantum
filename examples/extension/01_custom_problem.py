"""Defining a custom Problem subclass (MQ-05).

Problems are JSON-safe data containers; subclassing adds domain-specific
fields and validation while keeping the generic `validate` /
`to_dict` / `to_json` contract.
"""

from microquantum import Problem


class MaxCutProblem(Problem):
    """A graph problem: partition vertices to maximize cut edges."""

    def __init__(self, edges: list[tuple[int, int]], num_vertices: int, name: str = "maxcut") -> None:
        super().__init__(name=name, num_qubits=num_vertices)
        self.edges = list(edges)
        self.num_vertices = num_vertices

    def validate(self) -> list[str]:
        issues = super().validate()
        for a, b in self.edges:
            if a >= self.num_vertices or b >= self.num_vertices:
                issues.append(f"edge ({a},{b}) references a missing vertex")
        if not self.edges:
            issues.append("maxcut needs at least one edge")
        return issues

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["edges"] = [list(e) for e in self.edges]
        data["num_vertices"] = self.num_vertices
        return data


def main() -> None:
    print("=== Custom problem subclass ===\n")

    good = MaxCutProblem(edges=[(0, 1), (1, 2)], num_vertices=3, name="triangle")
    bad = MaxCutProblem(edges=[(0, 1), (1, 7)], num_vertices=3, name="broken")

    print(f"  valid?   {good.validate() == []}")
    print(f"  invalid: {bad.validate()}")
    print(f"  serialized: {good.to_dict()['type']} "
          f"edges={good.to_dict()['edges']}")


if __name__ == "__main__":
    main()