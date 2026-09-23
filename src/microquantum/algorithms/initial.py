"""Shared initial-point strategies for variational algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Union

import numpy as np

from ..core.parameter import Parameter

__all__ = [
    "InitialPoint",
    "initial_parameters",
]


@dataclass(frozen=True)
class InitialPoint:
    """How to choose an optimizer starting point.

    Attributes:
        strategy: ``"zeros"``, ``"random"`` (uniform in ``[-range,
            range]``) or ``"custom"`` (use :attr:`values`).
        seed: RNG seed for the ``"random"`` strategy.
        values: Explicit values for the ``"custom"`` strategy, keyed by
            parameter name or object.
        range: Half-width of the random interval.
    """

    strategy: str = "zeros"
    seed: Optional[int] = None
    values: Mapping[Union[str, Parameter], float] = field(default_factory=dict)
    range: float = float(np.pi)

    def __post_init__(self) -> None:
        if self.strategy not in ("zeros", "random", "custom"):
            raise ValueError(f"Unknown strategy '{self.strategy}'")
        if self.range <= 0:
            raise ValueError("range must be positive")
        if self.strategy == "custom" and not self.values:
            raise ValueError("'custom' strategy requires values")


def initial_parameters(
    parameters: Iterable[Parameter], spec: InitialPoint
) -> dict[Parameter, float]:
    """Build a starting-point mapping for *parameters* under *spec*."""
    ordered = list(parameters)
    if spec.strategy == "zeros":
        return {param: 0.0 for param in ordered}
    if spec.strategy == "random":
        rng = np.random.default_rng(spec.seed)
        draws = rng.uniform(-spec.range, spec.range, size=len(ordered))
        return {param: float(draw) for param, draw in zip(ordered, draws, strict=False)}
    resolved: dict[Parameter, float] = {}
    for param in ordered:
        if param in spec.values:
            resolved[param] = float(spec.values[param])
        elif param.name in spec.values:
            key: Union[str, Parameter] = param.name
            resolved[param] = float(spec.values[key])
        else:
            raise KeyError(f"No custom value for parameter '{param.name}'")
    return resolved
