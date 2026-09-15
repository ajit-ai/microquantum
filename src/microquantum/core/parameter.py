"""Symbolic parameters for parameterized quantum circuits."""

from __future__ import annotations

from typing import Union


class Parameter:
    """A symbolic scalar parameter for variational gates.

    Represents an unbound variable that can be used in rotation gates
    and bound to a numeric value later.

    Attributes:
        name: Human-readable parameter name.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Parameter name."""
        return self._name

    def __repr__(self) -> str:
        return f"Parameter('{self._name}')"

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parameter):
            return self._name == other._name
        return NotImplemented

    def __rmul__(self, scalar: complex) -> ParameterExpression:
        return ParameterExpression(self, coefficient=complex(scalar))

    def __mul__(self, scalar: complex) -> ParameterExpression:
        """Multiply this parameter by a scalar (``theta * 2``)."""
        return ParameterExpression(self, coefficient=complex(scalar))

    def __add__(
        self, other: Union[float, int, complex]
    ) -> ParameterExpression:
        """Add a scalar offset to this parameter (``theta + 0.5``)."""
        return ParameterExpression(self, constant=complex(other))

    def __radd__(
        self, other: Union[float, int, complex]
    ) -> ParameterExpression:
        return ParameterExpression(self, constant=complex(other))

    def __sub__(
        self, other: Union[float, int, complex]
    ) -> ParameterExpression:
        """Subtract a scalar offset from this parameter (``theta - 0.5``)."""
        return ParameterExpression(self, constant=-complex(other))

    def __rsub__(
        self, other: Union[float, int, complex]
    ) -> ParameterExpression:
        return ParameterExpression(self, coefficient=-1.0, constant=complex(other))

    def __neg__(self) -> ParameterExpression:
        return ParameterExpression(self, coefficient=-1.0)

    def gradient(self, parameter: Union["Parameter", str, None] = None) -> float:
        """Symbolic partial derivative of this parameter w.r.t. a variable.

        A :class:`Parameter` ``p`` is the identity scalar field in its own
        variable: ``d(p)/dp = 1`` and ``d(p)/dq = 0`` for any other
        variable ``q``.

        Args:
            parameter: The variable to differentiate with respect to, given
                as a :class:`Parameter` or its name.  ``None`` (default)
                differentiates with respect to the parameter itself.

        Returns:
            ``1.0`` if ``parameter`` denotes this parameter, ``0.0``
            otherwise.
        """
        if parameter is None:
            return 1.0
        other = parameter if isinstance(parameter, str) else parameter.name
        return 1.0 if other == self._name else 0.0

    def __hash__(self) -> int:
        return hash(self._name)


class ParameterExpression:
    """A simple linear expression over a Parameter.

    Supports operations like ``2 * theta``, ``-theta``, and
    ``theta + 0.5``.

    Attributes:
        parameter: The symbolic parameter.
        coefficient: Scalar coefficient multiplying the parameter.
        constant: Additive constant offset.
    """

    def __init__(
        self,
        parameter: Parameter,
        coefficient: complex = 1.0,
        constant: complex = 0.0,
    ) -> None:
        self._parameter = parameter
        self._coefficient = complex(coefficient)
        self._constant = complex(constant)

    @property
    def parameter(self) -> Parameter:
        """The symbolic parameter."""
        return self._parameter

    @property
    def coefficient(self) -> complex:
        """Scalar coefficient."""
        return self._coefficient

    @property
    def constant(self) -> complex:
        """Additive constant."""
        return self._constant

    def evaluate(self, param_map: dict[Union[str, Parameter], float]) -> float:
        """Evaluate the expression given a parameter binding.

        Args:
            param_map: Mapping from parameter name/object to numeric value.

        Returns:
            The evaluated scalar value.
        """
        key = self._resolve_key(param_map)
        value = param_map[key]
        result = self._coefficient * value + self._constant
        return float(result.real)

    def _resolve_key(
        self, param_map: dict[Union[str, Parameter], float]
    ) -> Union[str, Parameter]:
        """Find the matching key in param_map."""
        if self._parameter in param_map:
            return self._parameter
        if self._parameter.name in param_map:
            return self._parameter.name
        raise KeyError(
            f"Parameter '{self._parameter.name}' not found in param_map"
        )

    def __mul__(self, scalar: complex) -> ParameterExpression:
        return ParameterExpression(
            self._parameter,
            coefficient=self._coefficient * scalar,
            constant=self._constant * scalar,
        )

    def __rmul__(self, scalar: complex) -> ParameterExpression:
        return self.__mul__(scalar)

    def __add__(
        self, other: Union[float, int, complex, ParameterExpression]
    ) -> ParameterExpression:
        if isinstance(other, ParameterExpression):
            if self._parameter != other._parameter:
                raise ValueError("Cannot add expressions with different parameters")
            return ParameterExpression(
                self._parameter,
                coefficient=self._coefficient + other._coefficient,
                constant=self._constant + other._constant,
            )
        return ParameterExpression(
            self._parameter,
            coefficient=self._coefficient,
            constant=self._constant + complex(other),
        )

    def __radd__(
        self, other: Union[float, int, complex]
    ) -> ParameterExpression:
        return self.__add__(other)

    def __neg__(self) -> ParameterExpression:
        return ParameterExpression(
            self._parameter,
            coefficient=-self._coefficient,
            constant=-self._constant,
        )

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._coefficient != 1:
            parts.append(f"{self._coefficient}")
        parts.append(repr(self._parameter))
        if self._constant != 0:
            parts.append(f"+ {self._constant}")
        return " * ".join(parts) if len(parts) == 1 else " ".join(parts)

    def gradient(self, parameter: Union[Parameter, str, None] = None) -> float:
        """Symbolic derivative of the expression w.r.t. a variable.

        For the linear expression ``a * p + c`` the derivative is the
        coefficient: ``d(a*p + c)/dp = a``.  With respect to any other
        variable the derivative is ``0.0``.  The additive constant ``c``
        never contributes.

        Args:
            parameter: The variable to differentiate with respect to, given
                as a :class:`Parameter` or its name.  ``None`` (default)
                differentiates with respect to the expression's own
                parameter.

        Returns:
            Real part of the coefficient when ``parameter`` denotes the
            expression's parameter, ``0.0`` otherwise.
        """
        if parameter is None:
            return float(self._coefficient.real)
        other = parameter if isinstance(parameter, str) else parameter.name
        if other == self._parameter.name:
            return float(self._coefficient.real)
        return 0.0

    def __str__(self) -> str:
        return repr(self)
