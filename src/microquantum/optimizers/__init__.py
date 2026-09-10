"""Classical optimizers for variational quantum algorithms."""

from .base import Optimizer, OptimizerResult
from .gradient_descent import Adam, GradientDescent
from .gradient_free import COBYLA, NelderMead
from .quasi_newton import BFGS
from .spsa import QNSPSA, SPSA

__all__ = [
    "Optimizer",
    "OptimizerResult",
    "GradientDescent",
    "Adam",
    "BFGS",
    "COBYLA",
    "NelderMead",
    "SPSA",
    "QNSPSA",
]

try:
    from .quasi_newton import L_BFGS_B  # noqa: F401
    __all__.append("L_BFGS_B")
except ImportError:
    pass
