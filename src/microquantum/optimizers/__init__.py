"""Classical optimizers for variational quantum algorithms."""

from .base import Optimizer, OptimizerResult
from .bounds import Bounds
from .callbacks import CallbackProtocol, minimize_with_callbacks
from .gradient_descent import Adam, GradientDescent
from .gradient_free import COBYLA, NelderMead
from .quasi_newton import BFGS
from .spsa import QNSPSA, SPSA

__all__ = [
    "Optimizer",
    "OptimizerResult",
    "Bounds",
    "CallbackProtocol",
    "minimize_with_callbacks",
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
