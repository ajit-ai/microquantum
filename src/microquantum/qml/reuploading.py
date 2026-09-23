"""Data-reuploading quantum classifier with a training entry point.

:class:`DataReuploadingClassifier` interleaves the feature encoding with
variational layers (each layer re-uploads the input features), which
increases expressivity over single-uploading classifiers.  It follows
the :class:`VariationalClassifier` ``predict`` / ``score`` contract and
adds the missing :meth:`fit` training loop (cross-entropy minimization
through any :class:`Optimizer`, with optional early-stopping
callbacks).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.circuit import QuantumCircuit, _narrow_concrete
from ..core.parameter import Parameter
from ..core.tensor import expand_operator
from ..optimizers.base import Optimizer, OptimizerResult
from ..optimizers.callbacks import CallbackProtocol, minimize_with_callbacks
from ..optimizers.gradient_descent import GradientDescent
from .classifier import ClassifierResult
from .encoding import AngleEncoding, BaseEncoder

__all__ = [
    "DataReuploadingClassifier",
]


class DataReuploadingClassifier:
    """Classifier with per-layer data re-uploading plus ``fit`` training.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes (default 2).
        layers: Number of re-uploading layers (must be >= 1).
        encoder: Feature map re-uploaded each layer (default AngleEncoding).
        optimizer: Classical optimizer for :meth:`fit`.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        layers: int = 2,
        encoder: Optional[BaseEncoder] = None,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        if num_classes < 2:
            raise ValueError(f"Need >= 2 classes, got {num_classes}")
        if layers < 1:
            raise ValueError(f"Need >= 1 layer, got {layers}")
        self._num_features = num_features
        self._num_classes = num_classes
        self._layers = layers
        self._encoder = encoder or AngleEncoding(num_features)
        self._optimizer = optimizer or GradientDescent(learning_rate=0.1, max_iter=50)
        self._num_qubits = self._encoder.num_qubits + max(1, num_classes - 1)
        self._params: dict[Parameter, float] = {}
        for layer in range(layers):
            for qubit in range(self._num_qubits):
                self._params[Parameter(f"theta_{layer}_{qubit}")] = 0.0

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    @property
    def layers(self) -> int:
        """Number of re-uploading layers."""
        return self._layers

    @property
    def num_qubits(self) -> int:
        """Total qubits (encoding + output)."""
        return self._num_qubits

    @property
    def parameters(self) -> dict[Parameter, float]:
        """Current parameter values."""
        return dict(self._params)

    def build_circuit(
        self, features: list[float], params: dict[Parameter, float]
    ) -> QuantumCircuit:
        """Build the re-uploading circuit for one input."""
        if len(features) != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {len(features)}"
            )
        total = self._num_qubits
        circuit = QuantumCircuit(total)
        for layer in range(self._layers):
            encoding = self._encoder.encode(features)
            for gate_instr in encoding._gate_instructions:
                if QuantumCircuit._is_parameterized_gate(gate_instr):
                    continue
                op, targets = _narrow_concrete(gate_instr)
                expanded = expand_operator(op, [int(t) for t in targets], total)
                circuit.append(expanded, list(range(total)))
            for qubit in range(total):
                key = Parameter(f"theta_{layer}_{qubit}")
                circuit.ry(float(params.get(key, 0.0)), qubit)
            for qubit in range(total - 1):
                circuit.cx(qubit, qubit + 1)
        return circuit

    def _predict_probs(
        self, features: list[float], params: dict[Parameter, float]
    ) -> list[float]:
        """Class probabilities for a single input (output-qubit readout)."""
        circuit = self.build_circuit(features, params)
        state = circuit.run()
        probs = np.abs(state.amplitudes) ** 2
        n_enc = self._encoder.num_qubits
        dim = 2**circuit.num_qubits
        class_probs = [0.0] * self._num_classes
        for index in range(dim):
            out_bits = 0
            for position in range(circuit.num_qubits - n_enc):
                bit = (index >> (n_enc + position)) & 1
                out_bits |= bit << position
            if out_bits < self._num_classes:
                class_probs[out_bits] += float(probs[index])
        total = sum(class_probs)
        if total > 1e-12:
            class_probs = [p / total for p in class_probs]
        return class_probs

    def predict(
        self, X: list[list[float]], params: Optional[dict[Parameter, float]] = None
    ) -> ClassifierResult:
        """Predict class labels for inputs."""
        active = self._params if params is None else params
        predictions: list[int] = []
        probabilities: list[list[float]] = []
        for features in X:
            probs = self._predict_probs(features, active)
            predictions.append(int(np.argmax(probs)))
            probabilities.append(probs)
        return ClassifierResult(
            predictions=predictions,
            probabilities=probabilities,
            num_classes=self._num_classes,
        )

    def score(
        self,
        X: list[list[float]],
        y: list[int],
        params: Optional[dict[Parameter, float]] = None,
    ) -> float:
        """Classification accuracy."""
        result = self.predict(X, params)
        correct = sum(
            1 for pred, true in zip(result.predictions, y, strict=False) if pred == true
        )
        return correct / len(y) if y else 0.0

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        initial_params: Optional[dict[Parameter, float]] = None,
        callbacks: Sequence[CallbackProtocol] = (),
    ) -> ClassifierResult:
        """Train on labeled data by minimizing cross-entropy loss.

        Updates the classifier's parameters in place and returns the
        training-set result including the :class:`OptimizerResult`.
        """
        if len(X) != len(y):
            raise ValueError(f"Got {len(X)} inputs but {len(y)} labels")
        if not X:
            raise ValueError("Training set must be non-empty")
        if any(not 0 <= label < self._num_classes for label in y):
            raise ValueError(f"Labels must be in [0, {self._num_classes})")
        init = dict(self._params if initial_params is None else initial_params)

        def cost(params: dict[Parameter, float]) -> float:
            total = 0.0
            for features, label in zip(X, y, strict=False):
                probs = self._predict_probs(features, params)
                total += -np.log(max(probs[label], 1e-12))
            return total / len(X)

        if callbacks:
            opt_result = minimize_with_callbacks(
                self._optimizer, cost, initial_params=init, callbacks=callbacks
            )
        else:
            opt_result = self._optimizer.minimize(cost, initial_params=init)
        self._params = dict(opt_result.optimal_parameters)
        result = self.predict(X, self._params)
        result.accuracy = self.score(X, y, self._params)
        result.optimizer_result = opt_result if isinstance(opt_result, OptimizerResult) else None
        return result

    def __repr__(self) -> str:
        return (
            f"DataReuploadingClassifier(features={self._num_features}, "
            f"classes={self._num_classes}, layers={self._layers})"
        )
