"""Variational quantum classifier for quantum machine learning.

Implements a variational quantum classifier that combines a data
encoding feature map with a parameterized variational circuit for
binary and multi-class classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.parameter import Parameter
from ..core.tensor import expand_operator
from ..optimizers.base import Optimizer, OptimizerResult
from ..optimizers.gradient_descent import GradientDescent
from .encoding import AngleEncoding, BaseEncoder


@dataclass
class ClassifierResult:
    """Result from a variational classifier.

    Attributes:
        predictions: Predicted class labels for each input.
        probabilities: Class probabilities for each input.
        accuracy: Classification accuracy (if true labels provided).
        optimizer_result: Underlying VQE optimization result.
        num_classes: Number of output classes.
    """
    predictions: list[int] = field(default_factory=list)
    probabilities: list[list[float]] = field(default_factory=list)
    accuracy: float = 0.0
    optimizer_result: Optional[OptimizerResult] = None
    num_classes: int = 2


class VariationalClassifier:
    """Variational quantum classifier.

    Combines a data encoding feature map with a parameterized ansatz
    for classification. The classifier is trained by minimizing a
    cross-entropy loss on labeled data.

    For binary classification, the output is a single qubit measured
    in the Z-basis. For multi-class, multiple output qubits are used.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes (default 2).
        encoder: Feature map for data encoding. If None, uses AngleEncoding.
        ansatz_depth: Depth of the parameterized ansatz circuit.
        optimizer: Classical optimizer for training.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        encoder: Optional[BaseEncoder] = None,
        ansatz_depth: int = 2,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        if num_classes < 2:
            raise ValueError(f"Need >= 2 classes, got {num_classes}")

        self._num_features = num_features
        self._num_classes = num_classes
        self._encoder = encoder or AngleEncoding(num_features)
        self._ansatz_depth = ansatz_depth
        self._optimizer = optimizer or GradientDescent(
            learning_rate=0.1, max_iter=50
        )

        # Number of qubits = encoding qubits + output qubits
        self._num_qubits = self._encoder.num_qubits + max(1, num_classes - 1)

        # Initialize parameters
        self._params: dict[Parameter, float] = {}
        param_count = 0
        for _ in range(ansatz_depth):
            for _ in range(self._num_qubits):
                self._params[Parameter(f"theta_{param_count}")] = 0.0
                param_count += 1

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def parameters(self) -> dict[Parameter, float]:
        return dict(self._params)

    def _build_ansatz(
        self, params: dict[Parameter, float]
    ) -> QuantumCircuit:
        """Build the parameterized ansatz circuit.

        Args:
            params: Parameter values for the ansatz.

        Returns:
            QuantumCircuit implementing the ansatz.
        """
        qc = QuantumCircuit(self._num_qubits)
        param_idx = 0

        for _ in range(self._ansatz_depth):
            # Rotation layer
            for i in range(self._num_qubits):
                p = Parameter(f"theta_{param_idx}")
                val = params.get(p, 0.0)
                qc.ry(val, i)
                param_idx += 1

            # Entangling layer
            for i in range(self._num_qubits - 1):
                qc.cx(i, i + 1)
            if self._num_qubits > 2:
                qc.cx(self._num_qubits - 1, 0)

        return qc

    def _build_circuit(
        self,
        features: list[float],
        params: dict[Parameter, float],
    ) -> QuantumCircuit:
        """Build full classification circuit.

        Args:
            features: Input feature vector.
            params: Parameter values.

        Returns:
            Complete circuit: encoding + ansatz.
        """
        enc = self._encoder.encode(features)
        ansatz = self._build_ansatz(params)

        # Combine into single circuit
        total = enc.num_qubits + (self._num_qubits - enc.num_qubits)
        qc = QuantumCircuit(total)

        # Encoding on first n_enc qubits
        for gate_instr in enc._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            op = gate_instr[0]  # type: ignore[assignment]
            targets = gate_instr[1]  # type: ignore[assignment,union-attr]
            expanded = expand_operator(op, [int(t) for t in targets], total)  # type: ignore[arg-type,union-attr]
            qc.append(expanded, list(range(total)))

        # Ansatz on all qubits
        for gate_instr in ansatz._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            op = gate_instr[0]  # type: ignore[assignment]
            targets = gate_instr[1]  # type: ignore[assignment,union-attr]
            expanded = expand_operator(op, [int(t) for t in targets], total)  # type: ignore[arg-type,union-attr]
            qc.append(expanded, list(range(total)))

        return qc

    def _predict_probs(
        self, features: list[float], params: dict[Parameter, float]
    ) -> list[float]:
        """Compute class probabilities for a single input.

        Args:
            features: Input feature vector.
            params: Parameter values.

        Returns:
            List of class probabilities.
        """
        qc = self._build_circuit(features, params)
        state = qc.run()
        probs = np.abs(state.amplitudes) ** 2

        # For binary: probability of measuring qubit 0 in |1>
        # For multi-class: probability distribution over first n_classes qubits
        n_enc = self._encoder.num_qubits
        n_out_qubits = qc.num_qubits - n_enc
        dim = 2**qc.num_qubits
        n_out = self._num_classes
        class_probs = [0.0] * n_out

        if self._num_classes == 2 and n_out_qubits == 1:
            qubit_idx = n_enc
            prob1 = sum(
                probs[i] for i in range(dim) if (i >> qubit_idx) & 1
            )
            class_probs = [1.0 - prob1, prob1]
        else:
            for i in range(dim):
                bits = 0
                for c in range(n_out_qubits):
                    qubit_idx = n_enc + c
                    bit = (i >> qubit_idx) & 1
                    bits |= bit << c
                if bits < n_out:
                    class_probs[bits] += probs[i]

        # Normalize
        total = sum(class_probs)
        if total > 1e-12:
            class_probs = [p / total for p in class_probs]

        return class_probs

    def predict(
        self, X: list[list[float]], params: Optional[dict[Parameter, float]] = None
    ) -> ClassifierResult:
        """Predict class labels for a set of inputs.

        Args:
            X: Feature vectors to classify.
            params: Optional parameter override. Uses self.params if None.

        Returns:
            ClassifierResult with predictions and probabilities.
        """
        if params is None:
            params = self._params

        predictions = []
        probabilities = []

        for features in X:
            probs = self._predict_probs(features, params)
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
        """Compute classification accuracy.

        Args:
            X: Feature vectors.
            y: True class labels.
            params: Optional parameter override.

        Returns:
            Accuracy between 0 and 1.
        """
        result = self.predict(X, params)
        correct = sum(1 for pred, true in zip(result.predictions, y, strict=False) if pred == true)
        return correct / len(y) if y else 0.0

    def __repr__(self) -> str:
        return (
            f"VariationalClassifier(features={self._num_features}, "
            f"classes={self._num_classes}, qubits={self._num_qubits})"
        )
