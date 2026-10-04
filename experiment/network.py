"""Small NumPy MLP implementing the runtime ResidualPredictor contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ResidualMLP:
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 24, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.12, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(input_vector, dtype=float)
        if x.ndim != 1 or x.shape[0] != self.w1.shape[0] or not np.all(np.isfinite(x)):
            raise ValueError("invalid learned-appraisal input")
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        epochs: int = 250,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> dict[str, float]:
        if len(train_x) == 0 or len(validation_x) == 0:
            raise ValueError("training and validation sets must be nonempty")
        if train_x.shape[1] != self.w1.shape[0] or train_y.shape[1] != self.w2.shape[1]:
            raise ValueError("training tensors do not match the model")
        parameters = ("w1", "b1", "w2", "b2")
        moments = {name: np.zeros_like(getattr(self, name)) for name in parameters}
        variances = {name: np.zeros_like(getattr(self, name)) for name in parameters}
        nonzero = np.any(np.abs(train_y) > 1e-8, axis=1)
        positive_weight = min(10.0, max(1.0, float(np.sum(~nonzero)) / max(1, int(np.sum(nonzero)))))
        weights = np.where(nonzero, positive_weight, 1.0)
        weights /= np.sum(weights)
        best = float("inf")
        best_epoch = 0
        best_parameters = {name: getattr(self, name).copy() for name in parameters}
        for epoch in range(1, epochs + 1):
            hidden = np.tanh(train_x @ self.w1 + self.b1)
            prediction = np.tanh(hidden @ self.w2 + self.b2)
            error = prediction - train_y
            d_output = 2.0 * weights[:, None] * error / train_y.shape[1]
            d_output *= 1.0 - prediction * prediction
            gradients = {
                "w2": hidden.T @ d_output + 2.0 * weight_decay * self.w2,
                "b2": d_output.sum(axis=0),
            }
            d_hidden = (d_output @ self.w2.T) * (1.0 - hidden * hidden)
            gradients["w1"] = train_x.T @ d_hidden + 2.0 * weight_decay * self.w1
            gradients["b1"] = d_hidden.sum(axis=0)
            for name in parameters:
                gradient = np.clip(gradients[name], -1.0, 1.0)
                moments[name] = 0.9 * moments[name] + 0.1 * gradient
                variances[name] = 0.999 * variances[name] + 0.001 * gradient * gradient
                corrected_m = moments[name] / (1.0 - 0.9 ** epoch)
                corrected_v = variances[name] / (1.0 - 0.999 ** epoch)
                setattr(
                    self, name,
                    getattr(self, name) - learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8),
                )
            val_hidden = np.tanh(validation_x @ self.w1 + self.b1)
            val_prediction = np.tanh(val_hidden @ self.w2 + self.b2)
            val_loss = float(np.mean((val_prediction - validation_y) ** 2))
            if val_loss < best:
                best = val_loss
                best_epoch = epoch
                best_parameters = {name: getattr(self, name).copy() for name in parameters}
        for name, value in best_parameters.items():
            setattr(self, name, value)
        return {
            "validation_mse": best,
            "best_epoch": float(best_epoch),
            "nonzero_training_targets": float(np.sum(nonzero)),
            "positive_target_weight": positive_weight,
        }

    def save(self, path: str | Path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualMLP":
        with np.load(path, allow_pickle=False) as values:
            w1, b1, w2, b2 = (values[name].copy() for name in ("w1", "b1", "w2", "b2"))
        if w1.ndim != 2 or w2.ndim != 2 or b1.shape != (w1.shape[1],) or b2.shape != (w2.shape[1],):
            raise ValueError("invalid learned-appraisal artifact shapes")
        if w2.shape[0] != w1.shape[1] or not all(np.all(np.isfinite(x)) for x in (w1, b1, w2, b2)):
            raise ValueError("invalid learned-appraisal artifact values")
        model = cls(w1.shape[0], w2.shape[1], w1.shape[1])
        model.w1, model.b1, model.w2, model.b2 = w1, b1, w2, b2
        return model
