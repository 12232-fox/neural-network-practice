"""Полносвязная нейронная сеть на NumPy."""

import json
from typing import List, Tuple

import numpy as np

from .exceptions import (
    DataFileNotFoundError,
    InvalidLayerSizeError,
    MismatchedDataError,
)


class NeuralNetwork:
    """
    Полносвязная нейронная сеть с произвольным числом слоёв.

    Выходной слой всегда использует сигмоиду (для приведения выхода
    к диапазону [0, 1]); скрытые слои — выбранную функцию активации.
    """

    def __init__(self, layer_sizes: List[int], activation: str = "sigmoid") -> None:
        """
        :param layer_sizes: список размеров слоёв, включая входной и выходной.
        :param activation: функция активации скрытых слоёв: 'sigmoid' или 'relu'.
        """
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError(
                "Сеть должна содержать как минимум входной и выходной слой."
            )
        if any(size < 1 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть >= 1.")
        if activation not in ("sigmoid", "relu"):
            raise ValueError("Активация должна быть 'sigmoid' или 'relu'.")

        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            # He-инициализация: хорошо подходит для ReLU.
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self._set_activation(activation)
        self.loss_history: List[float] = []
        self.layer_inputs: List[np.ndarray] = []
        self.layer_outputs: List[np.ndarray] = []

    # Функции активации
    def _set_activation(self, activation: str) -> None:
        """Устанавливает указатели на функции активации и их производные."""
        if activation == "sigmoid":
            self._activation_func = self._sigmoid
            self._activation_deriv = self._sigmoid_deriv
        elif activation == "relu":
            self._activation_func = self._relu
            self._activation_deriv = self._relu_deriv
        else:
            raise ValueError(f"Неизвестная активация: {activation}")
        self.activation_name = activation

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Численно устойчивая сигмоида."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def _sigmoid_deriv(self, x: np.ndarray) -> np.ndarray:
        """Производная сигмоиды по её аргументу."""
        s = self._sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU."""
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv(x: np.ndarray) -> np.ndarray:
        """Производная ReLU."""
        return (x > 0).astype(float)

    # Прямой и обратный проход
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Прямой проход.

        :param X: входные данные формы (n_samples, n_features).
        :return: выход сети формы (n_samples, n_outputs).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидается {self.layer_sizes[0]} признаков, получено {X.shape[1]}."
            )

        self.layer_inputs = [X]
        self.layer_outputs = []
        A = X
        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]
            self.layer_inputs.append(Z)
            if i == len(self.weights) - 1:
                # Выходной слой — всегда сигмоида.
                A = self._sigmoid(Z)
            else:
                A = self._activation_func(Z)
            self.layer_outputs.append(A)
        return A

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Вычисляет градиенты весов и смещений для батча.

        :param X: батч признаков (m, n_features).
        :param y: батч целевых значений (m, n_outputs).
        :return: (grad_weights, grad_biases).
        """
        m = X.shape[0]
        y_pred = self.forward(X)
        dA = (y_pred - y) / m  # производная MSE по выходу

        grad_w: List[np.ndarray] = [None] * len(self.weights)  # type: ignore
        grad_b: List[np.ndarray] = [None] * len(self.biases)  # type: ignore

        for i in reversed(range(len(self.weights))):
            A_prev = X if i == 0 else self.layer_outputs[i - 1]
            if i == len(self.weights) - 1:
                dZ = dA * self._sigmoid_deriv(self.layer_inputs[i + 1])
            else:
                dZ = dA * self._activation_deriv(self.layer_inputs[i + 1])

            grad_w[i] = A_prev.T @ dZ
            grad_b[i] = np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                dA = dZ @ self.weights[i].T
        return grad_w, grad_b

    def _apply_gradients(
        self,
        grad_w: List[np.ndarray],
        grad_b: List[np.ndarray],
        learning_rate: float,
    ) -> None:
        """Обновляет веса и смещения по градиентам."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """Один шаг обратного распространения с обновлением весов."""
        grad_w, grad_b = self._compute_gradients(X, y)
        self._apply_gradients(grad_w, grad_b, learning_rate)

    # Обучение и предсказание
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 1,
    ) -> None:
        """
        Обучение сети.

        :param X: признаки (m, n_features).
        :param y: целевые значения (m, n_outputs).
        :param epochs: число эпох.
        :param learning_rate: скорость обучения.
        :param batch_size: размер батча.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise MismatchedDataError(
                f"X и y содержат разное число примеров: {X.shape[0]} vs {y.shape[0]}."
            )
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидается {self.layer_sizes[0]} признаков, получено {X.shape[1]}."
            )
        if y.shape[1] != self.layer_sizes[-1]:
            raise MismatchedDataError(
                f"Ожидается {self.layer_sizes[-1]} выходов, получено {y.shape[1]}."
            )
        if epochs < 1:
            raise MismatchedDataError("Число эпох должно быть >= 1.")
        if batch_size < 1:
            raise MismatchedDataError("Размер батча должен быть >= 1.")

        m = X.shape[0]
        self.loss_history = []
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0.0
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(X_batch)
                loss = float(np.mean((y_pred - y_batch) ** 2))
                epoch_loss += loss * (end - start)
                self.backward(X_batch, y_batch, learning_rate)
            avg_loss = epoch_loss / m
            self.loss_history.append(avg_loss)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Эпоха {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказания сети для входных данных."""
        return self.forward(X)

    # Сохранение и загрузка
    def save_weights(self, filepath: str) -> None:
        """Сохраняет веса, смещения и конфигурацию в JSON-файл."""
        data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_weights(self, filepath: str) -> None:
        """
        Загружает веса из JSON-файла.

        :raises DataFileNotFoundError: если файл не найден.
        :raises MismatchedDataError: если размеры слоёв не совпадают.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise DataFileNotFoundError(f"Файл {filepath} не найден.") from exc
        except json.JSONDecodeError as exc:
            raise MismatchedDataError(
                f"Файл {filepath} повреждён или не является JSON."
            ) from exc

        if data.get("layer_sizes") != self.layer_sizes:
            raise MismatchedDataError(
                f"Размеры слоёв не совпадают: "
                f"в файле {data.get('layer_sizes')}, в сети {self.layer_sizes}."
            )
        self.weights = [np.array(w, dtype=np.float64) for w in data["weights"]]
        self.biases = [np.array(b, dtype=np.float64) for b in data["biases"]]
        self._set_activation(data.get("activation", "sigmoid"))
