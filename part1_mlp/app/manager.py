import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from .exceptions import DataFileNotFoundError, MismatchedDataError

load_dotenv()


class DatasetManager:
    """Загрузка данных из CSV, нормализация, разбиение на train/test."""

    def __init__(
        self,
        filepath: str,
        target_col: Optional[str] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        :param filepath: путь к CSV-файлу.
        :param target_col: имя целевого столбца; если None — берётся последний.
        :param test_size: доля тестовой выборки; если None — из ENV или 0.2.
        :param random_state: seed разбиения; если None — из ENV или 42.
        """
        try:
            self.df = pd.read_csv(filepath)
        except FileNotFoundError as exc:
            raise DataFileNotFoundError(f"Файл {filepath} не найден.") from exc
        except pd.errors.EmptyDataError as exc:
            raise MismatchedDataError(f"Файл {filepath} пуст.") from exc

        if self.df.empty:
            raise MismatchedDataError("Загруженный датасет пуст.")

        self.target_col = target_col if target_col is not None else self.df.columns[-1]
        if self.target_col not in self.df.columns:
            raise MismatchedDataError(f"Столбец {self.target_col} не найден в данных.")

        self.X = self.df.drop(columns=[self.target_col]).values.astype(np.float64)
        self.y = self.df[self.target_col].values.reshape(-1, 1).astype(np.float64)

        self.test_size = (
            test_size if test_size is not None else float(os.getenv("TEST_SIZE", "0.2"))
        )
        self.random_state = (
            random_state
            if random_state is not None
            else int(os.getenv("RANDOM_STATE", "42"))
        )

        # Параметры нормализации будут заполнены в normalize().
        self.method: Optional[str] = None
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.mean_vals: Optional[np.ndarray] = None
        self.std_vals: Optional[np.ndarray] = None

    def normalize(self, method: Optional[str] = None) -> None:
        """
        Нормализует признаки.

        :param method: 'minmax' или 'standard'. Если None — берётся из ENV
                       NORMALIZE_METHOD или 'minmax'.
        """
        if method is None:
            method = os.getenv("NORMALIZE_METHOD", "minmax")
        self.method = method

        if method == "minmax":
            self.min_vals = np.min(self.X, axis=0)
            self.max_vals = np.max(self.X, axis=0)
            range_vals = self.max_vals - self.min_vals
            range_vals[range_vals == 0] = 1.0
            self.X = (self.X - self.min_vals) / range_vals
        elif method == "standard":
            self.mean_vals = np.mean(self.X, axis=0)
            self.std_vals = np.std(self.X, axis=0)
            self.std_vals[self.std_vals == 0] = 1.0
            self.X = (self.X - self.mean_vals) / self.std_vals
        else:
            raise ValueError("Метод нормализации должен быть 'minmax' или 'standard'.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Применяет ранее подобранную нормализацию к новым данным.

        :param X: массив признаков формы (n_samples, n_features).
        :return: нормализованный массив.
        :raises MismatchedDataError: если нормализация ещё не выполнена
                                     или размерность не совпадает.
        """
        if self.method is None:
            raise MismatchedDataError(
                "Нормализация не выполнена: вызовите normalize() сначала."
            )
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X.shape[1]:
            raise MismatchedDataError(
                f"Ожидается {self.X.shape[1]} признаков, получено {X.shape[1]}."
            )
        if self.method == "minmax":
            return (X - self.min_vals) / (self.max_vals - self.min_vals)
        if self.method == "standard":
            return (X - self.mean_vals) / self.std_vals
        raise MismatchedDataError(f"Неизвестный метод нормализации: {self.method}")

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разбивает выборку на train/test."""
        return train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
