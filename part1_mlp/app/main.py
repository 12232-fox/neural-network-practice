"""Консольное меню части 1."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from .exceptions import NeuralNetworkError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import get_output_dir, parse_int_list, safe_float_input, safe_int_input

load_dotenv()


def _create_network() -> NeuralNetwork:
    """Запрашивает параметры и создаёт сеть."""
    sizes = parse_int_list("Введите размеры слоёв через пробел (например, 4 10 1): ")
    act = input("Выберите активацию (sigmoid/relu): ").strip().lower()
    if act not in ("sigmoid", "relu"):
        print("Неверная активация, будет использована sigmoid.")
        act = "sigmoid"
    network = NeuralNetwork(sizes, act)
    print(f"Сеть создана: {sizes}, активация {act}")
    return network


def _load_dataset() -> DatasetManager:
    """Загружает CSV и выполняет нормализацию."""
    path = input("Введите путь к CSV: ").strip()
    manager = DatasetManager(path)
    method = (
        input("Метод нормализации (minmax/standard, Enter — minmax): ").strip().lower()
        or "minmax"
    )
    if method not in ("minmax", "standard"):
        print("Неверный метод, будет использован minmax.")
        method = "minmax"
    manager.normalize(method=method)
    return manager


def _train(
    network: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Запрашивает гиперпараметры и обучает сеть."""
    lr = safe_float_input("Скорость обучения: ")
    epochs = safe_int_input("Число эпох: ")
    batch_size = safe_int_input("Размер батча: ")
    network.fit(X_train, y_train, epochs, lr, batch_size)
    y_pred = network.predict(X_test)
    test_loss = float(np.mean((y_pred - y_test) ** 2))
    print(f"Test loss: {test_loss:.6f}")


def _predict(network: NeuralNetwork, manager: DatasetManager) -> None:
    """Предсказание для одного примера (вручную или из файла)."""
    mode = input("1 — ввести признаки вручную, 2 — загрузить из CSV: ").strip()
    if mode == "1":
        raw = input("Введите значения признаков через пробел: ").split()
        if not raw:
            raise ValueError("Нет данных.")
        X = np.array([list(map(float, raw))], dtype=np.float64)
    elif mode == "2":
        path = input("Путь к CSV с примерами (без целевого столбца): ").strip()
        import pandas as pd

        X = pd.read_csv(path).values.astype(np.float64)
    else:
        print("Неверный выбор.")
        return

    X_norm = manager.transform(X)
    pred = network.predict(X_norm)
    for i, row in enumerate(pred, start=1):
        print(f"Пример {i}: {row[0]:.4f}")


def _save_or_load_weights(network: NeuralNetwork) -> None:
    """Сохранение или загрузка весов."""
    output_dir = get_output_dir()
    sub = input("1 — сохранить, 2 — загрузить: ").strip()
    if sub == "1":
        fname = (
            input("Имя файла для сохранения (например, weights.json): ").strip()
            or "weights.json"
        )
        save_path = output_dir / fname
        network.save_weights(str(save_path))
        print(f"Веса сохранены в {save_path}")
    elif sub == "2":
        fname = input("Имя файла для загрузки: ").strip()
        load_path = output_dir / fname
        if not load_path.exists():
            load_path = Path(fname)
            if not load_path.exists():
                print(f"Файл {fname} не найден.")
                return
        network.load_weights(str(load_path))
        print(f"Веса загружены из {load_path}")
    else:
        print("Неверный выбор.")


def _plot_loss(network: NeuralNetwork) -> None:
    """Показывает график ошибки по эпохам."""
    if not network.loss_history:
        print("Нет истории ошибок.")
        return
    plt.plot(network.loss_history)
    plt.xlabel("Эпоха")
    plt.ylabel("Потери (MSE)")
    plt.title("График ошибки")
    plt.grid(True)
    plt.show()


def main_menu() -> None:
    """Главное меню программы."""
    network: NeuralNetwork | None = None
    manager: DatasetManager | None = None
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    while True:
        print("\n===== МЕНЮ =====")
        print("1. Создать сеть")
        print("2. Загрузить данные из CSV")
        print("3. Обучить сеть")
        print("4. Сделать предсказание")
        print("5. Сохранить/загрузить веса")
        print("6. Показать график ошибки")
        print("7. Выход")
        choice = input("Выберите действие: ").strip()

        try:
            if choice == "1":
                network = _create_network()

            elif choice == "2":
                manager = _load_dataset()
                data = manager.split()
                X_train, X_test, y_train, y_test = data
                print(
                    f"Данные загружены. Train: {X_train.shape[0]}, "
                    f"Test: {X_test.shape[0]}"
                )

            elif choice == "3":
                if network is None or data is None:
                    print("Сначала создайте сеть и загрузите данные.")
                    continue
                X_train, X_test, y_train, y_test = data
                _train(network, X_train, y_train, X_test, y_test)

            elif choice == "4":
                if network is None or manager is None:
                    print("Сначала создайте сеть и загрузите данные.")
                    continue
                _predict(network, manager)

            elif choice == "5":
                if network is None:
                    print("Создайте сеть сначала.")
                    continue
                _save_or_load_weights(network)

            elif choice == "6":
                if network is None:
                    print("Создайте сеть сначала.")
                    continue
                _plot_loss(network)

            elif choice == "7":
                print("Выход.")
                return

            else:
                print("Неверный пункт.")

        except NeuralNetworkError as exc:
            print(f"Ошибка нейросети: {exc}")
        except (ValueError, OSError) as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(0)
