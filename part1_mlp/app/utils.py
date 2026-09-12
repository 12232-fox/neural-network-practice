"""Вспомогательные функции части 1."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# utils.py -> app/ -> part1_mlp/ -> neural_network_practice/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_output_dir() -> Path:
    """Возвращает (и при необходимости создаёт) каталог для результатов."""
    output_dir = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_input_dir() -> Path:
    """Возвращает каталог с входными данными."""
    return PROJECT_ROOT / os.getenv("DATA_PATH", "data/input")


def get_weights_path(filename: str = "weights.json") -> Path:
    """Возвращает путь к файлу весов по умолчанию."""
    return get_output_dir() / filename


def safe_int_input(prompt: str, min_value: int = 1) -> int:
    """Считывает целое число с проверкой минимального значения."""
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("Ошибка: введите целое число.")
            continue
        if value < min_value:
            print(f"Значение должно быть не меньше {min_value}.")
            continue
        return value


def safe_float_input(prompt: str, min_value: float = 0.0) -> float:
    """Считывает число с плавающей точкой с проверкой минимума."""
    while True:
        try:
            value = float(input(prompt).strip())
        except ValueError:
            print("Ошибка: введите число.")
            continue
        if value <= min_value:
            print(f"Значение должно быть больше {min_value}.")
            continue
        return value


def parse_int_list(prompt: str, min_value: int = 1) -> List[int]:
    """Считывает список целых чисел через пробел."""
    while True:
        raw = input(prompt).strip().split()
        if not raw:
            print("Список не может быть пустым.")
            continue
        try:
            values = [int(x) for x in raw]
        except ValueError:
            print("Ошибка: введите целые числа через пробел.")
            continue
        if any(v < min_value for v in values):
            print(f"Все значения должны быть не меньше {min_value}.")
            continue
        return values
