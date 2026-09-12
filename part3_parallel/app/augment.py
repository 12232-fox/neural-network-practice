"""Аугментация изображений: повороты, отражения, шум."""

import random

import numpy as np


def augment_image(arr: np.ndarray) -> np.ndarray:
    """
    Применяет случайные аугментации: поворот на 90/180/270,
    горизонтальное отражение и гауссов шум.

    :param arr: массив (H, W) или (H, W, C) со значениями в [0, 1].
    :return: аугментированный массив той же формы.
    """
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Случайный поворот на 90/180/270.
    k = random.randint(0, 3)
    if k > 0:
        arr = np.rot90(arr, k, axes=(0, 1))

    # Случайное горизонтальное отражение.
    if random.random() > 0.5:
        arr = np.fliplr(arr)

    # Случайный гауссов шум с clip в [0, 1].
    if random.random() > 0.5:
        noise = np.random.normal(0.0, 0.05, arr.shape)
        arr = np.clip(arr + noise, 0.0, 1.0)

    return arr
