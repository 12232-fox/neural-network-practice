"""Сканирование каталогов, загрузка файлов и подготовка .npy-архива."""

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
CSV_EXTENSIONS = {".csv"}

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | CSV_EXTENSIONS


def scan_directory(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    """
    Рекурсивно обходит директорию и возвращает отсортированный список
    путей к файлам с указанными расширениями.

    :param directory: корневая директория поиска.
    :param extensions: список расширений (с точкой или без).
    :raises FileNotFoundError: если директория не существует.
    :raises NotADirectoryError: если путь указывает не на директорию.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Директория '{directory}' не найдена.")
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' не является директорией.")

    normalized: set[str] = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.add(ext)

    if not normalized:
        raise ValueError("Не задано ни одного расширения для поиска.")

    found: set[Path] = set()
    for ext in normalized:
        for path in directory.rglob(f"*{ext}"):
            if path.is_file():
                found.add(path)

    files = sorted(found)
    logger.info(
        "Найдено %d файлов с расширениями %s в '%s'.",
        len(files),
        sorted(normalized),
        directory,
    )
    return files


def _resize_image(img: Image.Image, size: Optional[tuple[int, int]]) -> Image.Image:
    """Приводит изображение к заданному размеру (если size не None)."""
    if size is None:
        return img
    return img.resize(size, Image.BILINEAR)


def load_file_as_array(
    filepath: Path,
    image_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Загружает файл как массив NumPy.

    :param filepath: путь к файлу.
    :param image_size: если задан — изображения ресайзятся к этому размеру (W, H).
    :return: массив NumPy.
    :raises ValueError: если расширение не поддерживается.
    :raises UnidentifiedImageError: если файл не удалось открыть как изображение.
    """
    ext = filepath.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        with Image.open(filepath) as img:
            img = _resize_image(img, image_size)
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            return arr

    if ext in CSV_EXTENSIONS:
        df = pd.read_csv(filepath)
        if df.isnull().values.any():
            logger.warning("В файле '%s' есть пропуски — заполняю нулями.", filepath)
            df = df.fillna(0.0)
        return df.values.astype(np.float64)

    raise ValueError(f"Неподдерживаемое расширение: '{ext}'.")


def normalize_image(arr: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """Нормализует изображение делением на max_val (по умолчанию 255)."""
    return arr.astype(np.float32) / max_val


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max нормализация массива в диапазон [0, 1] по столбцам."""
    arr = arr.astype(np.float64)
    mn = arr.min(axis=0, keepdims=True)
    mx = arr.max(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return ((arr - mn) / rng).astype(np.float32)


def prepare_data(
    directory: str | Path,
    extensions: Iterable[str],
    output_file: str | Path,
    normalize: bool = True,
    normalize_csv: bool = True,
    image_size: Optional[tuple[int, int]] = None,
    stack_axis: int = 0,
) -> dict:
    """
    Основной пайплайн: сканирует, загружает, нормализует, объединяет и сохраняет в .npy.

    :param directory: директория с файлами.
    :param extensions: список расширений.
    :param output_file: путь к выходному .npy.
    :param normalize: нормализовать ли изображения (деление на 255).
    :param normalize_csv: применять ли min-max к CSV.
    :param image_size: ресайзить изображения к этому размеру (W, H) или None.
    :param stack_axis: ось для объединения массивов (по умолчанию 0).
    :return: словарь со статистикой.
    :raises FileNotFoundError: если директория не найдена.
    :raises ValueError: если не найдено ни одного файла или массивы несовместимы.
    """
    start_time = time.time()
    files = scan_directory(directory, extensions)

    if not files:
        raise ValueError(
            f"В директории '{directory}' не найдено файлов с указанными расширениями."
        )

    arrays: list[np.ndarray] = []
    shapes: list[tuple[int, ...]] = []
    dtypes: Counter = Counter()
    skipped: list[tuple[Path, str]] = []

    for file in files:
        try:
            arr = load_file_as_array(file, image_size=image_size)
        except (UnidentifiedImageError, ValueError, OSError) as exc:
            logger.error("Пропускаю '%s': %s", file, exc)
            skipped.append((file, str(exc)))
            continue

        if normalize:
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                if arr.dtype == np.uint8:
                    arr = normalize_image(arr, 255.0)
            elif file.suffix.lower() in CSV_EXTENSIONS and normalize_csv:
                arr = normalize_minmax(arr)

        arrays.append(arr)
        shapes.append(arr.shape)
        dtypes[str(arr.dtype)] += 1

    if not arrays:
        raise ValueError("Не удалось загрузить ни одного файла.")

    try:
        combined = np.concatenate(arrays, axis=stack_axis)
    except ValueError as exc:
        raise ValueError(
            "Формы массивов несовместимы для объединения. "
            "Попробуйте задать --image-size для ресайза изображений. "
            f"Детали: {exc}"
        ) from exc

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, combined)

    elapsed = time.time() - start_time
    size_bytes = output_path.stat().st_size

    stats = {
        "num_files_scanned": len(files),
        "num_files_loaded": len(arrays),
        "num_files_skipped": len(skipped),
        "skipped": [{"file": str(p), "reason": r} for p, r in skipped],
        "shapes": {
            "min": list(min(shapes, key=lambda s: np.prod(s))),
            "max": list(max(shapes, key=lambda s: np.prod(s))),
            "unique_count": len(set(shapes)),
        },
        "dtypes": dict(dtypes),
        "combined_shape": list(combined.shape),
        "combined_dtype": str(combined.dtype),
        "output_file": str(output_path),
        "output_size_bytes": size_bytes,
        "time_seconds": round(elapsed, 3),
    }

    logger.info(
        "Сохранён массив %s в '%s' за %.2f с (%d файлов, пропущено %d).",
        combined.shape,
        output_path,
        elapsed,
        len(arrays),
        len(skipped),
    )
    return stats
