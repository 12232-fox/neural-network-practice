"""Чтение конфигурации для части 3."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# config.py -> app/ -> part3_parallel/ -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.bmp,.tiff"
DEFAULT_DATA_PATH = "data/input"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_CONSUMERS = 4
DEFAULT_NUM_PROCESSES = 4

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _normalize_extensions(raw: str) -> list[str]:
    """Нормализует строку расширений в список с ведущей точкой."""
    result = []
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        result.append(ext)
    return result


def _safe_int(value: str | None, default: int, minimum: int = 1) -> int:
    """Парсит int из строки, при ошибке возвращает default."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Не удалось разобрать число '%s', используется %d.", value, default
        )
        return default
    if parsed < minimum:
        logging.getLogger(__name__).warning(
            "Значение %d меньше %d, используется %d.", parsed, minimum, default
        )
        return default
    return parsed


def get_config() -> dict:
    """Возвращает настройки из ENV с fallback на значения по умолчанию."""
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if log_level not in VALID_LOG_LEVELS:
        log_level = DEFAULT_LOG_LEVEL

    return {
        "data_path": os.getenv("DATA_PATH", DEFAULT_DATA_PATH),
        "output_dir": os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        "extensions": _normalize_extensions(
            os.getenv("EXTENSIONS", DEFAULT_EXTENSIONS)
        ),
        "log_level": log_level,
        "num_consumers": _safe_int(os.getenv("NUM_CONSUMERS"), DEFAULT_NUM_CONSUMERS),
        "num_processes": _safe_int(os.getenv("NUM_PROCESSES"), DEFAULT_NUM_PROCESSES),
        "project_root": PROJECT_ROOT,
    }
