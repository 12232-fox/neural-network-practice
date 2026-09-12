"""Чтение конфигурации из переменных окружения и .env-файла."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# config.py -> app/ -> part2_prepare/ -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Явно указываем путь, чтобы .env находился независимо от текущей рабочей директории.
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.bmp,.tiff,.csv"
DEFAULT_DATA_PATH = "data/input"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_LOG_LEVEL = "INFO"

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _normalize_extensions(raw: str) -> list[str]:
    """Приводит строку расширений к списку с ведущей точкой и без пробелов."""
    result = []
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        result.append(ext)
    return result


def get_config() -> dict:
    """
    Возвращает словарь с настройками.

    Приоритет: переменные окружения → значения по умолчанию.
    """
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if log_level not in VALID_LOG_LEVELS:
        logging.getLogger(__name__).warning(
            "Неизвестный LOG_LEVEL '%s', используется %s.",
            log_level,
            DEFAULT_LOG_LEVEL,
        )
        log_level = DEFAULT_LOG_LEVEL

    return {
        "data_path": os.getenv("DATA_PATH", DEFAULT_DATA_PATH),
        "output_dir": os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        "extensions": _normalize_extensions(
            os.getenv("EXTENSIONS", DEFAULT_EXTENSIONS)
        ),
        "log_level": log_level,
        "project_root": PROJECT_ROOT,
    }
