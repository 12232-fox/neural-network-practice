"""Замер производительности: последовательная vs параллельная обработка."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .producer_consumer import DataProcessor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_output_dir() -> Path:
    """Возвращает каталог для результатов (OUTPUT_DIR или data/output)."""
    output_dir = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def measure_performance(
    input_dir: str,
    extensions: list[str],
    num_consumers: int = 4,
    num_processes: int = 4,
) -> dict:
    """
    Измеряет время последовательной и параллельной обработки,
    сохраняет отчёт в performance_report.json.

    :param input_dir: директория с изображениями.
    :param extensions: список расширений.
    :param num_consumers: число потоков-потребителей.
    :param num_processes: число процессов для аугментации.
    :return: словарь с результатами.
    """
    logger.info("=== Последовательная обработка ===")
    dp_seq = DataProcessor(input_dir, extensions, num_consumers, num_processes)
    count_seq, time_seq = dp_seq.run_sequential()
    logger.info("Последовательно: %d файлов за %.3f с.", count_seq, time_seq)

    logger.info("=== Параллельная обработка ===")
    dp_par = DataProcessor(input_dir, extensions, num_consumers, num_processes)
    count_par, time_par = dp_par.run_parallel()
    logger.info("Параллельно: %d файлов за %.3f с.", count_par, time_par)

    speedup = time_seq / time_par if time_par > 0 else 0.0
    report = {
        "input_dir": str(input_dir),
        "extensions": extensions,
        "num_consumers": num_consumers,
        "num_processes": num_processes,
        "sequential": {"count": count_seq, "time": round(time_seq, 4)},
        "parallel": {"count": count_par, "time": round(time_par, 4)},
        "speedup": round(speedup, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = _get_output_dir() / "performance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Отчёт сохранён в %s", report_path)
    print(f"Отчёт о производительности сохранён в {report_path}")
    return report
