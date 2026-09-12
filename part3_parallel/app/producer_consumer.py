"""Producer-consumer с потоками и пулом процессов для аугментации."""

import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .augment import augment_image

logger = logging.getLogger(__name__)


def _augment_worker(arr: np.ndarray) -> np.ndarray:
    """Функция аугментации на уровне модуля — безопасна для pickle."""
    return augment_image(arr)


class DataProcessor:
    """
    Обработка файлов по схеме producer-consumer:

    - один поток-производитель сканирует директорию и кладёт пути
      в ``file_queue``;
    - несколько потоков-потребителей читают файлы, нормализуют
      и кладут массивы в ``array_queue``;
    - пул процессов применяет аугментацию (CPU-bound) через
      ``Pool.apply_async``.
    """

    def __init__(
        self,
        input_dir: str | Path,
        extensions: Iterable[str],
        num_consumers: int = 4,
        num_processes: int = 4,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.extensions = list(extensions)
        self.num_consumers = num_consumers
        self.num_processes = num_processes

        # Свежие очереди на каждый запуск.
        self.file_queue: queue.Queue = queue.Queue()
        self.array_queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.processed_count = 0
        self.total_files = 0

    # ------------------------------------------------------------------ #
    # Producer / consumer
    # ------------------------------------------------------------------ #

    def _producer(self) -> None:
        """Кладёт пути к файлам в file_queue и завершает работу."""
        for ext in self.extensions:
            if not ext.startswith("."):
                ext = "." + ext
            for file in sorted(self.input_dir.rglob(f"*{ext}")):
                if file.is_file():
                    self.file_queue.put(file)
        # Один маркер на каждого потребителя — конец работы.
        for _ in range(self.num_consumers):
            self.file_queue.put(None)
        logger.info("Producer: завершён, файлов в очереди %d", self.file_queue.qsize())

    def _consumer(self) -> None:
        """Читает файлы, нормализует и кладёт массивы в array_queue."""
        try:
            while True:
                try:
                    file = self.file_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if file is None:
                    break
                try:
                    with Image.open(file) as img:
                        arr = np.array(img)
                    if arr.ndim == 2:
                        arr = arr[:, :, np.newaxis]
                    arr = arr.astype(np.float32) / 255.0
                    self.array_queue.put(arr)
                    with self.lock:
                        self.processed_count += 1
                        if self.processed_count % 10 == 0:
                            logger.info("Обработано %d файлов", self.processed_count)
                except (UnidentifiedImageError, OSError) as exc:
                    logger.error("Ошибка чтения %s: %s", file, exc)
        finally:
            # Всегда сигнализируем о завершении, даже при исключении.
            self.array_queue.put(None)
            logger.info("Consumer: завершён.")

    # ------------------------------------------------------------------ #
    # Последовательная версия
    # ------------------------------------------------------------------ #

    def run_sequential(self) -> tuple[int, float]:
        """Последовательная обработка (без потоков и процессов)."""
        start = time.perf_counter()
        count = 0
        for ext in self.extensions:
            if not ext.startswith("."):
                ext = "." + ext
            for file in sorted(self.input_dir.rglob(f"*{ext}")):
                if not file.is_file():
                    continue
                try:
                    with Image.open(file) as img:
                        arr = np.array(img)
                    if arr.ndim == 2:
                        arr = arr[:, :, np.newaxis]
                    arr = arr.astype(np.float32) / 255.0
                    augment_image(arr)
                    count += 1
                except (UnidentifiedImageError, OSError) as exc:
                    logger.error("Ошибка чтения %s: %s", file, exc)
        elapsed = time.perf_counter() - start
        return count, elapsed

    # ------------------------------------------------------------------ #
    # Параллельная версия
    # ------------------------------------------------------------------ #

    def run_parallel(self) -> tuple[int, float]:
        """Producer → consumers (threads) → augment (processes)."""
        start = time.perf_counter()

        producer_thread = threading.Thread(target=self._producer, name="producer")
        producer_thread.start()

        consumer_threads = [
            threading.Thread(target=self._consumer, name=f"consumer-{i}")
            for i in range(self.num_consumers)
        ]
        for t in consumer_threads:
            t.start()

        completed = 0
        failed = 0
        with mp.Pool(processes=self.num_processes) as pool:
            async_results = []
            active_consumers = self.num_consumers

            while active_consumers > 0 or not self.array_queue.empty():
                try:
                    arr = self.array_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if arr is None:
                    active_consumers -= 1
                    continue
                async_results.append(pool.apply_async(_augment_worker, (arr,)))

            for t in consumer_threads:
                t.join()
            producer_thread.join()

            for res in async_results:
                try:
                    res.get(timeout=30)
                    completed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.error("Ошибка аугментации: %s", exc)

        elapsed = time.perf_counter() - start
        logger.info(
            "Параллельно: успешно %d, ошибок %d, время %.2f с.",
            completed,
            failed,
            elapsed,
        )
        return completed, elapsed
