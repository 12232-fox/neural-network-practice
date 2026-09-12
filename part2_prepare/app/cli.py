"""CLI-утилита части 2: подготовка данных и диагностика окружения."""

import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import get_config
from .logging_config import setup_logging
from .scanner import prepare_data

logger = logging.getLogger(__name__)

# prepare
def _parse_image_size(raw: str | None) -> tuple[int, int] | None:
    """Парсит строку вида '224x224' в кортеж (W, H)."""
    if not raw:
        return None
    try:
        w_str, h_str = raw.lower().split("x")
        w, h = int(w_str), int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Размер изображения должен быть в формате WxH, например 224x224."
        ) from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Размеры изображения должны быть > 0.")
    return w, h


def _cmd_prepare(args: argparse.Namespace, config: dict) -> int:
    """Обработчик команды prepare. Возвращает код выхода."""
    path = args.path or config["data_path"]
    ext_raw = args.ext or ",".join(config["extensions"])
    ext_list = [e.strip() for e in ext_raw.split(",") if e.strip()]
    output = args.output or str(Path(config["output_dir"]) / "prepared_data.npy")
    normalize = not args.no_normalize
    normalize_csv = not args.no_normalize_csv
    image_size = _parse_image_size(args.image_size)

    logger.info("Команда prepare: path=%s, ext=%s, output=%s", path, ext_list, output)
    try:
        stats = prepare_data(
            directory=path,
            extensions=ext_list,
            output_file=output,
            normalize=normalize,
            normalize_csv=normalize_csv,
            image_size=image_size,
        )
    except FileNotFoundError as exc:
        logger.error("Директория не найдена: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Путь не является директорией: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Ошибка подготовки данных: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Ошибка ввода-вывода: %s", exc)
        return 1

    print("=== Статистика ===")
    for key, value in stats.items():
        if key == "skipped" and not value:
            continue
        print(f"{key}: {value}")
    return 0

# doctor
def _check_library(name: str, import_name: str | None = None) -> tuple[bool, str]:
    """
    Проверяет, установлена ли библиотека.

    :return: (установлена_ли, версия_или_сообщение).
    """
    import_name = import_name or name
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        return False, "не установлена"
    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def _check_cuda() -> tuple[bool, str]:
    """Проверяет доступность CUDA через nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi не установлен"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi не ответил за 10 секунд"
    except OSError as exc:
        return False, f"ошибка запуска nvidia-smi: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else "нет вывода"
        return False, f"nvidia-smi вернул код {result.returncode}: {tail}"

    first_line = (result.stdout or "").strip().splitlines()
    cuda_info = "доступна"
    for line in first_line:
        if "CUDA Version:" in line:
            cuda_info = line.strip()
            break
    return True, cuda_info


def _cmd_doctor(args: argparse.Namespace, config: dict) -> int:
    """Обработчик команды doctor. Возвращает код выхода."""
    print("=== Doctor ===")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Корень проекта: {config['project_root']}")

    print("\n--- Библиотеки ---")
    libs = [
        ("numpy", "numpy"),
        ("PIL (Pillow)", "PIL"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("tensorflow", "tensorflow"),
        ("cv2 (OpenCV)", "cv2"),
        ("dotenv", "dotenv"),
    ]
    missing: list[str] = []
    for display_name, import_name in libs:
        ok, info = _check_library(display_name, import_name)
        mark = "OK" if ok else "MISSING"
        print(f"[{mark:^7}] {display_name}: {info}")
        if not ok:
            missing.append(display_name)

    print("\n--- CUDA ---")
    cuda_ok, cuda_info = _check_cuda()
    print(f"nvidia-smi: {'доступна' if cuda_ok else 'недоступна'} ({cuda_info})")

    # Дополнительно: torch.cuda, если torch установлен.
    if "torch" not in missing:
        try:
            import torch  # type: ignore

            cuda_torch = torch.cuda.is_available()
            print(f"torch.cuda.is_available(): {cuda_torch}")
        except Exception as exc:  # noqa: BLE001
            print(f"torch.cuda: не удалось проверить ({exc})")

    print("\n--- Итог ---")
    if missing:
        print(f"Отсутствуют библиотеки: {', '.join(missing)}")
    else:
        print("Все проверяемые библиотеки установлены.")
    return 0

# main
def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="part2_prepare",
        description="Утилита для подготовки данных и диагностики окружения.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prepare_parser = subparsers.add_parser(
        "prepare", help="Подготовка данных из директории в .npy"
    )
    prepare_parser.add_argument(
        "--path", help="Путь к директории с данными (по умолчанию DATA_PATH)."
    )
    prepare_parser.add_argument(
        "--ext",
        help=(
            "Расширения через запятую, например '.jpg,.png,.csv' "
            "(по умолчанию EXTENSIONS)."
        ),
    )
    prepare_parser.add_argument(
        "--output",
        help="Выходной .npy-файл (по умолчанию OUTPUT_DIR/prepared_data.npy).",
    )
    prepare_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Отключить нормализацию изображений.",
    )
    prepare_parser.add_argument(
        "--no-normalize-csv",
        action="store_true",
        help="Отключить min-max нормализацию CSV.",
    )
    prepare_parser.add_argument(
        "--image-size",
        help="Ресайз изображений в формат WxH, например 224x224.",
    )

    # doctor
    subparsers.add_parser("doctor", help="Диагностика окружения.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = get_config()

    setup_logging(config["log_level"])

    if args.command == "prepare":
        return _cmd_prepare(args, config)
    if args.command == "doctor":
        return _cmd_doctor(args, config)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
