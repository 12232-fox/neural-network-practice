"""CLI части 3: запуск замера производительности."""

import argparse
import logging
import sys

from . import __version__
from .config import get_config
from .logging_config import setup_logging
from .performance import measure_performance

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="part3_parallel",
        description="Многопоточная и многопроцессорная аугментация данных.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser(
        "bench", help="Замер производительности последовательной и параллельной версий."
    )
    bench.add_argument(
        "--path", help="Директория с изображениями (по умолчанию DATA_PATH)."
    )
    bench.add_argument(
        "--ext",
        help=(
            "Расширения через запятую, например '.jpg,.png' "
            "(по умолчанию EXTENSIONS)."
        ),
    )
    bench.add_argument(
        "--consumers",
        type=int,
        help="Число потоков-потребителей (по умолчанию NUM_CONSUMERS или 4).",
    )
    bench.add_argument(
        "--processes",
        type=int,
        help="Число процессов для аугментации (по умолчанию NUM_PROCESSES или 4).",
    )

    return parser


def _cmd_bench(args: argparse.Namespace, config: dict) -> int:
    """Обработчик команды bench."""
    path = args.path or config["data_path"]
    ext_raw = args.ext or ",".join(config["extensions"])
    ext_list = [e.strip() for e in ext_raw.split(",") if e.strip()]
    num_consumers = args.consumers or config["num_consumers"]
    num_processes = args.processes or config["num_processes"]

    if num_consumers < 1 or num_processes < 1:
        logger.error("Число потоков и процессов должно быть >= 1.")
        return 1

    logger.info(
        "Команда bench: path=%s, ext=%s, consumers=%d, processes=%d",
        path,
        ext_list,
        num_consumers,
        num_processes,
    )
    try:
        report = measure_performance(
            input_dir=path,
            extensions=ext_list,
            num_consumers=num_consumers,
            num_processes=num_processes,
        )
    except FileNotFoundError as exc:
        logger.error("Директория не найдена: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Путь не является директорией: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Ошибка обработки: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Ошибка ввода-вывода: %s", exc)
        return 1

    print("=== Отчёт ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = get_config()

    setup_logging(config["log_level"])

    if args.command == "bench":
        return _cmd_bench(args, config)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
