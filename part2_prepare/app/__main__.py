"""Точка входа для запуска пакета как модуля: python -m part2_prepare.app."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
