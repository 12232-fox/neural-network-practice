# Часть 2. CLI-утилита для подготовки данных и диагностики

Модуль для сканирования каталога с файлами (изображения, CSV),
преобразования их в единый `.npy`-архив, пригодный для подачи в нейросеть,
а также для диагностики окружения.

## Возможности

- Команда `prepare`:
  - рекурсивный обход директории;
  - поддержка изображений (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) и CSV;
  - нормализация изображений делением на 255;
  - min-max нормализация CSV (опционально);
  - опциональный ресайз изображений к единому размеру;
  - сохранение объединённого массива в `.npy`;
  - вывод статистики: количество файлов, формы, размер, время.
- Команда `doctor`:
  - версия Python и путь к интерпретатору;
  - текущая рабочая директория;
  - проверка наличия библиотек (`numpy`, `PIL`, `pandas`, `torch`, `tensorflow`, `cv2`, `dotenv`);
  - доступность CUDA через `nvidia-smi` (через `subprocess.run`);
  - дополнительно `torch.cuda.is_available()`, если `torch` установлен.
- Настройки читаются из ENV, CLI-аргументов или значений по умолчанию.

## Структура

part2_prepare/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── cli.py
│ ├── config.py
│ ├── logging_config.py
│ └── scanner.py
└── README.md

## Установка

```
pip install -r requirements.txt
```

## Настройка

Скопируйте .env.example в .env в корне проекта и при необходимости отредактируйте.
| Переменная | Назвачение                 | По умолчанию                    |
|------------|----------------------------|---------------------------------|
| DATA_PATH  | каталог с входными данными | ./data/input                    |
| OUTPUT_DIR | каталог для результатов    | ./data/output                   |
| EXTENSIONS | расширения через запятую   | .jpg,.jpeg,.png,.bmp,.tiff,.csv |
| LOG_LEVEL  | уровень логирования        | INFO                            |



## Использование

Для начала нужно перейти в корень проекта и запускать оттуда.

*prepare*
```python -m part2_prepare.app prepare --path data/input --ext .csv --output data/output/test.npy```

Опции:
--path — директория с данными (по умолчанию DATA_PATH).
--ext — расширения через запятую (по умолчанию EXTENSIONS).
--output — выходной .npy (по умолчанию OUTPUT_DIR/prepared_data.npy).
--no-normalize — отключить нормализацию изображений.
--no-normalize-csv — отключить min-max нормализацию CSV.
--image-size 224x224 — ресайз всех изображений к единому размеру.


Пример вывода:
```
=== Статистика ===
num_files_scanned: 42
num_files_loaded: 42
num_files_skipped: 0
shapes: {'min': [28, 28, 1], 'max': [28, 28, 1], 'unique_count': 1}
dtypes: {'float32': 42}
combined_shape: [1176, 28, 28, 1]
combined_dtype: float32
output_file: data/output/prepared_data.npy
output_size_bytes: 3696928
time_seconds: 1.234
```

*doctor*
```python -m part2_prepare.app doctor```

Пример вывода:
```
=== Doctor ===
Python: 3.11.5 (/usr/bin/python3)
Текущая директория: /home/user/project
Корень проекта: /home/user/project

--- Библиотеки ---
[  OK   ] numpy: 1.26.0
[  OK   ] PIL (Pillow): 10.1.0
[  OK   ] pandas: 2.1.1
[MISSING] torch: не установлена
[MISSING] tensorflow: не установлена
[  OK   ] cv2 (OpenCV): 4.8.1
[  OK   ] dotenv: 1.0.0

--- CUDA ---
nvidia-smi: недоступна (nvidia-smi не установлен)

--- Итог ---
Отсутствуют библиотеки: torch, tensorflow
```

## Зависимости

См. requirements.txt в корне проекта. Ключевые: numpy, pandas, Pillow, python-dotenv.

## Обработка ошибок

Отсутствие директории → FileNotFoundError, код выхода 1.
Пустая выборка → ValueError, код выхода 1.
Несовместимые формы массивов → ValueError с подсказкой использовать --image-size.
Ошибка открытия конкретного файла → логируется, файл пропускается, остальные обрабатываются.