# Часть 3. Многопоточная и многопроцессорная аугментация данных

Модуль реализует схему producer-consumer для обработки изображений:
один поток-производитель сканирует каталог, несколько потоков-потребителей
читают файлы, а пул процессов применяет аугментацию (CPU-bound).
Дополнительно измеряется время последовательной и параллельной версий.

## Возможности

- **Producer** (1 поток): рекурсивный обход директории, кладёт пути
к файлам в `queue.Queue`.
- **Consumers** (N потоков): читают изображения через PIL, нормализуют
в [0, 1], кладут массивы во вторую очередь.
- **Пул процессов** (M процессов): применяет аугментацию
(`np.rot90`, `np.fliplr`, гауссов шум).
- **Синхронизация**: `threading.Lock` для счётчика обработанных файлов,
`multiprocessing.Pool.apply_async` для параллельной аугментации.
- **Замер производительности**: сохраняет
`data/output/performance_report.json` с временем и ускорением.

## Структура

part3_parallel/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── cli.py
│ ├── config.py
│ ├── logging_config.py
│ ├── augment.py
│ ├── producer_consumer.py
│ └── performance.py
└── README.md

## Настройка

Скопируйте .env.example в .env в корне проекта и при необходимости отредактируйте.
```cp .env.example .env```

| Переменная    | Назвачение                  | По умолчанию                    |
|---------------|-----------------------------|---------------------------------|
| DATA_PATH     | каталог с входными данными  | ./data/input                    |
| OUTPUT_DIR    | каталог для результатов     | ./data/output                   |
| EXTENSIONS    | расширения через запятую    | .jpg,.jpeg,.png,.bmp,.tiff,.csv |
| NUM_CONSUMERS | число потоков-потребителей  | 4                               |
| NUM_PROCESSES | число процессов аугментации | 4                               |
| LOG_LEVEL     | уровень логирования         | INFO                            |


## Использование

Для начала нужно перейти в корень проекта и запускать оттуда.

```
python -m part3_parallel.app bench --path data/input --ext .jpg --consumers 4 --processes 4
```

Вывод (пример):
```
2026-09-12 20:57:19,281 - part3_parallel.app.cli - INFO - Команда bench: path=data/input, ext=['.jpg'], consumers=4, processes=4
2026-09-12 20:57:19,281 - part3_parallel.app.performance - INFO - === Последовательная обработка ===
2026-09-12 20:57:19,487 - part3_parallel.app.performance - INFO - Последовательно: 10 файлов за 0.204 с.
2026-09-12 20:57:19,487 - part3_parallel.app.performance - INFO - === Параллельная обработка ===
2026-09-12 20:57:19,490 - part3_parallel.app.producer_consumer - INFO - Producer: завершён, файлов в очереди 9
2026-09-12 20:57:19,492 - part3_parallel.app.producer_consumer - INFO - Consumer: завершён.
2026-09-12 20:57:19,492 - part3_parallel.app.producer_consumer - INFO - Consumer: завершён.
2026-09-12 20:57:19,493 - part3_parallel.app.producer_consumer - INFO - Consumer: завершён.
2026-09-12 20:57:19,493 - part3_parallel.app.producer_consumer - INFO - Обработано 10 файлов
2026-09-12 20:57:19,493 - part3_parallel.app.producer_consumer - INFO - Consumer: завершён.
2026-09-12 20:57:19,812 - part3_parallel.app.producer_consumer - INFO - Параллельно: успешно 10, ошибок 0, время 0.33 с.
2026-09-12 20:57:19,813 - part3_parallel.app.performance - INFO - Параллельно: 10 файлов за 0.325 с.
2026-09-12 20:57:19,814 - part3_parallel.app.performance - INFO - Отчёт сохранён в C:\Users\user\Desktop\neural_network_practice\data\output\performance_report.json
Отчёт о производительности сохранён в C:\Users\user\Desktop\neural_network_practice\data\output\performance_report.json
=== Отчёт ===
input_dir: data/input
extensions: ['.jpg']
num_consumers: 4
num_processes: 4
sequential: {'count': 10, 'time': 0.2042}
parallel: {'count': 10, 'time': 0.325}
speedup: 0.628
timestamp: 2026-09-12T15:57:19.813666+00:00
```

## Формат performance_report.json
```
{
  "input_dir": "data/input",
  "extensions": [".jpg", ".png"],
  "num_consumers": 4,
  "num_processes": 4,
  "sequential": {"count": 100, "time": 0.512},
  "parallel": {"count": 100, "time": 0.198},
  "speedup": 2.586,
  "timestamp": "2026-09-12T20:00:00+00:00"
}
```

## Обработка ошибок

- Файлы, которые не удалось прочитать, логируются и пропускаются.
- Потоки-потребители всегда кладут маркер None в очередь (через try/finally), даже при исключении — это гарантирует, что run_parallel завершится.
- Пул процессов закрывается через with mp.Pool(...) as pool, что гарантирует корректное завершение процессов.