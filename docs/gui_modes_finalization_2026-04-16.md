# Финализация GUI-режимов и План Чистки (2026-04-16)

## 1) Зафиксированный контракт режимов GUI

В GUI оставляем только 3 режима:

1. `surya-surya` (значение mode: `surya`)
2. `chandra-chandra` (значение mode: `chandra`)
3. `chandra-surya` (значение mode: `chandra+surya`)

Текущее отображение режимов и маршрутизация заданы в:

- `src/uniscan/web/service.py` (селектор mode)
- `src/uniscan/app/ocr_pipeline.py` (`PDF_MODE_*`, маршрутизация OCR/сборки)

## 2) Зафиксированная логика гибрида `chandra-surya`

Для гибрида используем последнюю реализованную логику выбора геометрии:

- Near-tie override в `auto`:
  - если лучший кандидат `secondary` выигрывает у `primary` незначительно, выбираем `primary`.
  - порог: `_HYBRID_AUTO_PRIMARY_SCORE_MARGIN = 0.025`
- Ограниченный blend:
  - blend включается только при слабом `primary` (coverage ниже порога) и заметном преимуществе `secondary`.
  - порог преимущества: `_HYBRID_BLEND_SECONDARY_ADVANTAGE_MIN = 0.08`
- В лог добавлен флаг `auto_primary_override`.

Кодовые точки фиксации:

- `src/uniscan/ocr/artifact_searchable.py:60`
- `src/uniscan/ocr/artifact_searchable.py:61`
- `src/uniscan/ocr/artifact_searchable.py:993`
- `src/uniscan/ocr/artifact_searchable.py:1013`
- `src/uniscan/ocr/artifact_searchable.py:1995`
- `src/uniscan/ocr/artifact_searchable.py:2050`

Практический эффект по новым логам:

- `softline` перестал массово делать blend на нормальных страницах:
  - `gost`: было `blended=37`, стало `blended=0`
  - `book`: было `blended=33`, стало `blended=1`
- `auto` чаще оставляет Surya-геометрию при near-tie:
  - `gost`: `primary 2 -> 5`
  - `book`: `primary 3 -> 10`

Сравнения лежат в:

- `outputs/last_attempt_geometry_tuning_gost_v2`
- `outputs/last_attempt_geometry_tuning_book_v1`

## 3) Оценка скорости по логам (приближенно)

Важно:

- Ниже значения взяты из существующих логов/артефактов.
- Это не синтетический бенчмарк, а фактические прогоны.
- На скорость сильно влияют warm/cold cache, сеть HF, загрузка GPU/CPU.

### 3.1 OCR-этап (из `*_ocr_benchmark.json`)

| Документ | `surya-surya` OCR | `chandra-chandra` OCR |
|---|---:|---:|
| `book_handwritten` (33 стр.) | 387.855 s | 4385.856 s |
| `gost` (37 стр.) | 316.326 s | 19093.202 s |

Источники:

- `outputs/service_runs/book_handwritten__surya_native_20260415_123606/surya/book_handwritten__surya_native_ocr_benchmark.json`
- `outputs/service_runs/book_handwritten__chandra_native_20260415_124242/chandra/book_handwritten__chandra_native_ocr_benchmark.json`
- `outputs/service_runs/gost__surya_native_20260415_143018/surya/gost__surya_native_ocr_benchmark.json`
- `outputs/service_runs/gost__chandra_native_20260414_174002/chandra/gost__chandra_native_ocr_benchmark.json`

### 3.2 Сборка searchable PDF из артефактов (из `artifact_searchable_summary.json`)

| Документ | `surya-surya` build | `chandra-chandra` build | `chandra-surya` build (latest, `auto`) |
|---|---:|---:|---:|
| `book_handwritten` (33 стр.) | 5.578 s | 77.086 s | 192.204 s |
| `gost` (37 стр.) | 2.684 s | 15.882 s | 32.526 s |

Источники:

- `outputs/service_runs/book_handwritten__surya_native_20260415_123606/searchable_pdf_final/artifact_searchable_summary.json`
- `outputs/service_runs/book_handwritten__chandra_native_20260415_124242/searchable_pdf_final/artifact_searchable_summary.json`
- `outputs/service_runs/gost__surya_native_20260415_143018/searchable_pdf_final/artifact_searchable_summary.json`
- `outputs/service_runs/gost__chandra_native_20260414_174002/searchable_pdf_final/artifact_searchable_summary.json`
- `outputs/last_attempt_geometry_tuning_book_v1/auto/artifact_searchable_summary.json`
- `outputs/last_attempt_geometry_tuning_gost_v2/auto/artifact_searchable_summary.json`

### 3.3 Приближенный итог по режимам GUI

Для `chandra-surya` в GUI итоговая длительность оценивается как:

`OCR(chandra) + OCR(surya) + build(hybrid)`

На текущих логах:

- На `book_handwritten`: гибрид примерно `~4966 s` (около `82.8 мин`)
- На `gost`: гибрид примерно `~19442 s` (около `324.0 мин`, явно аномальный run с очень долгим Chandra OCR)

Вывод по скорости:

- Самый быстрый режим: `surya-surya`
- Самый медленный и наименее стабильный по времени: `chandra-chandra`
- `chandra-surya` обычно ближе к `chandra-chandra` по времени, т.к. включает два OCR-прохода

## 4) План работ на чистку и рефакторинг

### Фаза A. Freeze поведения (обязательно перед чисткой)

1. Зафиксировать 3 режима GUI как единственные поддерживаемые публичные сценарии.
2. Зафиксировать hybrid-логику (near-tie override + gating blend) как baseline.
3. Зафиксировать тесты на новую hybrid-эвристику.

Критерий завершения:

- Документация и тесты отражают текущее поведение, без “плавающих” веток логики.

### Фаза B. Нормализация структуры (под рефакторинг)

1. Разделить в репозитории runtime-пути и исследовательские артефакты:
   - runtime: `src/`, `tests/`, `scripts/`, `README.md`, `docs/`
   - исследовательское/временное: `outputs/**`, временные каталоги
2. Вынести временные/экспериментальные файлы в отдельный архивный корень (например `outputs_archive/` вне рабочего цикла).
3. Оставить в `outputs/` только минимальные эталонные прогоны для верификации.

Критерий завершения:

- Новому разработчику понятно, какие директории “боевые”, какие архивные.

### Фаза C. Минимально необходимый набор в репозитории

Оставить обязательно:

1. `src/uniscan/**` (runtime OCR pipeline)
2. `tests/**` (покрытие критичной логики)
3. `scripts/full_hybrid_geometry_eval.ps1` и/или один унифицированный eval-скрипт
4. `README.md` + актуальные docs по режимам и запуску

Проверить на удаление/перенос:

1. Дублирующие планы и исторические документы без актуальной ценности
2. Временные каталоги/логи/кэш, не нужные для воспроизводимости
3. Разрозненные экспериментальные скрипты без входа из README

Критерий завершения:

- В корне нет “мусорных” путей, у каждого оставшегося файла есть роль в runtime или поддержке runtime.

### Фаза D. Целевой рефакторинг после чистки

1. Упростить `artifact_searchable.py`:
   - выделить выбор кандидатов/скоры/blend в отдельные модули
2. Ввести единый слой конфигурации hybrid-политик (вместо распыления по env/CLI)
3. Закрыть рефакторинг тестами:
   - юнит-тесты на candidate selection
   - smoke-тесты сборки по 3 GUI режимам

Критерий завершения:

- Логика режима `chandra-surya` читается из 1-2 модулей, изменение порогов не ломает пайплайн.

## 5) Что считать “самым необходимым” для финальной ветки

Минимальный операционный контур:

1. Один вход: `python -m uniscan searchable-pdf ...`
2. Три режима GUI: `surya`, `chandra`, `chandra+surya`
3. Воспроизводимый benchmark-док по скорости и качеству
4. Набор тестов, который можно запускать локально без ручных шагов

