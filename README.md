# Surya_Chandra_PDF_OCR

Гибридный OCR-проект:

1. OCR-текст: `Chandra`
2. Геометрия/координаты: `Surya`
3. Сборка searchable PDF из артефактов

## Основные команды

```powershell
python -m uniscan benchmark-ocr --help
python -m uniscan prepare-compare-txt --help
python -m uniscan build-searchable-from-artifacts --help
python -m uniscan compare-chandra-geometry --help
python -m uniscan searchable-pdf --help
python -m uniscan serve-http --help
```

## Сервисный слой (web-ready)

Для будущего web GUI в проект вынесен application-layer:

- `src/uniscan/app/page_spec.py` — единый парсер страниц (`1,3,5-8`)
- `src/uniscan/app/ocr_pipeline.py` — оркестрация OCR/артефактных workflow

Базовый GUI использует именно этот слой, а не shell-команды напрямую.

## Контракт Вход/Выход PDF

По умолчанию система работает в режиме `chandra+surya`:

- OCR-текст: `chandra`
- Геометрия: `surya`

Поддерживаемые режимы (`mode`):

1. `chandra`
2. `surya`
3. `chandra+surya` (по умолчанию)

### Если на вход передан путь к PDF

Входной файл перезаписывается итоговым searchable PDF.

```powershell
python -m uniscan searchable-pdf `
  --pdf "D:\path\input.pdf" `
  --mode chandra+surya
```

### Если на вход переданы bytes PDF (например upload в web/gui)

Используйте Python API `build_searchable_pdf(...)` и получайте `output_pdf_bytes`.

```python
from uniscan.app import build_searchable_pdf

result = build_searchable_pdf(
    pdf_bytes=uploaded_pdf_bytes,
    mode="chandra+surya",  # default
)

searchable_pdf_bytes = result.output_pdf_bytes
```

### Дополнительно: обработка выбранных страниц

```powershell
python -m uniscan searchable-pdf `
  --pdf "D:\path\input.pdf" `
  --mode chandra `
  --pages 1,3,5-8
```

## Базовый GUI (минимальный)

```powershell
.\run_basic_gui.cmd
```

Что делает:

1. Создаёт `.venv` прямо в этом репозитории.
2. Устанавливает проект и OCR-движки в локальный `.venv` (`surya-ocr`, `chandra-ocr[hf]`, `pypdf`, `reportlab` и др.).
3. Запускает минимальный GUI:
   - выбор одного PDF,
   - выбор режима: `chandra+surya` (по умолчанию) / `chandra` / `surya`,
   - выбор страниц (например: `1,3,5-8`, пусто = все страницы),
   - индикатор прогресса выполнения,
   - итоговый результат: searchable PDF (входной файл перезаписывается).

## Web GUI + HTTP API

Запуск локального web-сервиса:

```powershell
python -m uniscan serve-http --host 127.0.0.1 --port 8000
```

Доступно:

1. `GET /` — web GUI (загрузка PDF, выбор режима, страницы, live-прогресс, скачивание результата).
2. `GET /health` — health-check JSON.
3. `POST /api/jobs` — создать async OCR-job (raw PDF bytes в body).
4. `GET /api/jobs/{id}` — статус job (`queued/running/done/error`) + progress.
5. `GET /api/jobs/{id}/result` — скачать готовый searchable PDF.
6. `POST /searchable-pdf` — синхронный endpoint (raw PDF bytes -> PDF bytes), для скриптов.

Параметры query string для `POST /api/jobs` и `POST /searchable-pdf`:

1. `mode`: `chandra`, `surya`, `chandra+surya` (по умолчанию).
2. `pages`: например `1,3,5-8` (опционально).
3. `lang`: OCR language, по умолчанию `rus+eng`.
4. `strict`: `1/0`, `true/false` (по умолчанию strict).
5. `filename`: имя для скачиваемого файла в web GUI.

Пример sync-вызова из PowerShell:

```powershell
$inPdf = "D:\Git_Code\PDFS\ГОСТ с плохим качеством скана.pdf"
$outPdf = "D:\Git_Code\PDFS\ГОСТ_web_result.pdf"

Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/searchable-pdf?mode=chandra%2Bsurya&pages=1-3" `
  -Method Post `
  -InFile $inPdf `
  -ContentType "application/pdf" `
  -OutFile $outPdf
```

Пример async-цикла из PowerShell:

```powershell
$inPdf = "D:\Git_Code\PDFS\ГОСТ с плохим качеством скана.pdf"

$create = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/jobs?mode=chandra%2Bsurya&pages=1-3&filename=gost.pdf" `
  -Method Post `
  -InFile $inPdf `
  -ContentType "application/pdf"

$jobId = $create.job_id
do {
  Start-Sleep -Milliseconds 800
  $status = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/jobs/$jobId"
  "$($status.status) $($status.progress)% $($status.message)"
} while ($status.status -in @("queued","running"))

if ($status.status -eq "done") {
  Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/jobs/$jobId/result" -OutFile "D:\Git_Code\PDFS\ГОСТ_async_result.pdf"
}
```

## Docker

Сборка образа:

```powershell
docker build -t uniscan-ocr:latest .
```

Запуск API в контейнере:

```powershell
docker run --rm -p 8000:8000 -v D:\Git_Code\Surya_Chandra_PDF_OCR\outputs:/app/outputs uniscan-ocr:latest
```

## Сравнение геометрии (Chandra text)

Собрать два searchable PDF с одинаковым `chandra`-текстом и разной геометрией:

```powershell
.\scripts\compare_chandra_geometry_variants.ps1 `
  -RunRoot "D:\Git_Code\Surya_Chandra_PDF_OCR\outputs\basic_gui_runs\ГОСТ с плохим качеством скана_20260410_182529" `
  -PdfRoot "O:\OBS_TEST\PDF2OBS\PDFS"
```

Подробный воспроизводимый путь зафиксирован в:
`docs/benchmark_runs/2026-04-13_chandra_geometry_compare_path.md`

### Политики гибридного позиционирования

Для `build-searchable-from-artifacts` добавлены опции управления геометрией `chandra`:

1. `--chandra-geometry-policy auto` — автоподбор на каждой странице.
2. `--chandra-geometry-policy surya_only` — только surya-геометрия.
3. `--chandra-geometry-policy softline` — мягкое построчное смешивание.
4. `--chandra-blend-weight 0.75` — вес первичной геометрии при смешивании (0..1).
5. `--geometry-debug-log` — записывает per-page лог выбора геометрии (`*_geometry_log.json`).

Пример:

```powershell
python -m uniscan build-searchable-from-artifacts `
  --compare-dir "D:\path\_compare_txt" `
  --pdf-root "D:\path\pdf_root" `
  --output "D:\path\out" `
  --engines chandra `
  --chandra-geometry-policy softline `
  --chandra-blend-weight 0.75 `
  --geometry-debug-log `
  --strict
```

### Полный прогон (ГОСТ + книга, все варианты)

Готовый скрипт запускает:

1. `surya` native и `chandra` native (4 PDF на 2 документа).
2. Гибрид A: `chandra text + surya geometry`.
3. Гибрид B: `auto per page`.
4. Гибрид C: `softline`.

Скрипт печатает живые логи в текущем окне PowerShell и сохраняет общий `full_run.log`.

```powershell
.\scripts\full_hybrid_geometry_eval.ps1 `
  -GostPdf "D:\Git_Code\PDFS\ГОСТ с плохим качеством скана.pdf" `
  -BookPdf "D:\Git_Code\PDFS\Старая книга с частично рукописным текстом.pdf" `
  -PdfRoot "D:\Git_Code\PDFS"
```

## Что внутри

- OCR benchmark matrix и артефактный pipeline
- Гибридный Surya+Chandra token alignment
- Тесты OCR-ветки
- Набор OCRmyPDF plugins/скетчей

## Разделение репозиториев

- `img_2_pdf` — только pre-OCR подготовка изображений/PDF
- `surya_hOCR_bridge` — только Surya bridge/HOCR
- `Surya_Chandra_PDF_OCR` — финальная гибридная OCR-ветка
