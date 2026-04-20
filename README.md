# Surya_Chandra_PDF_OCR

OCR-only репозиторий для сборки `searchable PDF` из входного PDF.

Поддерживаемые режимы:

1. `surya` (`surya-surya`)
2. `chandra` (`chandra-chandra`)
3. `chandra+surya` (гибрид)

В гибриде итоговый PDF строится как:

1. текст от `chandra`
2. геометрия от `surya`

## Быстрый старт (минимальный локальный GUI)

```powershell
.\run_basic_gui.cmd
```

Скрипт:

1. создает `.venv`
2. ставит проект и OCR-зависимости
3. запускает минимальный GUI `src/uniscan/ui/basic_ocr_gui.py`

## Основные CLI-команды

```powershell
python -m uniscan benchmark-ocr --help
python -m uniscan benchmark-ocr-canonical --help
python -m uniscan prepare-compare-txt --help
python -m uniscan build-searchable-from-artifacts --help
python -m uniscan compare-chandra-geometry --help
python -m uniscan searchable-pdf --help
python -m uniscan serve-http --help
```

## Базовый сценарий PDF -> searchable PDF

```powershell
python -m uniscan searchable-pdf `
  --pdf "D:\path\input.pdf" `
  --mode chandra+surya `
  --strict
```

## Артефактный режим (text/geometry из готовых прогонов)

```powershell
python -m uniscan build-searchable-from-artifacts `
  --compare-dir "D:\path\_compare_txt" `
  --pdf-root "D:\path\pdf_root" `
  --output "D:\path\out" `
  --engines chandra `
  --chandra-geometry-policy auto `
  --geometry-debug-log `
  --strict
```

Политики гибридной геометрии для `chandra`:

1. `auto`
2. `surya_only`
3. `softline`

## HTTP сервис (опционально)

```powershell
python -m uniscan serve-http --host 127.0.0.1 --port 8000
```

Доступно:

1. `GET /` — web GUI
2. `GET /health`
3. `POST /searchable-pdf`
4. `POST /api/jobs`
5. `GET /api/jobs/{id}`
6. `GET /api/jobs/{id}/result`

## Dual-Venv Mode (Recommended)

Use isolated environments to avoid dependency conflicts:

1. `.venv_surya` for Surya OCR + geometry.
2. `.venv_chandra` for Chandra OCR text.

One-time setup:

```powershell
.\setup_dual_venv.cmd
```

Launch GUI with dual routing:

```powershell
.\run_basic_gui.cmd
```

Runtime routing is controlled by:

1. `UNISCAN_SURYA_PYTHON`
2. `UNISCAN_CHANDRA_PYTHON`

When these variables are set, each OCR engine is executed in its own interpreter,
while the main orchestrator remains in the GUI process.
