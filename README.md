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
```

## Сервисный слой (web-ready)

Для будущего web GUI в проект вынесен application-layer:

- `src/uniscan/app/page_spec.py` — единый парсер страниц (`1,3,5-8`)
- `src/uniscan/app/ocr_pipeline.py` — оркестрация OCR/артефактных workflow

Базовый GUI использует именно этот слой, а не shell-команды напрямую.

## Базовый GUI (минимальный)

```powershell
.\run_basic_gui.cmd
```

Что делает:

1. Создаёт `.venv` прямо в этом репозитории.
2. Устанавливает проект и OCR-движки в локальный `.venv` (`surya-ocr`, `chandra-ocr[hf]`, `pypdf`, `reportlab` и др.).
3. Запускает минимальный GUI:
   - выбор одного PDF,
   - выбор режима: `Surya` / `Гибрид` / `Оба`,
   - выбор страниц (например: `1,3,5-8`, пусто = все страницы),
   - индикатор прогресса выполнения.

Примечание:

- `Гибрид` в этом GUI запускает движок `chandra` (гибридная ветка проекта: текст Chandra + Surya-геометрия в пайплайне).

## Сравнение геометрии (Chandra text)

Собрать два searchable PDF с одинаковым `chandra`-текстом и разной геометрией:

```powershell
.\scripts\compare_chandra_geometry_variants.ps1 `
  -RunRoot "D:\Git_Code\Surya_Chandra_PDF_OCR\outputs\basic_gui_runs\ГОСТ с плохим качеством скана_20260410_182529" `
  -PdfRoot "O:\OBS_TEST\PDF2OBS\PDFS"
```

Подробный воспроизводимый путь зафиксирован в:
`docs/benchmark_runs/2026-04-13_chandra_geometry_compare_path.md`

## Что внутри

- OCR benchmark matrix и артефактный pipeline
- Гибридный Surya+Chandra token alignment
- Тесты OCR-ветки
- Набор OCRmyPDF plugins/скетчей

## Разделение репозиториев

- `img_2_pdf` — только pre-OCR подготовка изображений/PDF
- `surya_hOCR_bridge` — только Surya bridge/HOCR
- `Surya_Chandra_PDF_OCR` — финальная гибридная OCR-ветка
