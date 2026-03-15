# img_2_pdf

Набор локальных Windows-скриптов (Tkinter GUI) для конвертации изображений/PDF в PDF с OCR.

## Что делает каждый скрипт

| Файл | Назначение | Вход -> Выход | Основные зависимости |
|---|---|---|---|
| `fast.py` | Основной универсальный GUI для OCR через `ocrmypdf`. Поддерживает 3 режима: папка изображений, один PDF, пакет PDF. | Images/PDF -> searchable PDF | `ocrmypdf`, `pypdf`, `img2pdf`, Tesseract, Ghostscript, qpdf |
| `img_2_pdf.py` | Конвертер фото в PDF с предобработкой OpenCV: авто-перспектива, deskew+crop, split разворотов. OCR опционально. | Папка фото -> PDF (опц. OCR) | `opencv-python`, `numpy`, `img2pdf`, (опц.) `ocrmypdf` |
| `only_tesseract.py` | OCR-пайплайн без `ocrmypdf`: прямой вызов `tesseract.exe`; для PDF-режима рендер/merge через Poppler. | Images/PDF -> searchable PDF | Tesseract, `pypdf`, Poppler (`pdftoppm`, `pdfunite`) |
| `imgs_and_pdfs_ocr_fast_STABLE.py` | Стабильная/предыдущая версия OCR GUI (ближайший аналог `fast.py`). | Images/PDF -> searchable PDF | `ocrmypdf`, `pypdf`, Tesseract |
| `prepare pdf to tesseract.py` | Подготовка PDF (например из Office Lens) перед OCR: растрирование, даунскейл, JPEG-сжатие. | PDF -> "prepared to tesseract.pdf" | `PyMuPDF (fitz)`, `Pillow` |
| `naps2-7.5.3-win.exe` | Установщик NAPS2 для сканирования/сборки документов. | - | - |

## Кратко: какой скрипт запускать

1. Нужен один рабочий инструмент для OCR из изображений и PDF: `fast.py`.
2. Нужно улучшить качество страниц с фото (выравнивание, обрезка, разделение разворотов): `img_2_pdf.py`.
3. Нельзя/не хочется ставить `ocrmypdf`, но есть `tesseract.exe`: `only_tesseract.py`.
4. Хочется более консервативный вариант `fast.py`: `imgs_and_pdfs_ocr_fast_STABLE.py`.
5. Исходный PDF "тяжелый" или плохо распознается: сначала `prepare pdf to tesseract.py`, потом OCR.

## Запуск

```powershell
python fast.py
python img_2_pdf.py
python only_tesseract.py
python "prepare pdf to tesseract.py"
```

## Зависимости

Python-пакеты (в зависимости от выбранного скрипта):

```powershell
pip install ocrmypdf pypdf img2pdf opencv-python numpy pillow pymupdf
```

Внешние утилиты (нужны не всем скриптам):

1. Tesseract OCR
2. Ghostscript
3. qpdf
4. Poppler (`pdftoppm`, `pdfunite`) для `only_tesseract.py` в PDF-режиме

## Что изменялось последним (по git)

Проверено по истории коммитов:

1. `img_2_pdf.py` -> коммит `0cfb6e7`, `2026-02-11 01:02:50 +03:00`
2. `fast.py`, `imgs_and_pdfs_ocr_fast_STABLE.py`, `only_tesseract.py`, `prepare pdf to tesseract.py` -> коммит `5c408eb`, `2026-02-11 01:01:40 +03:00`

То есть последним менялся именно `img_2_pdf.py`.
