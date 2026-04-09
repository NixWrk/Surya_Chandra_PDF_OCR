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
