# OCR Runtime Inventory (2026-04-16)

Документ фиксирует **минимально необходимый набор файлов и зависимостей** для рабочей OCR-ветки проекта
с тремя финальными режимами GUI:

1. `surya-surya` (`mode=surya`)
2. `chandra-chandra` (`mode=chandra`)
3. `chandra-surya` (`mode=chandra+surya`)

## 1) Scope и важная оговорка

Инвентаризация ниже сделана для текущего состояния кода **без дополнительного рефакторинга импортов**.

Сейчас `src/uniscan/cli.py` импортирует `ui` и `tools` на старте, поэтому даже для OCR-команд
часть UI/crop-модулей остается транзитивно обязательной.

## 2) Обязательные файлы (runtime)

### 2.1 Корень репозитория

Обязательно оставить:

1. `pyproject.toml`
2. `README.md`
3. `run_basic_gui.cmd` (основной Windows launcher)
4. `Dockerfile` (если нужен контейнерный запуск)
5. `src/` (весь пакет `uniscan`)

### 2.2 Обязательные runtime-модули в `src/uniscan`

Для текущей структуры CLI безопасно считать обязательным весь пакет:

1. `src/uniscan/__main__.py`
2. `src/uniscan/cli.py`
3. `src/uniscan/app/**`
4. `src/uniscan/ocr/**`
5. `src/uniscan/io/**`
6. `src/uniscan/core/**`
7. `src/uniscan/export/**`
8. `src/uniscan/web/**`
9. `src/uniscan/ui/**`
10. `src/uniscan/tools/**`
11. `src/uniscan/session/**`
12. `src/uniscan/storage/**`

Причина: `cli.py` сейчас делает eager-import этих веток через `app/ocr/tools/ui/web`.

### 2.3 Что обязательно для сопровождения (не runtime, но нужно оставить)

1. `tests/**`
2. `docs/gui_modes_finalization_2026-04-16.md`
3. `docs/hybrid_geometry_improvement_plan_2026-04-16.md`
4. Этот документ `docs/ocr_runtime_inventory_2026-04-16.md`

## 3) Python-зависимости

### 3.1 Базовые зависимости пакета (`pyproject.toml`)

1. `opencv-python>=4.8`
2. `numpy>=1.26`
3. `pillow>=10.0`
4. `img2pdf>=0.5`
5. `pymupdf>=1.24`
6. `customtkinter>=5.2`

### 3.2 OCR-библиотеки для финальной ветки (`build_searchable_pdf`)

1. `ocrmypdf>=16.0`
2. `pypdf>=4.0`
3. `reportlab>=4.0`

Важно: для трех финальных режимов (`surya-surya`, `chandra-chandra`, `chandra-surya`)
критичны именно `pypdf` и `reportlab` (сборка invisible-text PDF через `artifact_searchable`).
`ocrmypdf` ставится текущим bootstrap-путем (`.[ocr]`), но в этой финальной схеме
не является блокирующей runtime-зависимостью.

### 3.3 Обязательные engine-пакеты для 3 финальных режимов

1. `surya-ocr`
2. `chandra-ocr[hf]`
3. `requests`
4. `transformers==4.57.1`
5. `tokenizers==0.22.1`
6. `huggingface-hub==0.34.4`

Именно эти пакеты дополнительно ставятся в `run_basic_gui.cmd` и в `Dockerfile`.

### 3.4 Транзитивно важные runtime-пакеты/бинарники

1. `torch` (как зависимость Chandra-стека; нужен для стабильной работы `chandra`-движка)
2. CLI/модули `surya_ocr` (или `surya`/`marker`) для режима `surya`
3. CLI/модуль `chandra` для режима `chandra`

## 4) Системные зависимости

### 4.1 Linux/Docker (из текущего Dockerfile)

1. `libgl1`
2. `libglib2.0-0`
3. `tesseract-ocr` (не обязателен для Surya/Chandra режимов, но сейчас ставится в образ)

### 4.2 Windows

1. Рабочий Python 3.11
2. CUDA/GPU-драйверы по требованиям Surya/Chandra (если используется GPU)

## 5) Матрица зависимостей по финальным режимам

### `surya-surya`

Нужно:

1. Базовые зависимости пакета
2. `pypdf`, `reportlab`
3. `surya-ocr` + совместимые `transformers/tokenizers/huggingface-hub`

### `chandra-chandra`

Нужно:

1. Базовые зависимости пакета
2. `pypdf`, `reportlab`
3. `chandra-ocr[hf]` + совместимые `transformers/tokenizers/huggingface-hub`

### `chandra-surya` (гибрид)

Нужно одновременно:

1. Всё из `surya-surya`
2. Всё из `chandra-chandra`

## 6) Что можно выносить в архив при чистке

При сохранении runtime-контуров выше можно выносить из рабочего дерева:

1. `outputs/**` (кроме 1-2 эталонных run-наборов, если нужны для smoke-check)
2. Кэши: `.hf_cache`, `.surya_cache`, `.modelscope_cache`, `.uv_cache`, `.tmp*`, `pytest-cache-files-*`
3. Исторические одиночные скрипты в корне, не участвующие в текущем OCR-pipeline:
   - `camscan_hybrid_tool.py`
   - `fast.py`
   - `imgs_and_pdfs_ocr_fast_STABLE.py`
   - `only_tesseract.py`
   - `prepare pdf to tesseract.py`
   - `unified_pdf_tool.py`
4. Установщики/бинарники в корне (`naps2-7.5.3-win.exe`), если не используются процессом сборки
5. Экспериментальные внешние плагины в `OCRmypdf_plugins/**`, если не задействованы в финальном контуре

## 7) Быстрая проверка готовности среды

```powershell
.\.venv\Scripts\python.exe -m uniscan searchable-pdf --help
.\.venv\Scripts\python.exe -m uniscan serve-http --help
```

Проверка доступности движков:

```powershell
.\.venv\Scripts\python.exe -c "from uniscan.ocr import detect_ocr_engine_status as s; print('surya', s('surya')); print('chandra', s('chandra'))"
```

Если оба `ready=True`, финальные режимы GUI должны быть исполнимы.

## 8) Рекомендация перед глубокой чисткой

Сначала сделать рефакторинг `cli.py` на lazy-import `ui`/`tools`.
После этого можно будет существенно сократить обязательный runtime-набор и убрать лишние GUI/crop-зависимости из headless OCR-контура.
