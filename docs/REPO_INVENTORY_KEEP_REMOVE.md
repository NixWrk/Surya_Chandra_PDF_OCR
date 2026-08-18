# Repository Inventory: Keep/Remove Matrix

Last updated: 2026-04-20

Checkpoint reference:

`checkpoint/dual-venv-stable-20260420` (base commit `158698e`)

Status: historical cleanup record, not an active deletion checklist. The current
evidence-backed inventory is `docs/audit/REPOSITORY_INVENTORY.md`. Never delete
`outputs/`, `PDFs/`, model caches, retained incidents, resumable caches, or user
artifacts from this document alone; verify ownership and the current retention
policy first.

## Scope target

Target repository shape:

1. OCR pipeline only.
2. One production searchable-PDF mode: `chandra+surya`; single-engine modes are
   diagnostic/benchmark commands only.
3. Minimal GUI + CLI + optional HTTP entrypoint.

## Keep (required)

### Runtime code

1. `src/uniscan/app/*` (workflow orchestration)
2. `src/uniscan/ocr/*` (engines, benchmark, artifact-based build)
3. `src/uniscan/io/loaders.py` and `src/uniscan/io/__init__.py`
4. `src/uniscan/export/*` (PDF export helpers used by OCR flow)
5. `src/uniscan/ui/basic_ocr_gui.py` and `src/uniscan/ui/__init__.py`
6. `src/uniscan/web/*` (if HTTP mode is kept)
7. `src/uniscan/cli.py`, `src/uniscan/__main__.py`, `src/uniscan/__init__.py`

### Entrypoints and setup

1. `run_basic_gui.cmd`
2. `setup_dual_venv.cmd`
3. `pyproject.toml`
4. `pytest.ini`
5. `README.md`
6. `docs/*`

### Tests (current OCR scope)

1. `tests/test_app_searchable_pdf.py`
2. `tests/test_loaders.py`
3. `tests/test_ocr_artifact_searchable.py`
4. `tests/test_ocr_benchmark.py`
5. `tests/test_ocr_canonical.py`
6. `tests/test_ocr_engine.py`
7. `tests/test_ocr_preprocessing.py`
8. `tests/test_web_service.py` (if HTTP mode is kept)
9. `tests/conftest.py`

## Remove (safe immediate cleanup)

### Local runtime artifacts and caches

Only confirmed process-owned temporary files and regenerable tool caches may be
removed. Virtual environments, model caches, `outputs/`, `PDFs/`, `_run_exec/`,
retained failures, and resumable run data are not blanket cleanup targets. Use
the current runbooks and retention policy, never this historical list, before
deleting local data.

### Legacy metadata and local tool state

1. `.omc/`
2. `.claude/`
3. `CLAUDE.md`
4. `_source_filename_map.csv`

## Remove (after reference check in code/tests)

### Legacy UI/camera/session/storage/tooling modules

1. `src/uniscan/io/camera_service.py`
2. `src/uniscan/ui/app.py`
3. `src/uniscan/ui/camera_health.py`
4. `src/uniscan/ui/page_parse.py`
5. `src/uniscan/session/*`
6. `src/uniscan/storage/*`
7. `src/uniscan/tools/*`

### Legacy core adapters not needed in OCR-only target

1. `src/uniscan/core/geometry.py`
2. `src/uniscan/core/postprocess.py`
3. `src/uniscan/core/preprocess.py`
4. `src/uniscan/core/scanner_adapter.py`

### Legacy scripts replaced by OCR pipeline entrypoints

1. `run_gui.bat`
2. `run_gui_app.bat`
3. `run_gui_detailed.bat`

## Verification checklist before physical deletion commit

1. `rg` search returns no imports of remove-candidate modules.
2. CLI commands for OCR pipeline still work:
   - `benchmark-ocr`
   - `prepare-compare-txt`
   - `build-searchable-from-artifacts`
   - `searchable-pdf`
3. GUI works in the production `chandra+surya` mode after cleanup.
4. Hybrid output still uses Chandra text + Surya geometry.
