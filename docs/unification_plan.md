# Unified Scan Project Plan

## Goal

Build one application that combines the strongest functionality from:

- `camscan_hybrid_tool.py`
- `camscan_suhren/camscan/app.py`

Target capabilities:

1. Live camera preview and interactive capture
2. Batch camera capture (N shots + delay)
3. Import from folder, files, and PDF
4. Page/session management (preview, reorder, select, delete)
5. Export to merged PDF and separate image files
6. Processing controls (detect document, two-page split, postprocess, quality profile)
7. Background jobs with progress and cancellation
8. Optional OCR stage

## Architecture

Create a new root package:

- `src/uniscan/core`: processing pipeline + adapters
- `src/uniscan/io`: image/pdf/camera loaders
- `src/uniscan/session`: capture session model and operations
- `src/uniscan/export`: PDF and separate-file exporters
- `src/uniscan/ui`: unified CustomTkinter app
- `src/uniscan/ocr`: optional OCR integration

## Commit Roadmap

1. `chore(repo): add root pyproject, tooling, pytest config`
2. `chore(vendor): pin camscan source as tracked dependency and add THIRD_PARTY_NOTICES`
3. `refactor(core): create src/uniscan/core/postprocess.py and scanner_adapter.py`
4. `refactor(core): move image/pdf loading to src/uniscan/io/loaders.py`
5. `refactor(core): move processing pipeline to src/uniscan/core/pipeline.py`
6. `refactor(core): add camera service (live stream + burst capture) in src/uniscan/io/camera_service.py`
7. `feat(session): add CaptureSession model with add/remove/reorder/select`
8. `feat(ui): scaffold unified CTk app shell in src/uniscan/ui/app.py`
9. `feat(ui-camera): port live preview, free-capture mode, camera config dialog`
10. `feat(ui-import): add import folder/files/pdf + natural sorting + validation`
11. `feat(ui-pages): add thumbnails, full preview, select-all, delete, reorder`
12. `feat(ui-export): merged PDF export + separate files export + format selection`
13. `feat(ui-processing): quality profiles, detect toggle, two-page split, postprocess mode`
14. `feat(worker): background jobs, stage/progress bar, cancel support`
15. `feat(ocr): optional OCR stage (engine select, language, dependency checks)`
16. `chore(compat): convert old entrypoints to wrappers`
17. `test: add unit tests for loaders/pipeline/session/export and smoke UI test`
18. `docs(ci): rewrite README + add GitHub Actions for lint/test`

## Delivery Milestones

1. `v0.1`: commits 1-8 (single package and app skeleton)
2. `v0.2`: commits 9-14 (merged functionality from both projects)
3. `v0.3`: commits 15-18 (OCR and stabilization)
