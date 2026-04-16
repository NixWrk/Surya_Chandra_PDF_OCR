"""Reusable OCR workflow orchestration for desktop/web frontends."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
import uuid

from uniscan.ocr import (
    ArtifactSearchableResult,
    CompareTxtBuildResult,
    OcrBenchmarkResult,
    build_compare_txt_from_benchmark,
    detect_ocr_engine_status,
    run_artifact_searchable_package,
    run_ocr_benchmark,
)


DEFAULT_BASIC_GUI_LANG = "rus+eng"
MODE_SURYA = "surya"
MODE_HYBRID = "hybrid"
MODE_BOTH = "both"

PDF_MODE_CHANDRA = "chandra"
PDF_MODE_SURYA = "surya"
PDF_MODE_HYBRID = "chandra+surya"

MODE_TO_ENGINES: dict[str, tuple[str, ...]] = {
    MODE_SURYA: ("surya",),
    MODE_HYBRID: ("chandra",),
    MODE_BOTH: ("surya", "chandra"),
}

ProgressCallback = Callable[[int, str], None]


@dataclass(slots=True, frozen=True)
class BasicOcrRunSummary:
    run_dir: Path
    results: tuple[OcrBenchmarkResult, ...]
    result_files: tuple[Path, ...]
    failed_engines: tuple[str, ...]
    skipped_engines: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ChandraGeometryVariantsSummary:
    run_root: Path
    compare_dir: Path
    output_root: Path
    compare_results: tuple[CompareTxtBuildResult, ...]
    chandra_geometry_results: tuple[ArtifactSearchableResult, ...]
    surya_geometry_results: tuple[ArtifactSearchableResult, ...]


@dataclass(slots=True, frozen=True)
class SearchablePdfSummary:
    mode: str
    run_dir: Path
    compare_dir: Path
    output_pdf_path: Path
    output_pdf_bytes: bytes | None
    overwritten_input_path: Path | None
    benchmark: BasicOcrRunSummary
    compare_results: tuple[CompareTxtBuildResult, ...]
    artifact_results: tuple[ArtifactSearchableResult, ...]


def _emit_progress(cb: ProgressCallback | None, percent: int, status: str) -> None:
    if cb is None:
        return
    bounded = max(0, min(100, int(percent)))
    cb(bounded, status)


def _result_error_text(result: OcrBenchmarkResult) -> str:
    if result.error and result.error.strip():
        return result.error.strip()
    if result.note and result.note.strip():
        return result.note.strip()
    return "unknown error"


def _normalize_pdf_mode(raw: str | None) -> str:
    normalized = (raw or "").strip().lower()
    if not normalized:
        return PDF_MODE_HYBRID
    if normalized in {PDF_MODE_CHANDRA, PDF_MODE_SURYA, PDF_MODE_HYBRID}:
        return normalized
    if normalized == MODE_HYBRID:
        return PDF_MODE_HYBRID
    if normalized == MODE_SURYA:
        return PDF_MODE_SURYA
    if normalized == "both":
        return PDF_MODE_HYBRID
    raise ValueError(
        "Unsupported mode. Use one of: chandra, surya, chandra+surya."
    )


def _mode_to_benchmark_key(mode: str) -> str:
    if mode == PDF_MODE_CHANDRA:
        return MODE_HYBRID
    if mode == PDF_MODE_SURYA:
        return MODE_SURYA
    if mode == PDF_MODE_HYBRID:
        return MODE_BOTH
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _mode_to_prepare_engines(mode: str) -> tuple[str, ...]:
    if mode == PDF_MODE_CHANDRA:
        return ("chandra",)
    if mode == PDF_MODE_SURYA:
        return ("surya",)
    if mode == PDF_MODE_HYBRID:
        return ("chandra", "surya")
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _mode_to_build_engines(mode: str) -> tuple[str, ...]:
    if mode == PDF_MODE_CHANDRA:
        return ("chandra",)
    if mode == PDF_MODE_SURYA:
        return ("surya",)
    if mode == PDF_MODE_HYBRID:
        # Hybrid output is one PDF: chandra text aligned to surya geometry.
        return ("chandra",)
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _pick_ok_pdf(results: tuple[ArtifactSearchableResult, ...]) -> Path:
    for item in results:
        if item.status != "ok":
            continue
        raw = (item.searchable_pdf_path or "").strip()
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    raise RuntimeError("No successful searchable PDF output was produced.")


def run_basic_ocr_benchmark(
    *,
    pdf_path: Path,
    mode_key: str,
    page_numbers: tuple[int, ...] | None = None,
    lang: str = DEFAULT_BASIC_GUI_LANG,
    output_root: Path | None = None,
    progress: ProgressCallback | None = None,
) -> BasicOcrRunSummary:
    resolved_pdf = Path(pdf_path)
    if not resolved_pdf.exists() or not resolved_pdf.is_file():
        raise RuntimeError(f"PDF file not found: {resolved_pdf}")
    if resolved_pdf.suffix.lower() != ".pdf":
        raise RuntimeError("Only PDF input is supported.")

    selected_mode = mode_key.strip().lower()
    requested_engines = MODE_TO_ENGINES.get(selected_mode)
    if not requested_engines:
        known = ", ".join(sorted(MODE_TO_ENGINES))
        raise RuntimeError(f"Unknown mode '{mode_key}'. Supported: {known}.")

    ready_engines: list[str] = []
    skipped_engines: list[str] = []
    for engine in requested_engines:
        status = detect_ocr_engine_status(engine)
        if status.ready:
            ready_engines.append(engine)
            continue
        missing_deps = ", ".join(status.missing) if status.missing else "unknown"
        skipped_engines.append(f"{engine}: {missing_deps}")
    if not ready_engines:
        raise RuntimeError("No ready OCR engines:\n\n" + "\n".join(skipped_engines))

    run_root = Path(output_root) if output_root is not None else (Path.cwd() / "outputs" / "basic_gui_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (run_root / f"{resolved_pdf.stem}_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress, 0, "Preparing...")
    results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    failed_engines: list[str] = []
    total = max(1, len(ready_engines))
    sample_size = 999999 if page_numbers is None else max(len(page_numbers), 1)
    runtime_tmp = (Path.cwd() / ".tmp_runtime").resolve()
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    with _temporary_env("TEMP", str(runtime_tmp)), _temporary_env("TMP", str(runtime_tmp)):
        for index, engine in enumerate(ready_engines, start=1):
            start_percent = int(((index - 1) / total) * 100)
            end_percent = int((index / total) * 100)
            _emit_progress(progress, start_percent, f"Running: {engine}")

            engine_output = run_dir / engine
            engine_output.mkdir(parents=True, exist_ok=True)
            engine_results = run_ocr_benchmark(
                pdf_path=resolved_pdf,
                output_dir=engine_output,
                engines=(engine,),
                sample_size=sample_size,
                page_numbers=page_numbers,
                lang=lang,
            )
            if not engine_results:
                failed_engines.append(f"{engine}: benchmark returned no result")
                _emit_progress(progress, end_percent, f"Error: {engine}")
                continue

            result = engine_results[0]
            results.append(result)
            report_path = engine_output / f"{resolved_pdf.stem}_ocr_benchmark.json"
            if report_path.exists():
                result_files.append(report_path)
            if result.status != "ok":
                failed_engines.append(f"{engine}: {_result_error_text(result)}")
                _emit_progress(progress, end_percent, f"Error: {engine}")
                continue
            _emit_progress(progress, end_percent, f"Done: {engine}")

    if len(failed_engines) >= len(ready_engines):
        details = "\n\n".join(failed_engines)
        raise RuntimeError(f"No engine completed successfully.\n\n{details}")

    _emit_progress(progress, 100, "Completed")
    return BasicOcrRunSummary(
        run_dir=run_dir,
        results=tuple(results),
        result_files=tuple(result_files),
        failed_engines=tuple(failed_engines),
        skipped_engines=tuple(skipped_engines),
    )


def build_searchable_pdf(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    mode: str = PDF_MODE_HYBRID,
    lang: str = DEFAULT_BASIC_GUI_LANG,
    page_numbers: tuple[int, ...] | None = None,
    work_root: Path | None = None,
    overwrite_input_path: bool = True,
    return_bytes: bool | None = None,
    strict: bool = True,
    progress: ProgressCallback | None = None,
    delete_original_text_layer: bool = False,
) -> SearchablePdfSummary:
    """Build one searchable PDF from file-path or in-memory PDF input."""
    if (pdf_path is None and pdf_bytes is None) or (pdf_path is not None and pdf_bytes is not None):
        raise ValueError("Provide exactly one input: pdf_path or pdf_bytes.")

    normalized_mode = _normalize_pdf_mode(mode)
    benchmark_mode_key = _mode_to_benchmark_key(normalized_mode)
    prepare_engines = _mode_to_prepare_engines(normalized_mode)
    build_engines = _mode_to_build_engines(normalized_mode)

    resolved_work_root = Path(work_root) if work_root is not None else (Path.cwd() / "outputs" / "service_runs")
    resolved_work_root.mkdir(parents=True, exist_ok=True)

    input_path: Path
    if pdf_path is not None:
        input_path = Path(pdf_path).resolve()
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")
    else:
        if pdf_bytes is None or len(pdf_bytes) == 0:
            raise ValueError("Input pdf_bytes is empty.")
        staged_dir = (resolved_work_root / f"inline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}").resolve()
        staged_dir.mkdir(parents=True, exist_ok=True)
        input_path = staged_dir / "input.pdf"
        input_path.write_bytes(pdf_bytes)

    _emit_progress(progress, 2, "OCR benchmarking...")
    benchmark = run_basic_ocr_benchmark(
        pdf_path=input_path,
        mode_key=benchmark_mode_key,
        page_numbers=page_numbers,
        lang=lang,
        output_root=resolved_work_root,
        progress=progress,
    )

    compare_dir = benchmark.run_dir / "_compare_txt"
    _emit_progress(progress, 78, "Preparing compare artifacts...")
    compare_results = tuple(
        build_compare_txt_from_benchmark(
            benchmark_root=benchmark.run_dir,
            output_dir=compare_dir,
            engines=prepare_engines,
        )
    )
    if strict:
        _ensure_ok(compare_results, step="prepare-compare-txt")

    output_root = benchmark.run_dir / "searchable_pdf_final"
    _emit_progress(progress, 86, "Building searchable PDF...")
    if normalized_mode == PDF_MODE_HYBRID:
        geometry_override_dir = benchmark.run_dir / "surya"
    else:
        geometry_override_dir = None

    with _temporary_env(
        "UNISCAN_CHANDRA_GEOMETRY_DIR",
        str(geometry_override_dir) if geometry_override_dir is not None else None,
    ):
        artifact_results = tuple(
            run_artifact_searchable_package(
                compare_dir=compare_dir,
                pdf_root=input_path.parent,
                output_dir=output_root,
                engines=build_engines,
                require_page_markers=True,
            )
        )
    if strict:
        _ensure_ok(artifact_results, step="build-searchable-from-artifacts")
    produced_pdf = _pick_ok_pdf(artifact_results)

    overwritten_path: Path | None = None
    final_pdf_path = produced_pdf
    if pdf_path is not None and overwrite_input_path:
        # Если пользователь выбрал удаление исходного текстового слоя,
        # то полностью заменяем PDF независимо от того, какие страницы выбраны
        if delete_original_text_layer:
            # Полная замена: копируем новый PDF поверх старого
            shutil.copy2(produced_pdf, input_path)
            overwritten_path = input_path
            final_pdf_path = input_path
        else:
            # Старое поведение: обновляем только указанные страницы
            shutil.copy2(produced_pdf, input_path)
            overwritten_path = input_path
            final_pdf_path = input_path

    if return_bytes is None:
        need_bytes = pdf_bytes is not None
    else:
        need_bytes = bool(return_bytes)
    output_bytes = final_pdf_path.read_bytes() if need_bytes else None

    _emit_progress(progress, 100, "Done")
    return SearchablePdfSummary(
        mode=normalized_mode,
        run_dir=benchmark.run_dir,
        compare_dir=compare_dir,
        output_pdf_path=final_pdf_path,
        output_pdf_bytes=output_bytes,
        overwritten_input_path=overwritten_path,
        benchmark=benchmark,
        compare_results=compare_results,
        artifact_results=artifact_results,
    )


@contextmanager
def _temporary_env(name: str, value: str | None):
    had_old = name in os.environ
    old_value = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if had_old and old_value is not None:
            os.environ[name] = old_value
        else:
            os.environ.pop(name, None)


def _ensure_ok(results: tuple[CompareTxtBuildResult, ...] | tuple[ArtifactSearchableResult, ...], *, step: str) -> None:
    errors = [item for item in results if getattr(item, "status", "") != "ok"]
    if not errors:
        return
    preview = "; ".join(
        f"{getattr(item, 'document', 'doc')}[{getattr(item, 'engine', 'engine')}]: {getattr(item, 'error', '') or getattr(item, 'note', '') or 'failed'}"
        for item in errors[:3]
    )
    raise RuntimeError(f"{step} failed: {preview}")


def build_chandra_geometry_variants(
    *,
    run_root: Path,
    pdf_root: Path,
    output_root: Path | None = None,
    strict: bool = True,
) -> ChandraGeometryVariantsSummary:
    resolved_run = Path(run_root).resolve()
    resolved_pdf_root = Path(pdf_root).resolve()
    if not resolved_run.exists():
        raise FileNotFoundError(f"Run root not found: {resolved_run}")
    if not resolved_pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {resolved_pdf_root}")

    compare_dir = resolved_run / "_compare_txt"
    compare_results = tuple(
        build_compare_txt_from_benchmark(
            benchmark_root=resolved_run,
            output_dir=compare_dir,
            engines=("chandra", "surya"),
        )
    )
    if strict:
        _ensure_ok(compare_results, step="prepare-compare-txt")

    target_root = output_root if output_root is not None else (resolved_run / "searchable_pdf_geometry_compare")
    resolved_output_root = Path(target_root).resolve()
    output_chandra = resolved_output_root / "chandra_text__chandra_geometry"
    output_surya = resolved_output_root / "chandra_text__surya_geometry"

    chandra_results = tuple(
        run_artifact_searchable_package(
            compare_dir=compare_dir,
            pdf_root=resolved_pdf_root,
            output_dir=output_chandra,
            engines=("chandra",),
            require_page_markers=True,
        )
    )
    if strict:
        _ensure_ok(chandra_results, step="build chandra_text__chandra_geometry")

    surya_geometry_dir = resolved_run / "surya"
    with _temporary_env("UNISCAN_CHANDRA_GEOMETRY_DIR", str(surya_geometry_dir)):
        surya_results = tuple(
            run_artifact_searchable_package(
                compare_dir=compare_dir,
                pdf_root=resolved_pdf_root,
                output_dir=output_surya,
                engines=("chandra",),
                require_page_markers=True,
            )
        )
    if strict:
        _ensure_ok(surya_results, step="build chandra_text__surya_geometry")

    return ChandraGeometryVariantsSummary(
        run_root=resolved_run,
        compare_dir=compare_dir,
        output_root=resolved_output_root,
        compare_results=compare_results,
        chandra_geometry_results=chandra_results,
        surya_geometry_results=surya_results,
    )
