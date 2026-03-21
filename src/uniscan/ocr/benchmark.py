"""OCR benchmark helpers for sampled PDF fixtures."""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from uniscan.io import imwrite_unicode, render_pdf_page_indices

from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_VALUES,
    SEARCHABLE_PDF_ENGINES,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)

_DEFAULT_PADDLE_CACHE_HOME = Path(__file__).resolve().parents[3] / ".paddlex_cache"


@dataclass(slots=True)
class OcrBenchmarkResult:
    engine: str
    status: str
    sample_pages: list[int]
    elapsed_seconds: float
    artifact_path: str | None
    text_chars: int
    error: str | None = None
    note: str | None = None

    @property
    def label(self) -> str:
        return OCR_ENGINE_LABELS.get(self.engine, self.engine)


def sample_pdf_page_indices(page_count: int, *, sample_size: int = 5) -> list[int]:
    """Pick first/middle/last page windows without loading the whole PDF."""
    if page_count <= 0:
        return []

    window = max(1, int(sample_size))
    if page_count <= window:
        return list(range(page_count))

    indices: set[int] = set(range(0, min(window, page_count)))

    midpoint = page_count // 2
    mid_start = max(0, midpoint - window // 2)
    mid_end = min(page_count, mid_start + window)
    mid_start = max(0, mid_end - window)
    indices.update(range(mid_start, mid_end))

    last_start = max(0, page_count - window)
    indices.update(range(last_start, page_count))

    return sorted(indices)


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _collect_text_strings(value: Any) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        try:
            return [value.decode("utf-8", errors="ignore")]
        except Exception:
            return []
    if isinstance(value, dict):
        for item in value.values():
            texts.extend(_collect_text_strings(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_collect_text_strings(item))
        return texts

    for attr in ("text", "rec_text", "transcription", "content", "label"):
        if hasattr(value, attr):
            texts.extend(_collect_text_strings(getattr(value, attr)))
    return texts


def _paddleocr_lang(lang: str) -> str:
    """Map OCR language codes to PaddleOCR language identifiers."""
    normalized = lang.strip().lower()
    if normalized in {"eng", "en", "english"}:
        return "en"
    return normalized


def _render_sample_paths(
    pdf_path: Path,
    sample_pages: Sequence[int],
    *,
    dpi: int,
    tmp_dir: Path,
) -> list[Path]:
    rendered = render_pdf_page_indices(pdf_path, sample_pages, dpi=dpi)
    image_paths: list[Path] = []
    for idx, (_name, image) in enumerate(rendered, start=1):
        out_path = tmp_dir / f"{idx:05d}.png"
        if not imwrite_unicode(out_path, image):
            raise RuntimeError(f"Failed to write sampled page image: {out_path}")
        image_paths.append(out_path)
    return image_paths


def _extract_pdf_text_chars(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    try:
        total = 0
        for page in doc:
            total += len(page.get_text("text"))
        return total
    finally:
        doc.close()


def _run_paddleocr_direct(image_paths: Sequence[Path], *, lang: str) -> tuple[str, int]:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_DEFAULT_PADDLE_CACHE_HOME))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang=_paddleocr_lang(lang),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    collected: list[str] = []
    for path in image_paths:
        result = ocr.ocr(str(path))
        collected.extend(_collect_text_strings(result))

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _make_result(
    *,
    engine: str,
    status: str,
    sample_pages: Sequence[int],
    elapsed_seconds: float,
    artifact_path: Path | None,
    text_chars: int,
    error: str | None = None,
    note: str | None = None,
) -> OcrBenchmarkResult:
    return OcrBenchmarkResult(
        engine=engine,
        status=status,
        sample_pages=[page + 1 for page in sample_pages],
        elapsed_seconds=elapsed_seconds,
        artifact_path=None if artifact_path is None else str(artifact_path),
        text_chars=text_chars,
        error=error,
        note=note,
    )


def run_ocr_benchmark(
    *,
    pdf_path: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
    sample_size: int = 5,
    dpi: int = 160,
    lang: str = "eng",
) -> list[OcrBenchmarkResult]:
    """Run a sampled OCR benchmark against a PDF fixture."""
    resolved_pdf = Path(pdf_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    page_count = _pdf_page_count(resolved_pdf)
    sample_pages = sample_pdf_page_indices(page_count, sample_size=sample_size)
    if not sample_pages:
        raise ValueError("No PDF pages available for OCR benchmark.")

    selected_engines = tuple(engines) if engines is not None else OCR_ENGINE_VALUES
    results: list[OcrBenchmarkResult] = []

    with tempfile.TemporaryDirectory(prefix="uniscan_ocr_benchmark_") as tmp:
        tmp_dir = Path(tmp)
        sampled_image_paths = _render_sample_paths(
            resolved_pdf,
            sample_pages,
            dpi=dpi,
            tmp_dir=tmp_dir,
        )

        for engine in selected_engines:
            start = perf_counter()
            engine_status = detect_ocr_engine_status(engine)
            if not engine_status.ready:
                elapsed = perf_counter() - start
                results.append(
                    _make_result(
                        engine=engine,
                        status="skipped",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=None,
                        text_chars=0,
                        note=f"missing: {', '.join(engine_status.missing) if engine_status.missing else 'unknown'}",
                    )
                )
                continue

            if engine in SEARCHABLE_PDF_ENGINES:
                artifact_path = resolved_output / f"{resolved_pdf.stem}_{engine}.pdf"
                try:
                    output_pdf = image_paths_to_searchable_pdf(
                        sampled_image_paths,
                        out_pdf=artifact_path,
                        lang=lang,
                        engine_name=engine,
                    )
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="ok",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=output_pdf,
                            text_chars=_extract_pdf_text_chars(output_pdf),
                        )
                    )
                except Exception as exc:
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="error",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=artifact_path,
                            text_chars=0,
                            error=str(exc),
                        )
                    )
                continue

            if engine == OCR_ENGINE_PADDLEOCR:
                artifact_path = resolved_output / f"{resolved_pdf.stem}_{engine}.txt"
                try:
                    text, text_chars = _run_paddleocr_direct(sampled_image_paths, lang=lang)
                    artifact_path.write_text(text, encoding="utf-8")
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="ok",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=artifact_path,
                            text_chars=text_chars,
                        )
                    )
                except Exception as exc:
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="error",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=artifact_path,
                            text_chars=0,
                            error=str(exc),
                        )
                    )
                continue

            elapsed = perf_counter() - start
            results.append(
                _make_result(
                    engine=engine,
                    status="skipped",
                    sample_pages=sample_pages,
                    elapsed_seconds=elapsed,
                    artifact_path=None,
                    text_chars=0,
                    note="not wired for benchmark yet",
                )
            )

    report_path = resolved_output / f"{resolved_pdf.stem}_ocr_benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "pdf_path": str(resolved_pdf),
                "page_count": page_count,
                "sample_pages": [page + 1 for page in sample_pages],
                "results": [asdict(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return results


def summarize_ocr_benchmark(results: Sequence[OcrBenchmarkResult]) -> str:
    """Format a concise human-readable benchmark summary."""
    lines: list[str] = []
    for result in results:
        if result.status == "ok":
            lines.append(
                f"{result.engine}: ok {result.elapsed_seconds:.2f}s "
                f"text={result.text_chars} artifact={result.artifact_path}"
            )
            continue
        if result.status == "skipped":
            lines.append(
                f"{result.engine}: skipped {result.elapsed_seconds:.2f}s "
                f"{result.note or 'no note'}"
            )
            continue
        lines.append(
            f"{result.engine}: error {result.elapsed_seconds:.2f}s "
            f"{result.error or 'unknown error'}"
        )
    return "\n".join(lines)
