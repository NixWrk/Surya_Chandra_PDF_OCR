"""Build searchable PDFs from existing OCR text artifacts (artifact-first mode)."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence


@dataclass(slots=True)
class ArtifactSearchableResult:
    document: str
    engine: str
    status: str
    source_pdf_path: str | None
    text_artifact_path: str
    searchable_pdf_path: str | None
    page_count: int
    text_chars: int
    elapsed_seconds: float
    error: str | None = None


_PAGE_MARKER_RE = re.compile(r"^\s*\[SOURCE PAGE\s+(\d+)\]\s*$", re.IGNORECASE)


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_artifact_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(
            f"Invalid artifact filename '{path.name}'. Expected '<document>__<engine>.txt'."
        )
    document, engine = stem.rsplit("__", 1)
    document = document.strip()
    engine = engine.strip().lower()
    if not document or not engine:
        raise ValueError(
            f"Invalid artifact filename '{path.name}'. Expected '<document>__<engine>.txt'."
        )
    return document, engine


def _split_text_to_pages(text: str, page_count: int) -> list[str]:
    if page_count <= 0:
        return []
    if page_count == 1:
        return [text]

    marker_pages: dict[int, list[str]] = {}
    current_page: int | None = None
    preamble: list[str] = []

    for line in text.splitlines():
        match = _PAGE_MARKER_RE.match(line)
        if match:
            current_page = int(match.group(1))
            marker_pages.setdefault(current_page, [])
            continue
        if current_page is None:
            preamble.append(line)
        else:
            marker_pages.setdefault(current_page, []).append(line)

    if marker_pages:
        pages: list[str] = []
        for page_idx in range(1, page_count + 1):
            if page_idx == 1 and preamble:
                source = preamble + marker_pages.get(page_idx, [])
            else:
                source = marker_pages.get(page_idx, [])
            pages.append("\n".join(source).strip())
        return pages

    # Many OCR tools emit form-feed as page separator.
    if "\f" in text:
        chunks = [chunk.strip() for chunk in text.split("\f")]
        if len(chunks) == page_count:
            return chunks

    lines = text.splitlines()
    if not lines:
        return [""] * page_count

    total_chars = max(len(text), 1)
    target_chars = max(total_chars // page_count, 1)
    pages: list[str] = []
    current_lines: list[str] = []
    current_chars = 0

    for idx, line in enumerate(lines):
        remaining_lines = len(lines) - idx
        remaining_pages = page_count - len(pages)
        can_finalize = (
            len(pages) < (page_count - 1)
            and current_lines
            and current_chars >= target_chars
            and remaining_lines >= (remaining_pages - 1)
        )
        if can_finalize:
            pages.append("\n".join(current_lines).strip())
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += len(line) + 1

    pages.append("\n".join(current_lines).strip())

    if len(pages) < page_count:
        pages.extend([""] * (page_count - len(pages)))
    elif len(pages) > page_count:
        overflow = pages[page_count - 1 :]
        pages = pages[: page_count - 1] + ["\n".join(part for part in overflow if part)]
    return pages


def _extract_pdf_text(pdf_path: Path) -> str:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    try:
        parts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(part for part in parts if part)


def _build_searchable_pdf_from_text(
    *,
    source_pdf: Path,
    text: str,
    out_pdf: Path,
) -> tuple[int, int]:
    import fitz  # type: ignore

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(source_pdf))
    try:
        page_count = int(doc.page_count)
        page_texts = _split_text_to_pages(text, page_count)
        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = page_texts[page_idx] if page_idx < len(page_texts) else ""
            if not page_text.strip():
                continue
            rect = page.rect
            text_rect = fitz.Rect(
                rect.x0 + 4.0,
                rect.y0 + 4.0,
                rect.x1 - 4.0,
                rect.y1 - 4.0,
            )
            # Render mode 3 = invisible text; keeps scan visually unchanged.
            page.insert_textbox(
                text_rect,
                page_text,
                fontsize=1.0,
                fontname="helv",
                render_mode=3,
                overlay=True,
            )

        doc.save(str(out_pdf), garbage=3, deflate=True)
        return page_count, len(text)
    finally:
        doc.close()


def run_artifact_searchable_package(
    *,
    compare_dir: Path,
    pdf_root: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
) -> list[ArtifactSearchableResult]:
    resolved_compare = Path(compare_dir)
    resolved_pdf_root = Path(pdf_root)
    resolved_output = Path(output_dir)

    if not resolved_compare.exists():
        raise FileNotFoundError(f"Compare dir not found: {resolved_compare}")
    if not resolved_pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {resolved_pdf_root}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    allowed_engines = None if engines is None else {engine.strip().lower() for engine in engines if engine.strip()}

    artifact_files = sorted(path for path in resolved_compare.glob("*.txt") if path.name.lower() != "sources_map.txt")

    pdf_index: dict[str, Path] = {}
    for pdf_path in resolved_pdf_root.rglob("*.pdf"):
        key = _normalize_key(pdf_path.stem)
        pdf_index.setdefault(key, pdf_path)

    results: list[ArtifactSearchableResult] = []

    for artifact_path in artifact_files:
        start = perf_counter()
        try:
            document, engine = _parse_artifact_filename(artifact_path)
        except Exception as exc:
            results.append(
                ArtifactSearchableResult(
                    document=artifact_path.stem,
                    engine="unknown",
                    status="error",
                    source_pdf_path=None,
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=str(exc),
                )
            )
            continue

        if allowed_engines is not None and engine not in allowed_engines:
            continue

        source_pdf = pdf_index.get(_normalize_key(document))
        if source_pdf is None:
            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="error",
                    source_pdf_path=None,
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=f"Source PDF not found in pdf_root for document '{document}'.",
                )
            )
            continue

        try:
            text = artifact_path.read_text(encoding="utf-8", errors="ignore")
            out_pdf = resolved_output / document / f"{document}__{engine}_searchable.pdf"
            page_count, text_chars = _build_searchable_pdf_from_text(
                source_pdf=source_pdf,
                text=text,
                out_pdf=out_pdf,
            )
            extracted = _extract_pdf_text(out_pdf)
            if not extracted.strip():
                raise RuntimeError("Output PDF has empty extracted text layer.")

            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="ok",
                    source_pdf_path=str(source_pdf),
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=str(out_pdf),
                    page_count=page_count,
                    text_chars=text_chars,
                    elapsed_seconds=perf_counter() - start,
                )
            )
        except Exception as exc:
            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="error",
                    source_pdf_path=str(source_pdf),
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=str(exc),
                )
            )

    summary_json = resolved_output / "artifact_searchable_summary.json"
    summary_csv = resolved_output / "artifact_searchable_summary.csv"
    summary_json.write_text(
        json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "document",
                "engine",
                "status",
                "source_pdf_path",
                "text_artifact_path",
                "searchable_pdf_path",
                "page_count",
                "text_chars",
                "elapsed_seconds",
                "error",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))

    return results


def summarize_artifact_searchable_package(results: Sequence[ArtifactSearchableResult]) -> str:
    lines: list[str] = []
    for row in results:
        if row.status == "ok":
            lines.append(
                f"{row.document} [{row.engine}]: ok {row.elapsed_seconds:.2f}s "
                f"pages={row.page_count} text={row.text_chars} pdf={row.searchable_pdf_path}"
            )
        else:
            lines.append(
                f"{row.document} [{row.engine}]: error {row.elapsed_seconds:.2f}s "
                f"{row.error or 'unknown error'}"
            )
    return "\n".join(lines)

