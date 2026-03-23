"""Build a human-readable OCR comparison package in ./outputs.

Supports two input formats:
- **Canonical run** (``canonical_summary.json`` present): uses per-page
  ``canonical/{engine}/page_NNNN.txt`` files directly and copies source page
  images from ``source_pages/``.
- **Matrix run** (``summary.json`` present): extracts text from per-engine
  ``.pdf`` / ``.txt`` artifacts as before.

Outputs
-------
- ``texts/{engine}/page_NN.txt``  (one file per page per engine, canonical mode)
- ``texts/{engine}.txt``          (single file per engine, matrix mode)
- ``source_pages/``               (copied from canonical run when available)
- ``ocr_comparison_report.md``    (Markdown summary)
- ``ocr_comparison_report.html``  (self-contained side-by-side HTML)
- ``ocr_comparison_summary.json`` (machine-readable metadata)
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def _load_summary(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported summary payload format: {type(payload)!r}")


def _safe_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "run"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> str:
    fitz_error: Exception | None = None
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        try:
            parts = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        text = "\n".join(part for part in parts if part)
        if text.strip():
            return text
    except Exception as exc:  # pragma: no cover
        fitz_error = exc

    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as exc:  # pragma: no cover
        if fitz_error is not None:
            raise RuntimeError(
                f"PDF text extraction failed via fitz ({fitz_error}) and pypdf ({exc})."
            ) from exc
        raise RuntimeError(f"PDF text extraction failed via pypdf: {exc}") from exc


def _extract_text(artifact_path: Path) -> str:
    suffix = artifact_path.suffix.lower()
    if suffix == ".txt":
        return artifact_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(artifact_path)
    raise RuntimeError(f"Unsupported artifact extension: {artifact_path.suffix}")


# ---------------------------------------------------------------------------
# Anomaly classification
# ---------------------------------------------------------------------------

_EMPTY_THRESHOLD = 0          # chars per page
_SUSPICIOUS_LOW = 10          # very few chars
_SUSPICIOUS_HIGH = 100_000    # suspiciously many chars (e.g. MinerU JSON dump)


def _classify_chars(chars: int) -> str:
    """Return 'empty', 'suspicious', or 'ok'."""
    if chars <= _EMPTY_THRESHOLD:
        return "empty"
    if chars < _SUSPICIOUS_LOW or chars > _SUSPICIOUS_HIGH:
        return "suspicious"
    return "ok"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _normalize_for_similarity(text: str, max_chars: int = 200_000) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:max_chars]


def _pairwise_similarity(engine_texts: dict[str, str]) -> list[tuple[str, str, float]]:
    engines = sorted(engine_texts.keys())
    result: list[tuple[str, str, float]] = []
    for idx, left in enumerate(engines):
        left_text = _normalize_for_similarity(engine_texts[left])
        for right in engines[idx + 1 :]:
            right_text = _normalize_for_similarity(engine_texts[right])
            ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
            result.append((left, right, ratio))
    return result


# ---------------------------------------------------------------------------
# Canonical mode helpers
# ---------------------------------------------------------------------------

def _load_canonical_per_page(
    source_root: Path,
    summary_rows: list[dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    """Return {engine: [(page_name, text), ...]} from canonical/{engine}/page_NNNN.txt."""
    engine_pages: dict[str, list[tuple[str, str]]] = {}
    canonical_root = source_root / "canonical"
    if not canonical_root.is_dir():
        return engine_pages
    for row in summary_rows:
        engine = str(row.get("engine", "")).strip()
        if not engine:
            continue
        engine_dir = canonical_root / engine
        if not engine_dir.is_dir():
            continue
        pages: list[tuple[str, str]] = []
        for txt_file in sorted(engine_dir.glob("page_*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            pages.append((txt_file.stem, text))
        if pages:
            engine_pages[engine] = pages
    return engine_pages


def _copy_source_pages(source_root: Path, dest_dir: Path) -> list[Path]:
    """Copy source_pages/*.png from canonical run into dest_dir. Returns copied paths."""
    src_pages_dir = source_root / "source_pages"
    copied: list[Path] = []
    if not src_pages_dir.is_dir():
        return copied
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(src_pages_dir.glob("page_*.png")):
        dst = dest_dir / img.name
        shutil.copy2(img, dst)
        copied.append(dst)
    return copied


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_FLAG_COLORS: dict[str, str] = {
    "empty": "#ffcccc",
    "suspicious": "#fff3cc",
    "ok": "#ccffcc",
}

_FLAG_LABELS: dict[str, str] = {
    "empty": "⛔ empty",
    "suspicious": "⚠ suspicious",
    "ok": "✓ ok",
}


def _image_to_data_uri(img_path: Path) -> str:
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_html_report(
    *,
    run_dir: Path,
    source_root: Path,
    is_canonical: bool,
    engines: list[str],
    # canonical mode: {engine: [(page_stem, text)]}
    engine_pages: dict[str, list[tuple[str, str]]],
    # matrix mode: {engine: full_text}
    engine_texts: dict[str, str],
    source_page_images: list[Path],
    summary_rows: list[dict[str, Any]],
    similarity_rows: list[tuple[str, str, float]],
) -> str:
    esc = _html_escape

    engine_cols = "".join(f"<th>{esc(e)}</th>" for e in engines)

    # Build per-engine summary row
    def _status_badge(row: dict[str, Any]) -> str:
        status = str(row.get("status", ""))
        color = "#ccffcc" if status == "ok" else "#ffcccc"
        return f'<span style="background:{color};padding:2px 6px;border-radius:3px">{esc(status)}</span>'

    summary_trs = ""
    for row in summary_rows:
        engine = str(row.get("engine", ""))
        elapsed = row.get("elapsed_seconds", "")
        chars = row.get("text_chars", "")
        mem = row.get("memory_delta_mb", "")
        summary_trs += (
            f"<tr><td>{esc(engine)}</td><td>{_status_badge(row)}</td>"
            f"<td>{esc(str(elapsed))}</td><td>{esc(str(chars))}</td>"
            f"<td>{esc(str(mem))}</td></tr>\n"
        )

    # Similarity table
    sim_trs = ""
    for left, right, ratio in sorted(similarity_rows, key=lambda x: x[2], reverse=True):
        color = "#ccffcc" if ratio > 0.5 else ("#fff3cc" if ratio > 0.2 else "#ffcccc")
        sim_trs += (
            f"<tr><td>{esc(left)}</td><td>{esc(right)}</td>"
            f'<td style="background:{color}">{ratio:.4f}</td></tr>\n'
        )

    # Per-page sections
    page_sections = ""
    if is_canonical and engine_pages:
        # collect all page stems in order
        all_stems: list[str] = []
        for pages in engine_pages.values():
            for stem, _ in pages:
                if stem not in all_stems:
                    all_stems.append(stem)
        all_stems.sort()

        for page_idx, stem in enumerate(all_stems):
            # source image
            img_tag = ""
            if page_idx < len(source_page_images):
                img_path = source_page_images[page_idx]
                if img_path.exists():
                    uri = _image_to_data_uri(img_path)
                    img_tag = f'<img src="{uri}" style="max-width:300px;max-height:420px;border:1px solid #ccc">'
                else:
                    img_tag = "<em>image not found</em>"
            else:
                img_tag = "<em>no image</em>"

            cells = f'<td valign="top" style="min-width:220px">{img_tag}<br><small>{esc(stem)}</small></td>'
            for engine in engines:
                pages_list = engine_pages.get(engine, [])
                text = next((t for s, t in pages_list if s == stem), "")
                chars = len(text)
                flag = _classify_chars(chars)
                color = _FLAG_COLORS[flag]
                label = _FLAG_LABELS[flag]
                preview = esc(text[:800]) if text.strip() else "<em>—</em>"
                cells += (
                    f'<td valign="top" style="background:{color};min-width:220px;max-width:320px;'
                    f'padding:6px;font-size:12px;white-space:pre-wrap">'
                    f'<strong style="font-size:11px">{label} ({chars} chars)</strong><br>{preview}</td>'
                )

            page_sections += (
                f"<h3 style='margin-top:24px'>{esc(stem)}</h3>"
                f'<div style="overflow-x:auto"><table border="1" cellpadding="4" cellspacing="0" '
                f'style="border-collapse:collapse;font-family:monospace">'
                f'<tr><th>Source</th>{engine_cols}</tr>'
                f"<tr>{cells}</tr></table></div>\n"
            )
    else:
        # matrix mode: single snippet per engine
        cells = '<td valign="top"><em>source image not available in matrix mode</em></td>'
        for engine in engines:
            text = engine_texts.get(engine, "")
            chars = len(text)
            flag = _classify_chars(chars)
            color = _FLAG_COLORS[flag]
            label = _FLAG_LABELS[flag]
            preview = esc(text[:1200]) if text.strip() else "<em>—</em>"
            cells += (
                f'<td valign="top" style="background:{color};min-width:220px;max-width:320px;'
                f'padding:6px;font-size:12px;white-space:pre-wrap">'
                f'<strong style="font-size:11px">{label} ({chars} chars)</strong><br>{preview}</td>'
            )
        page_sections = (
            "<h3>Text Snippets (Matrix Mode)</h3>"
            f'<div style="overflow-x:auto"><table border="1" cellpadding="4" cellspacing="0" '
            f'style="border-collapse:collapse;font-family:monospace">'
            f'<tr><th>Source</th>{engine_cols}</tr>'
            f"<tr>{cells}</tr></table></div>\n"
        )

    mode_badge = "canonical" if is_canonical else "matrix"
    generated_at = datetime.now().isoformat(timespec="seconds")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Comparison Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background: #f9f9f9; }}
  h1, h2, h3 {{ color: #333; }}
  table {{ font-size: 13px; }}
  th {{ background: #ddd; padding: 6px 10px; }}
  td {{ padding: 4px 8px; }}
  .meta {{ color: #666; font-size: 12px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>OCR Comparison Report</h1>
<p class="meta">Generated: {esc(generated_at)} &nbsp;|&nbsp;
Mode: <strong>{esc(mode_badge)}</strong> &nbsp;|&nbsp;
Source: <code>{esc(str(source_root))}</code></p>

<h2>Engine Summary</h2>
<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse">
  <tr><th>Engine</th><th>Status</th><th>Elapsed (s)</th><th>Text chars</th><th>Memory ΔMB</th></tr>
  {summary_trs}
</table>

<h2>Pairwise Similarity</h2>
{'<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse"><tr><th>Engine A</th><th>Engine B</th><th>Similarity</th></tr>' + sim_trs + '</table>' if sim_trs else '<p><em>Not enough engines for comparison.</em></p>'}

<h2>Per-Page Comparison</h2>
{page_sections}

</body>
</html>
"""
    return html


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_markdown_report(
    *,
    run_dir: Path,
    source_root: Path,
    is_canonical: bool,
    summary_rows: list[dict[str, Any]],
    extracted_rows: list[dict[str, Any]],
    similarity_rows: list[tuple[str, str, float]],
    page_chars_table: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# OCR Comparison Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Mode: `{'canonical' if is_canonical else 'matrix'}`")
    lines.append(f"- Source run: `{source_root}`")
    lines.append(f"- Output bundle: `{run_dir}`")
    lines.append("")

    lines.append("## Engine Summary")
    lines.append("")
    lines.append("| Engine | Status | Elapsed (s) | Text chars | Memory delta (MB) | Artifact |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| {engine} | {status} | {elapsed} | {chars} | {mem} | `{artifact}` |".format(
                engine=row.get("engine", ""),
                status=row.get("status", ""),
                elapsed=row.get("elapsed_seconds", ""),
                chars=row.get("text_chars", ""),
                mem=row.get("memory_delta_mb", ""),
                artifact=row.get("artifact_path", row.get("searchable_pdf_path", "")),
            )
        )
    lines.append("")

    lines.append("## Extracted Text Files")
    lines.append("")
    lines.append("| Engine | Extracted chars | Flag | Text file | Note |")
    lines.append("|---|---:|---|---|---|")
    for row in extracted_rows:
        flag = _classify_chars(row["extracted_chars"])
        lines.append(
            "| {engine} | {chars} | {flag} | `{text_file}` | {note} |".format(
                engine=row["engine"],
                chars=row["extracted_chars"],
                flag=flag,
                text_file=row["text_file"],
                note=row["note"] or "",
            )
        )
    lines.append("")

    if page_chars_table:
        lines.append("## Per-Page Character Counts")
        lines.append("")
        engines_in_table = [k for k in page_chars_table[0] if k != "page"]
        header = "| Page | " + " | ".join(engines_in_table) + " |"
        sep = "|---|" + "---:|" * len(engines_in_table)
        lines.append(header)
        lines.append(sep)
        for row in page_chars_table:
            cells = [str(row.get(e, 0)) for e in engines_in_table]
            lines.append(f"| {row['page']} | " + " | ".join(cells) + " |")
        lines.append("")

    if similarity_rows:
        lines.append("## Pairwise Similarity (normalized text)")
        lines.append("")
        lines.append("| Engine A | Engine B | Similarity |")
        lines.append("|---|---|---:|")
        for left, right, ratio in sorted(similarity_rows, key=lambda item: item[2], reverse=True):
            lines.append(f"| {left} | {right} | {ratio:.4f} |")
        lines.append("")

    lines.append("## Snippets")
    lines.append("")
    for row in extracted_rows:
        lines.append(f"### {row['engine']}")
        lines.append("")
        lines.append(f"Source: `{row['artifact']}`")
        lines.append("")
        snippet = row["snippet"].strip()
        if snippet:
            lines.append("```text")
            lines.append(snippet)
            lines.append("```")
        else:
            lines.append("_No text extracted._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Package OCR comparison report into ./outputs.")
    parser.add_argument(
        "--input-root",
        default=str(_repo_root() / "artifacts" / "ocr_latest_matrix_full_run_final"),
        help="Path to OCR matrix or canonical run folder.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_repo_root() / "outputs"),
        help="Target root for comparison bundles.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional custom output folder name.",
    )
    args = parser.parse_args()

    source_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    # Auto-detect canonical vs matrix mode
    canonical_summary_path = source_root / "canonical_summary.json"
    matrix_summary_path = source_root / "summary.json"

    is_canonical = canonical_summary_path.exists()
    summary_path = canonical_summary_path if is_canonical else matrix_summary_path

    if not summary_path.exists():
        raise SystemExit(
            f"No summary.json or canonical_summary.json found in: {source_root}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{source_root.name}_compare_{timestamp}"
    run_name = _safe_slug(args.run_name) if args.run_name.strip() else default_name
    run_dir = output_root / run_name
    results_copy_dir = run_dir / "results"
    texts_dir = run_dir / "texts"
    source_pages_dest = run_dir / "source_pages"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    texts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, results_copy_dir)

    summary_rows = _load_summary(summary_path)
    engines: list[str] = [str(row.get("engine", "")).strip() for row in summary_rows if row.get("engine")]

    # Copy source page images (canonical only)
    source_page_images: list[Path] = []
    if is_canonical:
        source_page_images = _copy_source_pages(source_root, source_pages_dest)

    # Per-page texts (canonical mode)
    engine_pages: dict[str, list[tuple[str, str]]] = {}
    page_chars_table: list[dict[str, Any]] = []

    if is_canonical:
        engine_pages = _load_canonical_per_page(source_root, summary_rows)
        # Write per-page txt files to texts/{engine}/page_NN.txt
        for engine, pages in engine_pages.items():
            engine_texts_dir = texts_dir / engine
            engine_texts_dir.mkdir(parents=True, exist_ok=True)
            for stem, text in pages:
                (engine_texts_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        # Build page × engine chars table
        all_stems: list[str] = []
        for pages in engine_pages.values():
            for stem, _ in pages:
                if stem not in all_stems:
                    all_stems.append(stem)
        all_stems.sort()
        for stem in all_stems:
            row: dict[str, Any] = {"page": stem}
            for engine in engines:
                pages_list = engine_pages.get(engine, [])
                text = next((t for s, t in pages_list if s == stem), "")
                row[engine] = len(text)
            page_chars_table.append(row)

    # Matrix mode: extract full text per engine
    extracted_rows: list[dict[str, Any]] = []
    engine_texts: dict[str, str] = {}

    for row in summary_rows:
        engine = str(row.get("engine", "")).strip() or "unknown"
        note = ""
        extracted_text = ""

        if is_canonical:
            # Use per-page texts concatenated for snippet / similarity
            pages_list = engine_pages.get(engine, [])
            extracted_text = "\n\n".join(t for _, t in pages_list)
            text_file_str = str((texts_dir / engine).relative_to(run_dir)) + "/"
        else:
            artifact_rel = str(row.get("artifact_path", "")).strip()
            artifact_path = Path(artifact_rel)
            if artifact_rel and not artifact_path.is_absolute():
                artifact_path = _repo_root() / artifact_path

            text_output_path = texts_dir / f"{_safe_slug(engine)}.txt"
            text_file_str = str(text_output_path.relative_to(run_dir))

            if not artifact_rel:
                note = "no artifact path"
            elif not artifact_path.exists():
                note = f"artifact missing: {artifact_path}"
            else:
                try:
                    extracted_text = _extract_text(artifact_path)
                except Exception as exc:
                    note = f"extract failed: {exc}"

            text_output_path.write_text(extracted_text, encoding="utf-8")

        if extracted_text:
            engine_texts[engine] = extracted_text

        extracted_rows.append(
            {
                "engine": engine,
                "artifact": str(row.get("artifact_path", row.get("searchable_pdf_path", ""))),
                "text_file": text_file_str,
                "extracted_chars": len(extracted_text),
                "note": note,
                "snippet": extracted_text[:1200],
            }
        )

    similarity_rows = _pairwise_similarity(engine_texts)

    # Write Markdown report
    report_md = _render_markdown_report(
        run_dir=run_dir,
        source_root=source_root,
        is_canonical=is_canonical,
        summary_rows=summary_rows,
        extracted_rows=extracted_rows,
        similarity_rows=similarity_rows,
        page_chars_table=page_chars_table,
    )
    report_md_path = run_dir / "ocr_comparison_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")

    # Write HTML report
    report_html = _render_html_report(
        run_dir=run_dir,
        source_root=source_root,
        is_canonical=is_canonical,
        engines=engines,
        engine_pages=engine_pages,
        engine_texts=engine_texts,
        source_page_images=source_page_images,
        summary_rows=summary_rows,
        similarity_rows=similarity_rows,
    )
    report_html_path = run_dir / "ocr_comparison_report.html"
    report_html_path.write_text(report_html, encoding="utf-8")

    # Write summary JSON
    metadata = {
        "source_root": str(source_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "canonical" if is_canonical else "matrix",
        "run_dir": str(run_dir),
        "summary_rows": summary_rows,
        "extracted_rows": [
            {
                "engine": row["engine"],
                "artifact": row["artifact"],
                "text_file": row["text_file"],
                "extracted_chars": row["extracted_chars"],
                "flag": _classify_chars(row["extracted_chars"]),
                "note": row["note"],
            }
            for row in extracted_rows
        ],
        "page_chars_table": page_chars_table,
        "pairwise_similarity": [
            {"engine_a": left, "engine_b": right, "ratio": ratio}
            for left, right, ratio in similarity_rows
        ],
    }
    (run_dir / "ocr_comparison_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Comparison bundle: {run_dir}")
    print(f"Report (MD):       {report_md_path}")
    print(f"Report (HTML):     {report_html_path}")
    print(f"Results copy:      {results_copy_dir}")
    print(f"Texts:             {texts_dir}")
    if source_page_images:
        print(f"Source pages:      {source_pages_dest}  ({len(source_page_images)} images)")
    if page_chars_table:
        print(f"\nPer-page char counts ({len(page_chars_table)} pages × {len(engines)} engines):")
        header = f"{'Page':<16}" + "".join(f"{e:>14}" for e in engines)
        print(header)
        for row in page_chars_table:
            cells = "".join(f"{row.get(e, 0):>14}" for e in engines)
            print(f"{row['page']:<16}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
