#!/usr/bin/env python3
"""Generate and validate the offline UniScan synthetic benchmark corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
from pathlib import Path
from typing import Any


SCHEMA = "uniscan.synthetic-benchmark.v1"
GENERATOR_VERSION = "1.0.0"
SEED = 20260818

_TEXT_FONT = "cjk"
_FALLBACK_FONT = "helv"
_PAGE_WIDTH = 612.0
_PAGE_HEIGHT = 792.0
_MARGIN = 48.0
_LINE_HEIGHT = 22.0


def _fitz() -> Any:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - exercised on missing optional runtime
        raise RuntimeError("Synthetic corpus generation requires PyMuPDF.") from exc
    return fitz


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value), encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _fixed_metadata(fitz: Any, fixture_id: str) -> dict[str, str]:
    return {
        "format": "PDF 1.7",
        "title": f"UniScan synthetic fixture: {fixture_id}",
        "author": "UniScan synthetic benchmark",
        "subject": "Procedural OCR benchmark fixture",
        "keywords": "synthetic,ocr,benchmark,privacy-safe",
        "creator": f"uniscan.synthetic-benchmark/{GENERATOR_VERSION}",
        "producer": "uniscan.synthetic-benchmark",
        "creationDate": "D:20200101000000Z",
        "modDate": "D:20200101000000Z",
    }


def _insert_text(page: Any, point: tuple[float, float], text: str, *, size: float, color: tuple[float, float, float] = (0.08, 0.08, 0.08), render_mode: int = 0) -> str:
    """Insert Unicode text, preferring PyMuPDF's bundled CJK fallback."""
    for font_name in (_TEXT_FONT, _FALLBACK_FONT):
        try:
            page.insert_text(point, text, fontname=font_name, fontsize=size, color=color, render_mode=render_mode)
            return font_name
        except Exception:
            continue
    raise RuntimeError(f"PyMuPDF cannot render synthetic text: {text!r}")


def _line(text: str, *, line_id: str, order: int, y: float, x0: float = _MARGIN, x1: float = _PAGE_WIDTH - _MARGIN, language: str = "eng", region_type: str = "body") -> dict[str, Any]:
    return {
        "id": line_id,
        "order": order,
        "text": text,
        "language": language,
        "region_type": region_type,
        "bbox_norm": [round(x0 / _PAGE_WIDTH, 6), round((y - 16) / _PAGE_HEIGHT, 6), round(x1 / _PAGE_WIDTH, 6), round((y + 4) / _PAGE_HEIGHT, 6)],
    }


def _page_record(page_number: int, *, text: str = "", language: str = "eng", outcome: str = "text", features: list[str] | None = None, degradation: list[str] | None = None, rotation_deg: int = 0, native_text_layer: bool = False, lines: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "page": page_number,
        "text": text,
        "language": language,
        "expected_outcome": outcome,
        "features": list(features or []),
        "degradation": list(degradation or []),
        "rotation_deg": rotation_deg,
        "native_text_layer": native_text_layer,
        "lines": list(lines or []),
    }


def _draw_background(page: Any, *, color: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    page.draw_rect(page.rect, color=color, fill=color, width=0)


def _draw_text_page(page: Any, record: dict[str, Any], *, color: tuple[float, float, float] = (0.08, 0.08, 0.08), font_size: float = 14.0, x_shift: float = 0.0) -> None:
    y = 74.0
    for index, line in enumerate(record["lines"]):
        size = 18.0 if line["region_type"] == "heading" else font_size
        x0 = _MARGIN + x_shift
        _insert_text(page, (x0, y), str(line["text"]), size=size, color=color)
        y += _LINE_HEIGHT if index == 0 else 20.0


def _draw_two_columns(page: Any, record: dict[str, Any]) -> None:
    left = [line for line in record["lines"] if line["id"].startswith("left-")]
    right = [line for line in record["lines"] if line["id"].startswith("right-")]
    for column_index, lines in enumerate((left, right)):
        x = 48.0 + column_index * 270.0
        y = 92.0
        for line in lines:
            _insert_text(page, (x, y), str(line["text"]), size=13.0, color=(0.08, 0.08, 0.08))
            y += 25.0
    page.draw_line((306.0, 70.0), (306.0, 700.0), color=(0.7, 0.7, 0.7), width=0.5)


def _draw_table(page: Any, record: dict[str, Any]) -> None:
    x0, y0, cell_w, cell_h = 52.0, 420.0, 127.0, 34.0
    for row in range(3):
        for column in range(4):
            rect = _fitz().Rect(x0 + column * cell_w, y0 + row * cell_h, x0 + (column + 1) * cell_w, y0 + (row + 1) * cell_h)
            page.draw_rect(rect, color=(0.2, 0.2, 0.2), width=0.6)
            value = str(record["lines"][row * 4 + column]["text"])
            _insert_text(page, (rect.x0 + 8, rect.y0 + 22), value, size=11.0, color=(0.08, 0.08, 0.08))


def _draw_noise(page: Any, *, seed: int, low_contrast: bool = False) -> None:
    rng = random.Random(seed)
    color = (0.75, 0.75, 0.75) if low_contrast else (0.55, 0.55, 0.55)
    for _ in range(40):
        x = 36.0 + rng.randrange(540)
        y = 40.0 + rng.randrange(700)
        length = 2.0 + rng.randrange(9)
        page.draw_line((x, y), (x + length, y), color=color, width=0.25)


def _build_fixture_pdf(fixture_id: str, pages: list[dict[str, Any]], output: Path) -> None:
    fitz = _fitz()
    doc = fitz.open()
    try:
        doc.set_metadata(_fixed_metadata(fitz, fixture_id))
        for record in pages:
            page = doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
            outcome = str(record["expected_outcome"])
            if outcome == "verified_blank":
                _draw_background(page)
            elif outcome == "explicit_nontext":
                _draw_background(page, color=(0.96, 0.96, 0.96))
                page.draw_rect(fitz.Rect(100, 120, 510, 280), color=(0.15, 0.25, 0.55), fill=(0.65, 0.78, 0.95), width=2)
                page.draw_circle((306, 480), 90, color=(0.65, 0.2, 0.2), fill=(0.95, 0.75, 0.55), width=3)
                page.draw_line((100, 620), (510, 620), color=(0.15, 0.15, 0.15), width=4)
            else:
                _draw_background(page)
                features = set(record.get("features", []))
                if "columns" in features:
                    _draw_two_columns(page, record)
                elif "table" in features:
                    _draw_text_page(page, record, font_size=13.0)
                    _draw_table(page, record)
                else:
                    _draw_text_page(
                        page,
                        record,
                        color=(0.46, 0.46, 0.46) if "low-contrast" in features else (0.08, 0.08, 0.08),
                        font_size=13.0 if "low-resolution" in features else 14.0,
                        x_shift=3.0 if "skew" in features else 0.0,
                    )
                if "noise" in features:
                    _draw_noise(page, seed=SEED + int(record["page"]), low_contrast="low-contrast" in features)
                if bool(record.get("native_text_layer")):
                    # Deliberately invisible duplicate layer. Production mode must
                    # remove this layer before OCR rather than count it twice.
                    _insert_text(page, (48, 748), str(record["text"]), size=7.0, render_mode=3)
            if int(record.get("rotation_deg", 0)):
                page.set_rotation(int(record["rotation_deg"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output), garbage=4, deflate=True, clean=True, no_new_id=True)
    finally:
        doc.close()


def _clean_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


def _make_pages() -> dict[str, list[dict[str, Any]]]:
    en_lines = [
        _line("SYNTHETIC REPORT ALPHA", line_id="heading-1", order=1, y=74, region_type="heading"),
        _line("Invented text exercises searchable output and page identity.", line_id="body-1", order=2, y=112),
        _line("The quick copper fox records item 17 for review.", line_id="body-2", order=3, y=138),
        _line("Footnote: all words in this fixture are procedural.", line_id="footnote-1", order=4, y=690, region_type="footnote"),
    ]
    ru_lines = [
        _line("СИНТЕТИЧЕСКИЙ ОТЧЁТ БЕТА", line_id="heading-1", order=1, y=74, language="rus", region_type="heading"),
        _line("Придуманный текст проверяет русский слой поиска.", line_id="body-1", order=2, y=112, language="rus"),
        _line("Смешанная строка Alpha — раздел 17.", line_id="body-2", order=3, y=138, language="mixed"),
    ]
    mixed_lines = [
        _line("MIXED LAYOUT / СМЕШАННЫЙ МАКЕТ", line_id="heading-1", order=1, y=74, language="mixed", region_type="heading"),
        _line("left-one: Alpha column text", line_id="left-1", order=2, y=112, x1=285),
        _line("left-two: Delta column text", line_id="left-2", order=3, y=138, x1=285),
        _line("right-one: Бета колонка", line_id="right-1", order=4, y=112, x0=330, language="rus"),
        _line("right-two: Гамма колонка", line_id="right-2", order=5, y=138, x0=330, language="rus"),
        _line("—", line_id="table-dash", order=6, y=430, x0=52, x1=179, region_type="table-cell"),
        _line("42", line_id="table-42", order=7, y=430, x0=179, x1=306, region_type="table-cell"),
        _line("ALARM", line_id="table-alarm", order=8, y=430, x0=306, x1=433, region_type="table-cell"),
        _line("—", line_id="table-dash-2", order=9, y=430, x0=433, x1=560, region_type="table-cell"),
        _line("—", line_id="table-dash-3", order=10, y=464, x0=52, x1=179, region_type="table-cell"),
        _line("OK", line_id="table-ok", order=11, y=464, x0=179, x1=306, region_type="table-cell"),
        _line("18", line_id="table-18", order=12, y=464, x0=306, x1=433, region_type="table-cell"),
        _line("—", line_id="table-dash-4", order=13, y=464, x0=433, x1=560, region_type="table-cell"),
        _line("Synthetic footnote [1]", line_id="footnote-1", order=14, y=690, region_type="footnote"),
    ]
    retention_pages = [
        [_line("EXPECTED PAGE 1", line_id="body-1", order=1, y=100)],
        [_line("EXPECTED PAGE 2", line_id="body-1", order=1, y=100)],
        [_line("EXPECTED PAGE THREE", line_id="body-1", order=1, y=100)],
    ]
    long_pages = [[_line(f"LONG DOCUMENT PAGE {page}", line_id="heading-1", order=1, y=74, region_type="heading"), _line("Repeated procedural text supports chunk and resume checks.", line_id="body-1", order=2, y=112)] for page in range(1, 24)]
    return {
        "clean-en": [_page_record(1, text="\n".join(str(item["text"]) for item in en_lines), language="eng", features=["headings", "footnote"], lines=en_lines)],
        "clean-ru": [_page_record(1, text="\n".join(str(item["text"]) for item in ru_lines), language="mixed", features=["rus", "eng"], lines=ru_lines)],
        "mixed-layout": [_page_record(1, text="\n".join(str(item["text"]) for item in mixed_lines), language="mixed", features=["columns", "table", "footnote", "punctuation-only-cells"], lines=mixed_lines)],
        "degraded-vector-text": [_page_record(1, text="LOW CONTRAST SYNTHETIC VECTOR TEXT", features=["low-contrast", "noise", "skew", "low-resolution", "vector-text"], degradation=["contrast=0.46", "noise-seed=20260819", "skew-degrees=1.5", "source-dpi=96", "not-rasterized"], lines=[_line("LOW CONTRAST SYNTHETIC VECTOR TEXT", line_id="body-1", order=1, y=100)])],
        "rotated": [_page_record(1, text="ROTATED PAGE 90", rotation_deg=90, features=["rotation"], lines=[_line("ROTATED PAGE 90", line_id="body-1", order=1, y=100)]), _page_record(2, text="ROTATED PAGE 180", rotation_deg=180, features=["rotation"], lines=[_line("ROTATED PAGE 180", line_id="body-1", order=1, y=100)]), _page_record(3, text="ROTATED PAGE 270", rotation_deg=270, features=["rotation"], lines=[_line("ROTATED PAGE 270", line_id="body-1", order=1, y=100)])],
        "native-text-layer": [_page_record(1, text="NATIVE TEXT LAYER MUST BE REMOVED", native_text_layer=True, features=["existing-text-layer"], lines=[_line("NATIVE TEXT LAYER MUST BE REMOVED", line_id="body-1", order=1, y=100)])],
        "blank-graphics": [_page_record(1, outcome="verified_blank", features=["blank"]), _page_record(2, outcome="explicit_nontext", features=["graphics-only"])],
        "retention-3p": [_page_record(index + 1, text=str(lines[0]["text"]), features=["exact-retention"], lines=lines) for index, lines in enumerate(retention_pages)],
        "long-23p": [_page_record(index + 1, text="\n".join(str(item["text"]) for item in lines), features=["large-document"], lines=lines) for index, lines in enumerate(long_pages)],
    }


def _fixture_manifest(fixture_id: str, pages: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    pdf_path = root / fixture_id / "source.pdf"
    gt_path = root / fixture_id / "ground_truth.jsonl"
    gt_rows = [{"fixture": fixture_id, **page} for page in pages]
    _write_jsonl(gt_path, gt_rows)
    _build_fixture_pdf(fixture_id, pages, pdf_path)
    return {
        "id": fixture_id,
        "revision": 1,
        "source_pdf": str(pdf_path.relative_to(root)).replace("\\", "/"),
        "source_pdf_sha256": _sha256(pdf_path),
        "source_pdf_bytes": pdf_path.stat().st_size,
        "page_count": len(pages),
        "ground_truth": str(gt_path.relative_to(root)).replace("\\", "/"),
        "ground_truth_sha256": _sha256(gt_path),
        "pages": [{key: value for key, value in page.items() if key != "text"} | {"text_sha256": hashlib.sha256(str(page["text"]).encode("utf-8")).hexdigest()} for page in pages],
        "chunk_plan": {"pages_per_chunk": 10, "expected_chunks": (len(pages) + 9) // 10},
    }


def _write_cases(root: Path) -> list[dict[str, Any]]:
    cases = [
        {
            "schema": "uniscan.synthetic-case.v1",
            "id": "punctuation-only-chandra-v1",
            "kind": "reconciliation",
            "fixture": "mixed-layout",
            "page": 1,
            "synthetic_evidence": {"surya_lines": ["ALARM"], "chandra_lines": ["ALARM", "—", "—"]},
            "historical_expected": {"status": "error", "reason": "invalid_chandra_attempt_evidence"},
            "current_expected": {"status": "ok", "punctuation_lines_retained": True},
            "private_source_included": False,
        },
        {
            "schema": "uniscan.synthetic-case.v1",
            "id": "exact-retention-page3-v1",
            "kind": "merge-validation",
            "fixture": "retention-3p",
            "mutation": {"page": 3, "replace_text": "WRONG PAGE THREE"},
            "expected_error": "Output PDF page 3 failed exact searchable text retention",
            "output_published": False,
            "private_source_included": False,
        },
    ]
    for case in cases:
        _write_json(root / "cases" / f"{case['id']}.json", case)
    return [{"id": case["id"], "path": f"cases/{case['id']}.json", "sha256": _sha256(root / "cases" / f"{case['id']}.json")} for case in cases]


def generate_corpus(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = [_fixture_manifest(fixture_id, pages, output) for fixture_id, pages in _make_pages().items()]
    case_records = _write_cases(output)
    fitz = _fitz()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "corpus_id": "uniscan-synthetic-offline",
        "revision": "1.0.0",
        "provenance": {"source_kind": "procedural-original", "contains_private_data": False, "external_assets": [], "generated_pdf_policy": "caller-output-only"},
        "generator": {"name": "generate.py", "version": GENERATOR_VERSION, "seed": SEED, "python": platform.python_version(), "pymupdf": str(getattr(fitz, "VersionBind", "unknown"))},
        "metrics": {"model_status": "not_run", "text": {"cer": "levenshtein-codepoint-v1", "wer": "whitespace-token-v1"}, "searchability": {"exact_text": "nfkc-collapse-whitespace-v1", "page_bijection": "exact-page-set-v1"}, "geometry": {"bbox_iou": "axis-aligned-iou-v1", "reading_order": "exact-order-v1"}, "unavailable_without_models": ["engine CER/WER", "VRAM", "cold/warm model latency", "model cache-hit latency"]},
        "fixtures": fixtures,
        "cases": case_records,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def validate_corpus(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported synthetic benchmark schema")
    if manifest.get("metrics", {}).get("model_status") != "not_run":
        raise ValueError("Offline corpus must declare model_status=not_run")
    for fixture in manifest.get("fixtures", []):
        pdf = root / str(fixture["source_pdf"])
        gt = root / str(fixture["ground_truth"])
        if not pdf.is_file() or _sha256(pdf) != fixture["source_pdf_sha256"]:
            raise ValueError(f"Source PDF hash mismatch: {fixture['id']}")
        if not gt.is_file() or _sha256(gt) != fixture["ground_truth_sha256"]:
            raise ValueError(f"Ground Truth hash mismatch: {fixture['id']}")
        if len(json.loads("[" + ",".join(line for line in gt.read_text(encoding="utf-8").splitlines()) + "]")) != int(fixture["page_count"]):
            raise ValueError(f"Ground Truth page count mismatch: {fixture['id']}")
    for case in manifest.get("cases", []):
        path = root / str(case["path"])
        if not path.is_file() or _sha256(path) != case["sha256"]:
            raise ValueError(f"Case hash mismatch: {case['id']}")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Caller-owned temporary/ignored output directory.")
    parser.add_argument("--check", action="store_true", help="Validate an already-generated corpus without rewriting it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        validate_corpus(args.output)
    else:
        generate_corpus(args.output)
        validate_corpus(args.output)
    print(f"synthetic corpus: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
