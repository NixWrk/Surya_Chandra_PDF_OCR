from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr import OCR_ENGINE_PADDLEOCR, OCR_ENGINE_PYTESSERACT, OCR_ENGINE_SURYA
from uniscan.ocr.canonical import (
    _build_text_only_searchable_pdf,
    run_ocr_canonical_package,
)


def _build_sample_pdf(tmp_path: Path, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((120, 180, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=150)
    return pdf_path


def test_run_ocr_canonical_package_writes_outputs(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [20, 80, 140])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=engine_name,
            ready=True,
            missing=[],
            searchable_pdf=engine_name == OCR_ENGINE_PYTESSERACT,
            label=engine_name,
        )

    def fake_extract(engine: str, image_path: Path, *, lang: str, work_dir: Path) -> str:
        assert lang == "eng"
        assert work_dir.exists() or not work_dir.exists()
        return f"{engine}:{image_path.name}"

    monkeypatch.setattr("uniscan.ocr.canonical.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.canonical._extract_page_text", fake_extract)

    results = run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PYTESSERACT, OCR_ENGINE_PADDLEOCR),
        sample_size=2,
        dpi=100,
        lang="eng",
    )

    assert len(results) == 2
    assert all(result.status == "ok" for result in results)
    assert (output_dir / "source_pages" / "page_0001.png").exists()
    assert (output_dir / "source_pages" / "page_0003.png").exists()

    for engine in (OCR_ENGINE_PYTESSERACT, OCR_ENGINE_PADDLEOCR):
        assert (output_dir / "canonical" / engine / "page_0001.txt").exists()
        assert (output_dir / "canonical" / engine / "page_0003.txt").exists()
        assert (output_dir / "searchable_pdf" / f"{engine}.pdf").exists()
        all_pages = (output_dir / "canonical" / engine / "all_pages.txt").read_text(
            encoding="utf-8"
        )
        assert "[SOURCE PAGE 0001]" in all_pages
        assert "[SOURCE PAGE 0003]" in all_pages

    summary_json = output_dir / "canonical_summary.json"
    summary_csv = output_dir / "canonical_summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert {row["engine"] for row in payload} == {OCR_ENGINE_PYTESSERACT, OCR_ENGINE_PADDLEOCR}


def test_run_ocr_canonical_package_unready_engine_returns_error(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 60])
    output_dir = tmp_path / "out"

    def fake_status(_engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=OCR_ENGINE_SURYA,
            ready=False,
            missing=["dependency-x"],
            searchable_pdf=False,
            label=OCR_ENGINE_SURYA,
        )

    monkeypatch.setattr("uniscan.ocr.canonical.detect_ocr_engine_status", fake_status)

    results = run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_SURYA,),
        sample_size=1,
        dpi=72,
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert "dependency-x" in (results[0].error or "")


def test_run_ocr_canonical_package_uses_explicit_page_numbers(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [20, 40, 60, 80])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=engine_name,
            ready=True,
            missing=[],
            searchable_pdf=False,
            label=engine_name,
        )

    def fake_extract(engine: str, image_path: Path, *, lang: str, work_dir: Path) -> str:
        return f"{engine}:{image_path.name}:{lang}"

    monkeypatch.setattr("uniscan.ocr.canonical.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.canonical._extract_page_text", fake_extract)

    results = run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=99,
        page_numbers=(3, 1),
        dpi=90,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [3, 1]
    engine_dir = output_dir / "canonical" / OCR_ENGINE_PADDLEOCR
    assert (engine_dir / "page_0003.txt").exists()
    assert (engine_dir / "page_0001.txt").exists()
    markerized = (engine_dir / "all_pages.txt").read_text(encoding="utf-8")
    assert markerized.index("[SOURCE PAGE 0003]") < markerized.index("[SOURCE PAGE 0001]")


def test_text_only_searchable_pdf_embeds_cyrillic_font(tmp_path: Path) -> None:
    output_pdf = tmp_path / "canonical.pdf"

    _build_text_only_searchable_pdf(
        ["Русский текст сохраняется"],
        out_pdf=output_pdf,
        source_pages_1based=[7],
    )

    from pypdf import PdfReader

    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(output_pdf)).pages)
    normalized = " ".join(extracted.split())
    assert "[SOURCE PAGE 0007]" in normalized
    assert "Русский текст сохраняется" in normalized


def test_run_ocr_canonical_preprocessing_failure_is_fatal(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [40])

    def fail_preprocessing(*_args, **_kwargs):
        raise RuntimeError("broken preprocessing")

    monkeypatch.setattr(
        "uniscan.ocr.canonical.preprocess_image_file",
        fail_preprocessing,
    )

    with pytest.raises(RuntimeError, match="Preprocessing failed.*broken preprocessing"):
        run_ocr_canonical_package(
            pdf_path=pdf_path,
            output_dir=tmp_path / "out",
            engines=(OCR_ENGINE_PADDLEOCR,),
            sample_size=1,
            preprocessing="basic",
        )


def test_run_ocr_canonical_rejects_invalid_dpi_and_empty_engines(tmp_path) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [40])

    with pytest.raises(ValueError, match="DPI values must be positive"):
        run_ocr_canonical_package(
            pdf_path=pdf_path,
            output_dir=tmp_path / "bad-dpi",
            engines=(OCR_ENGINE_PADDLEOCR,),
            dpi=0,
        )

    with pytest.raises(ValueError, match="No OCR engines selected"):
        run_ocr_canonical_package(
            pdf_path=pdf_path,
            output_dir=tmp_path / "no-engines",
            engines=(),
            sample_size=1,
        )


def test_run_ocr_canonical_validates_before_replacing_outputs(tmp_path) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [40])
    output_dir = tmp_path / "out"
    sentinel = output_dir / "source_pages" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        run_ocr_canonical_package(
            pdf_path=pdf_path,
            output_dir=output_dir,
            engines=("../../outside",),
            sample_size=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "outside").exists()


def test_run_ocr_canonical_removes_stale_preprocessed_output(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [40])
    output_dir = tmp_path / "out"
    stale = output_dir / "preprocessed" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    monkeypatch.setattr(
        "uniscan.ocr.canonical.detect_ocr_engine_status",
        lambda engine: SimpleNamespace(engine_name=engine, ready=True, missing=[]),
    )
    monkeypatch.setattr(
        "uniscan.ocr.canonical._extract_page_text",
        lambda *_args, **_kwargs: "text",
    )

    results = run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        preprocessing="none",
    )

    assert results[0].status == "ok"
    assert not stale.parent.exists()


def test_cli_benchmark_ocr_canonical_uses_runner_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run(**kwargs):
        assert kwargs["pdf_path"] == pdf_path
        assert kwargs["output_dir"] == output_dir
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PYTESSERACT,
                status="ok",
                elapsed_seconds=1.0,
                sample_pages=[1],
                text_chars=12,
                canonical_dir="canonical/pytesseract",
                searchable_pdf_path="searchable_pdf/pytesseract.pdf",
                error=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_canonical_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_canonical_package", lambda results: f"rows={len(results)}")

    exit_code = main(
        ["benchmark-ocr-canonical", "--pdf", str(pdf_path), "--output", str(output_dir)]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "rows=1" in stdout


def test_cli_benchmark_ocr_canonical_parses_pages(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run(**kwargs):
        assert kwargs["page_numbers"] == (3, 9)
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PYTESSERACT,
                status="ok",
                elapsed_seconds=1.0,
                sample_pages=[3, 9],
                text_chars=12,
                canonical_dir="canonical/pytesseract",
                searchable_pdf_path="searchable_pdf/pytesseract.pdf",
                error=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_canonical_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_canonical_package", lambda _results: "canonical")

    exit_code = main(
        [
            "benchmark-ocr-canonical",
            "--pdf",
            str(pdf_path),
            "--output",
            str(output_dir),
            "--pages",
            "3,9",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "canonical" in stdout


def test_cli_benchmark_ocr_canonical_strict_fails(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run(**_kwargs):
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PYTESSERACT,
                status="ok",
                elapsed_seconds=1.0,
                sample_pages=[1],
                text_chars=12,
                canonical_dir="canonical/pytesseract",
                searchable_pdf_path="searchable_pdf/pytesseract.pdf",
                error=None,
            ),
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="error",
                elapsed_seconds=2.0,
                sample_pages=[1],
                text_chars=0,
                canonical_dir="canonical/paddleocr",
                searchable_pdf_path=None,
                error="broken",
            ),
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_canonical_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_canonical_package", lambda _results: "canonical")

    exit_code = main(
        [
            "benchmark-ocr-canonical",
            "--pdf",
            str(pdf_path),
            "--output",
            str(output_dir),
            "--strict",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 1
    assert "canonical" in stdout
