from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr import OCR_ENGINE_PADDLEOCR, OCR_ENGINE_PYTESSERACT, run_ocr_benchmark, sample_pdf_page_indices

FIXTURE_PDF = Path(r"J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf")


def _build_sample_pdf(tmp_path: Path, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for idx, value in enumerate(page_values, start=1):
        pages.append(np.full((120, 180, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=150)
    return pdf_path


def test_sample_pdf_page_indices_spans_first_middle_last_windows() -> None:
    assert sample_pdf_page_indices(12, sample_size=3) == [0, 1, 2, 5, 6, 7, 9, 10, 11]
    assert sample_pdf_page_indices(2, sample_size=5) == [0, 1]
    assert sample_pdf_page_indices(0, sample_size=5) == []


def test_run_ocr_benchmark_writes_report_and_artifacts(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        searchable = engine_name in {OCR_ENGINE_PYTESSERACT}
        return SimpleNamespace(
            engine_name=engine_name,
            ready=True,
            missing=[],
            searchable_pdf=searchable,
            label=engine_name,
        )

    def fake_searchable_pdf(image_paths, *, out_pdf, lang, engine_name):
        out_pdf.write_text(f"{engine_name}:{lang}:{len(image_paths)}", encoding="utf-8")
        return out_pdf

    def fake_extract_chars(_pdf_path: Path) -> int:
        return 321

    def fake_paddleocr(image_paths, *, lang):
        return f"{lang}:{len(image_paths)}", 12

    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark.image_paths_to_searchable_pdf", fake_searchable_pdf)
    monkeypatch.setattr("uniscan.ocr.benchmark._extract_pdf_text_chars", fake_extract_chars)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PYTESSERACT, OCR_ENGINE_PADDLEOCR),
        sample_size=2,
        dpi=120,
        lang="eng",
    )

    assert [result.engine for result in results] == [OCR_ENGINE_PYTESSERACT, OCR_ENGINE_PADDLEOCR]
    assert all(result.status == "ok" for result in results)
    assert results[0].artifact_path and Path(results[0].artifact_path).exists()
    assert results[1].artifact_path and Path(results[1].artifact_path).exists()
    assert results[0].sample_pages == [1, 2, 3]
    assert results[1].sample_pages == [1, 2, 3]

    report_path = output_dir / "fixture_ocr_benchmark.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["pdf_path"] == str(pdf_path)
    assert payload["sample_pages"] == [1, 2, 3]
    assert len(payload["results"]) == 2


@pytest.mark.skipif(not FIXTURE_PDF.exists(), reason="external OCR fixture is not available")
def test_run_ocr_benchmark_uses_external_fixture_smoke(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"

    def fake_sample(_page_count: int, *, sample_size: int = 5) -> list[int]:
        assert sample_size == 1
        return [0]

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=engine_name,
            ready=True,
            missing=[],
            searchable_pdf=False,
            label=engine_name,
        )

    def fake_paddleocr(image_paths, *, lang):
        assert len(image_paths) == 1
        return f"{lang}:fixture", 7

    monkeypatch.setattr("uniscan.ocr.benchmark.sample_pdf_page_indices", fake_sample)
    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=FIXTURE_PDF,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        dpi=72,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [1]
    assert results[0].artifact_path is not None
    assert Path(results[0].artifact_path).exists()
    assert (output_dir / "Imaging Edge Mobile_paddleocr_uvdoc_ocr_benchmark.json").exists()


def test_cli_benchmark_ocr_uses_runner_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**kwargs):
        assert kwargs["pdf_path"] == pdf_path
        assert kwargs["output_dir"] == output_dir
        assert kwargs["sample_size"] == 5
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[1],
                elapsed_seconds=1.23,
                artifact_path=str(output_dir / "fixture_paddleocr.txt"),
                text_chars=7,
                error=None,
                note=None,
            )
        ]

    def fake_summary(results):
        assert len(results) == 1
        return "paddleocr ok"

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", fake_summary)

    exit_code = main(["benchmark-ocr", "--pdf", str(pdf_path), "--output", str(output_dir)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "paddleocr ok" in stdout
