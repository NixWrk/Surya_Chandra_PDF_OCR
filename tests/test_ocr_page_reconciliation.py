from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from uniscan.app import ocr_pipeline
from uniscan.app.ocr_pipeline import BasicOcrRunSummary, _ensure_requested_engines_succeeded
from uniscan.ocr import OcrBenchmarkResult


def _markerized(page_rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in page_rows:
        blocks.append(f"[SOURCE PAGE {int(row['source_page']):04d}]")
        text = str(row.get("text") or "")
        if text:
            blocks.append(text)
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _build_reconciliation_run(
    tmp_path: Path,
    *,
    surya_rows: list[dict[str, Any]],
    chandra_rows: list[dict[str, Any]],
) -> tuple[Path, list[OcrBenchmarkResult], list[Path]]:
    run_dir = tmp_path / "run"
    results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    for engine, raw_rows in (("surya", surya_rows), ("chandra", chandra_rows)):
        output_dir = run_dir / engine
        engine_dir = output_dir / engine
        engine_dir.mkdir(parents=True)
        page_rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            source_page = int(raw_row["source_page"])
            text = str(raw_row.get("text") or "")
            page_file = engine_dir / f"page_{source_page:04d}.txt"
            page_file.write_text(text, encoding="utf-8")
            geometry_file = engine_dir / f"page_{source_page:04d}.{engine}.json"
            outcome = str(raw_row["ocr_outcome"])
            image_evidence: dict[str, Any] = {
                "image_name": f"page_{source_page:04d}.png",
                "ocr_outcome": outcome,
                "pages": [
                    {
                        "image_bbox": [0, 0, 100, 100],
                        "text_lines": (
                            [{"text": text, "bbox": [0, 0, 50, 10]}]
                            if text
                            else []
                        ),
                    }
                ],
            }
            if engine == "chandra":
                explicit_nontext = bool(raw_row.get("explicit_nontext", False))
                image_evidence["explicit_nontext"] = explicit_nontext
                image_evidence["chandra_non_text_labels"] = (
                    ["figure"] if explicit_nontext else []
                )
            geometry_file.write_text(
                json.dumps({"images": [image_evidence]}),
                encoding="utf-8",
            )
            row = {
                "source_page": source_page,
                "file": page_file.name,
                "geometry_file": geometry_file.name,
                "geometry_type": f"{engine}_text_lines",
                "text_chars": len(text),
                "ocr_outcome": str(raw_row["ocr_outcome"]),
                "alnum_line_count": int(raw_row.get("alnum_line_count", 0)),
                "alnum_chars": int(raw_row.get("alnum_chars", 0)),
            }
            if "explicit_nontext" in raw_row:
                row["explicit_nontext"] = bool(raw_row["explicit_nontext"])
            page_errors = list(raw_row.get("page_errors", []))
            if page_errors:
                error_records: list[dict[str, str]] = []
                for item in page_errors:
                    if isinstance(item, dict):
                        error_records.append({
                            "code": str(item["code"]),
                            "message": str(item["message"]),
                        })
                    else:
                        code = (
                            "zero_output"
                            if outcome in {"zero_output", "explicit_nontext"}
                            else "unrelated"
                        )
                        error_records.append({"code": code, "message": str(item)})
                row["page_errors"] = error_records
            page_rows.append(row)

        aggregate_text = _markerized(raw_rows)
        (engine_dir / "all_pages.txt").write_text(aggregate_text, encoding="utf-8")
        artifact_path = output_dir / f"document_{engine}.txt"
        artifact_path.write_text(aggregate_text, encoding="utf-8")
        total_chars = sum(int(row["text_chars"]) for row in page_rows)
        (engine_dir / "pages.json").write_text(
            json.dumps(
                {
                    "pdf_path": str(tmp_path / "document.pdf"),
                    "engine": engine,
                    "pages": page_rows,
                    "total_text_chars": total_chars,
                    "aggregate_file": "all_pages.txt",
                    "aggregate_has_page_markers": True,
                }
            ),
            encoding="utf-8",
        )
        has_candidate_errors = any(
            row.get("ocr_outcome") in {"zero_output", "explicit_nontext"}
            and bool(row.get("page_errors"))
            for row in raw_rows
        )
        result = OcrBenchmarkResult(
            engine=engine,
            status="reconciliation_pending" if has_candidate_errors else "ok",
            sample_pages=[int(row["source_page"]) for row in raw_rows],
            elapsed_seconds=0.1,
            artifact_path=str(artifact_path),
            text_chars=total_chars,
            page_error_count=sum(len(list(row.get("page_errors", []))) for row in raw_rows),
        )
        report_path = output_dir / "document_ocr_benchmark.json"
        report_path.write_text(
            json.dumps(
                {
                    "pdf_path": str(tmp_path / "document.pdf"),
                    "page_count": len(raw_rows),
                    "sample_pages": result.sample_pages,
                    "results": [asdict(result)],
                }
            ),
            encoding="utf-8",
        )
        results.append(result)
        result_files.append(report_path)
    return run_dir, results, result_files


def _explicit_graphics_page(source_page: int) -> dict[str, Any]:
    return {
        "source_page": source_page,
        "text": "",
        "ocr_outcome": "explicit_nontext",
        "explicit_nontext": True,
        "alnum_line_count": 0,
        "alnum_chars": 0,
        "page_errors": ["Chandra geometry sidecar has no text_lines"],
    }


def test_reconcile_knh_accepts_trivial_surya_without_substituting_its_text(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "(A)",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 1,
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.page_error_count == 0 for result in adjusted)
    assert all(result.text_chars == 0 for result in adjusted)
    for engine in ("surya", "chandra"):
        assert Path(next(result.artifact_path for result in adjusted if result.engine == engine)).read_text(
            encoding="utf-8"
        ) == "[SOURCE PAGE 0001]\n"
        pages = json.loads((run_dir / engine / engine / "pages.json").read_text(encoding="utf-8"))
        assert pages["pages"][0]["ocr_outcome"] == "textless_graphics"
        assert pages["pages"][0]["alnum_line_count"] == 0
        assert pages["pages"][0]["alnum_chars"] == 0
        assert pages["pages"][0]["page_errors"] == []
        geometry = json.loads(
            (run_dir / engine / engine / f"page_0001.{engine}.json").read_text(
                encoding="utf-8"
            )
        )
        assert geometry["images"][0]["pages"][0]["text_lines"] == []
    for report_path in result_files:
        row = json.loads(report_path.read_text(encoding="utf-8"))["results"][0]
        expected = next(result for result in adjusted if result.engine == row["engine"])
        assert row == asdict(expected)
    reconciliation = json.loads(
        (run_dir / "page_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["status"] == "ok"
    assert reconciliation["accepted_textless_graphics_pages"] == [1]


def test_reconcile_74_accepts_explicit_graphics_with_zero_surya(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["Surya geometry sidecar has no text_lines"],
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert {result.engine: result.page_error_count for result in adjusted} == {
        "surya": 0,
        "chandra": 0,
    }


def test_reconcile_29d_rejects_dense_surya_with_chandra_zero(tmp_path: Path) -> None:
    dense_text = "Dense OCR line one\nDense OCR line two"
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": dense_text,
                "ocr_outcome": "text",
                "alnum_line_count": 2,
                "alnum_chars": 30,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "explicit_nontext": False,
                "page_errors": ["Chandra geometry sidecar has no text_lines"],
            }
        ],
    )

    original_surya = Path(results[0].artifact_path).read_text(encoding="utf-8")
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    assert Path(results[0].artifact_path).read_text(encoding="utf-8") == original_surya
    reconciliation = json.loads(
        (run_dir / "page_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["status"] == "error"
    assert reconciliation["accepted_textless_graphics_pages"] == []


def test_reconcile_rejects_chandra_text_with_surya_zero(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["Surya geometry sidecar has no text_lines"],
            }
        ],
        chandra_rows=[
            {"source_page": 1, "text": "CHANDRA TEXT", "ocr_outcome": "text"}
        ],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"


def test_reconcile_requires_exact_page_bijection(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[{"source_page": 1, "text": "A", "ocr_outcome": "text"}],
        chandra_rows=[
            {"source_page": 1, "text": "A", "ocr_outcome": "text"},
            {"source_page": 2, "text": "B", "ocr_outcome": "text"},
        ],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error is not None
    assert "not bijective" in error
    report = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert report["status"] == "error"


def test_reconcile_preserves_unrelated_page_errors(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {"source_page": 1, "text": "text", "ocr_outcome": "text"},
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "text",
                "ocr_outcome": "text",
                "page_errors": ["unrelated geometry corruption"],
            },
        ],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    assert next(result for result in adjusted if result.engine == "chandra").page_error_count == 1


def test_reconcile_cap_evidence_is_internally_consistent(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": page,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["zero"],
            }
            for page in (1, 2)
        ],
        chandra_rows=[_explicit_graphics_page(page) for page in (1, 2)],
    )

    _, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error == "unresolved pages: [1, 2]"
    report = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert report["accepted_textless_graphics_pages"] == []
    assert all(row["accepted"] is False for row in report["pages"])
    assert {row["reason"] for row in report["pages"]} == {
        "graphics_recovery_cap_exceeded"
    }


def test_reconcile_artifact_failure_never_leaves_green_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["zero"],
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )
    durable_paths = sorted(path for path in run_dir.rglob("*") if path.is_file())
    original_bytes = {path: path.read_bytes() for path in durable_paths}
    monkeypatch.setattr(
        ocr_pipeline,
        "_publish_file_transaction",
        lambda _updates: (_ for _ in ()).throw(RuntimeError("publish failed")),
    )

    _, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error == "artifact reconciliation failed: publish failed"
    report = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert report["status"] == "error"
    for path, expected in original_bytes.items():
        assert path.read_bytes() == expected



def test_reconcile_rejects_unrelated_error_on_graphics_candidate(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": [
                    {"code": "zero_output", "message": "expected zero"},
                    {"code": "geometry_corrupt", "message": "unrelated corruption"},
                ],
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    assert next(result for result in adjusted if result.engine == "surya").page_error_count == 2


def test_reconcile_recomputes_surya_metrics_from_dense_artifact(tmp_path: Path) -> None:
    dense_text = "DENSE ALPHA LINE\nDENSE BETA LINE"
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": dense_text,
                "ocr_outcome": "text",
                "alnum_line_count": 0,
                "alnum_chars": 0,
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    report = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert "alnum evidence does not match" in report["pages"][0]["reason"]


def test_reconcile_caps_actual_source_chunks_not_sample_order(tmp_path: Path) -> None:
    pages = (1, 11)
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": page,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["zero"],
            }
            for page in pages
        ],
        chandra_rows=[_explicit_graphics_page(page) for page in pages],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.page_error_count == 0 for result in adjusted)
    report = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert report["accepted_textless_graphics_pages"] == [1, 11]


def test_file_transaction_rolls_back_mid_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first-original")
    second.write_bytes(b"second-original")
    real_replace = ocr_pipeline.os.replace
    failed = False

    def fail_second_once(source, target):
        nonlocal failed
        if Path(target).resolve() == second.resolve() and not failed:
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(ocr_pipeline.os, "replace", fail_second_once)

    with pytest.raises(OSError, match="injected publish failure"):
        ocr_pipeline._publish_file_transaction(
            {first: b"first-new", second: b"second-new"}
        )

    assert first.read_bytes() == b"first-original"
    assert second.read_bytes() == b"second-original"
    assert list(tmp_path.glob("*.transaction")) == []

def test_strict_app_path_rejects_reconciliation_failure() -> None:
    results = (
        OcrBenchmarkResult("surya", "ok", [1], 0.1, "surya.txt", 1),
        OcrBenchmarkResult("chandra", "ok", [1], 0.1, "chandra.txt", 1),
    )
    summary = BasicOcrRunSummary(
        run_dir=Path("run"),
        results=results,
        result_files=(),
        failed_engines=("page reconciliation: unresolved pages: [1]",),
        skipped_engines=(),
    )

    with pytest.raises(RuntimeError, match="page reconciliation"):
        _ensure_requested_engines_succeeded(
            summary,
            expected_engines=("chandra", "surya"),
        )


def test_hybrid_cache_identity_includes_retry_and_reconciliation_revision() -> None:
    config = ocr_pipeline._hybrid_runtime_config()
    assert ocr_pipeline._HYBRID_CHUNK_PIPELINE_REVISION == "chandra-surya-resumable-v2"
    assert config["zero_output_retry_policy"] == "original+autocontrast-cutoff-1-max2-v1"
    assert config["page_reconciliation_policy"] == (
        "explicit-chandra-nontext+quiet-surya-v1"
    )
