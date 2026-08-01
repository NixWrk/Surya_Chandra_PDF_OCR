from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            raw_row = dict(raw_row)
            source_page = int(raw_row["source_page"])
            text = str(raw_row.get("text") or "")
            raw_history = raw_row.get("attempt_history")
            if engine == "surya" and isinstance(raw_history, list):
                durable_history: list[dict[str, Any]] = []
                for raw_item in raw_history:
                    item = dict(raw_item)
                    attempt = int(item["attempt"])
                    attempt_dir = (
                        engine_dir / f"page_{source_page:04d}.retry" / f"attempt_{attempt}"
                    )
                    attempt_dir.mkdir(parents=True)
                    image_path = attempt_dir / f"page_{source_page:04d}.png"
                    if attempt == 3:
                        attempt_one_path = (
                            engine_dir
                            / f"page_{source_page:04d}.retry"
                            / "attempt_1"
                            / f"page_{source_page:04d}.png"
                        )
                        with Image.open(attempt_one_path) as original:
                            content = original.resize((50, 50), Image.Resampling.LANCZOS)
                        candidate = Image.new("RGB", (100, 100), (255, 255, 255))
                        candidate.paste(content, (25, 25))
                        candidate.save(image_path, format="PNG")
                    else:
                        Image.new("RGB", (100, 100), color=(120, 120, 120)).save(
                            image_path,
                            format="PNG",
                        )
                    attempt_text = text if attempt == 3 else ""
                    attempt_bbox = [30, 30, 50, 40]
                    sidecar_path = attempt_dir / "surya_page_lines.json"
                    sidecar_path.write_text(
                        json.dumps(
                            {
                                "execution_path": "module",
                                "images": [
                                    {
                                        "image_name": image_path.name,
                                        "pages": [
                                            {
                                                "image_bbox": [0, 0, 100, 100],
                                                "text_lines": (
                                                    [{"text": attempt_text, "bbox": attempt_bbox}]
                                                    if attempt_text
                                                    else []
                                                ),
                                            }
                                        ],
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    item.update(
                        {
                            "image_size": [100, 100],
                            "image_path": str(image_path.resolve()),
                            "image_sha256": _sha256(image_path),
                            "image_bytes": image_path.stat().st_size,
                            "sidecar_path": str(sidecar_path.resolve()),
                            "sidecar_sha256": _sha256(sidecar_path),
                            "sidecar_bytes": sidecar_path.stat().st_size,
                        }
                    )
                    durable_history.append(item)
                raw_row["attempt_history"] = durable_history
            page_file = engine_dir / f"page_{source_page:04d}.txt"
            page_file.write_text(text, encoding="utf-8")
            geometry_file = engine_dir / f"page_{source_page:04d}.{engine}.json"
            outcome = str(raw_row["ocr_outcome"])
            omit_attempt_count = raw_row.get("omit_attempt_count") is True
            attempt_count = raw_row.get("attempt_count", 1)
            image_evidence: dict[str, Any] = {
                "image_name": f"page_{source_page:04d}.png",
                "ocr_outcome": outcome,
                "pages": [
                    {
                        "image_bbox": [0, 0, 100, 100],
                        "text_lines": (
                            [{"text": text, "bbox": [10, 10, 50, 30]}]
                            if text and isinstance(raw_history, list)
                            else ([{"text": text, "bbox": [0, 0, 50, 10]}] if text else [])
                        ),
                    }
                ],
            }
            if not omit_attempt_count:
                image_evidence["attempt_count"] = attempt_count
            if engine == "chandra":
                explicit_nontext = bool(raw_row.get("explicit_nontext", False))
                image_evidence["explicit_nontext"] = explicit_nontext
                image_evidence["chandra_non_text_labels"] = ["figure"] if explicit_nontext else []
            for key in (
                "retry_preprocessing",
                "retry_policy",
                "selected_attempt",
                "attempt_history",
                "geometry_coordinate_space",
                "geometry_transform",
            ):
                if key in raw_row:
                    image_evidence[key] = raw_row[key]
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
            if not omit_attempt_count:
                row["attempt_count"] = attempt_count
            if "explicit_nontext" in raw_row:
                row["explicit_nontext"] = bool(raw_row["explicit_nontext"])
            for key in (
                "retry_preprocessing",
                "retry_policy",
                "selected_attempt",
                "attempt_history",
                "geometry_coordinate_space",
                "geometry_transform",
            ):
                if key in raw_row:
                    row[key] = raw_row[key]
            page_errors = list(raw_row.get("page_errors", []))
            if page_errors:
                error_records: list[dict[str, str]] = []
                for item in page_errors:
                    if isinstance(item, dict):
                        error_records.append(
                            {
                                "code": str(item["code"]),
                                "message": str(item["message"]),
                            }
                        )
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


def _third_retry_fields() -> dict[str, Any]:
    return {
        "attempt_count": 3,
        "retry_preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
        "retry_policy": (
            "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
        ),
        "selected_attempt": 3,
        "geometry_coordinate_space": "source-image-v1",
        "geometry_transform": "inverse-actual-content-size-strict-v1",
        "attempt_history": [
            {
                "attempt": 1,
                "preprocessing": "original",
                "ocr_outcome": "zero_output",
                "image_size": [120, 80],
                "image_sha256": "1" * 64,
                "sidecar_sha256": "a" * 64,
            },
            {
                "attempt": 2,
                "preprocessing": "autocontrast-cutoff-1",
                "ocr_outcome": "zero_output",
                "image_size": [120, 80],
                "image_sha256": "2" * 64,
                "sidecar_sha256": "b" * 64,
            },
            {
                "attempt": 3,
                "preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
                "ocr_outcome": "text",
                "image_size": [120, 80],
                "image_sha256": "3" * 64,
                "sidecar_sha256": "c" * 64,
                "content_scale": 0.5,
                "content_size": [50, 50],
                "content_offset": [25, 25],
                "resampling": "lanczos",
                "canvas_fill_rgb": [255, 255, 255],
            },
        ],
    }


def _third_zero_retry_fields() -> dict[str, Any]:
    fields = _third_retry_fields()
    fields.pop("geometry_coordinate_space")
    fields.pop("geometry_transform")
    fields["attempt_history"][2]["ocr_outcome"] = "zero_output"
    return fields


def test_inverse_scaled_retry_geometry_uses_actual_odd_axis_scales() -> None:
    raw = ocr_pipeline._SealedPageGeometry(
        image_name="00001.png",
        image_bbox=(0.0, 0.0, 1301.0, 1313.0),
        lines=(
            ocr_pipeline._SealedTextLine("SOLD", (351.0, 366.0, 938.0, 597.0)),
            ocr_pipeline._SealedTextLine("OUT", (468.0, 675.0, 867.0, 870.0)),
        ),
        canonical_text="soldout",
    )

    mapped = ocr_pipeline._inverse_scaled_retry_geometry(
        raw,
        source_size=[1301, 1313],
        content_size=[650, 656],
        content_offset=[325, 328],
    )

    assert mapped.image_bbox == (0.0, 0.0, 1301.0, 1313.0)
    assert mapped.lines[0].bbox == pytest.approx(
        (52.04, 76.0579268292683, 1226.943076923077, 538.4100609756098)
    )
    assert mapped.lines[1].bbox == pytest.approx(
        (286.22, 694.5289634146342, 1084.833846153846, 1084.8262195121952)
    )


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
        assert (
            Path(
                next(result.artifact_path for result in adjusted if result.engine == engine)
            ).read_text(encoding="utf-8")
            == "[SOURCE PAGE 0001]\n"
        )
        pages = json.loads((run_dir / engine / engine / "pages.json").read_text(encoding="utf-8"))
        assert pages["pages"][0]["ocr_outcome"] == "textless_graphics"
        assert pages["pages"][0]["alnum_line_count"] == 0
        assert pages["pages"][0]["alnum_chars"] == 0
        assert pages["pages"][0]["page_errors"] == []
        geometry = json.loads(
            (run_dir / engine / engine / f"page_0001.{engine}.json").read_text(encoding="utf-8")
        )
        assert geometry["images"][0]["pages"][0]["text_lines"] == []
    for report_path in result_files:
        row = json.loads(report_path.read_text(encoding="utf-8"))["results"][0]
        expected = next(result for result in adjusted if result.engine == row["engine"])
        assert row == asdict(expected)
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["status"] == "ok"
    assert reconciliation["accepted_textless_graphics_pages"] == [1]


def test_reconcile_74_accepts_explicit_graphics_with_third_zero_surya(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["Surya geometry sidecar has no text_lines"],
                **_third_zero_retry_fields(),
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
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == ("explicit_chandra_nontext_with_quiet_surya")
    assert reconciliation["accepted_textless_graphics_pages"] == [1]


@pytest.mark.parametrize(
    "defect",
    [
        "history-outcome",
        "third-sidecar-text",
        "third-image-pixels",
        "selected-sidecar-text",
        "zero-transform-marker",
    ],
)
def test_reconcile_rejects_tampered_third_zero_lineage(
    tmp_path: Path,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["Surya geometry sidecar has no text_lines"],
                **_third_zero_retry_fields(),
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    selected_path = engine_dir / "page_0001.surya.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    row = pages_payload["pages"][0]
    selected_image = selected_payload["images"][0]
    histories = (row["attempt_history"], selected_image["attempt_history"])
    third_image = Path(histories[0][2]["image_path"])
    third_sidecar = Path(histories[0][2]["sidecar_path"])

    if defect == "history-outcome":
        for history in histories:
            history[2]["ocr_outcome"] = "text"
    elif defect == "third-sidecar-text":
        payload = json.loads(third_sidecar.read_text(encoding="utf-8"))
        payload["images"][0]["pages"][0]["text_lines"] = [
            {"text": "FORGED", "bbox": [30, 30, 70, 45]}
        ]
        third_sidecar.write_text(json.dumps(payload), encoding="utf-8")
        for history in histories:
            history[2]["sidecar_sha256"] = _sha256(third_sidecar)
            history[2]["sidecar_bytes"] = third_sidecar.stat().st_size
    elif defect == "third-image-pixels":
        with Image.open(third_image) as source:
            forged = source.copy()
        forged.putpixel((0, 0), (0, 0, 0))
        forged.save(third_image, format="PNG")
        for history in histories:
            history[2]["image_sha256"] = _sha256(third_image)
            history[2]["image_bytes"] = third_image.stat().st_size
    elif defect == "selected-sidecar-text":
        selected_image["pages"][0]["text_lines"] = [{"text": "FORGED", "bbox": [10, 10, 50, 30]}]
    else:
        for payload in (row, selected_image):
            payload["geometry_coordinate_space"] = "source-image-v1"
            payload["geometry_transform"] = "inverse-actual-content-size-strict-v1"

    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_payload), encoding="utf-8")
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"].startswith("invalid_candidate_evidence:")


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
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
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
        chandra_rows=[{"source_page": 1, "text": "CHANDRA TEXT", "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"


@pytest.mark.parametrize(
    ("surya_text", "chandra_text"),
    [
        ("ＳＯＬＤ — out", "Sold\nOUT"),
        ("Straße", "STRASSE"),
        ("Cafe\u0301", "CAFÉ"),
    ],
)
def test_reconcile_accepts_third_retry_only_with_normalized_exact_agreement(
    tmp_path: Path,
    surya_text: str,
    chandra_text: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": surya_text,
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": sum(char.isalnum() for char in surya_text),
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": chandra_text, "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["accepted"] is True
    assert reconciliation["pages"][0]["reason"] == "both_text_retry_geometry_agreement"
    agreement = reconciliation["pages"][0]["retry_text_agreement"]
    assert agreement["algorithm"] == "nfkc-casefold-unicode-alnum-exact-v1"
    assert agreement["matched"] is True
    assert agreement["surya_sha256"] == agreement["chandra_sha256"]
    assert len(agreement["surya_sha256"]) == 64


def test_reconcile_canonical_text_uses_benchmark_overlay_cleanup(
    tmp_path: Path,
) -> None:
    tagged_text = "<b>SOLD</b>"
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": tagged_text,
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 4,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": tagged_text,
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 4,
            }
        ],
    )
    for engine in ("surya", "chandra"):
        (run_dir / engine / engine / "page_0001.txt").write_text(
            "SOLD",
            encoding="utf-8",
        )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "both_text_retry_geometry_agreement"


def test_reconcile_rejects_compatibility_character_that_is_not_equal_after_nfkc(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "ǰ",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 1,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "j", "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "retry_text_mismatch"


@pytest.mark.parametrize(
    ("surya_text", "attempt_count", "retry_preprocessing", "expected_reason"),
    [
        ("SOLD NOW", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
        ("SОLD OUT", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
        ("OUT SOLD", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
        ("---", 3, "rgb-scale-0.5-center-white-lanczos-v1", "invalid_retry_text_evidence"),
        ("SOLD OUT", 3, None, "invalid_retry_text_evidence"),
        ("SOLD OUT", 4, "rgb-scale-0.5-center-white-lanczos-v1", "invalid_retry_text_evidence"),
        ("SOLD OUT", 0, None, "invalid_retry_text_evidence"),
        ("SOLD OUT", -1, None, "invalid_retry_text_evidence"),
        ("SOLD OUT", False, None, "invalid_retry_text_evidence"),
        ("SOLD OUT", "1", None, "invalid_retry_text_evidence"),
        ("SOLD OUT", 1, "autocontrast-cutoff-1", "invalid_retry_text_evidence"),
        ("SOLD OUT", 2, None, "invalid_retry_text_evidence"),
        ("SOLD OUT", 2, "forged", "invalid_retry_text_evidence"),
    ],
)
def test_reconcile_rejects_unconfirmed_or_invalid_third_retry_text(
    tmp_path: Path,
    surya_text: str,
    attempt_count: object,
    retry_preprocessing: str | None,
    expected_reason: str,
) -> None:
    surya_row: dict[str, object] = {
        "source_page": 1,
        "text": surya_text,
        "ocr_outcome": "text",
        "attempt_count": attempt_count,
        "alnum_line_count": int(any(char.isalnum() for char in surya_text)),
        "alnum_chars": sum(char.isalnum() for char in surya_text),
    }
    if retry_preprocessing is not None:
        surya_row["retry_preprocessing"] = retry_preprocessing
    if attempt_count == 3 and retry_preprocessing == "rgb-scale-0.5-center-white-lanczos-v1":
        surya_row.update(_third_retry_fields())
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[surya_row],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["accepted"] is False
    assert reconciliation["pages"][0]["reason"] == expected_reason


@pytest.mark.parametrize(
    ("attempt_count", "retry_preprocessing"),
    [(1, None), (2, "autocontrast-cutoff-1")],
)
def test_reconcile_keeps_legacy_both_text_for_attempts_one_and_two(
    tmp_path: Path,
    attempt_count: int,
    retry_preprocessing: str | None,
) -> None:
    surya_row: dict[str, object] = {
        "source_page": 1,
        "text": "SURYA TEXT",
        "ocr_outcome": "text",
        "attempt_count": attempt_count,
        "alnum_line_count": 1,
        "alnum_chars": 9,
    }
    if retry_preprocessing is not None:
        surya_row["retry_preprocessing"] = retry_preprocessing
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[surya_row],
        chandra_rows=[{"source_page": 1, "text": "CHANDRA TEXT", "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "both_text"


def test_reconcile_missing_attempt_count_does_not_fall_through_both_text(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "omit_attempt_count": True,
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_retry_text_evidence"


@pytest.mark.parametrize(
    "defect",
    [
        "row-marker",
        "sidecar-attempt",
        "sidecar-marker",
        "sidecar-text",
        "artifact-text",
        "bbox-overflow",
        "bbox-bool",
        "coherent-dimension-tamper",
        "history-sha",
        "history-bytes",
        "history-sidecar-content",
        "symlink-sidecar",
        "hardlink-sidecar",
        "extra-image",
    ],
)
def test_reconcile_third_retry_durable_evidence_tamper_fails_closed(
    tmp_path: Path,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.surya.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    image = geometry_payload["images"][0]
    line = image["pages"][0]["text_lines"][0]
    row_history = pages_payload["pages"][0]["attempt_history"]
    sidecar_history = image["attempt_history"]
    if defect == "row-marker":
        pages_payload["pages"][0]["retry_preprocessing"] = "forged"
        pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    elif defect == "sidecar-attempt":
        image["attempt_count"] = 2
    elif defect == "sidecar-marker":
        image["retry_preprocessing"] = "forged"
    elif defect == "sidecar-text":
        line["text"] = "OUT SOLD"
    elif defect == "artifact-text":
        (engine_dir / "page_0001.txt").write_text("OUT SOLD", encoding="utf-8")
    elif defect == "bbox-overflow":
        line["bbox"] = [1, 2, 101, 20]
    elif defect == "bbox-bool":
        line["bbox"] = [True, 2, 50, 20]
    elif defect == "coherent-dimension-tamper":
        image["pages"][0]["image_bbox"] = [0, 0, 90, 90]
        line["bbox"] = [1, 2, 40, 20]
    elif defect == "history-sha":
        row_history[2]["image_sha256"] = "f" * 64
        sidecar_history[2]["image_sha256"] = "f" * 64
    elif defect == "history-bytes":
        row_history[2]["sidecar_bytes"] = True
        sidecar_history[2]["sidecar_bytes"] = True
    elif defect == "history-sidecar-content":
        history_sidecar = Path(sidecar_history[2]["sidecar_path"])
        history_sidecar.write_text(
            history_sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif defect in {"symlink-sidecar", "hardlink-sidecar"}:
        history_sidecar = Path(sidecar_history[1]["sidecar_path"])
        target = engine_dir / f"{defect}.json"
        target.write_bytes(history_sidecar.read_bytes())
        history_sidecar.unlink()
        try:
            if defect == "symlink-sidecar":
                os.symlink(target, history_sidecar)
            else:
                os.link(target, history_sidecar)
        except OSError as exc:
            pytest.skip(f"link creation is unavailable: {exc}")
    else:
        geometry_payload["images"].append(dict(image))
    if defect in {"history-sha", "history-bytes"}:
        pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    if defect not in {
        "row-marker",
        "history-sidecar-content",
        "symlink-sidecar",
        "hardlink-sidecar",
    }:
        geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_retry_text_evidence"


@pytest.mark.parametrize("artifact_kind", ["image", "sidecar"])
def test_reconcile_retry_parses_the_same_bytes_that_were_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
    )
    attempt_dir = run_dir / "surya" / "surya" / "page_0001.retry" / "attempt_3"
    target = (
        attempt_dir / "page_0001.png"
        if artifact_kind == "image"
        else attempt_dir / "surya_page_lines.json"
    )
    original_stable_file_bytes = ocr_pipeline._stable_file_bytes
    mutated = False

    def _mutating_stable_file_bytes(
        path: Path,
        *,
        max_bytes: int | None = None,
    ) -> tuple[bytes, dict[str, object]]:
        nonlocal mutated
        payload, fingerprint = original_stable_file_bytes(path, max_bytes=max_bytes)
        if Path(path) == target and not mutated:
            target.write_bytes(b"changed after the sealed read")
            mutated = True
        return payload, fingerprint

    monkeypatch.setattr(ocr_pipeline, "_stable_file_bytes", _mutating_stable_file_bytes)
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert mutated is True
    assert error is None
    assert all(result.status == "ok" for result in adjusted)


def test_reconcile_retry_sidecar_size_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
    )
    target = run_dir / "surya" / "surya" / "page_0001.retry" / "attempt_3" / "surya_page_lines.json"
    original_stable_file_bytes = ocr_pipeline._stable_file_bytes

    def _oversized_sidecar(
        path: Path,
        *,
        max_bytes: int | None = None,
    ) -> tuple[bytes, dict[str, object]]:
        if Path(path) == target:
            raise ValueError("sidecar exceeds the permitted size")
        return original_stable_file_bytes(path, max_bytes=max_bytes)

    monkeypatch.setattr(ocr_pipeline, "_stable_file_bytes", _oversized_sidecar)
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"


@pytest.mark.parametrize(
    "defect",
    [
        "relocated-attempt",
        "selected-split",
        "selected-merge",
        "selected-reorder",
        "selected-image-name",
    ],
)
def test_reconcile_third_retry_requires_deterministic_inverse_geometry(
    tmp_path: Path,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    selected_path = engine_dir / "page_0001.surya.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    row_history = pages_payload["pages"][0]["attempt_history"]
    selected_image = selected_payload["images"][0]
    selected_history = selected_image["attempt_history"]
    selected_page = selected_image["pages"][0]
    raw_sidecar = Path(row_history[2]["sidecar_path"])
    raw_payload = json.loads(raw_sidecar.read_text(encoding="utf-8"))
    raw_page = raw_payload["images"][0]["pages"][0]

    if defect == "relocated-attempt":
        relocated = engine_dir / "page_0001.retry" / "relocated" / raw_sidecar.name
        relocated.parent.mkdir(parents=True)
        relocated.write_bytes(raw_sidecar.read_bytes())
        for history in (row_history, selected_history):
            history[2]["sidecar_path"] = str(relocated.resolve())
            history[2]["sidecar_sha256"] = _sha256(relocated)
            history[2]["sidecar_bytes"] = relocated.stat().st_size
    elif defect == "selected-split":
        selected_page["text_lines"] = [
            {"text": "SOLD", "bbox": [0, 0, 25, 10]},
            {"text": "OUT", "bbox": [25, 0, 50, 10]},
        ]
    elif defect == "selected-merge":
        raw_page["text_lines"] = [
            {"text": "SOLD", "bbox": [0, 0, 25, 10]},
            {"text": "OUT", "bbox": [25, 0, 50, 10]},
        ]
        raw_sidecar.write_text(json.dumps(raw_payload), encoding="utf-8")
        for history in (row_history, selected_history):
            history[2]["sidecar_sha256"] = _sha256(raw_sidecar)
            history[2]["sidecar_bytes"] = raw_sidecar.stat().st_size
    elif defect == "selected-reorder":
        ordered_lines = [
            {"text": "SOLD", "bbox": [0, 0, 25, 10]},
            {"text": "SOLD", "bbox": [0, 20, 25, 30]},
        ]
        raw_page["text_lines"] = ordered_lines
        selected_page["text_lines"] = list(reversed(ordered_lines))
        (engine_dir / "page_0001.txt").write_text("SOLD\nSOLD", encoding="utf-8")
        raw_sidecar.write_text(json.dumps(raw_payload), encoding="utf-8")
        for history in (row_history, selected_history):
            history[2]["sidecar_sha256"] = _sha256(raw_sidecar)
            history[2]["sidecar_bytes"] = raw_sidecar.stat().st_size
    else:
        selected_image["image_name"] = "forged-page.png"

    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_retry_text_evidence"


def test_reconcile_rejects_self_sealed_third_retry_pixels_not_derived_from_attempt_one(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    selected_path = engine_dir / "page_0001.surya.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    histories = (
        pages_payload["pages"][0]["attempt_history"],
        selected_payload["images"][0]["attempt_history"],
    )
    candidate_path = Path(histories[0][2]["image_path"])
    with Image.open(candidate_path) as source:
        forged = source.copy()
    forged.putpixel((0, 0), (0, 0, 0))
    forged.save(candidate_path, format="PNG")
    for history in histories:
        history[2]["image_sha256"] = _sha256(candidate_path)
        history[2]["image_bytes"] = candidate_path.stat().st_size
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_retry_text_evidence"


def test_reconcile_chandra_text_uses_benchmark_bbox_reading_order(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD\nOUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD\nOUT", "ocr_outcome": "text"}],
    )
    geometry_path = run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["images"][0]["pages"][0]["text_lines"] = [
        {"text": "OUT", "bbox": [0, 20, 30, 30]},
        {"text": "SOLD", "bbox": [0, 0, 40, 10]},
    ]
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)


def test_reconcile_accepts_chandra_model_space_geometry(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    geometry_path = run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    page = payload["images"][0]["pages"][0]
    page["image_bbox"] = [0, 0, 1535, 1550]
    page["text_lines"] = [{"text": "SOLD OUT", "bbox": [191, 103, 1310, 714]}]
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["accepted"] is True
    assert reconciliation["pages"][0]["reason"] == "both_text_retry_geometry_agreement"


@pytest.mark.parametrize("defect", ["wrong-name", "line-out-of-bounds", "invalid-bbox"])
def test_reconcile_rejects_invalid_chandra_identity_or_geometry(
    tmp_path: Path,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **_third_retry_fields(),
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    geometry_path = run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    image = payload["images"][0]
    if defect == "wrong-name":
        image["image_name"] = "other-page.png"
    elif defect == "line-out-of-bounds":
        image["pages"][0]["image_bbox"] = [0, 0, 40, 100]
    else:
        image["pages"][0]["text_lines"][0]["bbox"] = [0, 0, 0, 10]
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_retry_text_evidence"


@pytest.mark.parametrize(
    ("surya_row", "chandra_row", "expected_reason"),
    [
        (
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "verified_blank",
                **_third_retry_fields(),
            },
            {"source_page": 1, "text": "", "ocr_outcome": "verified_blank"},
            "third_retry_nontext_forbidden",
        ),
        (
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "page_errors": ["zero"],
                **_third_retry_fields(),
            },
            _explicit_graphics_page(1),
            "invalid_candidate_evidence: Surya zero-output retry has a coordinate transform marker",
        ),
        (
            {
                "source_page": 1,
                "text": "(A)",
                "ocr_outcome": "text",
                "omit_attempt_count": True,
                "alnum_line_count": 1,
                "alnum_chars": 1,
            },
            _explicit_graphics_page(1),
            "invalid_surya_attempt_evidence",
        ),
        (
            {
                "source_page": 1,
                "text": "(A)",
                "ocr_outcome": "text",
                "attempt_count": 1,
                "retry_preprocessing": "forged",
                "alnum_line_count": 1,
                "alnum_chars": 1,
            },
            _explicit_graphics_page(1),
            "invalid_surya_attempt_evidence",
        ),
        (
            {
                "source_page": 1,
                "text": "(A)",
                "ocr_outcome": "text",
                "attempt_count": 1,
                "attempt_history": _third_retry_fields()["attempt_history"],
                "alnum_line_count": 1,
                "alnum_chars": 1,
            },
            _explicit_graphics_page(1),
            "invalid_surya_attempt_evidence",
        ),
        (
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "attempt_count": 3,
                "retry_preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
                "retry_policy": (
                    "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
                ),
                "selected_attempt": 3,
                "page_errors": ["zero"],
            },
            _explicit_graphics_page(1),
            "invalid_surya_attempt_evidence",
        ),
        (
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "attempt_count": 3,
                "retry_preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
                "retry_policy": (
                    "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
                ),
                "selected_attempt": 3,
                "attempt_history": [],
                "page_errors": ["zero"],
            },
            _explicit_graphics_page(1),
            "invalid_surya_attempt_evidence",
        ),
    ],
)
def test_reconcile_never_falls_through_invalid_attempt_metadata_in_nontext_branches(
    tmp_path: Path,
    surya_row: dict[str, Any],
    chandra_row: dict[str, Any],
    expected_reason: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[surya_row],
        chandra_rows=[chandra_row],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == expected_reason


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
    assert {row["reason"] for row in report["pages"]} == {"graphics_recovery_cap_exceeded"}


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
        ocr_pipeline._publish_file_transaction({first: b"first-new", second: b"second-new"})

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
    assert ocr_pipeline._HYBRID_CHUNK_PIPELINE_REVISION == "chandra-surya-resumable-v6"
    assert ocr_pipeline._HYBRID_CHUNK_MANIFEST_SCHEMA == "uniscan.hybrid-chunks.v4"
    assert config["zero_output_retry_policy"] == (
        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
    )
    assert config["page_reconciliation_policy"] == (
        "explicit-chandra-nontext+quiet-surya+scaled-terminal-lineage-v5"
    )


def test_hybrid_cache_identity_changes_from_legacy_v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"sealed source")
    kwargs = {
        "input_path": source,
        "mode": "chandra+surya",
        "lang": "rus+eng",
        "strict": True,
        "delete_original_text_layer": True,
        "chunk_pages": 10,
        "page_count": 1,
    }
    _, current_key = ocr_pipeline._hybrid_run_identity(**kwargs)
    current_config = ocr_pipeline._hybrid_runtime_config()

    monkeypatch.setattr(
        ocr_pipeline,
        "_HYBRID_CHUNK_PIPELINE_REVISION",
        "chandra-surya-resumable-v3",
    )
    monkeypatch.setattr(
        ocr_pipeline,
        "_hybrid_runtime_config",
        lambda: {**current_config, "zero_output_retry_policy": "legacy-v1"},
    )
    _, legacy_key = ocr_pipeline._hybrid_run_identity(**kwargs)

    assert current_key != legacy_key
