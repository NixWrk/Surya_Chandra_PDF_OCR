from __future__ import annotations

from dataclasses import asdict
import html
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import unicodedata

import pytest
from PIL import Image, ImageOps

from uniscan.app import ocr_pipeline
from uniscan.app.ocr_pipeline import BasicOcrRunSummary, _ensure_requested_engines_succeeded
import uniscan.ocr.benchmark as ocr_benchmark
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


_CHANDRA_PROMPT_SHA256 = {
    "ocr_layout": "025935f3e1de1acdfadd4c7d581ab17eb82e8caaffef7b64962621c80b7ca9a8",
    "ocr": "5bddc652e7f39d5c5501e5455ea2487ebb5cdbf71832b5dfb749ec2413b5a69b",
}
_CHANDRA_RETRY_POLICY = "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
_SURYA_RETRY_POLICY = "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
_CHANDRA_CONTENT_FILTERS = {
    "ocr_layout": "skip-graphic-labels-v1",
    "ocr": "visible-text-tags-exclude-media-description-v1",
}


def _empty_alternative_text_evidence() -> dict[str, str]:
    return ocr_benchmark._chandra_alternative_text_evidence(
        raw_result={"html": "", "markdown": ""},
        texts=[],
        ignored_graphic_lines=[],
    )


def _canonical_alnum(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = unicodedata.normalize("NFKC", normalized.casefold())
    return "".join(char for char in normalized if char.isalnum())


def _test_source_raster_image(source_page: int, *, verified_blank: bool) -> Image.Image:
    if verified_blank:
        return Image.new("RGB", (100, 100), color=(255, 255, 255))
    base = 60 + (source_page % 40)
    image = Image.new("RGB", (100, 100), color=(base, base, base))
    image.paste((120, 120, 120), (20, 20, 80, 80))
    return image


def _test_source_raster_identity(
    source_page: int,
    *,
    verified_blank: bool,
) -> dict[str, object]:
    image = _test_source_raster_image(source_page, verified_blank=verified_blank)
    return {
        "pixel_sha256": ocr_benchmark._canonical_rgb_pixel_sha256(image),
        "width": image.width,
        "height": image.height,
        "name": f"page_{source_page:04d}.png",
        "source_page": source_page,
        "verified_blank": verified_blank,
    }


def _test_chandra_input_image(source: Image.Image) -> Image.Image:
    min_dim = ocr_benchmark._CHANDRA_MIN_IMAGE_DIM
    if source.width >= min_dim and source.height >= min_dim:
        return source.copy()
    scale = min_dim / float(min(source.width, source.height))
    expected_size = (int(source.width * scale), int(source.height * scale))
    return source.resize(expected_size, Image.Resampling.LANCZOS)


def _fixture_geometry_lines(
    text: str,
    *,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    lines = text.splitlines() or ([text] if text else [])
    if not lines:
        return []
    x0, y0, x1, y1 = bbox
    line_height = (y1 - y0) / len(lines)
    return [
        {
            "text": line,
            "bbox": [x0, y0 + index * line_height, x1, y0 + (index + 1) * line_height],
        }
        for index, line in enumerate(lines)
    ]


def _write_source_raster_artifact(
    *,
    engine_dir: Path,
    source_page: int,
    engine: str,
    source_image: Image.Image,
) -> dict[str, object]:
    source_path = engine_dir / f"page_{source_page:04d}.{engine}-source" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.save(source_path, format="PNG")
    return {
        "path": str(source_path.resolve()),
        "sha256": _sha256(source_path),
        "bytes": source_path.stat().st_size,
    }


def _chandra_v8_fields(
    *,
    text: str,
    outcome: str,
    attempt_count: int | None,
    source_raster_identity: dict[str, object],
) -> dict[str, Any]:
    if attempt_count is None:
        attempt_count = 1 if outcome in {"text", "verified_blank"} else 3
    prompts = ["ocr_layout", "ocr_layout", "ocr"][:attempt_count]
    preprocessing = ["original", "autocontrast-cutoff-1", "original"][:attempt_count]
    attempts: list[dict[str, Any]] = []
    for index, (prompt_type, preprocessing_name) in enumerate(
        zip(prompts, preprocessing, strict=True),
        start=1,
    ):
        selected_text = text if outcome == "text" and index == attempt_count else ""
        if selected_text:
            labels = ["text"] if prompt_type == "ocr_layout" else ["image"]
        elif outcome == "explicit_nontext":
            labels = ["image"]
        else:
            labels = []
        explicit_nontext = bool(labels) and not selected_text
        selected_evidence_lines = [
            cleaned
            for line in selected_text.splitlines() or [selected_text]
            if (cleaned := ocr_pipeline._clean_overlay_line(line))
        ]
        canonical_text = _canonical_alnum("\n".join(selected_evidence_lines))
        attempts.append(
            {
                "attempt": index,
                "source_raster_identity": dict(source_raster_identity),
                "prompt_type": prompt_type,
                "prompt_sha256": _CHANDRA_PROMPT_SHA256[prompt_type],
                "preprocessing": preprocessing_name,
                "content_filter_policy": _CHANDRA_CONTENT_FILTERS[prompt_type],
                "alternative_text_evidence": _empty_alternative_text_evidence(),
                "labels": labels,
                "text_chars": sum(len(line) for line in selected_evidence_lines),
                "canonical_alnum_chars": len(canonical_text),
                "canonical_alnum_sha256": hashlib.sha256(
                    canonical_text.encode("utf-8")
                ).hexdigest(),
                "geometry_lines": len(selected_evidence_lines),
                "explicit_nontext": explicit_nontext,
                "ocr_outcome": (
                    "text"
                    if selected_text
                    else ("explicit_nontext" if explicit_nontext else "zero_output")
                ),
            }
        )
    fields: dict[str, Any] = {
        "attempt_count": attempt_count,
        "terminal_attempt": attempt_count,
        "chandra_retry_policy": _CHANDRA_RETRY_POLICY,
        "attempts": attempts,
        "explicit_nontext": outcome == "explicit_nontext",
        "chandra_non_text_labels": (
            ["image"] if any("image" in attempt["labels"] for attempt in attempts) else []
        ),
    }
    if outcome == "text":
        fields["selected_attempt"] = attempt_count
    if attempt_count == 2:
        fields["retry_preprocessing"] = "autocontrast-cutoff-1"
    elif attempt_count == 3:
        fields["retry_preprocessing"] = "plain-ocr-original-v1"
    return fields


def _write_chandra_attempt_history(
    *,
    engine_dir: Path,
    source_page: int,
    attempts: list[dict[str, Any]],
    selected_text: str,
    source_raster_identity: dict[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, object]], dict[str, object]]:
    original = _test_source_raster_image(
        source_page,
        verified_blank=bool(source_raster_identity["verified_blank"]),
    )
    source_artifact = _write_source_raster_artifact(
        engine_dir=engine_dir,
        source_page=source_page,
        engine="chandra",
        source_image=original,
    )
    model_original = _test_chandra_input_image(original)
    images = [model_original]
    if len(attempts) >= 2:
        images.append(ImageOps.autocontrast(model_original, cutoff=1))
    if len(attempts) == 3:
        images.append(model_original.copy())
    history: list[dict[str, Any]] = []
    terminal_lines: list[dict[str, object]] = []
    for attempt, (evidence, attempt_image) in enumerate(
        zip(attempts, images, strict=True),
        start=1,
    ):
        attempt_dir = engine_dir / f"page_{source_page:04d}.chandra-attempts" / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True)
        image_path = attempt_dir / "input.png"
        attempt_image.save(image_path, format="PNG")
        texts: list[str] = []
        text_lines: list[dict[str, object]] = []
        chunks: list[dict[str, object]] = []
        labels = list(evidence["labels"])
        if evidence["ocr_outcome"] == "text":
            cleaned_lines = [
                cleaned
                for line in selected_text.splitlines() or [selected_text]
                if (cleaned := ocr_pipeline._clean_overlay_line(line))
            ]
            if evidence["prompt_type"] == "ocr":
                content = "<p>" + "<br/>".join(html.escape(line) for line in cleaned_lines) + "</p>"
                label = "Image"
            else:
                content = "<br/>".join(html.escape(line) for line in cleaned_lines)
                label = "Text"
            chunks = [{"label": label, "content": content, "bbox": [0.0, 0.0, 50.0, 10.0]}]
            texts = cleaned_lines
            text_lines = ocr_benchmark._chandra_expand_chunk_to_line_boxes(
                lines=texts,
                bbox=[0.0, 0.0, 50.0, 10.0],
            )
        elif labels:
            chunks = [
                {
                    "label": label.title(),
                    "content": "",
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                }
                for label in labels
            ]
        raw_result = {"error": False, "chunks": chunks, "html": "", "markdown": ""}
        sidecar_path = attempt_dir / "chandra_attempt.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema": "uniscan.chandra-attempt.v2",
                    "source_raster_identity": dict(source_raster_identity),
                    "image_name": "input.png",
                    "image_bbox": [
                        0.0,
                        0.0,
                        float(attempt_image.width),
                        float(attempt_image.height),
                    ],
                    "raw_result": raw_result,
                    "parsed": {
                        "texts": texts,
                        "text_lines": text_lines,
                        "labels": sorted(set(labels)),
                    },
                    "evidence": evidence,
                }
            ),
            encoding="utf-8",
        )
        history.append(
            {
                "attempt": attempt,
                "image_size": [attempt_image.width, attempt_image.height],
                "image_path": str(image_path.resolve()),
                "image_sha256": _sha256(image_path),
                "image_bytes": image_path.stat().st_size,
                "sidecar_path": str(sidecar_path.resolve()),
                "sidecar_sha256": _sha256(sidecar_path),
                "sidecar_bytes": sidecar_path.stat().st_size,
            }
        )
        if attempt == len(attempts):
            terminal_lines = text_lines
    return history, terminal_lines, source_artifact


def _build_reconciliation_run(
    tmp_path: Path,
    *,
    surya_rows: list[dict[str, Any]],
    chandra_rows: list[dict[str, Any]],
    retry_leaf_metadata: bool = False,
) -> tuple[Path, list[OcrBenchmarkResult], list[Path]]:
    run_dir = tmp_path / "run"
    results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    all_input_rows = [*surya_rows, *chandra_rows]
    source_identities: dict[int, dict[str, object]] = {}
    for source_page in {int(row["source_page"]) for row in all_input_rows}:
        page_rows = [row for row in all_input_rows if int(row["source_page"]) == source_page]
        explicit_blank = [
            bool(row["source_verified_blank"])
            for row in page_rows
            if "source_verified_blank" in row
        ]
        verified_blank = (
            explicit_blank[-1]
            if explicit_blank
            else any(str(row.get("ocr_outcome") or "") == "verified_blank" for row in page_rows)
        )
        source_identities[source_page] = _test_source_raster_identity(
            source_page,
            verified_blank=verified_blank,
        )
    for engine, raw_rows in (("surya", surya_rows), ("chandra", chandra_rows)):
        output_dir = run_dir / engine
        engine_dir = output_dir / engine
        engine_dir.mkdir(parents=True)
        page_rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            raw_row = dict(raw_row)
            source_page = int(raw_row["source_page"])
            text = str(raw_row.get("text") or "")
            source_raster_identity = dict(source_identities[source_page])
            raw_row["source_raster_identity"] = source_raster_identity
            selected_image_size = [100, 100]
            if engine == "chandra":
                chandra_defaults = _chandra_v8_fields(
                    text=text,
                    outcome=str(raw_row["ocr_outcome"]),
                    attempt_count=raw_row.get("attempt_count"),
                    source_raster_identity=source_raster_identity,
                )
                for key, value in chandra_defaults.items():
                    raw_row.setdefault(key, value)
                (
                    chandra_history,
                    chandra_terminal_lines,
                    chandra_source_artifact,
                ) = _write_chandra_attempt_history(
                    engine_dir=engine_dir,
                    source_page=source_page,
                    attempts=raw_row["attempts"],
                    selected_text=text,
                    source_raster_identity=source_raster_identity,
                )
                raw_row["attempt_history"] = chandra_history
                raw_row["source_raster_artifact"] = chandra_source_artifact
                selected_image_size = list(chandra_history[-1]["image_size"])
            else:
                chandra_terminal_lines = []
                raw_row["source_raster_artifact"] = _write_source_raster_artifact(
                    engine_dir=engine_dir,
                    source_page=source_page,
                    engine="surya",
                    source_image=_test_source_raster_image(
                        source_page,
                        verified_blank=bool(source_raster_identity["verified_blank"]),
                    ),
                )
                attempt_count = raw_row.get("attempt_count", 1)
                if attempt_count == 1:
                    raw_row.setdefault("retry_policy", _SURYA_RETRY_POLICY)
                    raw_row.setdefault("selected_attempt", 1)
                    if (
                        "attempt_history" not in raw_row
                        and raw_row.get("omit_attempt_history") is not True
                    ):
                        raw_row["attempt_history"] = [
                            {
                                "attempt": 1,
                                "preprocessing": "original",
                                "ocr_outcome": str(raw_row["ocr_outcome"]),
                            }
                        ]
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
                    if attempt == 1:
                        candidate = _test_source_raster_image(
                            source_page,
                            verified_blank=bool(source_raster_identity["verified_blank"]),
                        )
                        candidate.save(image_path, format="PNG")
                    elif attempt == 2:
                        attempt_one_path = (
                            engine_dir
                            / f"page_{source_page:04d}.retry"
                            / "attempt_1"
                            / f"page_{source_page:04d}.png"
                        )
                        with Image.open(attempt_one_path) as original:
                            candidate = ImageOps.autocontrast(original.convert("RGB"), cutoff=1)
                        candidate.save(image_path, format="PNG")
                    else:
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
                    attempt_text = text if attempt == len(raw_history) else ""
                    sidecar_path = attempt_dir / "surya_page_lines.json"
                    attempt_image: dict[str, Any] = {
                        "image_name": image_path.name,
                        "pages": [
                            {
                                "image_bbox": [0, 0, 100, 100],
                                "text_lines": _fixture_geometry_lines(
                                    attempt_text,
                                    bbox=(30.0, 30.0, 50.0, 40.0),
                                ),
                            }
                        ],
                    }
                    attempt_image.update(
                        {
                            "ocr_outcome": item["ocr_outcome"],
                            "attempt_count": attempt,
                        }
                    )
                    if attempt > 1:
                        attempt_image["retry_preprocessing"] = item["preprocessing"]
                    sidecar_path.write_text(
                        json.dumps(
                            {
                                "execution_path": "module",
                                "images": [attempt_image],
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
            page_file.write_bytes(text.encode("utf-8"))
            geometry_file = engine_dir / f"page_{source_page:04d}.{engine}.json"
            outcome = str(raw_row["ocr_outcome"])
            omit_attempt_count = raw_row.get("omit_attempt_count") is True
            attempt_count = raw_row.get("attempt_count", 1)
            image_evidence: dict[str, Any] = {
                "image_name": f"page_{source_page:04d}.png",
                "source_raster_identity": source_raster_identity,
                "source_raster_artifact": raw_row["source_raster_artifact"],
                "ocr_outcome": outcome,
                "pages": [
                    {
                        "image_bbox": [0, 0, *selected_image_size],
                        "ocr_outcome": outcome,
                        "text_lines": (
                            chandra_terminal_lines
                            if engine == "chandra"
                            else _fixture_geometry_lines(
                                text,
                                bbox=(
                                    (
                                        (10.0, 10.0, 50.0, 30.0)
                                        if attempt_count == 3
                                        else (30.0, 30.0, 50.0, 40.0)
                                    )
                                    if isinstance(raw_history, list)
                                    else (0.0, 0.0, 50.0, 10.0)
                                ),
                            )
                        ),
                    }
                ],
            }
            if not omit_attempt_count:
                image_evidence["attempt_count"] = attempt_count
            if engine == "chandra":
                explicit_nontext = bool(raw_row.get("explicit_nontext", False))
                image_evidence["explicit_nontext"] = explicit_nontext
                image_evidence["chandra_non_text_labels"] = raw_row["chandra_non_text_labels"]
                image_evidence["terminal_attempt"] = raw_row["terminal_attempt"]
                image_evidence["chandra_retry_policy"] = raw_row["chandra_retry_policy"]
                image_evidence["attempts"] = raw_row["attempts"]
            for key in (
                "retry_preprocessing",
                "terminal_attempt",
                "chandra_retry_policy",
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
                "source_raster_identity": source_raster_identity,
                "source_raster_artifact": raw_row["source_raster_artifact"],
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
                "terminal_attempt",
                "chandra_retry_policy",
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
        (engine_dir / "all_pages.txt").write_bytes(aggregate_text.encode("utf-8"))
        artifact_path = output_dir / f"document_{engine}.txt"
        artifact_path.write_bytes(aggregate_text.encode("utf-8"))
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


def _second_retry_fields() -> dict[str, Any]:
    fields = _third_retry_fields()
    fields["attempt_count"] = 2
    fields["retry_preprocessing"] = "autocontrast-cutoff-1"
    fields["selected_attempt"] = 2
    fields.pop("geometry_coordinate_space")
    fields.pop("geometry_transform")
    fields["attempt_history"] = fields["attempt_history"][:2]
    fields["attempt_history"][1]["ocr_outcome"] = "text"
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


@pytest.mark.parametrize("retry_leaf_metadata", [False, True])
def test_reconcile_74_accepts_explicit_graphics_with_third_zero_surya(
    tmp_path: Path,
    retry_leaf_metadata: bool,
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
        retry_leaf_metadata=retry_leaf_metadata,
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
    surya_dir = run_dir / "surya" / "surya"
    surya_pages = json.loads((surya_dir / "pages.json").read_text(encoding="utf-8"))
    surya_geometry = json.loads((surya_dir / "page_0001.surya.json").read_text(encoding="utf-8"))
    assert "selected_attempt" not in surya_pages["pages"][0]
    assert "selected_attempt" not in surya_geometry["images"][0]
    assert len(surya_pages["pages"][0]["attempt_history"]) == 3
    assert len(surya_geometry["images"][0]["attempt_history"]) == 3
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == ("explicit_chandra_nontext_with_quiet_surya")
    assert reconciliation["accepted_textless_graphics_pages"] == [1]


def test_reconcile_accepts_sparse_surya_when_chandra_exhausts_with_zero_output(
    tmp_path: Path,
) -> None:
    surya_text = "Profession of\nPhonester"
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": surya_text,
                "ocr_outcome": "text",
                "alnum_line_count": 2,
                "alnum_chars": 21,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "alnum_line_count": 0,
                "alnum_chars": 0,
                "page_errors": ["Chandra geometry sidecar has no text_lines"],
            }
        ],
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
    for engine in ("surya", "chandra"):
        engine_dir = run_dir / engine / engine
        pages = json.loads((engine_dir / "pages.json").read_text(encoding="utf-8"))
        assert pages["pages"][0]["ocr_outcome"] == "textless_graphics"
        assert (engine_dir / f"page_0001.{engine}.json").is_file()
        assert (engine_dir / "page_0001.txt").read_text(encoding="utf-8") == ""
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == (
        "exhausted_chandra_zero_with_sparse_surya"
    )
    assert reconciliation["accepted_textless_graphics_pages"] == [1]


@pytest.mark.parametrize(
    ("surya_text", "alnum_lines", "alnum_chars"),
    [
        ("one\ntwo\nthree", 3, 11),
        ("abcdefghijklmnopqrstuvwxy", 1, 25),
    ],
)
def test_reconcile_rejects_dense_surya_when_chandra_exhausts_with_zero_output(
    tmp_path: Path,
    surya_text: str,
    alnum_lines: int,
    alnum_chars: int,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": surya_text,
                "ocr_outcome": "text",
                "alnum_line_count": alnum_lines,
                "alnum_chars": alnum_chars,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "",
                "ocr_outcome": "zero_output",
                "alnum_line_count": 0,
                "alnum_chars": 0,
                "page_errors": ["Chandra geometry sidecar has no text_lines"],
            }
        ],
    )

    _adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["pages"][0]["reason"] == (
        "exhausted_chandra_zero_with_dense_surya"
    )
    assert reconciliation["accepted_textless_graphics_pages"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "history-outcome",
        "third-sidecar-text",
        "third-image-pixels",
        "selected-sidecar-text",
        "zero-transform-marker",
        "leaf-outcome",
        "leaf-count",
        "leaf-preprocessing",
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
    elif defect == "zero-transform-marker":
        for payload in (row, selected_image):
            payload["geometry_coordinate_space"] = "source-image-v1"
            payload["geometry_transform"] = "inverse-actual-content-size-strict-v1"
    else:
        payload = json.loads(third_sidecar.read_text(encoding="utf-8"))
        leaf_image = payload["images"][0]
        field, value = {
            "leaf-outcome": ("ocr_outcome", "text"),
            "leaf-count": ("attempt_count", 2),
            "leaf-preprocessing": ("retry_preprocessing", "forged"),
        }[defect]
        leaf_image[field] = value
        third_sidecar.write_text(json.dumps(payload), encoding="utf-8")
        for history in histories:
            history[2]["sidecar_sha256"] = _sha256(third_sidecar)
            history[2]["sidecar_bytes"] = third_sidecar.stat().st_size

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
    reason = reconciliation["pages"][0]["reason"]
    assert reason == "invalid_surya_attempt_evidence"


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


def test_reconcile_t45_rejects_dense_surya_with_explicit_chandra_graphics(
    tmp_path: Path,
) -> None:
    dense_text = "\n".join(f"hallucinated line {index}" for index in range(1, 52))
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": dense_text,
                "ocr_outcome": "text",
                "alnum_line_count": 51,
                "alnum_chars": sum(char.isalnum() for char in dense_text),
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
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == ("explicit_chandra_nontext_with_dense_surya")


@pytest.mark.parametrize(
    "defect",
    [
        "prompt-type",
        "prompt-sha",
        "preprocessing",
        "attempt-order",
        "union-label",
        "selected-attempt",
        "retry-policy",
        "attempt-count",
    ],
)
def test_reconcile_rejects_tampered_chandra_v8_attempt_lineage(
    tmp_path: Path,
    defect: str,
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
    engine_dir = run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.chandra.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    row = pages_payload["pages"][0]
    image = geometry_payload["images"][0]
    attempts = image["attempts"]

    if defect == "prompt-type":
        attempts[2]["prompt_type"] = "ocr_layout"
    elif defect == "prompt-sha":
        attempts[2]["prompt_sha256"] = "0" * 64
    elif defect == "preprocessing":
        attempts[1]["preprocessing"] = "original"
    elif defect == "attempt-order":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    elif defect == "union-label":
        attempts[1].update(
            {
                "labels": [],
                "explicit_nontext": False,
                "ocr_outcome": "zero_output",
            }
        )
    elif defect == "selected-attempt":
        row["selected_attempt"] = 3
        image["selected_attempt"] = 3
    elif defect == "retry-policy":
        row["chandra_retry_policy"] = "forged-v1"
        image["chandra_retry_policy"] = "forged-v1"
    else:
        row["attempt_count"] = 2
        row["terminal_attempt"] = 2
        image["attempt_count"] = 2
        image["terminal_attempt"] = 2

    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_chandra_attempt_evidence"


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


def test_reconcile_rejects_tampered_chandra_text_on_nonretry_surya(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    geometry_path = run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry["images"][0]["pages"][0]["text_lines"][0]["text"] = "FORGED"
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_chandra_attempt_evidence"


def test_reconcile_rejects_missing_surya_attempt_one_history(tmp_path: Path) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
        chandra_rows=[{"source_page": 1, "text": "SOLD OUT", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.surya.json"
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    pages["pages"][0].pop("attempt_history")
    geometry["images"][0].pop("attempt_history")
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

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
    ("surya_text", "chandra_text"),
    [
        ("Ｄｏｓｅ 1.5 mg", "Dose 1.5 mg"),
        ("ＳＯＬＤ — out", "SOLD — out"),
        ("Straße", "Straße"),
        ("Cafe\u0301", "Café"),
    ],
)
@pytest.mark.parametrize("retry_leaf_metadata", [False, True])
def test_reconcile_accepts_third_retry_only_with_normalized_exact_agreement(
    tmp_path: Path,
    surya_text: str,
    chandra_text: str,
    retry_leaf_metadata: bool,
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
        retry_leaf_metadata=retry_leaf_metadata,
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
    assert agreement["algorithm"] == "nfkc-layout-whitespace-exact-v1"
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
        engine_dir = run_dir / engine / engine
        (engine_dir / "page_0001.txt").write_text(
            "SOLD",
            encoding="utf-8",
        )
        pages_path = engine_dir / "pages.json"
        pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
        pages_payload["pages"][0]["text_chars"] = len("SOLD")
        pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")

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
        ("sold out", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
        ("SOLDOUT", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
        ("SOLD OUT!", 3, "rgb-scale-0.5-center-white-lanczos-v1", "retry_text_mismatch"),
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
    if attempt_count == 2:
        surya_row.update(_second_retry_fields())
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
    assert error is not None
    assert "seal is invalid" in error
    assert adjusted == results


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


@pytest.mark.parametrize("attempt_count", [2, 3])
@pytest.mark.parametrize("history_index", [0, 1])
def test_reconcile_rejects_self_sealed_surya_source_or_autocontrast_lineage(
    tmp_path: Path,
    attempt_count: int,
    history_index: int,
) -> None:
    retry_fields = _second_retry_fields() if attempt_count == 2 else _third_retry_fields()
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "SOLD OUT",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
                **retry_fields,
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
    image_path = Path(histories[0][history_index]["image_path"])
    with Image.open(image_path) as source:
        forged = source.convert("RGB")
    forged.putpixel((0, 0), (1, 2, 3))
    forged.save(image_path, format="PNG")
    for history in histories:
        history[history_index]["image_sha256"] = _sha256(image_path)
        history[history_index]["image_bytes"] = image_path.stat().st_size
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    row = reconciliation["pages"][0]
    assert row["reason"] == "invalid_retry_text_evidence"
    assert f"attempt {history_index + 1} pixels" in row["retry_evidence_error"]


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
    pages_path = run_dir / "chandra" / "chandra" / "pages.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    image = payload["images"][0]
    reordered_lines = [
        {"text": "OUT", "bbox": [0, 20, 30, 30]},
        {"text": "SOLD", "bbox": [0, 0, 40, 10]},
    ]
    image["pages"][0]["text_lines"] = reordered_lines
    image["attempts"][-1]["geometry_lines"] = 2
    attempt_path = Path(image["attempt_history"][-1]["sidecar_path"])
    attempt_payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_payload["raw_result"]["chunks"] = [
        {"label": "Text", "content": "OUT", "bbox": [0, 20, 30, 30]},
        {"label": "Text", "content": "SOLD", "bbox": [0, 0, 40, 10]},
    ]
    attempt_payload["parsed"]["texts"] = ["OUT", "SOLD"]
    attempt_payload["parsed"]["text_lines"] = reordered_lines
    attempt_payload["evidence"] = image["attempts"][-1]
    attempt_path.write_text(json.dumps(attempt_payload), encoding="utf-8")
    for history in (
        image["attempt_history"],
        pages_payload["pages"][0]["attempt_history"],
    ):
        history[-1]["sidecar_sha256"] = _sha256(attempt_path)
        history[-1]["sidecar_bytes"] = attempt_path.stat().st_size
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert all(result.status == "ok" for result in adjusted)


@pytest.mark.parametrize(
    "forged_text",
    [
        "DOSE 1.5 MG",
        "Dose;1.5 mg",
        "Dose 1,5 mg",
    ],
)
@pytest.mark.parametrize(
    "engine,expected_reason",
    [
        ("surya", "invalid_retry_text_evidence"),
        ("chandra", "invalid_chandra_attempt_evidence"),
    ],
)
def test_reconcile_rejects_same_canonical_but_non_deterministic_page_bytes(
    tmp_path: Path,
    forged_text: str,
    engine: str,
    expected_reason: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "Dose 1.5 mg",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 8,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "Dose 1.5 mg",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 8,
            }
        ],
    )
    assert _canonical_alnum(forged_text) == _canonical_alnum("Dose 1.5 mg")
    (run_dir / engine / engine / "page_0001.txt").write_bytes(forged_text.encode("utf-8"))

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == expected_reason


def test_reconcile_accepts_official_chandra_dehyphenated_page_artifact(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "biomechanics",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 12,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "bio-\nmechanics",
                "ocr_outcome": "text",
                "alnum_line_count": 2,
                "alnum_chars": 12,
            }
        ],
    )
    chandra_dir = run_dir / "chandra"
    (chandra_dir / "chandra" / "page_0001.txt").write_bytes(b"biomechanics")
    pages_path = chandra_dir / "chandra" / "pages.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    pages_payload["pages"][0]["text_chars"] = len("biomechanics")
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    aggregate = _markerized([{"source_page": 1, "text": "biomechanics"}]).encode("utf-8")
    (chandra_dir / "chandra" / "all_pages.txt").write_bytes(aggregate)
    Path(str(results[1].artifact_path)).write_bytes(aggregate)

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
    pages_path = run_dir / "chandra" / "chandra" / "pages.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    image = payload["images"][0]
    page = image["pages"][0]
    page["image_bbox"] = [0, 0, 1536, 1536]
    page["text_lines"] = [{"text": "SOLD OUT", "bbox": [190, 100, 1310, 700]}]
    image_path = Path(image["attempt_history"][0]["image_path"])
    source_path = Path(image["source_raster_artifact"]["path"])
    with Image.open(source_path) as source:
        model_image = source.convert("RGB").resize((1536, 1536), Image.Resampling.LANCZOS)
    model_image.save(image_path, format="PNG")
    attempt_path = Path(image["attempt_history"][0]["sidecar_path"])
    attempt_payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_payload["image_bbox"] = [0, 0, 1536, 1536]
    attempt_payload["raw_result"]["chunks"][0]["bbox"] = [190, 100, 1310, 700]
    attempt_payload["parsed"]["text_lines"] = page["text_lines"]
    attempt_path.write_text(json.dumps(attempt_payload), encoding="utf-8")
    for history in (
        image["attempt_history"],
        pages_payload["pages"][0]["attempt_history"],
    ):
        history[0].update(
            {
                "image_size": [1536, 1536],
                "image_sha256": _sha256(image_path),
                "image_bytes": image_path.stat().st_size,
                "sidecar_sha256": _sha256(attempt_path),
                "sidecar_bytes": attempt_path.stat().st_size,
            }
        )
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")

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


@pytest.mark.parametrize(
    ("defect", "expected_reason"),
    [
        ("wrong-name", "invalid_chandra_attempt_evidence"),
        ("line-out-of-bounds", "invalid_chandra_attempt_evidence"),
        ("invalid-bbox", "invalid_chandra_attempt_evidence"),
    ],
)
def test_reconcile_rejects_invalid_chandra_identity_or_geometry(
    tmp_path: Path,
    defect: str,
    expected_reason: str,
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
    assert reconciliation["pages"][0]["reason"] == expected_reason


def test_reconcile_rejects_chandra_page_swap_by_source_raster_identity(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {"source_page": 1, "text": "PAGE ONE", "ocr_outcome": "text"},
            {"source_page": 2, "text": "PAGE TWO", "ocr_outcome": "text"},
        ],
        chandra_rows=[
            {"source_page": 1, "text": "PAGE ONE", "ocr_outcome": "text"},
            {"source_page": 2, "text": "PAGE TWO", "ocr_outcome": "text"},
        ],
    )
    engine_dir = run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    page_one_path = engine_dir / "page_0001.chandra.json"
    page_two = json.loads((engine_dir / "page_0002.chandra.json").read_text("utf-8"))
    pages = json.loads(pages_path.read_text("utf-8"))
    page_one = json.loads(page_one_path.read_text("utf-8"))
    swapped_identity = page_two["images"][0]["source_raster_identity"]
    pages["pages"][0]["source_raster_identity"] = swapped_identity
    page_one["images"][0]["source_raster_identity"] = swapped_identity
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    page_one_path.write_text(json.dumps(page_one), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    report = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert report["pages"][0]["reason"] == "invalid_chandra_attempt_evidence"


def test_reconcile_rejects_same_name_with_different_source_pixel_digest(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[{"source_page": 1, "text": "VISIBLE", "ocr_outcome": "text"}],
        chandra_rows=[{"source_page": 1, "text": "VISIBLE", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.chandra.json"
    pages = json.loads(pages_path.read_text("utf-8"))
    geometry = json.loads(geometry_path.read_text("utf-8"))
    forged = dict(pages["pages"][0]["source_raster_identity"])
    forged["pixel_sha256"] = "f" * 64
    pages["pages"][0]["source_raster_identity"] = forged
    geometry["images"][0]["source_raster_identity"] = forged
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"


@pytest.mark.parametrize("layer", ["row", "selected", "attempt"])
def test_reconcile_rejects_source_raster_identity_layer_divergence(
    tmp_path: Path,
    layer: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[{"source_page": 1, "text": "VISIBLE", "ocr_outcome": "text"}],
        chandra_rows=[{"source_page": 1, "text": "VISIBLE", "ocr_outcome": "text"}],
    )
    engine_dir = run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.chandra.json"
    pages = json.loads(pages_path.read_text("utf-8"))
    geometry = json.loads(geometry_path.read_text("utf-8"))
    if layer == "row":
        pages["pages"][0]["source_raster_identity"]["name"] = "other.png"
    elif layer == "selected":
        geometry["images"][0]["source_raster_identity"]["name"] = "other.png"
    else:
        geometry["images"][0]["attempts"][0]["source_raster_identity"]["name"] = "other.png"
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"


def test_reconcile_rejects_nonblank_pixels_forged_as_verified_blank(
    tmp_path: Path,
) -> None:
    blank_row = {
        "source_page": 1,
        "text": "",
        "ocr_outcome": "verified_blank",
        "source_verified_blank": False,
    }
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[dict(blank_row)],
        chandra_rows=[dict(blank_row)],
    )
    for engine in ("surya", "chandra"):
        engine_dir = run_dir / engine / engine
        pages_path = engine_dir / "pages.json"
        geometry_path = engine_dir / f"page_0001.{engine}.json"
        pages = json.loads(pages_path.read_text("utf-8"))
        geometry = json.loads(geometry_path.read_text("utf-8"))
        pages["pages"][0]["source_raster_identity"]["verified_blank"] = True
        geometry["images"][0]["source_raster_identity"]["verified_blank"] = True
        pages_path.write_text(json.dumps(pages), encoding="utf-8")
        geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    report = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert report["pages"][0]["reason"] == "invalid_surya_attempt_evidence"


@pytest.mark.parametrize("defect", ["extra-page", "invalid-image-bbox"])
def test_reconcile_rejects_malformed_chandra_nontext_geometry(
    tmp_path: Path,
    defect: str,
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
    geometry_path = run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    page = payload["images"][0]["pages"][0]
    if defect == "extra-page":
        payload["images"][0]["pages"].append(dict(page))
    else:
        page["image_bbox"] = [0, 0, 0, 100]
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_chandra_attempt_evidence"


def test_reconcile_requires_agreement_when_chandra_plain_attempt_recovers_text(
    tmp_path: Path,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "DENSE SURYA HALLUCINATION",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 23,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "VISIBLE DIALOGUE",
                "ocr_outcome": "text",
                "attempt_count": 3,
                "alnum_line_count": 1,
                "alnum_chars": 15,
            }
        ],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    row = reconciliation["pages"][0]
    assert row["reason"] == "retry_text_mismatch"
    assert row["retry_text_agreement"]["matched"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "history-sha",
        "row-history-divergence",
        "raw-content",
        "parsed-text",
        "sidecar-evidence",
        "attempt2-pixels",
        "attempt2-dimensions",
        "attempt3-pixels",
        "attempt3-dimensions",
        "declared-image-bytes",
        "declared-image-dimensions",
        "non-png-image",
        "multi-frame-png",
    ],
)
def test_reconcile_chandra_durable_attempt_tamper_fails_closed(
    tmp_path: Path,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "VISIBLE DIALOGUE",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 15,
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "VISIBLE DIALOGUE",
                "ocr_outcome": "text",
                "attempt_count": 3,
                "alnum_line_count": 1,
                "alnum_chars": 15,
            }
        ],
    )
    engine_dir = run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.chandra.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    row_history = pages_payload["pages"][0]["attempt_history"]
    image_history = geometry_payload["images"][0]["attempt_history"]

    if defect == "history-sha":
        row_history[0]["image_sha256"] = "0" * 64
        image_history[0]["image_sha256"] = "0" * 64
    elif defect == "row-history-divergence":
        row_history[0]["image_bytes"] += 1
    elif defect in {"raw-content", "parsed-text", "sidecar-evidence"}:
        sidecar_path = Path(image_history[2]["sidecar_path"])
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if defect == "raw-content":
            sidecar["raw_result"]["chunks"][0]["content"] = "<p>FORGED</p>"
        elif defect == "parsed-text":
            sidecar["parsed"]["texts"] = ["FORGED"]
        else:
            sidecar["evidence"]["prompt_type"] = "ocr_layout"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        for history in (row_history, image_history):
            history[2]["sidecar_sha256"] = _sha256(sidecar_path)
            history[2]["sidecar_bytes"] = sidecar_path.stat().st_size
    elif defect in {"declared-image-bytes", "declared-image-dimensions"}:
        for history in (row_history, image_history):
            if defect == "declared-image-bytes":
                history[0]["image_bytes"] = ocr_pipeline._MAX_CHANDRA_ATTEMPT_IMAGE_BYTES + 1
            else:
                history[0]["image_size"] = [
                    ocr_pipeline._MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION + 1,
                    1,
                ]
    elif defect in {"non-png-image", "multi-frame-png"}:
        image_path = Path(image_history[0]["image_path"])
        if defect == "non-png-image":
            Image.new("RGB", (100, 100), (80, 80, 80)).save(image_path, format="JPEG")
        else:
            first = Image.new("RGB", (100, 100), (80, 80, 80))
            second = Image.new("RGB", (100, 100), (120, 120, 120))
            first.save(
                image_path,
                format="PNG",
                save_all=True,
                append_images=[second],
                duration=100,
                loop=0,
            )
        for history in (row_history, image_history):
            history[0]["image_sha256"] = _sha256(image_path)
            history[0]["image_bytes"] = image_path.stat().st_size
    else:
        index = 1 if defect.startswith("attempt2-") else 2
        image_path = Path(image_history[index]["image_path"])
        with Image.open(image_path) as source:
            forged = source.convert("RGB")
        if defect.endswith("-dimensions"):
            forged = forged.crop((0, 0, forged.width - 1, forged.height))
        else:
            forged.putpixel((0, 0), (1, 2, 3))
        forged.save(image_path, format="PNG")
        for history in (row_history, image_history):
            history[index]["image_size"] = [forged.width, forged.height]
            history[index]["image_sha256"] = _sha256(image_path)
            history[index]["image_bytes"] = image_path.stat().st_size

    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_chandra_attempt_evidence"


@pytest.mark.parametrize("engine", ["surya", "chandra"])
@pytest.mark.parametrize(
    "defect",
    [
        "declared-bytes",
        "declared-dimensions",
        "non-png",
        "multi-frame-png",
        "pixel-mismatch",
    ],
)
def test_reconcile_durable_source_raster_tamper_fails_closed(
    tmp_path: Path,
    engine: str,
    defect: str,
) -> None:
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": "VISIBLE",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 7,
            }
        ],
        chandra_rows=[_explicit_graphics_page(1)],
    )
    engine_dir = run_dir / engine / engine
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / f"page_0001.{engine}.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    row = pages_payload["pages"][0]
    image = geometry_payload["images"][0]
    row_artifact = row["source_raster_artifact"]
    image_artifact = image["source_raster_artifact"]
    source_path = Path(row_artifact["path"])

    if defect == "declared-bytes":
        forged_bytes = ocr_pipeline._MAX_CHANDRA_ATTEMPT_IMAGE_BYTES + 1
        row_artifact["bytes"] = forged_bytes
        image_artifact["bytes"] = forged_bytes
    elif defect == "declared-dimensions":
        forged_width = ocr_pipeline._MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION + 1
        row["source_raster_identity"]["width"] = forged_width
        image["source_raster_identity"]["width"] = forged_width
    else:
        if defect == "non-png":
            Image.new("RGB", (100, 100), (80, 80, 80)).save(source_path, format="JPEG")
        elif defect == "multi-frame-png":
            first = Image.new("RGB", (100, 100), (80, 80, 80))
            second = Image.new("RGB", (100, 100), (120, 120, 120))
            first.save(
                source_path,
                format="PNG",
                save_all=True,
                append_images=[second],
                duration=100,
                loop=0,
            )
        else:
            with Image.open(source_path) as source:
                forged = source.convert("RGB")
            forged.putpixel((0, 0), (1, 2, 3))
            forged.save(source_path, format="PNG")
        for artifact in (row_artifact, image_artifact):
            artifact["sha256"] = _sha256(source_path)
            artifact["bytes"] = source_path.stat().st_size

    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")
    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    expected_reason = (
        "invalid_surya_attempt_evidence"
        if engine == "surya"
        else "invalid_chandra_attempt_evidence"
    )
    assert reconciliation["pages"][0]["reason"] == expected_reason


@pytest.mark.parametrize(
    "defect",
    [
        "outcome-divergence",
        "extra-sidecar-page",
        "nontext-artifact",
        "nontext-geometry",
    ],
)
def test_reconcile_surya_selected_evidence_invariants_fail_closed(
    tmp_path: Path,
    defect: str,
) -> None:
    nontext = defect.startswith("nontext-")
    surya_row = {
        "source_page": 1,
        "text": "" if nontext else "VISIBLE",
        "ocr_outcome": "zero_output" if nontext else "text",
        "alnum_line_count": 0 if nontext else 1,
        "alnum_chars": 0 if nontext else 7,
    }
    if nontext:
        surya_row["page_errors"] = ["zero"]
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[surya_row],
        chandra_rows=[_explicit_graphics_page(1)],
    )
    engine_dir = run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.surya.json"
    pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    image = geometry_payload["images"][0]
    if defect == "outcome-divergence":
        image["ocr_outcome"] = "zero_output"
    elif defect == "extra-sidecar-page":
        image["pages"].append(json.loads(json.dumps(image["pages"][0])))
    elif defect == "nontext-artifact":
        (engine_dir / "page_0001.txt").write_bytes(b"FORGED")
    else:
        image["pages"][0]["text_lines"] = [{"text": "FORGED", "bbox": [0, 0, 50, 10]}]
    pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert adjusted == results
    assert error == "unresolved pages: [1]"
    reconciliation = json.loads((run_dir / "page_reconciliation.json").read_text("utf-8"))
    assert reconciliation["pages"][0]["reason"] == "invalid_surya_attempt_evidence"


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
            "invalid_surya_attempt_evidence",
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
            "invalid_surya_attempt_evidence",
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
    assert ocr_pipeline._HYBRID_CHUNK_PIPELINE_REVISION == "chandra-surya-resumable-v10"
    assert ocr_pipeline._HYBRID_CHUNK_MANIFEST_SCHEMA == "uniscan.hybrid-chunks.v4"
    assert config["chandra_min_image_dim"] == 1536
    assert config["zero_output_retry_policy"] == (
        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
    )
    assert config["chandra_zero_output_retry_policy"] == (
        "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
    )
    assert config["page_reconciliation_policy"] == (
        "explicit-chandra-nontext+quiet-surya+scaled-terminal-lineage+"
        "durable-source-rasters+durable-chandra-attempts+exact-text-artifacts+"
        "bounded-png+accounted-alternative-text+plain-cross-engine-agreement-v9"
    )


def test_hybrid_cache_identity_changes_from_v7(
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
    monkeypatch.setattr(
        ocr_pipeline,
        "_HYBRID_CHUNK_PIPELINE_REVISION",
        "chandra-surya-resumable-v7",
    )
    _, v7_key = ocr_pipeline._hybrid_run_identity(**kwargs)

    assert current_key != v7_key


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


def test_final_state_recomputes_unicode_text_and_rejects_counter_tamper(
    tmp_path: Path,
) -> None:
    text = "\u041a\u0438\u0440\u0438\u043b\u043b\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442"
    run_dir, results, result_files = _build_reconciliation_run(
        tmp_path,
        surya_rows=[
            {
                "source_page": 1,
                "text": text,
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": sum(char.isalnum() for char in text),
            }
        ],
        chandra_rows=[
            {
                "source_page": 1,
                "text": "ASCII text",
                "ocr_outcome": "text",
                "alnum_line_count": 1,
                "alnum_chars": 9,
            }
        ],
    )

    adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )

    assert error is None
    assert {result.engine: result.text_chars for result in adjusted} == {
        "surya": len(text),
        "chandra": len("ASCII text"),
    }
    assert ocr_pipeline._validate_final_mode_both_state(run_dir=run_dir) == frozenset()

    report_path = run_dir / "page_reconciliation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["result_text_chars"]["surya"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="text counters were not recomputed"):
        ocr_pipeline._validate_final_mode_both_state(run_dir=run_dir)


def test_final_state_recomputes_graphics_decision_instead_of_trusting_report(
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
    _adjusted, error = ocr_pipeline._reconcile_mode_both_pages(
        run_dir=run_dir,
        results=results,
        result_files=result_files,
    )
    assert error is None
    assert ocr_pipeline._validate_final_mode_both_state(run_dir=run_dir) == frozenset({1})

    report_path = run_dir / "page_reconciliation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["pages"][0]["surya_alnum_chars"] = 8
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="page decisions are not canonical"):
        ocr_pipeline._validate_final_mode_both_state(run_dir=run_dir)
