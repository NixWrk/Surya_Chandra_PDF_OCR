from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest
from PIL import Image, ImageOps

from uniscan.app import ocr_pipeline
from uniscan.app.ocr_pipeline import (
    BasicOcrRunSummary,
    SearchablePdfSummary,
    _ensure_requested_engines_succeeded,
    build_searchable_pdf,
    run_basic_ocr_benchmark,
)
from uniscan.ocr import ArtifactSearchableResult, CompareTxtBuildResult, OcrBenchmarkResult
import uniscan.ocr.benchmark as ocr_benchmark


def _ok_compare_result(engine: str, compare_path: Path) -> CompareTxtBuildResult:
    return CompareTxtBuildResult(
        engine=engine,
        status="ok",
        source_artifact_path=str(compare_path),
        compare_txt_path=str(compare_path),
    )


def _ok_artifact_result(searchable_pdf: Path, *, engine: str) -> ArtifactSearchableResult:
    return ArtifactSearchableResult(
        document=searchable_pdf.stem,
        engine=engine,
        status="ok",
        source_pdf_path=str(searchable_pdf),
        text_artifact_path=str(searchable_pdf.with_suffix(".txt")),
        searchable_pdf_path=str(searchable_pdf),
        page_count=1,
        text_chars=100,
        elapsed_seconds=0.01,
    )


def _ok_benchmark_result(engine: str) -> OcrBenchmarkResult:
    return OcrBenchmarkResult(
        engine=engine,
        status="ok",
        sample_pages=[1],
        elapsed_seconds=0.1,
        artifact_path=f"{engine}.txt",
        text_chars=100,
    )


def _fixture_source_raster_identity(
    source_page: int,
    source_image: Image.Image,
) -> dict[str, object]:
    image = source_image.convert("RGB")
    return {
        "pixel_sha256": ocr_benchmark._canonical_rgb_pixel_sha256(image),
        "width": image.width,
        "height": image.height,
        "name": f"{source_page:05d}.png",
        "source_page": source_page,
        "verified_blank": ocr_benchmark._is_effectively_blank_rgb_image(image),
    }


def _write_fixture_source_raster_artifact(
    *,
    engine_dir: Path,
    source_page: int,
    engine: str,
    source_image: Image.Image,
) -> dict[str, object]:
    source_path = engine_dir / f"page_{source_page:04d}.{engine}-source" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.convert("RGB").save(source_path, format="PNG")
    return {
        "path": str(source_path.resolve()),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "bytes": source_path.stat().st_size,
    }


def _chandra_attempt_one(
    text: str,
    *,
    source_raster_identity: dict[str, object],
) -> dict[str, object]:
    canonical = "".join(char for char in text.casefold() if char.isalnum())
    return {
        "explicit_nontext": False,
        "chandra_non_text_labels": [],
        "attempt_count": 1,
        "terminal_attempt": 1,
        "selected_attempt": 1,
        "chandra_retry_policy": (
            "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
        ),
        "attempts": [
            {
                "attempt": 1,
                "source_raster_identity": dict(source_raster_identity),
                "prompt_type": "ocr_layout",
                "prompt_sha256": (
                    "025935f3e1de1acdfadd4c7d581ab17eb82e8caaffef7b64962621c80b7ca9a8"
                ),
                "preprocessing": "original",
                "content_filter_policy": "skip-graphic-labels-v1",
                "alternative_text_evidence": ocr_benchmark._chandra_alternative_text_evidence(
                    raw_result={"html": "", "markdown": ""},
                    texts=[],
                    ignored_graphic_lines=[],
                ),
                "labels": ["text"],
                "text_chars": len(text),
                "canonical_alnum_chars": len(canonical),
                "canonical_alnum_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "geometry_lines": 1,
                "explicit_nontext": False,
                "ocr_outcome": "text",
            }
        ],
    }


def _write_chandra_attempt_one_artifact(
    *,
    engine_dir: Path,
    source_page: int,
    text: str,
    evidence: dict[str, object],
    source_raster_identity: dict[str, object],
    source_image: Image.Image,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    attempt_dir = engine_dir / f"page_{source_page:04d}.chandra-attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    image_path = attempt_dir / "input.png"
    source_image = source_image.convert("RGB")
    source_width, source_height = source_image.size
    if min(source_width, source_height) < ocr_benchmark._CHANDRA_MIN_IMAGE_DIM:
        scale = ocr_benchmark._CHANDRA_MIN_IMAGE_DIM / float(min(source_width, source_height))
        model_size = (int(source_width * scale), int(source_height * scale))
    else:
        model_size = source_image.size
    attempt_image = (
        source_image.copy()
        if model_size == source_image.size
        else source_image.resize(model_size, Image.Resampling.LANCZOS)
    )
    attempt_image.save(image_path, format="PNG")
    source_path = engine_dir / f"page_{source_page:04d}.chandra-source" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.save(source_path, format="PNG")
    source_artifact = {
        "path": str(source_path.resolve()),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "bytes": source_path.stat().st_size,
    }
    line = {"text": text, "bbox": [0.0, 0.0, 80.0, 10.0]}
    sidecar_path = attempt_dir / "chandra_attempt.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "uniscan.chandra-attempt.v2",
                "source_raster_identity": dict(source_raster_identity),
                "image_name": "input.png",
                "image_bbox": [0.0, 0.0, float(model_size[0]), float(model_size[1])],
                "raw_result": {
                    "error": False,
                    "chunks": [
                        {
                            "label": "Text",
                            "content": text,
                            "bbox": [0.0, 0.0, 80.0, 10.0],
                        }
                    ],
                    "html": "",
                    "markdown": "",
                },
                "parsed": {"texts": [text], "text_lines": [line], "labels": ["text"]},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    return [
        {
            "attempt": 1,
            "image_size": [model_size[0], model_size[1]],
            "image_path": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_bytes": image_path.stat().st_size,
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            "sidecar_bytes": sidecar_path.stat().st_size,
        }
    ], source_artifact


def _write_surya_attempt_one_artifact(
    *,
    engine_dir: Path,
    source_page: int,
    text: str,
    source_raster_identity: dict[str, object],
    source_image: Image.Image,
) -> list[dict[str, object]]:
    attempt_dir = engine_dir / f"page_{source_page:04d}.retry" / "attempt_1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    image_path = attempt_dir / str(source_raster_identity["name"])
    source_image.convert("RGB").save(image_path, format="PNG")
    sidecar_path = attempt_dir / "surya_page_lines.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "execution_path": "module",
                "images": [
                    {
                        "image_name": image_path.name,
                        "ocr_outcome": "text",
                        "attempt_count": 1,
                        "pages": [
                            {
                                "image_bbox": [0, 0, source_image.width, source_image.height],
                                "text_lines": [{"text": text, "bbox": [0, 0, 80, 10]}],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return [
        {
            "attempt": 1,
            "preprocessing": "original",
            "ocr_outcome": "text",
            "image_size": [source_image.width, source_image.height],
            "image_path": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_bytes": image_path.stat().st_size,
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            "sidecar_bytes": sidecar_path.stat().st_size,
        }
    ]


def _new_test_dir() -> Path:
    root = Path.cwd() / "outputs" / "_pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"searchable_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_numbered_pdf(path: Path, *, page_count: int) -> None:
    import fitz

    document = fitz.open()
    try:
        for page_index in range(page_count):
            page = document.new_page(
                width=595 + (page_index % 2),
                height=842 + (page_index % 3),
            )
            page.insert_text((72, 72), f"PAGE {page_index + 1}")
        document.save(str(path))
    finally:
        document.close()


def _write_complete_hybrid_summary(
    *,
    chunk_pdf: Path,
    run_dir: Path,
    output_pdf: Path | None = None,
    include_retry: bool = False,
    delete_original_text_layer: bool = False,
) -> SearchablePdfSummary:
    import fitz

    chunk_pdf = chunk_pdf.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_source = chunk_pdf
    if delete_original_text_layer:
        artifact_source = (
            run_dir.parent / "_source_pdf_without_text_fixture" / chunk_pdf.name
        ).resolve()
        ocr_pipeline._build_textless_source_pdf(
            source_pdf=chunk_pdf,
            out_pdf=artifact_source,
            dpi=ocr_pipeline._resolve_textless_dpi(),
        )
    compare_dir = run_dir / "_compare_txt"
    compare_dir.mkdir(parents=True, exist_ok=True)
    if output_pdf is None:
        output_pdf = run_dir / "result.pdf"
    output_pdf = output_pdf.resolve()
    document = fitz.open(str(chunk_pdf))
    try:
        page_count = int(document.page_count)
    finally:
        document.close()
    local_pages = list(range(1, page_count + 1))
    rendered_dir = run_dir / "_fixture_source_renders"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths = ocr_benchmark._render_sample_paths(
        artifact_source,
        list(range(page_count)),
        dpi=ocr_pipeline._resolve_ocr_render_dpi(),
        tmp_dir=rendered_dir,
    )
    source_images: list[Image.Image] = []
    for rendered_path in rendered_paths:
        with Image.open(rendered_path) as rendered:
            rendered.load()
            source_images.append(rendered.convert("RGB").copy())
    reconciliation = run_dir / "page_reconciliation.json"

    benchmark_results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    benchmark_artifacts: dict[str, Path] = {}
    engine_text_chars: dict[str, int] = {}
    for engine in ("chandra", "surya"):
        output_dir = run_dir / engine
        engine_dir = output_dir / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        page_rows: list[dict[str, object]] = []
        aggregate_blocks: list[str] = []
        for source_page in local_pages:
            text = f"{engine.upper()} PAGE {source_page}"
            if include_retry and engine == "surya" and source_page == 1:
                text = f"CHANDRA PAGE {source_page}"
            source_image = source_images[source_page - 1]
            source_raster_identity = _fixture_source_raster_identity(
                source_page,
                source_image,
            )
            source_raster_artifact = _write_fixture_source_raster_artifact(
                engine_dir=engine_dir,
                source_page=source_page,
                engine=engine,
                source_image=source_image,
            )
            text_path = engine_dir / f"page_{source_page:04d}.txt"
            text_path.write_bytes(text.encode("utf-8"))
            geometry_path = engine_dir / f"page_{source_page:04d}.{engine}.json"
            image_evidence: dict[str, object] = {
                "image_name": str(source_raster_identity["name"]),
                "source_raster_identity": source_raster_identity,
                "source_raster_artifact": source_raster_artifact,
                "ocr_outcome": "text",
                "pages": [
                    {
                        "image_bbox": [0, 0, source_image.width, source_image.height],
                        "ocr_outcome": "text",
                        "text_lines": [{"text": text, "bbox": [0, 0, 80, 10]}],
                    }
                ],
            }
            if engine == "chandra":
                chandra_fields = _chandra_attempt_one(
                    text,
                    source_raster_identity=source_raster_identity,
                )
                attempt_history, source_raster_artifact = _write_chandra_attempt_one_artifact(
                    engine_dir=engine_dir,
                    source_page=source_page,
                    text=text,
                    evidence=chandra_fields["attempts"][0],
                    source_raster_identity=source_raster_identity,
                    source_image=source_image,
                )
                chandra_fields["attempt_history"] = attempt_history
                chandra_fields["source_raster_artifact"] = source_raster_artifact
                image_evidence.update(chandra_fields)
                image_evidence["pages"][0]["image_bbox"] = [
                    0,
                    0,
                    *attempt_history[-1]["image_size"],
                ]
            else:
                chandra_fields = None
            row: dict[str, object] = {
                "source_page": source_page,
                "source_raster_identity": source_raster_identity,
                "source_raster_artifact": source_raster_artifact,
                "file": text_path.name,
                "geometry_file": geometry_path.name,
                "ocr_outcome": "text",
                "text_chars": len(text),
            }
            if engine == "surya":
                row["attempt_count"] = 1
                image_evidence["attempt_count"] = 1
                if not (include_retry and source_page == 1):
                    attempt_history = _write_surya_attempt_one_artifact(
                        engine_dir=engine_dir,
                        source_page=source_page,
                        text=text,
                        source_raster_identity=source_raster_identity,
                        source_image=source_image,
                    )
                    attempt_fields: dict[str, object] = {
                        "retry_policy": (
                            "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
                        ),
                        "selected_attempt": 1,
                        "attempt_history": attempt_history,
                    }
                    row.update(attempt_fields)
                    image_evidence.update(attempt_fields)
            else:
                assert chandra_fields is not None
                row.update(
                    {
                        key: value
                        for key, value in chandra_fields.items()
                        if key not in {"attempts", "chandra_non_text_labels"}
                    }
                )
            if include_retry and engine == "surya" and source_page == 1:
                retry_history: list[dict[str, object]] = []
                preprocessings = (
                    "original",
                    "autocontrast-cutoff-1",
                    "rgb-scale-0.5-center-white-lanczos-v1",
                )
                source_image = source_images[0].copy()
                autocontrast_image = ImageOps.autocontrast(source_image, cutoff=1)
                content_size = (
                    max(1, round(source_image.width * 0.5)),
                    max(1, round(source_image.height * 0.5)),
                )
                content_offset = (
                    (source_image.width - content_size[0]) // 2,
                    (source_image.height - content_size[1]) // 2,
                )
                scaled_content = source_image.resize(content_size, Image.Resampling.LANCZOS)
                scaled_image = Image.new("RGB", source_image.size, color=(255, 255, 255))
                scaled_image.paste(scaled_content, content_offset)
                scaled_line_bbox = [
                    float(content_offset[0]),
                    float(content_offset[1]),
                    float(content_offset[0])
                    + (80.0 * float(content_size[0]) / float(source_image.width)),
                    float(content_offset[1])
                    + (10.0 * float(content_size[1]) / float(source_image.height)),
                ]
                image_evidence["pages"][0]["text_lines"][0]["bbox"] = [
                    0.0,
                    0.0,
                    (scaled_line_bbox[2] - float(content_offset[0]))
                    * float(source_image.width)
                    / float(content_size[0]),
                    (scaled_line_bbox[3] - float(content_offset[1]))
                    * float(source_image.height)
                    / float(content_size[1]),
                ]
                retry_images = (source_image, autocontrast_image, scaled_image)
                for attempt, (preprocessing, retry_payload) in enumerate(
                    zip(preprocessings, retry_images, strict=True),
                    start=1,
                ):
                    retry_dir = engine_dir / "page_0001.retry" / f"attempt_{attempt}"
                    retry_dir.mkdir(parents=True)
                    retry_image = retry_dir / "00001.png"
                    retry_payload.save(retry_image, format="PNG")
                    retry_sidecar = retry_dir / "surya_page_lines.json"
                    attempt_outcome = "text" if attempt == 3 else "zero_output"
                    retry_sidecar.write_text(
                        json.dumps(
                            {
                                "execution_path": "module",
                                "images": [
                                    {
                                        "image_name": retry_image.name,
                                        "ocr_outcome": attempt_outcome,
                                        "attempt_count": attempt,
                                        "pages": [
                                            {
                                                "image_bbox": [0, 0, *source_image.size],
                                                "text_lines": (
                                                    [
                                                        {
                                                            "text": text,
                                                            "bbox": scaled_line_bbox,
                                                        }
                                                    ]
                                                    if attempt == 3
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
                    if attempt > 1:
                        retry_payload_json = json.loads(retry_sidecar.read_text(encoding="utf-8"))
                        retry_payload_json["images"][0]["retry_preprocessing"] = preprocessing
                        retry_sidecar.write_text(
                            json.dumps(retry_payload_json),
                            encoding="utf-8",
                        )
                    history_item: dict[str, object] = {
                        "attempt": attempt,
                        "preprocessing": preprocessing,
                        "ocr_outcome": attempt_outcome,
                        "image_size": list(source_image.size),
                        "image_path": str(retry_image.resolve()),
                        "image_sha256": hashlib.sha256(retry_image.read_bytes()).hexdigest(),
                        "image_bytes": retry_image.stat().st_size,
                        "sidecar_path": str(retry_sidecar.resolve()),
                        "sidecar_sha256": hashlib.sha256(retry_sidecar.read_bytes()).hexdigest(),
                        "sidecar_bytes": retry_sidecar.stat().st_size,
                    }
                    if attempt == 3:
                        history_item.update(
                            {
                                "content_scale": 0.5,
                                "content_size": list(content_size),
                                "content_offset": list(content_offset),
                                "resampling": "lanczos",
                                "canvas_fill_rgb": [255, 255, 255],
                            }
                        )
                    retry_history.append(history_item)
                retry_fields: dict[str, object] = {
                    "attempt_count": 3,
                    "retry_preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
                    "retry_policy": (
                        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
                    ),
                    "selected_attempt": 3,
                    "attempt_history": retry_history,
                    "geometry_coordinate_space": "source-image-v1",
                    "geometry_transform": "inverse-actual-content-size-strict-v1",
                }
                row.update(retry_fields)
                image_evidence.update(retry_fields)
            geometry_path.write_text(
                json.dumps({"images": [image_evidence]}),
                encoding="utf-8",
            )
            page_rows.append(row)
            aggregate_blocks.extend([f"[SOURCE PAGE {source_page:04d}]", text, ""])
        aggregate_path = engine_dir / "all_pages.txt"
        aggregate_text = "\n".join(aggregate_blocks).strip() + "\n"
        aggregate_path.write_bytes(aggregate_text.encode("utf-8"))
        pages_path = engine_dir / "pages.json"
        pages_path.write_text(
            json.dumps(
                {
                    "pdf_path": str(artifact_source),
                    "engine": engine,
                    "pages": page_rows,
                    "total_text_chars": sum(int(row["text_chars"]) for row in page_rows),
                    "aggregate_file": aggregate_path.name,
                    "aggregate_has_page_markers": True,
                }
            ),
            encoding="utf-8",
        )
        artifact_path = output_dir / f"document_{engine}.txt"
        artifact_path.write_bytes(aggregate_text.encode("utf-8"))
        benchmark_artifacts[engine] = artifact_path
        engine_text_chars[engine] = sum(int(row["text_chars"]) for row in page_rows)
        result = OcrBenchmarkResult(
            engine=engine,
            status="ok",
            sample_pages=list(local_pages),
            elapsed_seconds=0.1,
            artifact_path=str(artifact_path),
            text_chars=engine_text_chars[engine],
        )
        benchmark_results.append(result)
        report_path = output_dir / "document_ocr_benchmark.json"
        report_path.write_text(
            json.dumps({"results": [asdict(result)]}),
            encoding="utf-8",
        )
        result_files.append(report_path)
    reconciliation_pages: list[dict[str, object]] = []
    for source_page in local_pages:
        row: dict[str, object] = {
            "source_page": source_page,
            "surya_outcome": "text",
            "chandra_outcome": "text",
            "surya_alnum_line_count": None,
            "surya_alnum_chars": None,
            "surya_page_error_count": 0,
            "chandra_page_error_count": 0,
            "accepted": True,
            "reason": "both_text",
        }
        if include_retry and source_page == 1:
            canonical = ocr_pipeline._canonical_retry_agreement_text(f"CHANDRA PAGE {source_page}")
            canonical_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            row.update(
                {
                    "reason": "both_text_retry_geometry_agreement",
                    "retry_text_agreement": {
                        "algorithm": "nfkc-layout-whitespace-exact-v1",
                        "matched": True,
                        "surya_sha256": canonical_sha256,
                        "chandra_sha256": canonical_sha256,
                    },
                }
            )
        reconciliation_pages.append(row)
    reconciliation.write_text(
        json.dumps(
            {
                "schema": "uniscan.page-reconciliation.v1",
                "status": "ok",
                "exact_page_bijection": True,
                "accepted_textless_graphics_pages": [],
                "unresolved_pages": [],
                "pages": reconciliation_pages,
                "reconciled_page_error_counts": {"surya": 0, "chandra": 0},
                "result_text_chars": dict(engine_text_chars),
                "reconciliation_original_evidence": {},
            }
        ),
        encoding="utf-8",
    )

    compare_results: list[CompareTxtBuildResult] = []
    for engine in ("chandra", "surya"):
        compare_path = compare_dir / f"{chunk_pdf.stem}__{engine}.txt"
        compare_path.write_bytes(benchmark_artifacts[engine].read_bytes())
        compare_results.append(
            CompareTxtBuildResult(
                engine=engine,
                status="ok",
                source_artifact_path=str(benchmark_artifacts[engine]),
                compare_txt_path=str(compare_path),
            )
        )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    searchable = fitz.open(str(artifact_source))
    try:
        for source_page in local_pages:
            page = searchable[source_page - 1]
            page.insert_text(
                (36, 36),
                f"CHANDRA PAGE {source_page}",
                fontsize=6,
                render_mode=3,
            )
        searchable.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        searchable.close()
    geometry_log = run_dir / "searchable_pdf_final" / "geometry.json"
    geometry_log.parent.mkdir(parents=True, exist_ok=True)
    geometry_log.write_text("{}", encoding="utf-8")
    artifact_result = ArtifactSearchableResult(
        document=chunk_pdf.stem,
        engine="chandra",
        status="ok",
        source_pdf_path=str(artifact_source),
        text_artifact_path=str(compare_results[0].compare_txt_path),
        searchable_pdf_path=str(output_pdf),
        page_count=page_count,
        text_chars=engine_text_chars["chandra"],
        elapsed_seconds=0.01,
        geometry_log_path=str(geometry_log),
    )
    return SearchablePdfSummary(
        mode="chandra+surya",
        run_dir=run_dir,
        compare_dir=compare_dir,
        output_pdf_path=output_pdf,
        output_pdf_bytes=None,
        overwritten_input_path=None,
        benchmark=BasicOcrRunSummary(
            run_dir=run_dir,
            results=tuple(benchmark_results),
            result_files=tuple(result_files),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        ),
        compare_results=tuple(compare_results),
        artifact_results=(artifact_result,),
    )


def test_hybrid_chunk_pages_defaults_to_ten_and_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNISCAN_HYBRID_CHUNK_PAGES", raising=False)
    assert ocr_pipeline._resolve_hybrid_chunk_pages() == 10
    monkeypatch.setenv("UNISCAN_HYBRID_CHUNK_PAGES", "0")
    assert ocr_pipeline._resolve_hybrid_chunk_pages() == 0
    monkeypatch.setenv("UNISCAN_HYBRID_CHUNK_PAGES", "250")
    assert ocr_pipeline._resolve_hybrid_chunk_pages() == 100
    monkeypatch.setenv("UNISCAN_HYBRID_CHUNK_PAGES", "invalid")
    assert ocr_pipeline._resolve_hybrid_chunk_pages() == 10


def test_production_pdf_mode_rejects_single_engine_modes() -> None:
    assert ocr_pipeline.normalize_pdf_mode(None) == "chandra+surya"
    assert ocr_pipeline.normalize_pdf_mode("chandra surya") == "chandra+surya"
    for mode in ("chandra", "surya"):
        with pytest.raises(ValueError, match="requires chandra\\+surya"):
            ocr_pipeline.normalize_pdf_mode(mode)


def test_production_pdf_pipeline_rejects_non_strict_execution() -> None:
    with pytest.raises(ValueError, match="strict cannot be disabled"):
        build_searchable_pdf(pdf_bytes=b"not-opened", strict=False)


def test_split_and_merge_pdf_chunks_preserves_page_order_and_sizes() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=23)

    chunks = ocr_pipeline._split_pdf_chunks(
        source_pdf=source_pdf,
        output_root=tmp_path / "chunks",
        pages_per_chunk=10,
    )

    assert [(chunk.start_page, chunk.end_page) for chunk in chunks] == [(1, 10), (11, 20), (21, 23)]
    first_fingerprints = [ocr_pipeline._stable_file_fingerprint(chunk.path) for chunk in chunks]
    repeated_chunks = ocr_pipeline._split_pdf_chunks(
        source_pdf=source_pdf,
        output_root=tmp_path / "chunks",
        pages_per_chunk=10,
    )
    assert [
        ocr_pipeline._stable_file_fingerprint(chunk.path) for chunk in repeated_chunks
    ] == first_fingerprints

    merged_pdf = ocr_pipeline._merge_pdf_chunks(
        source_pdf=source_pdf,
        chunks=[(chunk, chunk.path) for chunk in chunks],
        output_pdf=tmp_path / "merged.pdf",
    )
    source = fitz.open(str(source_pdf))
    merged = fitz.open(str(merged_pdf))
    try:
        assert merged.page_count == source.page_count == 23
        for page_index in range(23):
            assert merged[page_index].get_text().strip() == f"PAGE {page_index + 1}"
            assert merged[page_index].rect == source[page_index].rect
    finally:
        source.close()
        merged.close()


def test_chunked_hybrid_resolves_one_runtime_config_and_passes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"sealed source")
    snapshot = ocr_pipeline._resolve_hybrid_run_config()
    resolve_calls = 0
    identity_snapshots: list[object] = []
    unlocked_snapshots: list[object] = []
    sentinel = object()

    def fake_resolve() -> object:
        nonlocal resolve_calls
        resolve_calls += 1
        return snapshot

    def fake_identity(**kwargs: object) -> tuple[dict[str, object], str]:
        identity_snapshots.append(kwargs["runtime_config"])
        return {"source": {"sha256": "a" * 64, "size": 13}}, "b" * 64

    def fake_unlocked(**kwargs: object) -> object:
        unlocked_snapshots.append(kwargs["runtime_config"])
        return sentinel

    monkeypatch.setattr(ocr_pipeline, "_resolve_hybrid_run_config", fake_resolve)
    monkeypatch.setattr(ocr_pipeline, "_hybrid_run_identity", fake_identity)
    monkeypatch.setattr(
        ocr_pipeline,
        "_build_searchable_pdf_chunked_unlocked",
        fake_unlocked,
    )

    result = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=11,
    )

    assert result is sentinel
    assert resolve_calls == 1
    assert identity_snapshots == [snapshot]
    assert unlocked_snapshots == [snapshot]


def test_chunked_hybrid_rejects_runtime_environment_drift_before_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=2)
    monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "72")
    calls = 0

    def fake_build_searchable_pdf(**kwargs: object) -> SearchablePdfSummary:
        nonlocal calls
        calls += 1
        chunk_pdf = Path(str(kwargs["pdf_path"]))
        summary = _write_complete_hybrid_summary(
            chunk_pdf=chunk_pdf,
            run_dir=Path(str(kwargs["work_root"])) / "run",
            delete_original_text_layer=False,
        )
        if calls == 1:
            monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "73")
        return summary

    monkeypatch.setattr(ocr_pipeline, "build_searchable_pdf", fake_build_searchable_pdf)

    with pytest.raises(
        RuntimeError,
        match="hybrid runtime configuration changed during the run",
    ):
        ocr_pipeline._build_searchable_pdf_chunked(
            input_path=source_pdf,
            mode="chandra+surya",
            lang="rus+eng",
            work_root=tmp_path / "work",
            overwrite_target=None,
            return_bytes=False,
            strict=True,
            progress=None,
            delete_original_text_layer=False,
            chunk_pages=1,
            page_count=2,
        )

    assert calls == 1


def test_chunked_hybrid_pipeline_uses_ten_page_hybrid_jobs_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fitz

    # This test exercises chunk lifecycle and evidence reuse, not production raster sizing.
    # Keep its 23-page fixture small enough that repeated evidence validation stays fast.
    monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "72")
    monkeypatch.setenv("UNISCAN_TEXTLESS_DPI", "72")
    monkeypatch.setattr(ocr_benchmark, "_CHANDRA_MIN_IMAGE_DIM", 72)
    monkeypatch.setattr(ocr_pipeline, "_CHANDRA_MIN_IMAGE_DIM", 72)

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=23)
    calls: list[dict[str, object]] = []

    def fake_build_searchable_pdf(**kwargs: object) -> SearchablePdfSummary:
        chunk_pdf = Path(str(kwargs["pdf_path"]))
        work_root = Path(str(kwargs["work_root"]))
        run_dir = work_root / "run"
        chunk_document = fitz.open(str(chunk_pdf))
        try:
            pages = int(chunk_document.page_count)
        finally:
            chunk_document.close()
        calls.append(
            {
                "mode": kwargs["mode"],
                "pages": pages,
                "chunk_setting": os.environ.get("UNISCAN_HYBRID_CHUNK_PAGES"),
                "delete_original_text_layer": kwargs["delete_original_text_layer"],
            }
        )
        callback = kwargs["progress"]
        assert callable(callback)
        callback(0, "starting")
        callback(100, "done")
        return _write_complete_hybrid_summary(
            chunk_pdf=chunk_pdf,
            run_dir=run_dir,
            delete_original_text_layer=bool(kwargs["delete_original_text_layer"]),
        )

    monkeypatch.setattr(ocr_pipeline, "build_searchable_pdf", fake_build_searchable_pdf)
    progress_values: list[int] = []
    summary = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=True,
        strict=True,
        progress=lambda value, _status: progress_values.append(value),
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=23,
    )

    assert calls == [
        {
            "mode": "chandra+surya",
            "pages": 10,
            "chunk_setting": "0",
            "delete_original_text_layer": True,
        },
        {
            "mode": "chandra+surya",
            "pages": 10,
            "chunk_setting": "0",
            "delete_original_text_layer": True,
        },
        {
            "mode": "chandra+surya",
            "pages": 3,
            "chunk_setting": "0",
            "delete_original_text_layer": True,
        },
    ]
    assert summary.chunk_count == 3
    assert summary.chunk_pages == 10
    assert summary.output_pdf_bytes is not None
    assert progress_values == sorted(progress_values)
    assert progress_values[-1] == 100
    assert [result.sample_pages for result in summary.benchmark.results] == [
        list(range(1, 11)),
        list(range(1, 11)),
        list(range(11, 21)),
        list(range(11, 21)),
        list(range(21, 24)),
        list(range(21, 24)),
    ]
    manifest = json.loads(summary.chunk_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert [item["status"] for item in manifest["chunks"]] == ["done", "done", "done"]
    assert manifest["schema"] == "uniscan.hybrid-chunks.v4"
    assert all(item["output_sha256"] for item in manifest["chunks"])

    calls_before_resume = len(calls)
    progress_values.clear()
    resumed = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=True,
        strict=True,
        progress=lambda value, _status: progress_values.append(value),
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=23,
    )

    assert len(calls) == calls_before_resume
    assert resumed.run_dir == summary.run_dir
    assert progress_values == sorted(progress_values)
    resumed_manifest = json.loads(resumed.chunk_manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["resume_count"] == 1
    assert [item["reused"] for item in resumed_manifest["chunks"]] == [True, True, True]

    first_output = Path(str(resumed_manifest["chunks"][0]["output_pdf"]))
    first_output.write_bytes(b"tampered")
    repaired = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=23,
    )

    assert len(calls) == calls_before_resume + 1
    repaired_manifest = json.loads(repaired.chunk_manifest_path.read_text(encoding="utf-8"))
    assert [item["reused"] for item in repaired_manifest["chunks"]] == [False, True, True]

    repaired.chunk_manifest_path.write_text("{broken", encoding="utf-8")
    recovered = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=23,
    )

    assert len(calls) == calls_before_resume + 4
    recovered_manifest = json.loads(recovered.chunk_manifest_path.read_text(encoding="utf-8"))
    assert "ignored unreadable manifest" in recovered_manifest["recovery_reason"]
    assert [item["reused"] for item in recovered_manifest["chunks"]] == [
        False,
        False,
        False,
    ]

    changed_pdf = source_pdf.with_name("changed.pdf")
    changed = fitz.open(str(source_pdf))
    try:
        changed[0].insert_text((72, 110), "SOURCE CONTENT CHANGED")
        changed.save(str(changed_pdf))
    finally:
        changed.close()
    os.replace(changed_pdf, source_pdf)
    changed_summary = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=23,
    )

    assert len(calls) == calls_before_resume + 7
    assert changed_summary.run_dir != recovered.run_dir


def test_chunk_manifest_v4_rejects_stale_v3_record() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    chunk_pdf = tmp_path / "chunk.pdf"
    source_pdf.write_bytes(b"source")
    chunk_pdf.write_bytes(b"chunk")
    manifest_path = tmp_path / "chunk_manifest.json"
    identity = {"pipeline_revision": "chandra-surya-resumable-v4"}
    run_key = "f" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "uniscan.hybrid-chunks.v3",
                "status": "done",
                "run_key": run_key,
                "identity": identity,
                "chunks": [
                    {
                        "index": 1,
                        "start_page": 1,
                        "end_page": 1,
                        "status": "done",
                        "output_pdf": str(tmp_path / "stale.pdf"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest, records = ocr_pipeline._prepare_chunk_manifest(
        manifest_path=manifest_path,
        identity=identity,
        run_key=run_key,
        input_path=source_pdf,
        input_chunks=[ocr_pipeline._PdfChunk(1, 1, 1, chunk_pdf)],
        mode="chandra+surya",
        page_count=1,
        chunk_pages=10,
    )

    assert manifest["schema"] == "uniscan.hybrid-chunks.v4"
    assert manifest["recovery_reason"] == "ignored incompatible manifest identity"
    assert records[0]["status"] == "pending"
    assert "output_pdf" not in records[0]


@pytest.mark.parametrize(
    "defect",
    [
        "reconciliation-tamper",
        "pages-tamper",
        "geometry-delete",
        "retry-image-tamper",
        "retry-sidecar-tamper",
        "unexpected-sidecar-add",
    ],
)
def test_reusable_chunk_rejects_tampered_critical_evidence(defect: str) -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        include_retry=True,
    )
    run_dir = summary.run_dir
    reconciliation = run_dir / "page_reconciliation.json"
    engine_dir = run_dir / "surya" / "surya"
    retry_dir = engine_dir / "page_0001.retry" / "attempt_3"
    pages_path = engine_dir / "pages.json"
    geometry_path = engine_dir / "page_0001.surya.json"
    retry_image = retry_dir / "page_0001.png"
    retry_sidecar = retry_dir / "surya_page_lines.json"
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    chunk = ocr_pipeline._PdfChunk(1, 1, 1, source_pdf)
    record: dict[str, object] = {}

    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=chunk,
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
    )

    evidence_manifest = Path(str(record["evidence_manifest"]))
    assert evidence_manifest.is_file()
    assert len(str(record["evidence_manifest_sha256"])) == 64
    assert (
        ocr_pipeline._reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )
        is not None
    )

    if defect == "reconciliation-tamper":
        reconciliation.write_text('{"status":"tampered"}', encoding="utf-8")
    elif defect == "pages-tamper":
        pages_path.write_text('{"engine":"surya","pages":[1]}', encoding="utf-8")
    elif defect == "geometry-delete":
        geometry_path.unlink()
    elif defect == "retry-image-tamper":
        retry_image.write_bytes(b"tampered retry png evidence")
    elif defect == "retry-sidecar-tamper":
        retry_sidecar.write_text('{"images":["tampered"]}', encoding="utf-8")
    else:
        (engine_dir / "page_0002.surya.json").write_text(
            '{"images":[]}',
            encoding="utf-8",
        )

    assert (
        ocr_pipeline._reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )
        is None
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing-benchmark-result",
        "out-of-tree-compare",
        "missing-page-geometry",
        "missing-reconciliation",
        "missing-retry-history",
        "empty-retry-history",
        "chandra-prompt-tamper",
        "aggregate-canonical-tamper",
    ],
)
def test_complete_chunk_refuses_incomplete_or_out_of_tree_claims(defect: str) -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        include_retry=True,
    )
    if defect == "missing-benchmark-result":
        summary.benchmark.result_files[0].unlink()
    elif defect == "out-of-tree-compare":
        escaped = tmp_path / "escaped-compare.txt"
        escaped.write_text("escaped", encoding="utf-8")
        first_compare = replace(
            summary.compare_results[0],
            compare_txt_path=str(escaped),
        )
        summary = replace(
            summary,
            compare_results=(first_compare, *summary.compare_results[1:]),
        )
    elif defect == "missing-page-geometry":
        (summary.run_dir / "chandra" / "chandra" / "page_0001.chandra.json").unlink()
    elif defect == "missing-reconciliation":
        (summary.run_dir / "page_reconciliation.json").unlink()
    elif defect == "chandra-prompt-tamper":
        geometry_path = summary.run_dir / "chandra" / "chandra" / "page_0001.chandra.json"
        geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        geometry_payload["images"][0]["attempts"][0]["prompt_sha256"] = "0" * 64
        geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")
    elif defect == "aggregate-canonical-tamper":
        aggregate_path = summary.run_dir / "chandra" / "chandra" / "all_pages.txt"
        aggregate_path.write_bytes(b"[SOURCE PAGE 0001]\nCHANDRA PAGE 1!\n")
    else:
        pages_path = summary.run_dir / "surya" / "surya" / "pages.json"
        pages_payload = json.loads(pages_path.read_text(encoding="utf-8"))
        first_page = pages_payload["pages"][0]
        assert isinstance(first_page, dict)
        if defect == "missing-retry-history":
            first_page.pop("attempt_history")
        else:
            first_page["attempt_history"] = []
        pages_path.write_text(json.dumps(pages_payload), encoding="utf-8")
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    record: dict[str, object] = {"status": "running"}

    with pytest.raises(RuntimeError):
        ocr_pipeline._complete_chunk_record(
            record=record,
            chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
            summary=summary,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )

    assert record.get("status") != "done"


def test_complete_chunk_rejects_pages_changed_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    pages_path = summary.run_dir / "surya" / "surya" / "pages.json"
    original_entries = ocr_pipeline._complete_chunk_evidence_entries
    mutated = False

    def _mutating_entries(
        *,
        required: ocr_pipeline._RequiredChunkEvidence,
        manifest_path: Path,
    ) -> list[dict[str, object]]:
        nonlocal mutated
        if not mutated:
            payload = json.loads(pages_path.read_text(encoding="utf-8"))
            payload["engine"] = "tampered-after-validation"
            pages_path.write_text(json.dumps(payload), encoding="utf-8")
            mutated = True
        return original_entries(required=required, manifest_path=manifest_path)

    monkeypatch.setattr(
        ocr_pipeline,
        "_complete_chunk_evidence_entries",
        _mutating_entries,
    )
    record: dict[str, object] = {"status": "running"}
    with pytest.raises(RuntimeError, match="changed before it could be sealed"):
        ocr_pipeline._complete_chunk_record(
            record=record,
            chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
            summary=summary,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )

    assert mutated is True
    assert record.get("status") != "done"


def test_chunk_evidence_manifest_records_required_and_outer_paths() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        include_retry=True,
    )
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
    )

    manifest_path = Path(str(record["evidence_manifest"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_paths = set(payload["required_paths"])
    sealed_paths = {item["path"] for item in payload["files"]}
    outer_required = set(payload["outer_required_paths"])
    outer_sealed = {item["path"] for item in payload["outer_files"]}

    assert payload["evidence_root"] == str(summary.run_dir)
    assert payload["outer_root"] == str(tmp_path)
    assert required_paths <= sealed_paths
    assert {
        "page_reconciliation.json",
        "surya/surya/pages.json",
        "chandra/chandra/pages.json",
        "surya/surya/page_0001.txt",
        "chandra/chandra/page_0001.chandra.json",
    } <= required_paths
    assert outer_required == outer_sealed
    assert len(outer_required) == 1
    assert all(len(item["sha256"]) == 64 and item["size"] > 0 for item in payload["outer_files"])


@pytest.mark.parametrize(
    "defect",
    [
        "partial-count-string",
        "status-bool",
        "sample-out-of-range",
        "path-not-string",
        "artifact-count-float",
        "unexpected-key",
    ],
)
def test_reusable_chunk_rejects_malformed_strict_summary(defect: str) -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    chunk = ocr_pipeline._PdfChunk(1, 1, 1, source_pdf)
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=chunk,
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
    )
    summary_payload = record["summary"]
    assert isinstance(summary_payload, dict)
    benchmark = summary_payload["benchmark"]
    assert isinstance(benchmark, dict)
    results = benchmark["results"]
    artifacts = summary_payload["artifact_results"]
    assert isinstance(results, list) and isinstance(results[0], dict)
    assert isinstance(artifacts, list) and isinstance(artifacts[0], dict)
    if defect == "partial-count-string":
        summary_payload["partial_page_failures"] = "0"
    elif defect == "status-bool":
        results[0]["status"] = True
    elif defect == "sample-out-of-range":
        results[0]["sample_pages"] = [2]
    elif defect == "path-not-string":
        benchmark["run_dir"] = 7
    elif defect == "artifact-count-float":
        artifacts[0]["page_count"] = 1.0
    else:
        summary_payload["unexpected"] = True

    assert (
        ocr_pipeline._reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )
        is None
    )


def test_complete_chunk_rejects_hardlinked_required_evidence() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    geometry = summary.run_dir / "surya" / "surya" / "page_0001.surya.json"
    original = tmp_path / "hardlink-target.json"
    original.write_bytes(geometry.read_bytes())
    geometry.unlink()
    try:
        os.link(original, geometry)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()

    with pytest.raises(RuntimeError, match="hard-linked"):
        ocr_pipeline._complete_chunk_record(
            record={},
            chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
            summary=summary,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )


def test_reusable_chunk_rejects_hardlinked_cached_output() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    chunk = ocr_pipeline._PdfChunk(1, 1, 1, source_pdf)
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=chunk,
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
    )
    output_pdf = summary.output_pdf_path
    hardlink_target = tmp_path / "cached-output-target.pdf"
    shutil.copy2(output_pdf, hardlink_target)
    output_pdf.unlink()
    try:
        os.link(hardlink_target, output_pdf)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")

    assert (
        ocr_pipeline._reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )
        is None
    )


def test_complete_chunk_rejects_symlinked_directory_component() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    compare_dir = summary.compare_dir
    compare_target = summary.run_dir / "_compare_txt_target"
    compare_dir.rename(compare_target)
    try:
        os.symlink(compare_target, compare_dir, target_is_directory=True)
    except OSError as exc:
        compare_target.rename(compare_dir)
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()

    with pytest.raises(RuntimeError, match="link|reparse"):
        ocr_pipeline._complete_chunk_record(
            record={},
            chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
            summary=summary,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )


def test_reusable_chunk_rejects_non_root_evidence_manifest_path() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
    )
    source = fitz.open(str(source_pdf))
    try:
        source_sizes = [(float(source[0].rect.width), float(source[0].rect.height))]
    finally:
        source.close()
    chunk = ocr_pipeline._PdfChunk(1, 1, 1, source_pdf)
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=chunk,
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
    )
    relocated = tmp_path / "relocated-evidence-manifest.json"
    shutil.copy2(Path(str(record["evidence_manifest"])), relocated)
    fingerprint = ocr_pipeline._stable_file_fingerprint(relocated)
    record["evidence_manifest"] = str(relocated)
    record["evidence_manifest_sha256"] = fingerprint["sha256"]
    record["evidence_manifest_size"] = fingerprint["size"]

    assert (
        ocr_pipeline._reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=tmp_path,
            source_sizes=source_sizes,
        )
        is None
    )


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_chunk_manifest_reader_rejects_linked_file(
    tmp_path: Path,
    link_kind: str,
) -> None:
    target = tmp_path / "manifest-target.json"
    target.write_text('{"schema":"uniscan.hybrid-chunks.v4"}', encoding="utf-8")
    manifest = tmp_path / "chunk_manifest.json"
    try:
        if link_kind == "hardlink":
            os.link(target, manifest)
        else:
            os.symlink(target, manifest)
    except OSError as exc:
        pytest.skip(f"{link_kind} creation is unavailable: {exc}")

    payload, error = ocr_pipeline._read_chunk_manifest(manifest)

    assert payload is None
    assert error is not None
    assert "link" in error.lower() or "reparse" in error.lower()


def test_chunked_hybrid_pipeline_records_failed_page_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=21)
    attempts = 0

    def fail_second_chunk(**kwargs: object) -> SearchablePdfSummary:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("CUDA out of memory")
        chunk_pdf = Path(str(kwargs["pdf_path"]))
        return _write_complete_hybrid_summary(
            chunk_pdf=chunk_pdf,
            run_dir=Path(str(kwargs["work_root"])) / "run",
            delete_original_text_layer=bool(kwargs["delete_original_text_layer"]),
        )

    monkeypatch.setattr(ocr_pipeline, "build_searchable_pdf", fail_second_chunk)
    with pytest.raises(RuntimeError, match="pages 11-20"):
        ocr_pipeline._build_searchable_pdf_chunked(
            input_path=source_pdf,
            mode="chandra+surya",
            lang="rus+eng",
            work_root=tmp_path / "failed_work",
            overwrite_target=None,
            return_bytes=False,
            strict=True,
            progress=None,
            delete_original_text_layer=True,
            chunk_pages=10,
            page_count=21,
        )
    manifest_path = next(
        (tmp_path / "failed_work").glob("hybrid_chunk_cache/hybrid_*/chunk_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["failed_chunk"] == 2
    assert [item["status"] for item in manifest["chunks"]] == ["done", "error", "pending"]
    assert manifest["chunks"][1]["start_page"] == 11
    assert manifest["chunks"][1]["end_page"] == 20
    first_output = Path(str(manifest["chunks"][0]["output_pdf"]))
    assert first_output.is_file()

    resumed = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "failed_work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=True,
        chunk_pages=10,
        page_count=21,
    )

    assert attempts == 4
    assert resumed.chunk_manifest_path == manifest_path
    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["status"] == "done"
    assert resumed_manifest["resume_count"] == 1
    assert [item["reused"] for item in resumed_manifest["chunks"]] == [
        True,
        False,
        False,
    ]


def test_build_searchable_pdf_overwrites_input_path(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"ORIGINAL")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    produced_pdf = tmp_path / "produced.pdf"
    produced_pdf.write_bytes(b"SEARCHABLE")
    seen: dict[str, Path] = {}

    def fake_build_textless_source_pdf(*, source_pdf: Path, out_pdf: Path, dpi: int = 300) -> Path:
        assert source_pdf == input_pdf.resolve()
        assert dpi == 300
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(b"TEXTLESS")
        seen["textless_pdf"] = out_pdf
        return out_pdf

    def fake_run_basic_ocr_benchmark(**kwargs) -> BasicOcrRunSummary:
        assert kwargs["pdf_path"] == seen["textless_pdf"]
        return BasicOcrRunSummary(
            run_dir=run_dir,
            results=(
                OcrBenchmarkResult(
                    engine="chandra",
                    status="ok",
                    sample_pages=[1],
                    elapsed_seconds=0.1,
                    artifact_path="chandra.txt",
                    text_chars=100,
                    page_error_count=2,
                ),
                OcrBenchmarkResult(
                    engine="surya",
                    status="ok",
                    sample_pages=[1],
                    elapsed_seconds=0.1,
                    artifact_path="surya.txt",
                    text_chars=100,
                    page_error_count=3,
                ),
            ),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    def fake_build_compare_txt_from_benchmark(**_kwargs):
        return [_ok_compare_result("chandra", run_dir / "_compare_txt" / "doc__chandra.txt")]

    def fake_run_artifact_searchable_package(**kwargs):
        assert kwargs["engines"] == ("chandra",)
        assert kwargs["pdf_root"] == seen["textless_pdf"].parent
        assert kwargs["reconciliation_root"] == run_dir
        expected_geometry = str((run_dir / "surya").resolve())
        assert os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR") == expected_geometry
        return [_ok_artifact_result(produced_pdf, engine="chandra")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(
        ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark
    )
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(
        ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package
    )
    monkeypatch.setattr(ocr_pipeline, "_pdf_page_count", lambda _path: 1)

    summary = build_searchable_pdf(
        pdf_path=input_pdf,
        mode="chandra+surya",
        work_root=tmp_path / "work",
        overwrite_input_path=True,
        return_bytes=False,
        strict=True,
    )

    assert isinstance(summary, SearchablePdfSummary)
    assert summary.mode == "chandra+surya"
    assert summary.overwritten_input_path == input_pdf.resolve()
    assert summary.output_pdf_path == input_pdf.resolve()
    assert summary.partial_page_failures == 3
    assert input_pdf.read_bytes() == b"SEARCHABLE"


def test_atomic_pdf_overwrite_preserves_original_when_replace_fails(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    source = tmp_path / "produced.pdf"
    target = tmp_path / "input.pdf"
    source.write_bytes(b"SEARCHABLE")
    target.write_bytes(b"ORIGINAL")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ocr_pipeline.os, "replace", fail_replace)

    try:
        ocr_pipeline._atomic_copy_file(source, target)
    except OSError as exc:
        assert "simulated replace failure" in str(exc)
    else:
        raise AssertionError("Expected atomic replacement to fail")

    assert target.read_bytes() == b"ORIGINAL"
    assert list(tmp_path.glob(".input.pdf.*.tmp")) == []


def test_build_searchable_pdf_from_bytes_returns_bytes(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    produced_pdf = tmp_path / "produced_bytes.pdf"
    produced_pdf.write_bytes(b"PDF-BYTES-RESULT")
    seen_pdf_path: dict[str, Path] = {}
    seen_textless_path: dict[str, Path] = {}

    def fake_build_textless_source_pdf(*, source_pdf: Path, out_pdf: Path, dpi: int = 300) -> Path:
        assert source_pdf.exists()
        assert source_pdf.read_bytes() == b"INLINE-PDF"
        assert dpi == 300
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(b"TEXTLESS-INLINE-PDF")
        seen_textless_path["value"] = out_pdf
        return out_pdf

    def fake_run_basic_ocr_benchmark(**kwargs) -> BasicOcrRunSummary:
        staged_pdf = Path(kwargs["pdf_path"])
        seen_pdf_path["value"] = staged_pdf
        assert staged_pdf.exists()
        assert staged_pdf.read_bytes() == b"TEXTLESS-INLINE-PDF"

        run_dir = tmp_path / "inline_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return BasicOcrRunSummary(
            run_dir=run_dir,
            results=(_ok_benchmark_result("chandra"), _ok_benchmark_result("surya")),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    def fake_build_compare_txt_from_benchmark(**_kwargs):
        return [_ok_compare_result("chandra", tmp_path / "doc__chandra.txt")]

    def fake_run_artifact_searchable_package(**kwargs):
        assert kwargs["engines"] == ("chandra",)
        assert kwargs["pdf_root"] == seen_textless_path["value"].parent
        assert kwargs["reconciliation_root"] == tmp_path / "inline_run"
        assert os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR") == str(
            tmp_path / "inline_run" / "surya"
        )
        return [_ok_artifact_result(produced_pdf, engine="chandra")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(
        ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark
    )
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(
        ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package
    )
    monkeypatch.setattr(ocr_pipeline, "_pdf_page_count", lambda _path: 1)

    summary = build_searchable_pdf(
        pdf_bytes=b"INLINE-PDF",
        mode="chandra+surya",
        work_root=tmp_path / "work_inline",
        overwrite_input_path=False,
        return_bytes=True,
        strict=True,
    )

    assert isinstance(summary, SearchablePdfSummary)
    assert summary.mode == "chandra+surya"
    assert summary.overwritten_input_path is None
    assert summary.output_pdf_bytes == b"PDF-BYTES-RESULT"
    assert summary.output_pdf_path == produced_pdf
    assert seen_pdf_path["value"].suffix.lower() == ".pdf"


def test_build_searchable_pdf_uses_textless_source_when_delete_enabled(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"ORIGINAL")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    produced_pdf = tmp_path / "produced.pdf"
    produced_pdf.write_bytes(b"SEARCHABLE")
    seen: dict[str, Path] = {}

    def fake_run_basic_ocr_benchmark(**kwargs) -> BasicOcrRunSummary:
        assert kwargs["pdf_path"] == seen["textless_pdf"]
        return BasicOcrRunSummary(
            run_dir=run_dir,
            results=(_ok_benchmark_result("chandra"), _ok_benchmark_result("surya")),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    def fake_build_compare_txt_from_benchmark(**_kwargs):
        return [_ok_compare_result("chandra", run_dir / "_compare_txt" / "doc__chandra.txt")]

    def fake_build_textless_source_pdf(*, source_pdf: Path, out_pdf: Path, dpi: int = 300) -> Path:
        assert source_pdf == input_pdf.resolve()
        assert dpi == 300
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(b"TEXTLESS")
        seen["textless_pdf"] = out_pdf
        return out_pdf

    def fake_run_artifact_searchable_package(**kwargs):
        assert kwargs["engines"] == ("chandra",)
        assert kwargs["pdf_root"] == seen["textless_pdf"].parent
        assert kwargs["reconciliation_root"] == run_dir
        assert seen["textless_pdf"].exists()
        assert os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR") == str(run_dir / "surya")
        return [_ok_artifact_result(produced_pdf, engine="chandra")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(
        ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark
    )
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(
        ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package
    )
    monkeypatch.setattr(ocr_pipeline, "_pdf_page_count", lambda _path: 1)

    summary = build_searchable_pdf(
        pdf_path=input_pdf,
        mode="chandra+surya",
        work_root=tmp_path / "work",
        overwrite_input_path=True,
        return_bytes=False,
        strict=True,
        delete_original_text_layer=True,
    )

    assert isinstance(summary, SearchablePdfSummary)
    assert summary.mode == "chandra+surya"
    assert summary.overwritten_input_path == input_pdf.resolve()
    assert summary.output_pdf_path == input_pdf.resolve()
    assert input_pdf.read_bytes() == b"SEARCHABLE"


def test_strict_hybrid_rejects_missing_engine() -> None:
    summary = BasicOcrRunSummary(
        run_dir=Path("run"),
        results=(_ok_benchmark_result("chandra"),),
        result_files=tuple(),
        failed_engines=tuple(),
        skipped_engines=("surya: unavailable",),
    )

    with pytest.raises(RuntimeError, match="missing successful engines: surya"):
        _ensure_requested_engines_succeeded(
            summary,
            expected_engines=("chandra", "surya"),
        )


def test_build_searchable_pdf_reports_monotonic_progress(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF input")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    produced_pdf = tmp_path / "produced.pdf"
    produced_pdf.write_bytes(b"%PDF output")
    progress_values: list[int] = []

    def fake_benchmark(**kwargs) -> BasicOcrRunSummary:
        callback = kwargs["progress"]
        callback(0, "starting")
        callback(40, "running")
        callback(100, "complete")
        return BasicOcrRunSummary(
            run_dir=run_dir,
            results=(_ok_benchmark_result("chandra"), _ok_benchmark_result("surya")),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_benchmark)
    monkeypatch.setattr(
        ocr_pipeline,
        "build_compare_txt_from_benchmark",
        lambda **_kwargs: [_ok_compare_result("chandra", tmp_path / "doc__chandra.txt")],
    )
    monkeypatch.setattr(
        ocr_pipeline,
        "run_artifact_searchable_package",
        lambda **_kwargs: [_ok_artifact_result(produced_pdf, engine="chandra")],
    )
    monkeypatch.setattr(ocr_pipeline, "_pdf_page_count", lambda _path: 1)

    build_searchable_pdf(
        pdf_path=input_pdf,
        mode="chandra+surya",
        work_root=tmp_path / "work",
        overwrite_input_path=False,
        return_bytes=False,
        strict=True,
        progress=lambda value, _status: progress_values.append(value),
        delete_original_text_layer=False,
    )

    assert progress_values == sorted(progress_values)
    assert progress_values[-1] == 100
    assert 77 in progress_values
    assert 78 in progress_values


def test_run_basic_ocr_benchmark_supports_engine_python_override(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% fake")
    output_root = tmp_path / "runs"

    fake_surya_python = tmp_path / "surya_python.exe"
    fake_surya_python.write_text("python", encoding="utf-8")
    monkeypatch.setenv("UNISCAN_SURYA_PYTHON", str(fake_surya_python))
    monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "275")

    def fake_detect_status(_engine: str):
        # Local venv may be incomplete; override should bypass this.
        return SimpleNamespace(ready=False, missing=["local deps missing"])

    def fake_subprocess_run(cmd, cwd, capture_output, text, encoding, errors, env, timeout):
        assert capture_output is True
        assert text is True
        assert encoding == "utf-8"
        assert errors == "replace"
        assert timeout is None
        assert cmd[0] == str(fake_surya_python)
        assert cmd[cmd.index("--dpi") + 1] == "275"
        assert "--internal-reconciliation-token" not in cmd
        output_dir = Path(cmd[cmd.index("--output") + 1])
        engine = cmd[cmd.index("--engines") + 1]
        report_path = output_dir / f"{pdf_path.stem}_ocr_benchmark.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "pdf_path": str(pdf_path),
                    "page_count": 1,
                    "sample_pages": [1],
                    "results": [
                        {
                            "engine": engine,
                            "status": "ok",
                            "sample_pages": [1],
                            "elapsed_seconds": 0.25,
                            "artifact_path": str(output_dir / f"{pdf_path.stem}_{engine}.txt"),
                            "text_chars": 123,
                            "memory_delta_mb": None,
                            "error": None,
                            "note": None,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ocr_pipeline, "detect_ocr_engine_status", fake_detect_status)
    monkeypatch.setattr(ocr_pipeline.subprocess, "run", fake_subprocess_run)

    summary = run_basic_ocr_benchmark(
        pdf_path=pdf_path,
        mode_key="surya",
        page_numbers=(1,),
        lang="rus+eng",
        output_root=output_root,
    )

    assert summary.results
    assert summary.results[0].engine == "surya"
    assert summary.results[0].status == "ok"
    assert summary.failed_engines == tuple()


def test_engine_subprocess_defer_uses_matching_internal_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "engine"

    def fake_subprocess_run(cmd, **kwargs):
        token_index = cmd.index("--internal-reconciliation-token") + 1
        token = cmd[token_index]
        assert len(token) == 32
        assert kwargs["env"]["UNISCAN_INTERNAL_RECONCILIATION_TOKEN"] == token
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "input_ocr_benchmark.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "engine": "surya",
                            "status": "reconciliation_pending",
                            "sample_pages": [1],
                            "elapsed_seconds": 0.1,
                            "artifact_path": str(output_dir / "input_surya.txt"),
                            "text_chars": 0,
                            "page_error_count": 1,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(ocr_pipeline.subprocess, "run", fake_subprocess_run)

    result = ocr_pipeline._run_engine_benchmark_subprocess(
        python_exe=tmp_path / "python.exe",
        engine="surya",
        pdf_path=pdf_path,
        output_dir=output_dir,
        sample_size=1,
        page_numbers=(1,),
        lang="eng",
        dpi=220,
        defer_empty_pages=True,
    )

    assert result.status == "reconciliation_pending"
    assert result.page_error_count == 1


def test_engine_subprocess_timeout_raises_runtime_error(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    pdf_path = tmp_path / "input.pdf"
    output_dir = tmp_path / "engine"
    monkeypatch.setenv("UNISCAN_ENGINE_SUBPROCESS_TIMEOUT_SECONDS", "12.5")

    def fake_subprocess_run(cmd, **kwargs):
        raise ocr_pipeline.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(ocr_pipeline.subprocess, "run", fake_subprocess_run)

    try:
        ocr_pipeline._run_engine_benchmark_subprocess(
            python_exe=tmp_path / "python.exe",
            engine="surya",
            pdf_path=pdf_path,
            output_dir=output_dir,
            sample_size=1,
            page_numbers=(1,),
            lang="rus+eng",
            dpi=220,
        )
    except RuntimeError as exc:
        assert "subprocess timed out after 12.5 seconds" in str(exc)
    else:
        raise AssertionError("Expected timeout to become a RuntimeError")


def test_ocr_render_dpi_defaults_and_clamps(monkeypatch) -> None:
    monkeypatch.delenv("UNISCAN_OCR_RENDER_DPI", raising=False)
    assert ocr_pipeline._resolve_ocr_render_dpi() == 220
    monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "999")
    assert ocr_pipeline._resolve_ocr_render_dpi() == 400
    monkeypatch.setenv("UNISCAN_OCR_RENDER_DPI", "bad")
    assert ocr_pipeline._resolve_ocr_render_dpi() == 220


def test_engine_python_override_accepts_relative_path(monkeypatch) -> None:
    tmp_path = _new_test_dir()
    fake_surya_python = tmp_path / "runtime" / "python.exe"
    fake_surya_python.parent.mkdir(parents=True, exist_ok=True)
    fake_surya_python.write_text("python", encoding="utf-8")

    relative_python = fake_surya_python.relative_to(Path.cwd())
    monkeypatch.setenv("UNISCAN_SURYA_PYTHON", str(relative_python))

    assert ocr_pipeline._resolve_engine_python("surya") == fake_surya_python.resolve()


def test_engine_subprocess_env_maps_surya_gpu_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UNISCAN_SURYA_TORCH_DEVICE", "cuda:0")
    monkeypatch.setenv("UNISCAN_SURYA_REQUIRE_GPU", "1")

    env = ocr_pipeline._build_engine_subprocess_env(engine="surya", repo_root=tmp_path)

    assert env["TORCH_DEVICE"] == "cuda:0"
    assert env["UNISCAN_SURYA_REQUIRE_GPU"] == "1"


def test_merge_candidate_validator_runs_before_publish() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    chunks = ocr_pipeline._split_pdf_chunks(
        source_pdf=source_pdf,
        output_root=tmp_path / "chunks",
        pages_per_chunk=10,
    )
    output_pdf = tmp_path / "merged.pdf"
    validated: list[Path] = []

    def _reject_candidate(candidate: Path) -> None:
        validated.append(candidate)
        assert candidate.is_file()
        assert not output_pdf.exists()
        raise RuntimeError("candidate mutation rejected")

    with pytest.raises(RuntimeError, match="candidate mutation rejected"):
        ocr_pipeline._merge_pdf_chunks(
            source_pdf=source_pdf,
            chunks=[(chunk, chunk.path) for chunk in chunks],
            output_pdf=output_pdf,
            validate_candidate=_reject_candidate,
        )

    assert len(validated) == 1
    assert not output_pdf.exists()




def test_merge_reproduces_page_three_exact_text_retention_failure() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    chunk_pdf = tmp_path / "chunk.pdf"
    output_pdf = tmp_path / "merged.pdf"
    source = fitz.open()
    chunk = fitz.open()
    try:
        for page_number in range(1, 4):
            source.new_page(width=200, height=200)
            page = chunk.new_page(width=200, height=200)
            text = "WRONG PAGE THREE" if page_number == 3 else f"EXPECTED PAGE {page_number}"
            page.insert_text((20, 20), text, fontsize=6, render_mode=3)
        source.save(str(source_pdf), garbage=4, deflate=True)
        chunk.save(str(chunk_pdf), garbage=4, deflate=True)
    finally:
        source.close()
        chunk.close()

    expected_page_texts = ("EXPECTED PAGE 1", "EXPECTED PAGE 2", "EXPECTED PAGE THREE")
    pdf_chunk = ocr_pipeline._PdfChunk(1, 1, 3, chunk_pdf)

    with pytest.raises(
        RuntimeError,
        match="Output PDF page 3 failed exact searchable text retention",
    ):
        ocr_pipeline._merge_pdf_chunks(
            source_pdf=source_pdf,
            chunks=[(pdf_chunk, chunk_pdf)],
            output_pdf=output_pdf,
            validate_candidate=lambda candidate: ocr_pipeline._validate_merged_candidate(
                candidate_pdf=candidate,
                expected_page_texts=expected_page_texts,
                textless_visual_seals={},
            ),
        )

    assert not output_pdf.exists()
def test_private_chunk_snapshot_rejects_record_fingerprint_mismatch() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    target = tmp_path / "snapshots" / "chunk.snapshot.pdf"

    with pytest.raises(RuntimeError, match="fingerprint changed before merge"):
        ocr_pipeline._stable_snapshot_chunk_output(
            source=source_pdf,
            target=target,
            expected_sha256="0" * 64,
            expected_size=source_pdf.stat().st_size,
        )

    assert not target.exists()


def test_merge_rejects_source_mutation_after_candidate_validation() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    chunks = ocr_pipeline._split_pdf_chunks(
        source_pdf=source_pdf,
        output_root=tmp_path / "chunks",
        pages_per_chunk=10,
    )
    expected_source = ocr_pipeline._stable_file_fingerprint(source_pdf)
    output_pdf = tmp_path / "merged.pdf"

    def _mutate_source(_candidate: Path) -> None:
        source_pdf.write_bytes(b"changed during merge validation")

    with pytest.raises(RuntimeError, match="Source PDF changed during chunk merge"):
        ocr_pipeline._merge_pdf_chunks(
            source_pdf=source_pdf,
            chunks=[(chunk, chunk.path) for chunk in chunks],
            output_pdf=output_pdf,
            validate_candidate=_mutate_source,
            expected_source_fingerprint=expected_source,
        )

    assert not output_pdf.exists()


def test_premerge_rejects_processing_source_changed_after_outer_validation() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        delete_original_text_layer=False,
    )
    chunk = ocr_pipeline._PdfChunk(1, 1, 1, source_pdf)
    source_sizes = ocr_pipeline._source_page_sizes(source_pdf)
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=chunk,
        summary=summary,
        run_dir=tmp_path,
        source_sizes=source_sizes,
        delete_original_text_layer=False,
    )
    required = ocr_pipeline._required_chunk_evidence(
        summary=summary,
        chunk=chunk,
        run_dir=tmp_path,
        delete_original_text_layer=False,
    )
    source_pdf.write_bytes(source_pdf.read_bytes() + b"\n% mutation after validation\n")

    with pytest.raises(
        RuntimeError,
        match="Validated outer semantic evidence changed before merge",
    ):
        ocr_pipeline._premerge_chunk_state(
            chunk=chunk,
            record=record,
            summary=summary,
            required=required,
            snapshot_root=tmp_path / "snapshots",
        )


def test_chunked_false_mode_preserves_native_and_overlay_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=2)

    def fake_build_searchable_pdf(**kwargs: object) -> SearchablePdfSummary:
        chunk_pdf = Path(str(kwargs["pdf_path"]))
        return _write_complete_hybrid_summary(
            chunk_pdf=chunk_pdf,
            run_dir=Path(str(kwargs["work_root"])) / "run",
            delete_original_text_layer=False,
        )

    monkeypatch.setattr(ocr_pipeline, "build_searchable_pdf", fake_build_searchable_pdf)
    summary = ocr_pipeline._build_searchable_pdf_chunked(
        input_path=source_pdf,
        mode="chandra+surya",
        lang="rus+eng",
        work_root=tmp_path / "work",
        overwrite_target=None,
        return_bytes=False,
        strict=True,
        progress=None,
        delete_original_text_layer=False,
        chunk_pages=1,
        page_count=2,
    )

    merged_texts = ocr_pipeline._extract_pdf_page_texts(summary.output_pdf_path)
    assert len(merged_texts) == 2
    assert "PAGE 1" in merged_texts[0]
    assert "PAGE 2" in merged_texts[1]
    assert all("CHANDRA PAGE 1" in text for text in merged_texts)


def test_global_graphics_cap_spans_non_ten_page_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=7)

    def fake_build_searchable_pdf(**kwargs: object) -> SearchablePdfSummary:
        chunk_pdf = Path(str(kwargs["pdf_path"]))
        return _write_complete_hybrid_summary(
            chunk_pdf=chunk_pdf,
            run_dir=Path(str(kwargs["work_root"])) / "run",
            delete_original_text_layer=False,
        )

    def fake_premerge_chunk_state(
        *,
        chunk: ocr_pipeline._PdfChunk,
        record: dict[str, object],
        summary: SearchablePdfSummary,
        required: ocr_pipeline._RequiredChunkEvidence,
        snapshot_root: Path,
    ) -> tuple[Path, tuple[str, ...], dict[int, dict[str, object]]]:
        del record, required, snapshot_root
        local_page_count = chunk.end_page - chunk.start_page + 1
        global_page = 6 if chunk.index == 1 else 7
        return (
            summary.output_pdf_path,
            tuple("" for _ in range(local_page_count)),
            {global_page: {"sealed": True}},
        )

    monkeypatch.setattr(ocr_pipeline, "build_searchable_pdf", fake_build_searchable_pdf)
    monkeypatch.setattr(ocr_pipeline, "_premerge_chunk_state", fake_premerge_chunk_state)

    with pytest.raises(RuntimeError, match="Global textless-graphics recovery cap exceeded"):
        ocr_pipeline._build_searchable_pdf_chunked(
            input_path=source_pdf,
            mode="chandra+surya",
            lang="rus+eng",
            work_root=tmp_path / "work",
            overwrite_target=None,
            return_bytes=False,
            strict=True,
            progress=None,
            delete_original_text_layer=False,
            chunk_pages=6,
            page_count=7,
        )


def _complete_c2_fixture(
    source_pdf: Path,
    summary: SearchablePdfSummary,
    run_dir: Path,
    *,
    delete_original_text_layer: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {}
    ocr_pipeline._complete_chunk_record(
        record=record,
        chunk=ocr_pipeline._PdfChunk(1, 1, 1, source_pdf),
        summary=summary,
        run_dir=run_dir,
        source_sizes=ocr_pipeline._source_page_sizes(source_pdf),
        delete_original_text_layer=delete_original_text_layer,
    )
    return record


@pytest.mark.parametrize("delete_original_text_layer", [False, True])
def test_c2_complete_chunk_accepts_exact_processing_source_derivation(
    delete_original_text_layer: bool,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        delete_original_text_layer=delete_original_text_layer,
    )
    record = _complete_c2_fixture(
        source_pdf,
        summary,
        tmp_path,
        delete_original_text_layer=delete_original_text_layer,
    )
    assert record["status"] == "done"


@pytest.mark.parametrize(
    ("fixture_delete", "validation_delete", "message"),
    [
        (False, True, "must be textless"),
        (True, False, "must exactly equal the fresh chunk"),
    ],
)
def test_c2_complete_chunk_rejects_wrong_processing_source_mode(
    fixture_delete: bool,
    validation_delete: bool,
    message: str,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        delete_original_text_layer=fixture_delete,
    )
    with pytest.raises(RuntimeError, match=message):
        _complete_c2_fixture(
            source_pdf,
            summary,
            tmp_path,
            delete_original_text_layer=validation_delete,
        )


@pytest.mark.parametrize("defect", ["pages-source", "render-name"])
def test_c2_complete_chunk_rejects_unbound_pages_source_evidence(defect: str) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(chunk_pdf=source_pdf, run_dir=tmp_path / "run")
    pages_path = summary.run_dir / "surya" / "surya" / "pages.json"
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    assert pages["pages"][0]["source_raster_identity"]["name"] == "00001.png"
    if defect == "pages-source":
        alien = tmp_path / "alien.pdf"
        shutil.copy2(source_pdf, alien)
        pages["pdf_path"] = str(alien.resolve())
        message = "pages source PDF disagrees"
    else:
        pages["pages"][0]["source_raster_identity"]["name"] = "alien.png"
        message = "source raster identity is invalid"
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        _complete_c2_fixture(source_pdf, summary, tmp_path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("compare-source", "does not exactly equal its benchmark artifact"),
        ("artifact-text", "is not the Chandra compare text"),
        ("compare-bytes", "compare text differs from its benchmark aggregate"),
    ],
)
def test_c2_complete_chunk_rejects_cross_stage_text_binding(
    defect: str,
    message: str,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(chunk_pdf=source_pdf, run_dir=tmp_path / "run")
    chandra = next(item for item in summary.compare_results if item.engine == "chandra")
    surya = next(item for item in summary.compare_results if item.engine == "surya")
    if defect == "compare-source":
        alien = summary.run_dir / "alien-benchmark.txt"
        shutil.copy2(Path(str(chandra.source_artifact_path)), alien)
        summary = replace(
            summary,
            compare_results=tuple(
                replace(item, source_artifact_path=str(alien)) if item.engine == "chandra" else item
                for item in summary.compare_results
            ),
        )
    elif defect == "artifact-text":
        summary = replace(
            summary,
            artifact_results=(
                replace(
                    summary.artifact_results[0],
                    text_artifact_path=str(surya.compare_txt_path),
                ),
            ),
        )
    else:
        compare_path = Path(str(chandra.compare_txt_path))
        payload = compare_path.read_bytes()
        compare_path.write_bytes(b"X" + payload[1:])
    with pytest.raises(RuntimeError, match=message):
        _complete_c2_fixture(source_pdf, summary, tmp_path)


def test_c2_complete_chunk_rejects_same_length_wrong_hidden_text() -> None:
    import fitz

    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(chunk_pdf=source_pdf, run_dir=tmp_path / "run")
    wrong_output = summary.run_dir / "wrong-hidden-text.pdf"
    document = fitz.open(str(source_pdf))
    try:
        document[0].insert_text((36, 36), "CHANDRA PAGE X", fontsize=6, render_mode=3)
        document.save(str(wrong_output), garbage=4, deflate=True)
    finally:
        document.close()
    assert len(ocr_pipeline._extract_pdf_page_texts(wrong_output)[0]) == len(
        ocr_pipeline._extract_pdf_page_texts(summary.output_pdf_path)[0]
    )
    summary = replace(
        summary,
        output_pdf_path=wrong_output,
        artifact_results=(
            replace(summary.artifact_results[0], searchable_pdf_path=str(wrong_output)),
        ),
    )
    with pytest.raises(RuntimeError, match="failed exact searchable text retention"):
        _complete_c2_fixture(source_pdf, summary, tmp_path)


def test_c2_complete_chunk_rejects_unaccounted_chandra_alternative_text() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(chunk_pdf=source_pdf, run_dir=tmp_path / "run")
    engine_dir = summary.run_dir / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    row = pages["pages"][0]
    geometry_path = engine_dir / str(row["geometry_file"])
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    image = geometry["images"][0]
    sidecar_path = Path(str(row["attempt_history"][0]["sidecar_path"]))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_result"]["html"] = "ALIEN ALTERNATIVE TEXT"
    alternative = ocr_benchmark._chandra_alternative_text_evidence(
        raw_result=sidecar["raw_result"],
        texts=["CHANDRA PAGE 1"],
        ignored_graphic_lines=[],
    )
    assert alternative["accounting"] == "unaccounted"
    sidecar["evidence"]["alternative_text_evidence"] = alternative
    image["attempts"][0]["alternative_text_evidence"] = alternative
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    fingerprint = ocr_pipeline._stable_file_fingerprint(sidecar_path)
    for history in (row["attempt_history"], image["attempt_history"]):
        history[0]["sidecar_sha256"] = fingerprint["sha256"]
        history[0]["sidecar_bytes"] = fingerprint["size"]
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unaccounted alternative text"):
        _complete_c2_fixture(source_pdf, summary, tmp_path)


def test_c2_complete_chunk_rejects_bool_retry_history_size() -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(
        chunk_pdf=source_pdf,
        run_dir=tmp_path / "run",
        include_retry=True,
    )
    engine_dir = summary.run_dir / "surya" / "surya"
    pages_path = engine_dir / "pages.json"
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    geometry_path = engine_dir / str(pages["pages"][0]["geometry_file"])
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    pages["pages"][0]["attempt_history"][0]["image_bytes"] = True
    geometry["images"][0]["attempt_history"][0]["image_bytes"] = True
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte counts|history seal"):
        _complete_c2_fixture(source_pdf, summary, tmp_path)


@pytest.mark.parametrize("semantic_file", ["main-geometry", "attempt-sidecar"])
def test_c2_complete_chunk_rejects_semantic_mutation_before_seal(
    monkeypatch: pytest.MonkeyPatch,
    semantic_file: str,
) -> None:
    tmp_path = _new_test_dir()
    source_pdf = tmp_path / "source.pdf"
    _write_numbered_pdf(source_pdf, page_count=1)
    summary = _write_complete_hybrid_summary(chunk_pdf=source_pdf, run_dir=tmp_path / "run")
    engine_dir = summary.run_dir / "chandra" / "chandra"
    pages = json.loads((engine_dir / "pages.json").read_text(encoding="utf-8"))
    row = pages["pages"][0]
    target = (
        engine_dir / str(row["geometry_file"])
        if semantic_file == "main-geometry"
        else Path(str(row["attempt_history"][0]["sidecar_path"]))
    )
    original = ocr_pipeline._complete_chunk_evidence_entries
    mutated = False

    def mutate_then_enumerate(**kwargs: object) -> list[dict[str, object]]:
        nonlocal mutated
        if not mutated:
            target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
            mutated = True
        return original(**kwargs)

    monkeypatch.setattr(
        ocr_pipeline,
        "_complete_chunk_evidence_entries",
        mutate_then_enumerate,
    )
    with pytest.raises(RuntimeError, match="changed before it could be sealed"):
        _complete_c2_fixture(source_pdf, summary, tmp_path)
    assert mutated is True


def test_c2_accepted_textless_graphics_pages_are_an_exact_subset() -> None:
    assert ocr_pipeline._accepted_textless_graphics_pages(
        {"accepted_textless_graphics_pages": [3, 1]},
        expected_pages=3,
    ) == frozenset({1, 3})
    with pytest.raises(RuntimeError, match="duplicates"):
        ocr_pipeline._accepted_textless_graphics_pages(
            {"accepted_textless_graphics_pages": [1, 1]},
            expected_pages=3,
        )
    with pytest.raises(RuntimeError, match="invalid"):
        ocr_pipeline._accepted_textless_graphics_pages(
            {"accepted_textless_graphics_pages": [True]},
            expected_pages=3,
        )
