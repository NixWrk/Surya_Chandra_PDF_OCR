from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest

from uniscan.app import ocr_pipeline
from uniscan.app.ocr_pipeline import (
    BasicOcrRunSummary,
    SearchablePdfSummary,
    _ensure_requested_engines_succeeded,
    build_searchable_pdf,
    run_basic_ocr_benchmark,
)
from uniscan.ocr import ArtifactSearchableResult, CompareTxtBuildResult, OcrBenchmarkResult


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
) -> SearchablePdfSummary:
    import fitz

    chunk_pdf = chunk_pdf.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_source = run_dir.parent / "_source_pdf_without_text_fixture" / chunk_pdf.name
    artifact_source.parent.mkdir(parents=True, exist_ok=True)
    if not artifact_source.exists():
        shutil.copy2(chunk_pdf, artifact_source)
    artifact_source = artifact_source.resolve()
    compare_dir = run_dir / "_compare_txt"
    compare_dir.mkdir(parents=True, exist_ok=True)
    if output_pdf is None:
        output_pdf = run_dir / "result.pdf"
    output_pdf = output_pdf.resolve()
    shutil.copy2(chunk_pdf, output_pdf)
    document = fitz.open(str(chunk_pdf))
    try:
        page_count = int(document.page_count)
    finally:
        document.close()
    local_pages = list(range(1, page_count + 1))
    reconciliation = run_dir / "page_reconciliation.json"
    reconciliation.write_text(
        json.dumps({"schema": "uniscan.page-reconciliation.v1", "status": "ok"}),
        encoding="utf-8",
    )

    benchmark_results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    benchmark_artifacts: dict[str, Path] = {}
    for engine in ("chandra", "surya"):
        output_dir = run_dir / engine
        engine_dir = output_dir / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        page_rows: list[dict[str, object]] = []
        aggregate_blocks: list[str] = []
        for source_page in local_pages:
            text = f"{engine.upper()} PAGE {source_page}"
            text_path = engine_dir / f"page_{source_page:04d}.txt"
            text_path.write_text(text, encoding="utf-8")
            geometry_path = engine_dir / f"page_{source_page:04d}.{engine}.json"
            geometry_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image_name": f"page_{source_page:04d}.png",
                                "ocr_outcome": "text",
                                "pages": [
                                    {
                                        "image_bbox": [0, 0, 100, 100],
                                        "text_lines": [{"text": text, "bbox": [0, 0, 80, 10]}],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            row: dict[str, object] = {
                "source_page": source_page,
                "file": text_path.name,
                "geometry_file": geometry_path.name,
                "ocr_outcome": "text",
            }
            if engine == "surya":
                row["attempt_count"] = 1
            if include_retry and engine == "surya" and source_page == 1:
                retry_history: list[dict[str, object]] = []
                preprocessings = (
                    "original",
                    "autocontrast-cutoff-1",
                    "rgb-scale-0.5-center-white-lanczos-v1",
                )
                for attempt, preprocessing in enumerate(preprocessings, start=1):
                    retry_dir = engine_dir / "page_0001.retry" / f"attempt_{attempt}"
                    retry_dir.mkdir(parents=True)
                    retry_image = retry_dir / "page_0001.png"
                    retry_image.write_bytes(f"durable retry png evidence {attempt}".encode())
                    retry_sidecar = retry_dir / "surya_page_lines.json"
                    retry_sidecar.write_text(
                        json.dumps({"images": [{"image_name": retry_image.name}]}),
                        encoding="utf-8",
                    )
                    retry_history.append(
                        {
                            "attempt": attempt,
                            "preprocessing": preprocessing,
                            "image_path": str(retry_image.resolve()),
                            "sidecar_path": str(retry_sidecar.resolve()),
                        }
                    )
                row.update(
                    {
                        "attempt_count": 3,
                        "retry_preprocessing": "rgb-scale-0.5-center-white-lanczos-v1",
                        "retry_policy": (
                            "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
                        ),
                        "selected_attempt": 3,
                        "attempt_history": retry_history,
                    }
                )
            page_rows.append(row)
            aggregate_blocks.extend([f"[SOURCE PAGE {source_page:04d}]", text, ""])
        aggregate_path = engine_dir / "all_pages.txt"
        aggregate_text = "\n".join(aggregate_blocks).strip() + "\n"
        aggregate_path.write_text(aggregate_text, encoding="utf-8")
        pages_path = engine_dir / "pages.json"
        pages_path.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "pages": page_rows,
                    "aggregate_file": aggregate_path.name,
                }
            ),
            encoding="utf-8",
        )
        artifact_path = output_dir / f"document_{engine}.txt"
        artifact_path.write_text(aggregate_text, encoding="utf-8")
        benchmark_artifacts[engine] = artifact_path
        result = OcrBenchmarkResult(
            engine=engine,
            status="ok",
            sample_pages=list(local_pages),
            elapsed_seconds=0.1,
            artifact_path=str(artifact_path),
            text_chars=len(aggregate_text),
        )
        benchmark_results.append(result)
        report_path = output_dir / "document_ocr_benchmark.json"
        report_path.write_text(
            json.dumps({"results": [asdict(result)]}),
            encoding="utf-8",
        )
        result_files.append(report_path)
    result_files.append(reconciliation)

    compare_results: list[CompareTxtBuildResult] = []
    for engine in ("chandra", "surya"):
        compare_path = compare_dir / f"chunk__{engine}.txt"
        compare_path.write_text(f"{engine} compare", encoding="utf-8")
        compare_results.append(
            CompareTxtBuildResult(
                engine=engine,
                status="ok",
                source_artifact_path=str(benchmark_artifacts[engine]),
                compare_txt_path=str(compare_path),
            )
        )
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
        text_chars=page_count,
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


def test_chunked_hybrid_pipeline_uses_ten_page_hybrid_jobs_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fitz

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
