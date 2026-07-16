from __future__ import annotations

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

    assert [
        (chunk.start_page, chunk.end_page)
        for chunk in chunks
    ] == [(1, 10), (11, 20), (21, 23)]

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
        compare_dir = run_dir / "_compare_txt"
        compare_dir.mkdir(parents=True, exist_ok=True)
        output_pdf = run_dir / "result.pdf"
        shutil.copy2(chunk_pdf, output_pdf)
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
        benchmark = BasicOcrRunSummary(
            run_dir=run_dir,
            results=(
                OcrBenchmarkResult(
                    engine="chandra",
                    status="ok",
                    sample_pages=list(range(1, pages + 1)),
                    elapsed_seconds=0.1,
                    artifact_path=str(output_pdf),
                    text_chars=pages,
                ),
            ),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )
        return SearchablePdfSummary(
            mode="chandra+surya",
            run_dir=run_dir,
            compare_dir=compare_dir,
            output_pdf_path=output_pdf,
            output_pdf_bytes=None,
            overwritten_input_path=None,
            benchmark=benchmark,
            compare_results=(
                _ok_compare_result("chandra", compare_dir / "chunk__chandra.txt"),
            ),
            artifact_results=(_ok_artifact_result(output_pdf, engine="chandra"),),
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
        list(range(11, 21)),
        list(range(21, 24)),
    ]
    manifest = json.loads(summary.chunk_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert [item["status"] for item in manifest["chunks"]] == ["done", "done", "done"]


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
        run_dir = Path(str(kwargs["work_root"])) / "run"
        compare_dir = run_dir / "_compare_txt"
        compare_dir.mkdir(parents=True, exist_ok=True)
        output_pdf = run_dir / "result.pdf"
        shutil.copy2(chunk_pdf, output_pdf)
        return SearchablePdfSummary(
            mode="chandra+surya",
            run_dir=run_dir,
            compare_dir=compare_dir,
            output_pdf_path=output_pdf,
            output_pdf_bytes=None,
            overwritten_input_path=None,
            benchmark=BasicOcrRunSummary(
                run_dir=run_dir,
                results=(_ok_benchmark_result("chandra"),),
                result_files=tuple(),
                failed_engines=tuple(),
                skipped_engines=tuple(),
            ),
            compare_results=tuple(),
            artifact_results=tuple(),
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
    manifest_path = next((tmp_path / "failed_work").glob("hybrid_chunks_*/chunk_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["failed_chunk"] == 2
    assert [item["status"] for item in manifest["chunks"]] == ["done", "error", "pending"]
    assert manifest["chunks"][1]["start_page"] == 11
    assert manifest["chunks"][1]["end_page"] == 20
    first_output = Path(str(manifest["chunks"][0]["output_pdf"]))
    assert first_output.is_file()


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
        expected_geometry = str((run_dir / "surya").resolve())
        assert os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR") == expected_geometry
        return [_ok_artifact_result(produced_pdf, engine="chandra")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark)
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package)
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
            results=(_ok_benchmark_result("surya"),),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    def fake_build_compare_txt_from_benchmark(**_kwargs):
        return [_ok_compare_result("surya", tmp_path / "doc__surya.txt")]

    def fake_run_artifact_searchable_package(**kwargs):
        assert kwargs["engines"] == ("surya",)
        assert kwargs["pdf_root"] == seen_textless_path["value"].parent
        assert os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR") is None
        return [_ok_artifact_result(produced_pdf, engine="surya")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark)
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package)

    summary = build_searchable_pdf(
        pdf_bytes=b"INLINE-PDF",
        mode="surya",
        work_root=tmp_path / "work_inline",
        overwrite_input_path=False,
        return_bytes=True,
        strict=True,
    )

    assert isinstance(summary, SearchablePdfSummary)
    assert summary.mode == "surya"
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
            results=(_ok_benchmark_result("surya"),),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    def fake_build_compare_txt_from_benchmark(**_kwargs):
        return [_ok_compare_result("surya", run_dir / "_compare_txt" / "doc__surya.txt")]

    def fake_build_textless_source_pdf(*, source_pdf: Path, out_pdf: Path, dpi: int = 300) -> Path:
        assert source_pdf == input_pdf.resolve()
        assert dpi == 300
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(b"TEXTLESS")
        seen["textless_pdf"] = out_pdf
        return out_pdf

    def fake_run_artifact_searchable_package(**kwargs):
        assert kwargs["engines"] == ("surya",)
        assert kwargs["pdf_root"] == seen["textless_pdf"].parent
        assert seen["textless_pdf"].exists()
        return [_ok_artifact_result(produced_pdf, engine="surya")]

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_run_basic_ocr_benchmark)
    monkeypatch.setattr(ocr_pipeline, "build_compare_txt_from_benchmark", fake_build_compare_txt_from_benchmark)
    monkeypatch.setattr(ocr_pipeline, "_build_textless_source_pdf", fake_build_textless_source_pdf)
    monkeypatch.setattr(ocr_pipeline, "run_artifact_searchable_package", fake_run_artifact_searchable_package)

    summary = build_searchable_pdf(
        pdf_path=input_pdf,
        mode="surya",
        work_root=tmp_path / "work",
        overwrite_input_path=True,
        return_bytes=False,
        strict=True,
        delete_original_text_layer=True,
    )

    assert isinstance(summary, SearchablePdfSummary)
    assert summary.mode == "surya"
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
            results=(_ok_benchmark_result("surya"),),
            result_files=tuple(),
            failed_engines=tuple(),
            skipped_engines=tuple(),
        )

    monkeypatch.setattr(ocr_pipeline, "run_basic_ocr_benchmark", fake_benchmark)
    monkeypatch.setattr(
        ocr_pipeline,
        "build_compare_txt_from_benchmark",
        lambda **_kwargs: [_ok_compare_result("surya", tmp_path / "doc__surya.txt")],
    )
    monkeypatch.setattr(
        ocr_pipeline,
        "run_artifact_searchable_package",
        lambda **_kwargs: [_ok_artifact_result(produced_pdf, engine="surya")],
    )

    build_searchable_pdf(
        pdf_path=input_pdf,
        mode="surya",
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
