from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from uniscan.web.service import _JobState, _JobStore, _parse_job_request, _query_bool


def test_query_bool_parsing() -> None:
    assert _query_bool("1", default=False) is True
    assert _query_bool("true", default=False) is True
    assert _query_bool("yes", default=False) is True
    assert _query_bool("0", default=True) is False
    assert _query_bool("false", default=True) is False
    assert _query_bool("no", default=True) is False
    assert _query_bool("unknown", default=True) is True


def test_parse_job_request_defaults() -> None:
    parsed = urlparse("/api/jobs")
    mode, pages_raw, lang, strict, filename, delete_original_text_layer = _parse_job_request(
        parsed,
        default_lang="rus+eng",
    )
    assert mode == "chandra+surya"
    assert pages_raw == ""
    assert lang == "rus+eng"
    assert strict is True
    assert filename == "document.pdf"
    assert delete_original_text_layer is True


def test_parse_job_request_applies_filename_extension() -> None:
    parsed = urlparse(
        "/api/jobs?mode=surya&pages=1-3&lang=eng&strict=0&filename=my_file&delete_text_layer=0"
    )
    mode, pages_raw, lang, strict, filename, delete_original_text_layer = _parse_job_request(
        parsed,
        default_lang="rus+eng",
    )
    assert mode == "surya"
    assert pages_raw == "1-3"
    assert lang == "eng"
    assert strict is False
    assert filename == "my_file.pdf"
    assert delete_original_text_layer is False


def test_parse_job_request_accepts_legacy_delete_text_layer_name() -> None:
    parsed = urlparse("/api/jobs?delete_original_text_layer=false")
    *_, delete_original_text_layer = _parse_job_request(parsed, default_lang="rus+eng")
    assert delete_original_text_layer is False


def test_job_store_persists_done_result_metadata(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = _JobStore(root)
    job = _JobState(
        job_id="JOB1",
        status="queued",
        progress=0,
        message="Queued",
        mode="chandra+surya",
        pages="",
        lang="rus+eng",
        strict=True,
        delete_original_text_layer=True,
        filename="input.pdf",
        input_bytes=10,
    )

    store.create(job)
    result_path = root / "JOB1" / "result.pdf"
    result_path.write_bytes(b"%PDF result")
    store.update(
        "JOB1",
        {
            "status": "done",
            "progress": 100,
            "message": "Done",
            "result_path": result_path,
            "result_bytes": result_path.stat().st_size,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
        event="done",
        message="Done",
    )

    reloaded = _JobStore(root)
    metadata = reloaded.metadata("JOB1")

    assert metadata is not None
    assert metadata["status"] == "done"
    assert metadata["result_bytes"] == len(b"%PDF result")
    assert metadata["result_url"] == "/api/jobs/JOB1/result"
    assert metadata["metadata_url"] == "/api/jobs/JOB1/metadata"
    assert (root / "JOB1" / "metadata.json").exists()
    assert (root / "JOB1" / "events.jsonl").exists()


def test_job_store_marks_active_jobs_interrupted_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = _JobStore(root)
    store.create(
        _JobState(
            job_id="JOB2",
            status="running",
            progress=42,
            message="OCR is running",
            mode="chandra+surya",
            pages="",
            lang="rus+eng",
            strict=True,
            delete_original_text_layer=True,
            filename="input.pdf",
            input_bytes=10,
        )
    )

    reloaded = _JobStore(root)
    metadata = reloaded.metadata("JOB2")

    assert metadata is not None
    assert metadata["status"] == "interrupted"
    assert metadata["error"] == "Interrupted by OCR API restart."
