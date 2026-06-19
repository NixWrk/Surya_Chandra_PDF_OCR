from __future__ import annotations

import http.client
import json
import sqlite3
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from uniscan.web.service import (
    UNISCAN_OCR_WORKER_CONCURRENCY,
    UNISCAN_JOB_PROTOCOL_VERSION,
    _build_handler,
    _JobState,
    _JobStore,
    _parse_job_request,
    _parse_protocol_metadata,
    _query_bool,
    _request_fingerprint,
)


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


def test_parse_protocol_metadata_from_headers_and_query() -> None:
    parsed = urlparse(
        "/api/jobs?project_id=query-project&priority=batch&estimated_pages=12&ttl_seconds=3600"
    )
    headers = {
        "X-Project-ID": "zotero",
        "X-Service-ID": "zotero-worker",
        "X-Task-ID": "zotero:item:ABCD1234:ocr",
        "X-Request-ID": "request-1",
        "X-Idempotency-Key": "zotero:item:ABCD1234:ocr:v1",
        "X-GPU-Policy": "auto",
        "X-Estimated-VRAM-GB": "8",
    }

    metadata = _parse_protocol_metadata(parsed, headers)

    assert metadata.protocol_version == UNISCAN_JOB_PROTOCOL_VERSION
    assert metadata.project_id == "zotero"
    assert metadata.service_id == "zotero-worker"
    assert metadata.task_id == "zotero:item:ABCD1234:ocr"
    assert metadata.request_id == "request-1"
    assert metadata.idempotency_key == "zotero:item:ABCD1234:ocr:v1"
    assert metadata.priority == "batch"
    assert metadata.gpu_policy == "auto"
    assert metadata.estimated_vram_gb == 8
    assert metadata.estimated_pages == 12
    assert metadata.ttl_seconds == 3600


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
        project_id="zotero",
        service_id="zotero-worker",
        task_id="zotero:item:ABCD1234:ocr",
        idempotency_key="zotero:item:ABCD1234:ocr:v1",
        priority="batch",
        gpu_policy="auto",
        estimated_vram_gb=8,
        estimated_pages=42,
        ttl_seconds=86400,
        input_sha256="abc",
        request_fingerprint=_request_fingerprint(
            input_sha256="abc",
            mode="chandra+surya",
            pages="",
            lang="rus+eng",
            strict=True,
            delete_original_text_layer=True,
        ),
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
    assert metadata["project_id"] == "zotero"
    assert metadata["service_id"] == "zotero-worker"
    assert metadata["task_id"] == "zotero:item:ABCD1234:ocr"
    assert metadata["idempotency_key"] == "zotero:item:ABCD1234:ocr:v1"
    assert metadata["priority"] == "batch"
    assert metadata["estimated_vram_gb"] == 8
    assert metadata["estimated_pages"] == 42
    assert metadata["ttl_seconds"] == 86400
    assert reloaded.find_by_idempotency_key("zotero:item:ABCD1234:ocr:v1") is not None
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
    assert metadata["finished_at"]


def test_job_store_keeps_queued_job_with_input_requeueable_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = _JobStore(root)
    input_path = store.write_input("JOBQ", b"%PDF queued")
    store.create(
        _JobState(
            job_id="JOBQ",
            status="queued",
            progress=0,
            message="Queued",
            mode="chandra+surya",
            pages="",
            lang="rus+eng",
            strict=True,
            delete_original_text_layer=True,
            filename="input.pdf",
            input_bytes=len(b"%PDF queued"),
            input_path=input_path,
        )
    )

    reloaded = _JobStore(root)
    metadata = reloaded.metadata("JOBQ")
    requeueable = reloaded.requeueable_jobs()

    assert metadata is not None
    assert metadata["status"] == "queued"
    assert metadata["input_path"] == str(input_path)
    assert [job.job_id for job in requeueable] == ["JOBQ"]


def test_job_store_summary_reports_counts_and_active_jobs(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = _JobStore(root)
    store.create(
        _JobState(
            job_id="JOB3",
            status="running",
            progress=55,
            message="OCR is running",
            mode="chandra+surya",
            pages="",
            lang="rus+eng",
            strict=True,
            delete_original_text_layer=True,
            filename="input.pdf",
            input_bytes=10,
            priority="interactive",
        )
    )

    summary = store.summary()

    assert summary["protocol_version"] == UNISCAN_JOB_PROTOCOL_VERSION
    assert summary["counts"] == {"running": 1}
    assert summary["worker_concurrency"] == UNISCAN_OCR_WORKER_CONCURRENCY
    assert summary["active_jobs"][0]["job_id"] == "JOB3"
    assert summary["active_jobs"][0]["priority"] == "interactive"


def test_job_store_writes_sqlite_index(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = _JobStore(root)
    store.create(
        _JobState(
            job_id="JOBSQL",
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
            idempotency_key="jobsql:v1",
            priority="batch",
        )
    )

    with sqlite3.connect(root / "jobs.sqlite3") as conn:
        row = conn.execute(
            "SELECT status, idempotency_key, priority, metadata_json FROM jobs WHERE job_id = ?",
            ("JOBSQL",),
        ).fetchone()

    assert row is not None
    assert row[0] == "queued"
    assert row[1] == "jobsql:v1"
    assert row[2] == "batch"
    assert json.loads(row[3])["job_id"] == "JOBSQL"


def test_job_store_cleanup_expired_removes_terminal_jobs_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UNISCAN_JOB_RETENTION_DAYS", "0")
    monkeypatch.setenv("UNISCAN_FAILED_JOB_RETENTION_DAYS", "0")
    root = tmp_path / "jobs"
    store = _JobStore(root)
    old_time = "2026-01-01T00:00:00+00:00"
    for job_id, status in (("DONE", "done"), ("ERR", "error"), ("ACTIVE", "queued")):
        input_path = store.write_input(job_id, f"%PDF {job_id}".encode("ascii"))
        store.create(
            _JobState(
                job_id=job_id,
                status=status,
                progress=100 if status != "queued" else 0,
                message=status,
                mode="chandra+surya",
                pages="",
                lang="rus+eng",
                strict=True,
                delete_original_text_layer=True,
                filename="input.pdf",
                input_bytes=10,
                input_path=input_path,
                created_at=old_time,
                updated_at=old_time,
                finished_at=old_time if status != "queued" else None,
            )
        )

    result = store.cleanup_expired()

    assert result["removed_count"] == 2
    assert set(result["removed"]) == {"DONE", "ERR"}
    assert store.metadata("DONE") is None
    assert store.metadata("ERR") is None
    assert store.metadata("ACTIVE") is not None
    assert not (root / "DONE").exists()
    assert not (root / "ERR").exists()
    assert (root / "ACTIVE").exists()


def test_http_jobs_are_processed_one_document_at_a_time(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()
    call_count = 0
    active_count = 0
    max_active = 0

    def fake_build_searchable_pdf(**_kwargs):
        nonlocal active_count, call_count, max_active
        with lock:
            call_count += 1
            call_index = call_count
            active_count += 1
            max_active = max(max_active, active_count)
        try:
            if call_index == 1:
                first_started.set()
                if not release_first.wait(timeout=5):
                    raise RuntimeError("Timed out waiting to release first OCR job.")
            elif call_index == 2:
                second_started.set()
            return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)
        finally:
            with lock:
                active_count -= 1

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]

        first_conn = http.client.HTTPConnection("127.0.0.1", port)
        first_conn.request(
            "POST",
            "/api/jobs?filename=first.pdf",
            body=b"%PDF first",
            headers={
                "Content-Type": "application/pdf",
                "X-Project-ID": "project-a",
                "X-Service-ID": "worker-a",
            },
        )
        first_response = first_conn.getresponse()
        first_payload = json.loads(first_response.read().decode("utf-8"))
        first_conn.close()

        assert first_response.status == HTTPStatus.ACCEPTED
        assert first_started.wait(timeout=2)

        second_conn = http.client.HTTPConnection("127.0.0.1", port)
        second_conn.request(
            "POST",
            "/api/jobs?filename=second.pdf",
            body=b"%PDF second",
            headers={
                "Content-Type": "application/pdf",
                "X-Project-ID": "project-b",
                "X-Service-ID": "worker-b",
            },
        )
        second_response = second_conn.getresponse()
        second_payload = json.loads(second_response.read().decode("utf-8"))
        second_conn.close()

        assert second_response.status == HTTPStatus.ACCEPTED
        assert first_payload["job_id"] != second_payload["job_id"]
        assert not second_started.wait(timeout=0.2)

        status_conn = http.client.HTTPConnection("127.0.0.1", port)
        status_conn.request("GET", f"/api/jobs/{second_payload['job_id']}")
        status_response = status_conn.getresponse()
        status_payload = json.loads(status_response.read().decode("utf-8"))
        status_conn.close()

        assert status_response.status == HTTPStatus.OK
        assert status_payload["status"] == "queued"

        summary_conn = http.client.HTTPConnection("127.0.0.1", port)
        summary_conn.request("GET", "/api/jobs")
        summary_response = summary_conn.getresponse()
        summary_payload = json.loads(summary_response.read().decode("utf-8"))
        summary_conn.close()

        assert summary_response.status == HTTPStatus.OK
        assert summary_payload["worker_concurrency"] == UNISCAN_OCR_WORKER_CONCURRENCY
        assert summary_payload["counts"] == {"queued": 1, "running": 1}

        release_first.set()
        assert second_started.wait(timeout=2)
        with lock:
            assert max_active == 1
    finally:
        release_first.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_job_persists_input_pdf_before_accepting(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_build_searchable_pdf(**_kwargs):
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        body = b"%PDF durable input"
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request(
            "POST",
            "/api/jobs?filename=input.pdf",
            body=body,
            headers={"Content-Type": "application/pdf", "X-Idempotency-Key": "input:v1"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()

        input_path = tmp_path / "work" / "jobs" / payload["job_id"] / "input.pdf"
        metadata_path = tmp_path / "work" / "jobs" / payload["job_id"] / "metadata.json"

        assert response.status == HTTPStatus.ACCEPTED
        assert input_path.read_bytes() == body
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["input_path"] == str(input_path.resolve())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_startup_requeues_persisted_queued_job(tmp_path: Path, monkeypatch) -> None:
    work_root = tmp_path / "work"
    jobs_root = work_root / "jobs"
    store = _JobStore(jobs_root)
    input_path = store.write_input("RESTORE1", b"%PDF restore")
    store.create(
        _JobState(
            job_id="RESTORE1",
            status="queued",
            progress=0,
            message="Queued",
            mode="chandra+surya",
            pages="",
            lang="rus+eng",
            strict=True,
            delete_original_text_layer=True,
            filename="input.pdf",
            input_bytes=len(b"%PDF restore"),
            input_path=input_path,
        )
    )
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    started = threading.Event()
    seen_pdf_path: list[Path] = []

    def fake_build_searchable_pdf(**kwargs):
        seen_pdf_path.append(Path(kwargs["pdf_path"]))
        started.set()
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=work_root, default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        assert started.wait(timeout=2)
        assert seen_pdf_path == [input_path]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_priority_orders_waiting_jobs(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first_started = threading.Event()
    release_first = threading.Event()
    processed: list[str] = []
    processed_lock = threading.Lock()

    def fake_build_searchable_pdf(**kwargs):
        job_id = Path(kwargs["pdf_path"]).parent.name
        with processed_lock:
            processed.append(job_id)
            call_index = len(processed)
        if call_index == 1:
            first_started.set()
            if not release_first.wait(timeout=5):
                raise RuntimeError("Timed out waiting to release first OCR job.")
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def submit(filename: str, priority: str) -> str:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        conn.request(
            "POST",
            f"/api/jobs?filename={filename}",
            body=f"%PDF {filename}".encode("ascii"),
            headers={"Content-Type": "application/pdf", "X-Priority": priority},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        assert response.status == HTTPStatus.ACCEPTED
        return str(payload["job_id"])

    try:
        first_id = submit("first.pdf", "normal")
        assert first_started.wait(timeout=2)
        low_id = submit("low.pdf", "low")
        interactive_id = submit("interactive.pdf", "interactive")

        release_first.set()
        deadline = threading.Event()
        for _ in range(20):
            with processed_lock:
                if len(processed) >= 3:
                    break
            deadline.wait(timeout=0.05)

        with processed_lock:
            assert processed[:3] == [first_id, interactive_id, low_id]
    finally:
        release_first.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_cancel_queued_job(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    call_count = 0
    lock = threading.Lock()

    def fake_build_searchable_pdf(**_kwargs):
        nonlocal call_count
        with lock:
            call_count += 1
            call_index = call_count
        if call_index == 1:
            first_started.set()
            if not release_first.wait(timeout=5):
                raise RuntimeError("Timed out waiting to release first OCR job.")
        elif call_index == 2:
            second_started.set()
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def submit(filename: str) -> str:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        conn.request(
            "POST",
            f"/api/jobs?filename={filename}",
            body=f"%PDF {filename}".encode("ascii"),
            headers={"Content-Type": "application/pdf"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        assert response.status == HTTPStatus.ACCEPTED
        return str(payload["job_id"])

    try:
        submit("first.pdf")
        assert first_started.wait(timeout=2)
        second_id = submit("second.pdf")

        cancel_conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        cancel_conn.request("POST", f"/api/jobs/{second_id}/cancel")
        cancel_response = cancel_conn.getresponse()
        cancel_payload = json.loads(cancel_response.read().decode("utf-8"))
        cancel_conn.close()

        assert cancel_response.status == HTTPStatus.OK
        assert cancel_payload["status"] == "cancelled"

        release_first.set()
        assert not second_started.wait(timeout=0.3)

        status_conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        status_conn.request("GET", f"/api/jobs/{second_id}")
        status_response = status_conn.getresponse()
        status_payload = json.loads(status_response.read().decode("utf-8"))
        status_conn.close()

        assert status_response.status == HTTPStatus.OK
        assert status_payload["status"] == "cancelled"
    finally:
        release_first.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_job_idempotency_replays_existing_job(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_build_searchable_pdf(**_kwargs):
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        body = b"%PDF-1.4\n"
        headers = {
            "Content-Type": "application/pdf",
            "X-Project-ID": "zotero",
            "X-Service-ID": "zotero-worker",
            "X-Task-ID": "zotero:item:ABCD1234:ocr",
            "X-Idempotency-Key": "zotero:item:ABCD1234:ocr:v1",
            "X-Priority": "batch",
        }

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/jobs?filename=input.pdf", body=body, headers=headers)
        first_response = conn.getresponse()
        first_payload = json.loads(first_response.read().decode("utf-8"))
        conn.close()

        assert first_response.status == HTTPStatus.ACCEPTED

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/jobs?filename=input.pdf", body=body, headers=headers)
        second_response = conn.getresponse()
        second_payload = json.loads(second_response.read().decode("utf-8"))
        conn.close()

        assert second_response.status == HTTPStatus.OK
        assert second_payload["idempotent_replay"] is True
        assert second_payload["job_id"] == first_payload["job_id"]
        assert second_payload["project_id"] == "zotero"
        assert second_payload["priority"] == "batch"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_job_idempotency_rejects_conflicting_request(tmp_path: Path, monkeypatch) -> None:
    result_pdf = tmp_path / "result.pdf"
    result_pdf.write_bytes(b"%PDF result")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_build_searchable_pdf(**_kwargs):
        return SimpleNamespace(output_pdf_path=result_pdf, run_dir=run_dir)

    monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", fake_build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        headers = {
            "Content-Type": "application/pdf",
            "X-Idempotency-Key": "same-key",
        }

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/jobs?mode=chandra+surya", body=b"%PDF A", headers=headers)
        first_response = conn.getresponse()
        first_response.read()
        conn.close()

        assert first_response.status == HTTPStatus.ACCEPTED

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/jobs?mode=surya", body=b"%PDF A", headers=headers)
        second_response = conn.getresponse()
        second_payload = json.loads(second_response.read().decode("utf-8"))
        conn.close()

        assert second_response.status == HTTPStatus.CONFLICT
        assert "different OCR request" in second_payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
