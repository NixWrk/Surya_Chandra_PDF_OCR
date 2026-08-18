from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import fitz
import pytest

from uniscan.web.service import _build_handler


@contextmanager
def _http_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    build_searchable_pdf=None,
) -> Iterator[ThreadingHTTPServer]:
    """Run the API with the watchdog disabled for bounded resource probes."""
    monkeypatch.setenv("OCR_WORKER_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("UNISCAN_JOB_CLEANUP_INTERVAL_SECONDS", "0")
    if build_searchable_pdf is not None:
        monkeypatch.setattr("uniscan.web.service.build_searchable_pdf", build_searchable_pdf)
    handler = _build_handler(work_root=tmp_path / "work", default_lang="rus+eng")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post_pdf(server: ThreadingHTTPServer, payload: bytes) -> tuple[int, dict[str, object]]:
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        conn.request(
            "POST",
            "/api/jobs?filename=resource-audit.pdf",
            body=payload,
            headers={"Content-Type": "application/pdf"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body
    finally:
        conn.close()


def _valid_pdf_bytes(marker: str = "") -> bytes:
    document = fitz.open()
    try:
        page = document.new_page(width=72, height=72)
        if marker:
            page.insert_text((10, 20), marker)
        return document.tobytes()
    finally:
        document.close()


def _encrypted_pdf_bytes() -> bytes:
    document = fitz.open()
    try:
        document.new_page(width=72, height=72)
        return document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    finally:
        document.close()


def test_http_should_reject_malformed_pdf_before_accepting_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _http_server(tmp_path, monkeypatch) as server:
        status, payload = _post_pdf(
            server,
            b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n",
        )

    assert status == HTTPStatus.BAD_REQUEST
    assert "PDF" in str(payload.get("error", ""))
    assert not list((tmp_path / "work" / "jobs").glob("*/input.pdf"))


def test_http_should_reject_encrypted_pdf_before_accepting_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _http_server(tmp_path, monkeypatch) as server:
        status, payload = _post_pdf(server, _encrypted_pdf_bytes())

    assert status == HTTPStatus.BAD_REQUEST
    assert "encrypt" in str(payload.get("error", "")).lower()
    assert not list((tmp_path / "work" / "jobs").glob("*/input.pdf"))


def test_http_accepts_pdf_above_chunk_size_without_page_count_admission_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_build(**_kwargs: object) -> None:
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("resource-limit audit stop")

    document = fitz.open()
    try:
        for _ in range(101):
            document.new_page(width=72, height=72)
        payload = document.tobytes()
    finally:
        document.close()

    try:
        with _http_server(
            tmp_path,
            monkeypatch,
            build_searchable_pdf=blocked_build,
        ) as server:
            status, response_payload = _post_pdf(server, payload)
            assert status == HTTPStatus.ACCEPTED
            assert response_payload["status"] in {"queued", "running"}
            assert started.wait(timeout=2)
    finally:
        release.set()


def test_http_queue_accepts_jobs_beyond_worker_concurrency_without_admission_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_build(**_kwargs: object) -> None:
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("resource-limit audit stop")

    try:
        with _http_server(
            tmp_path,
            monkeypatch,
            build_searchable_pdf=blocked_build,
        ) as server:
            queue_payload = _valid_pdf_bytes("queue-pressure-audit")
            submissions = []
            for _ in range(32):
                status, payload = _post_pdf(server, queue_payload)
                assert status == HTTPStatus.ACCEPTED
                submissions.append(payload)
            assert started.wait(timeout=2)

            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
            try:
                conn.request("GET", "/api/jobs")
                response = conn.getresponse()
                summary = json.loads(response.read().decode("utf-8"))
            finally:
                conn.close()

            assert response.status == HTTPStatus.OK
            assert summary["counts"]["running"] == 1
            assert summary["counts"]["queued"] == len(submissions) - 1
    finally:
        release.set()
