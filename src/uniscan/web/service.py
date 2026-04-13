"""HTTP service and web GUI for PDF-in / PDF-out OCR processing."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from uniscan.app import (
    DEFAULT_BASIC_GUI_LANG,
    PDF_MODE_HYBRID,
    SearchablePdfSummary,
    build_searchable_pdf,
    parse_page_numbers,
)


@dataclass(slots=True)
class _JobState:
    job_id: str
    status: str
    progress: int
    message: str
    mode: str
    pages: str
    lang: str
    strict: bool
    filename: str
    run_dir: str | None = None
    result_path: Path | None = None
    error: str | None = None


def _query_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_job_request(parsed, *, default_lang: str) -> tuple[str, str, str, bool, str]:
    query = parse_qs(parsed.query, keep_blank_values=True)
    mode = (query.get("mode", [PDF_MODE_HYBRID])[0] or PDF_MODE_HYBRID).strip()
    pages_raw = (query.get("pages", [""])[0] or "").strip()
    lang = (query.get("lang", [default_lang])[0] or default_lang).strip()
    strict = _query_bool(query.get("strict", ["1"])[0], default=True)
    filename = (query.get("filename", ["document.pdf"])[0] or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return mode, pages_raw, lang, strict, filename


def _html_ui() -> bytes:
    return (
        """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>UniScan Web GUI</title>
  <style>
    :root {
      --bg: #f1efe8;
      --paper: #fffdf7;
      --ink: #1d2b34;
      --muted: #52626b;
      --line: #d8d4c8;
      --accent: #1d7d6e;
      --accent-ink: #f6fffc;
      --warn: #b94242;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at 15% 10%, #fff8df 0%, var(--bg) 48%, #e9e4d6 100%);
      color: var(--ink);
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .wrap {
      max-width: 920px;
      margin: 0 auto;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 12px 28px rgba(32, 40, 48, 0.12);
      overflow: hidden;
    }
    .head {
      padding: 1.2rem 1.5rem;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #e6f3ef, #f3efe3 72%);
    }
    h1 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: 0.01em;
    }
    .sub {
      margin-top: 0.3rem;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      padding: 1.3rem 1.5rem;
    }
    .field label {
      display: block;
      margin-bottom: 0.35rem;
      font-weight: 600;
      font-size: 0.92rem;
    }
    .field input, .field select {
      width: 100%;
      padding: 0.55rem 0.7rem;
      border: 1px solid #c8c3b4;
      border-radius: 10px;
      background: #fff;
      font-size: 0.95rem;
    }
    .field.full { grid-column: 1 / -1; }
    .actions {
      padding: 0 1.5rem 1rem;
      display: flex;
      gap: 0.8rem;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      border: none;
      border-radius: 10px;
      padding: 0.65rem 1rem;
      font-size: 0.95rem;
      cursor: pointer;
      background: var(--accent);
      color: var(--accent-ink);
      font-weight: 600;
      transition: transform 0.06s ease, opacity 0.15s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    button.secondary { background: #304750; }
    .status {
      padding: 0 1.5rem 1.2rem;
    }
    progress {
      width: 100%;
      height: 18px;
      border-radius: 99px;
      overflow: hidden;
    }
    .line {
      margin-top: 0.55rem;
      font-size: 0.94rem;
      color: var(--muted);
      min-height: 1.2rem;
    }
    .line.error { color: var(--warn); }
    .foot {
      border-top: 1px solid var(--line);
      padding: 0.85rem 1.5rem 1rem;
      color: var(--muted);
      font-size: 0.85rem;
    }
    code {
      background: #ece8db;
      border-radius: 6px;
      padding: 0.1rem 0.35rem;
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>UniScan Web GUI</h1>
      <div class="sub">PDF in -> searchable PDF out. По умолчанию: <b>chandra+surya</b>.</div>
    </div>
    <div class="grid">
      <div class="field full">
        <label>PDF файл</label>
        <input id="pdfFile" type="file" accept=".pdf,application/pdf">
      </div>
      <div class="field">
        <label>Режим</label>
        <select id="mode">
          <option value="chandra+surya" selected>chandra+surya (default)</option>
          <option value="chandra">chandra</option>
          <option value="surya">surya</option>
        </select>
      </div>
      <div class="field">
        <label>Язык OCR</label>
        <input id="lang" value="rus+eng">
      </div>
      <div class="field">
        <label>Страницы (опционально)</label>
        <input id="pages" placeholder="1,3,5-8">
      </div>
      <div class="field">
        <label>Strict</label>
        <select id="strict">
          <option value="1" selected>true</option>
          <option value="0">false</option>
        </select>
      </div>
    </div>
    <div class="actions">
      <button id="runBtn">Запустить OCR</button>
      <button id="downloadBtn" class="secondary" disabled>Скачать результат</button>
      <span id="jobId"></span>
    </div>
    <div class="status">
      <progress id="bar" max="100" value="0"></progress>
      <div id="line" class="line">Готово.</div>
    </div>
    <div class="foot">
      API: <code>POST /api/jobs</code>, <code>GET /api/jobs/{id}</code>, <code>GET /api/jobs/{id}/result</code>
    </div>
  </div>
  <script>
    const fileEl = document.getElementById("pdfFile");
    const modeEl = document.getElementById("mode");
    const pagesEl = document.getElementById("pages");
    const langEl = document.getElementById("lang");
    const strictEl = document.getElementById("strict");
    const runBtn = document.getElementById("runBtn");
    const downloadBtn = document.getElementById("downloadBtn");
    const barEl = document.getElementById("bar");
    const lineEl = document.getElementById("line");
    const jobIdEl = document.getElementById("jobId");

    let pollTimer = null;
    let lastJobId = null;
    let lastResultUrl = null;
    let lastFilename = "document.searchable.pdf";

    const setLine = (text, isError=false) => {
      lineEl.textContent = text;
      lineEl.classList.toggle("error", !!isError);
    };

    const stopPolling = () => {
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    };

    const setRunning = (running) => {
      runBtn.disabled = running;
      if (running) {
        downloadBtn.disabled = true;
      }
    };

    const pollJob = async () => {
      if (!lastJobId) return;
      try {
        const res = await fetch(`/api/jobs/${lastJobId}`);
        const data = await res.json();
        if (!res.ok) {
          setLine(`Ошибка статуса: ${data.error || res.statusText}`, true);
          setRunning(false);
          return;
        }
        barEl.value = Number(data.progress || 0);
        const msg = data.error ? `${data.message}: ${data.error}` : data.message;
        setLine(msg || data.status, data.status === "error");
        if (data.status === "done") {
          setRunning(false);
          lastResultUrl = data.result_url;
          downloadBtn.disabled = !lastResultUrl;
          return;
        }
        if (data.status === "error") {
          setRunning(false);
          return;
        }
        pollTimer = setTimeout(pollJob, 900);
      } catch (err) {
        setLine("Потеряна связь с сервером: " + err, true);
        setRunning(false);
      }
    };

    runBtn.addEventListener("click", async () => {
      stopPolling();
      const file = fileEl.files[0];
      if (!file) {
        setLine("Сначала выберите PDF файл.", true);
        return;
      }
      setRunning(true);
      barEl.value = 0;
      downloadBtn.disabled = true;
      lastResultUrl = null;
      lastFilename = file.name.replace(/\\.pdf$/i, "") + ".searchable.pdf";
      setLine("Отправка файла...");
      const params = new URLSearchParams({
        mode: modeEl.value,
        lang: langEl.value.trim() || "rus+eng",
        strict: strictEl.value,
        filename: file.name
      });
      const pages = pagesEl.value.trim();
      if (pages) params.set("pages", pages);
      try {
        const payload = await file.arrayBuffer();
        const res = await fetch(`/api/jobs?${params.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "application/pdf" },
          body: payload
        });
        const data = await res.json();
        if (!res.ok) {
          setLine(data.error || `HTTP ${res.status}`, true);
          setRunning(false);
          return;
        }
        lastJobId = data.job_id;
        jobIdEl.textContent = `job: ${lastJobId}`;
        setLine("Задача создана, OCR выполняется...");
        pollTimer = setTimeout(pollJob, 300);
      } catch (err) {
        setLine("Ошибка запроса: " + err, true);
        setRunning(false);
      }
    });

    downloadBtn.addEventListener("click", async () => {
      if (!lastResultUrl) return;
      try {
        const res = await fetch(lastResultUrl);
        if (!res.ok) {
          const txt = await res.text();
          setLine(`Ошибка скачивания: ${txt}`, true);
          return;
        }
        const blob = await res.blob();
        const href = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = href;
        a.download = lastFilename;
        a.click();
        URL.revokeObjectURL(href);
        setLine("Результат скачан.");
      } catch (err) {
        setLine("Ошибка скачивания: " + err, true);
      }
    });
  </script>
</body>
</html>
"""
    ).encode("utf-8")


def _build_handler(*, work_root: Path, default_lang: str):
    jobs_root = (work_root / "jobs").resolve()
    pipeline_root = (work_root / "runs").resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    pipeline_root.mkdir(parents=True, exist_ok=True)

    jobs: dict[str, _JobState] = {}
    jobs_lock = threading.Lock()

    def _serialize_job(job: _JobState) -> dict[str, object]:
        payload: dict[str, object] = {
            "job_id": job.job_id,
            "status": job.status,
            "progress": int(max(0, min(100, job.progress))),
            "message": job.message,
            "mode": job.mode,
            "pages": job.pages,
            "lang": job.lang,
            "strict": job.strict,
            "filename": job.filename,
        }
        if job.run_dir:
            payload["run_dir"] = job.run_dir
        if job.error:
            payload["error"] = job.error
        if job.result_path is not None and job.result_path.exists() and job.status == "done":
            payload["result_url"] = f"/api/jobs/{job.job_id}/result"
        return payload

    def _run_job(job_id: str, *, payload: bytes, mode: str, pages_raw: str, lang: str, strict: bool) -> None:
        def _set_state(
            *,
            status: str | None = None,
            progress: int | None = None,
            message: str | None = None,
            run_dir: str | None = None,
            result_path: Path | None = None,
            error: str | None = None,
        ) -> None:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is None:
                    return
                if status is not None:
                    job.status = status
                if progress is not None:
                    job.progress = int(max(0, min(100, progress)))
                if message is not None:
                    job.message = message
                if run_dir is not None:
                    job.run_dir = run_dir
                if result_path is not None:
                    job.result_path = result_path
                if error is not None:
                    job.error = error

        def _progress_cb(value: int, status: str) -> None:
            _set_state(status="running", progress=value, message=status)

        _set_state(status="running", progress=1, message="Queued")
        try:
            page_numbers = parse_page_numbers(pages_raw)
            summary: SearchablePdfSummary = build_searchable_pdf(
                pdf_bytes=payload,
                mode=mode,
                lang=lang,
                page_numbers=page_numbers,
                work_root=pipeline_root,
                overwrite_input_path=False,
                return_bytes=False,
                strict=strict,
                progress=_progress_cb,
            )
            result_target = jobs_root / job_id / "result.pdf"
            result_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary.output_pdf_path, result_target)
            _set_state(
                status="done",
                progress=100,
                message="Done",
                run_dir=str(summary.run_dir),
                result_path=result_target.resolve(),
                error=None,
            )
        except Exception as exc:
            _set_state(status="error", progress=100, message="Failed", error=str(exc))

    class SearchablePdfApiHandler(BaseHTTPRequestHandler):
        server_version = "UniScanHTTP/0.2"

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_pdf(self, *, status: int, payload: bytes, filename: str = "searchable.pdf") -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def _read_request_body(self) -> bytes:
            raw_len = self.headers.get("Content-Length", "")
            if not raw_len:
                return b""
            try:
                length = int(raw_len)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def _handle_sync_searchable_pdf(self, parsed) -> None:
            try:
                payload = self._read_request_body()
                if not payload:
                    raise ValueError("Request body is empty. Send raw PDF bytes.")

                mode, pages_raw, lang, strict, _filename = _parse_job_request(parsed, default_lang=default_lang)
                page_numbers = parse_page_numbers(pages_raw)
                summary = build_searchable_pdf(
                    pdf_bytes=payload,
                    mode=mode,
                    lang=lang,
                    page_numbers=page_numbers,
                    work_root=pipeline_root,
                    overwrite_input_path=False,
                    return_bytes=True,
                    strict=strict,
                )
                output_bytes = summary.output_pdf_bytes
                if output_bytes is None:
                    raise RuntimeError("Searchable PDF bytes were not returned by service pipeline.")
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(output_bytes)))
            self.send_header("Content-Disposition", 'attachment; filename="searchable.pdf"')
            self.send_header("X-UniScan-Mode", summary.mode)
            self.send_header("X-UniScan-Run-Dir", str(summary.run_dir))
            self.end_headers()
            self.wfile.write(output_bytes)

        def _handle_create_job(self, parsed) -> None:
            try:
                payload = self._read_request_body()
                if not payload:
                    raise ValueError("Request body is empty. Send raw PDF bytes.")
                mode, pages_raw, lang, strict, filename = _parse_job_request(parsed, default_lang=default_lang)
                parse_page_numbers(pages_raw)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            job_id = uuid.uuid4().hex[:12]
            job = _JobState(
                job_id=job_id,
                status="queued",
                progress=0,
                message="Queued",
                mode=mode,
                pages=pages_raw,
                lang=lang,
                strict=bool(strict),
                filename=filename,
            )
            with jobs_lock:
                jobs[job_id] = job

            worker = threading.Thread(
                target=_run_job,
                kwargs={
                    "job_id": job_id,
                    "payload": payload,
                    "mode": mode,
                    "pages_raw": pages_raw,
                    "lang": lang,
                    "strict": strict,
                },
                daemon=True,
            )
            worker.start()
            self._send_json(HTTPStatus.ACCEPTED, _serialize_job(job))

        def _handle_get_job(self, job_id: str) -> None:
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            self._send_json(HTTPStatus.OK, _serialize_job(job))

        def _handle_get_job_result(self, job_id: str) -> None:
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            if job.status != "done" or job.result_path is None or not job.result_path.exists():
                self._send_json(HTTPStatus.CONFLICT, {"error": "Result is not ready yet."})
                return
            filename = job.filename
            safe_name = filename[:-4] if filename.lower().endswith(".pdf") else filename
            download_name = f"{safe_name}.searchable.pdf"
            self._send_pdf(status=HTTPStatus.OK, payload=job.result_path.read_bytes(), filename=download_name)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"", "/", "/index.html"}:
                self._send_html(HTTPStatus.OK, _html_ui())
                return
            if parsed.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "uniscan",
                        "mode_default": PDF_MODE_HYBRID,
                    },
                )
                return

            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "jobs":
                self._handle_get_job(parts[2])
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "result":
                self._handle_get_job_result(parts[2])
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/searchable-pdf":
                self._handle_sync_searchable_pdf(parsed)
                return
            if parsed.path == "/api/jobs":
                self._handle_create_job(parsed)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def log_message(self, _format: str, *_args: object) -> None:
            # Keep stdout clean in CLI runs; operational logs can be added later.
            return

    return SearchablePdfApiHandler


def run_http_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    work_root: Path | None = None,
    lang: str = DEFAULT_BASIC_GUI_LANG,
) -> None:
    resolved_work_root = Path(work_root) if work_root is not None else (Path.cwd() / "outputs" / "web_runs")
    resolved_work_root.mkdir(parents=True, exist_ok=True)

    handler = _build_handler(work_root=resolved_work_root.resolve(), default_lang=lang)
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"UniScan HTTP API listening on http://{host}:{port}")
    print("GUI: GET /")
    print("Async API: POST /api/jobs, GET /api/jobs/{job_id}, GET /api/jobs/{job_id}/result")
    print("Sync API: POST /searchable-pdf")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
