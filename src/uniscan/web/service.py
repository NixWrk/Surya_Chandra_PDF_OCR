"""Minimal HTTP service for PDF-in / PDF-out OCR processing."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from uniscan.app import (
    DEFAULT_BASIC_GUI_LANG,
    PDF_MODE_CHANDRA,
    PDF_MODE_HYBRID,
    PDF_MODE_SURYA,
    build_searchable_pdf,
    parse_page_numbers,
)


def _query_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _html_ui() -> bytes:
    return (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>UniScan OCR API</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 2rem; max-width: 920px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    label { display: block; margin-top: 0.6rem; font-weight: 600; }
    input, select, button { font-size: 1rem; padding: 0.45rem; margin-top: 0.2rem; }
    button { cursor: pointer; }
    code { background: #f4f4f4; padding: 0.15rem 0.3rem; border-radius: 4px; }
    pre { background: #f7f7f7; padding: 0.8rem; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>UniScan OCR API</h1>
  <p>POST raw PDF bytes to <code>/searchable-pdf</code> and get searchable PDF bytes back.</p>
  <div class="card">
    <h2>Quick Form</h2>
    <label>PDF file</label>
    <input id="pdfFile" type="file" accept=".pdf,application/pdf">
    <label>Mode</label>
    <select id="mode">
      <option value="chandra+surya" selected>chandra+surya (default)</option>
      <option value="chandra">chandra</option>
      <option value="surya">surya</option>
    </select>
    <label>Pages (optional)</label>
    <input id="pages" placeholder="1,3,5-8">
    <div style="margin-top:1rem;">
      <button id="runBtn">Process PDF</button>
    </div>
    <pre id="log">Ready.</pre>
  </div>
  <script>
    const log = (text) => { document.getElementById("log").textContent = text; };
    document.getElementById("runBtn").addEventListener("click", async () => {
      const fileInput = document.getElementById("pdfFile");
      const file = fileInput.files[0];
      if (!file) {
        log("Select a PDF file first.");
        return;
      }
      const mode = document.getElementById("mode").value;
      const pages = document.getElementById("pages").value.trim();
      const params = new URLSearchParams({ mode });
      if (pages) params.set("pages", pages);
      log("Processing...");
      try {
        const payload = await file.arrayBuffer();
        const res = await fetch(`/searchable-pdf?${params.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "application/pdf" },
          body: payload
        });
        if (!res.ok) {
          const errorText = await res.text();
          log(`HTTP ${res.status}: ${errorText}`);
          return;
        }
        const blob = await res.blob();
        const href = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = href;
        a.download = file.name.replace(/\\.pdf$/i, "") + ".searchable.pdf";
        a.click();
        URL.revokeObjectURL(href);
        log("Done. Download started.");
      } catch (error) {
        log("Failed: " + error);
      }
    });
  </script>
</body>
</html>
"""
    ).encode("utf-8")


def _build_handler(*, work_root: Path, default_lang: str):
    class SearchablePdfApiHandler(BaseHTTPRequestHandler):
        server_version = "UniScanHTTP/0.1"

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
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/searchable-pdf":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            try:
                payload = self._read_request_body()
                if not payload:
                    raise ValueError("Request body is empty. Send raw PDF bytes.")

                query = parse_qs(parsed.query, keep_blank_values=True)
                mode = (query.get("mode", [PDF_MODE_HYBRID])[0] or PDF_MODE_HYBRID).strip()
                pages_raw = (query.get("pages", [""])[0] or "").strip()
                lang = (query.get("lang", [default_lang])[0] or default_lang).strip()
                strict = _query_bool(query.get("strict", ["1"])[0], default=True)

                page_numbers = parse_page_numbers(pages_raw)
                summary = build_searchable_pdf(
                    pdf_bytes=payload,
                    mode=mode,
                    lang=lang,
                    page_numbers=page_numbers,
                    work_root=work_root,
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
    print("Endpoints: GET /health, POST /searchable-pdf (raw PDF bytes), GET /")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
