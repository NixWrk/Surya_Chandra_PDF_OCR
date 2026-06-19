"""HTTP service and web GUI for PDF-in / PDF-out OCR processing."""

from __future__ import annotations

import json
import hashlib
import queue
import shutil
import sqlite3
import threading
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from uniscan.app import (
    DEFAULT_BASIC_GUI_LANG,
    PDF_MODE_HYBRID,
    SearchablePdfSummary,
    build_searchable_pdf,
    parse_page_numbers,
)

DEFAULT_DELETE_ORIGINAL_TEXT_LAYER = True
UNISCAN_JOB_PROTOCOL_VERSION = "uniscan-ocr-job.v1"
UNISCAN_OCR_WORKER_CONCURRENCY = 1

_DEFAULT_PRIORITY = "normal"
_KNOWN_PRIORITIES = {"interactive", "normal", "batch", "low"}
_PRIORITY_RANK = {"interactive": 0, "normal": 10, "batch": 20, "low": 30}
_DEFAULT_GPU_POLICY = "cuda"
_KNOWN_GPU_POLICIES = {"auto", "cuda"}
_TERMINAL_JOB_STATUSES = {"done", "error", "interrupted", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    delete_original_text_layer: bool
    filename: str
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    input_bytes: int = 0
    result_bytes: int = 0
    input_path: Path | None = None
    completed_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    run_dir: str | None = None
    result_path: Path | None = None
    error: str | None = None
    protocol_version: str = UNISCAN_JOB_PROTOCOL_VERSION
    request_id: str | None = None
    project_id: str | None = None
    service_id: str | None = None
    task_id: str | None = None
    idempotency_key: str | None = None
    priority: str = _DEFAULT_PRIORITY
    gpu_policy: str = _DEFAULT_GPU_POLICY
    estimated_vram_gb: float | None = None
    estimated_pages: int | None = None
    ttl_seconds: int | None = None
    input_sha256: str | None = None
    request_fingerprint: str | None = None


@dataclass(slots=True, frozen=True)
class _ProtocolMetadata:
    protocol_version: str = UNISCAN_JOB_PROTOCOL_VERSION
    request_id: str | None = None
    project_id: str | None = None
    service_id: str | None = None
    task_id: str | None = None
    idempotency_key: str | None = None
    priority: str = _DEFAULT_PRIORITY
    gpu_policy: str = _DEFAULT_GPU_POLICY
    estimated_vram_gb: float | None = None
    estimated_pages: int | None = None
    ttl_seconds: int | None = None


@dataclass(slots=True, frozen=True)
class _QueuedJob:
    job_id: str
    input_path: Path
    mode: str
    pages_raw: str
    lang: str
    strict: bool
    delete_original_text_layer: bool


_ACTIVE_JOB_STATUSES = {"queued", "running"}


class _JobStore:
    def __init__(self, jobs_root: Path):
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.jobs_root / "jobs.sqlite3"
        self._jobs: dict[str, _JobState] = {}
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._init_db()
        self._load_existing_jobs()
        if _env_bool("UNISCAN_JOB_CLEANUP_ON_START", default=False):
            self.cleanup_expired()

    def create(self, job: _JobState) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._write_metadata_locked(job)
            self._upsert_sqlite_locked(job)
            self._append_event_locked(
                job,
                "created",
                "OCR job was created.",
                {"status": job.status, "input_bytes": job.input_bytes},
            )

    def get(self, job_id: str) -> _JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def find_by_idempotency_key(self, idempotency_key: str | None) -> _JobState | None:
        key = (idempotency_key or "").strip()
        if not key:
            return None
        with self._lock:
            matches = [job for job in self._jobs.values() if job.idempotency_key == key]
            if not matches:
                return None
            return sorted(matches, key=lambda job: job.created_at)[0]

    def update(
        self,
        job_id: str,
        updates: dict[str, Any],
        *,
        event: str | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _utc_now()
            self._write_metadata_locked(job)
            self._upsert_sqlite_locked(job)
            if event:
                self._append_event_locked(job, event, message or event, metadata or updates)
            return job

    def cancel(self, job_id: str) -> tuple[_JobState | None, bool, str | None]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, False, f"Job not found: {job_id}"
            if job.status == "queued":
                now = _utc_now()
                job.status = "cancelled"
                job.progress = 100
                job.message = "Cancelled"
                job.finished_at = now
                job.heartbeat_at = now
                job.updated_at = now
                self._write_metadata_locked(job)
                self._upsert_sqlite_locked(job)
                self._append_event_locked(
                    job,
                    "cancelled",
                    "OCR job was cancelled before processing started.",
                    {"status": job.status},
                )
                return job, True, None
            if job.status == "running":
                return job, False, "Running OCR jobs cannot be cancelled safely yet."
            return job, False, f"Job is already terminal: {job.status}"

    def metadata(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            payload = _serialize_job(job)
            payload["metadata_path"] = str(self._metadata_path(job.job_id))
            payload["events_path"] = str(self._events_path(job.job_id))
            if job.input_path is not None:
                payload["input_path"] = str(job.input_path)
            if job.result_path is not None:
                payload["result_path"] = str(job.result_path)
            return payload

    def health(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return {
                "ok": True,
                "root": str(self.jobs_root),
                "jobs": len(self._jobs),
                "counts": counts,
                "durable": True,
                "sqlite_path": str(self.db_path),
                "worker_concurrency": UNISCAN_OCR_WORKER_CONCURRENCY,
            }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at)
            for job in jobs:
                counts[job.status] = counts.get(job.status, 0) + 1
            active_jobs = [job for job in jobs if job.status in _ACTIVE_JOB_STATUSES]
            recent_jobs = jobs[-20:]
            return {
                "ok": True,
                "protocol_version": UNISCAN_JOB_PROTOCOL_VERSION,
                "root": str(self.jobs_root),
                "sqlite_path": str(self.db_path),
                "jobs": len(jobs),
                "counts": counts,
                "worker_concurrency": UNISCAN_OCR_WORKER_CONCURRENCY,
                "active_jobs": [_serialize_job(job) for job in active_jobs],
                "recent_jobs": [_serialize_job(job) for job in recent_jobs],
            }

    def write_input(self, job_id: str, payload: bytes) -> Path:
        input_path = self._input_path(job_id)
        _atomic_write_bytes(input_path, payload)
        return input_path.resolve()

    def requeueable_jobs(self) -> list[_JobState]:
        with self._lock:
            return [
                job
                for job in sorted(self._jobs.values(), key=_job_queue_sort_key)
                if job.status == "queued"
                and job.input_path is not None
                and Path(job.input_path).exists()
            ]

    def cleanup_expired(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        removed: list[str] = []
        skipped_active: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status in _ACTIVE_JOB_STATUSES:
                    skipped_active.append(job_id)
                    continue
                if job.status not in _TERMINAL_JOB_STATUSES:
                    continue
                if not _job_expired(job, now=current):
                    continue
                job_dir = self._job_dir(job_id)
                if _safe_remove_job_dir(root=self.jobs_root, target=job_dir):
                    removed.append(job_id)
                    self._jobs.pop(job_id, None)
                    self._delete_sqlite_locked(job_id)
            return {
                "ok": True,
                "removed": removed,
                "removed_count": len(removed),
                "skipped_active": skipped_active,
            }

    def _load_existing_jobs(self) -> None:
        for metadata_path in sorted(self.jobs_root.glob("*/metadata.json")):
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                job = _job_from_metadata(raw)
            except Exception:
                continue

            result_candidate = metadata_path.parent / "result.pdf"
            input_candidate = metadata_path.parent / "input.pdf"
            if job.input_path is None and input_candidate.exists():
                job.input_path = input_candidate.resolve()
            if job.result_path is None and result_candidate.exists():
                job.result_path = result_candidate.resolve()
            if job.result_path is not None and Path(job.result_path).exists():
                job.result_path = Path(job.result_path).resolve()
                job.result_bytes = Path(job.result_path).stat().st_size
                if job.status in _ACTIVE_JOB_STATUSES:
                    job.status = "done"
                    job.progress = 100
                    job.message = "Recovered completed OCR result."
                    job.completed_at = job.completed_at or _utc_now()
                    job.finished_at = job.finished_at or job.completed_at
                    job.heartbeat_at = job.heartbeat_at or job.completed_at
            elif job.status == "queued" and job.input_path is not None and Path(job.input_path).exists():
                job.progress = 0
                job.message = "Recovered queued OCR job."
                job.started_at = None
                job.finished_at = None
                job.heartbeat_at = None
            elif job.status in _ACTIVE_JOB_STATUSES:
                interrupted_at = _utc_now()
                job.status = "interrupted"
                job.message = "OCR API restarted before this job finished."
                job.error = job.error or "Interrupted by OCR API restart."
                job.finished_at = job.finished_at or interrupted_at
                job.heartbeat_at = job.heartbeat_at or interrupted_at
            job.updated_at = _utc_now()
            self._jobs[job.job_id] = job
            self._write_metadata_locked(job)
            self._upsert_sqlite_locked(job)

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _input_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "input.pdf"

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "metadata.json"

    def _events_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "events.jsonl"

    def _write_metadata_locked(self, job: _JobState) -> None:
        job_dir = self._job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self._metadata_path(job.job_id), _serialize_job(job))

    def _init_db(self) -> None:
        with self._connect_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    idempotency_key TEXT,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs(idempotency_key)"
            )

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _upsert_sqlite_locked(self, job: _JobState) -> None:
        payload = json.dumps(_serialize_job(job), ensure_ascii=False, sort_keys=True)
        with self._db_lock, self._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, idempotency_key, priority,
                    created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    idempotency_key=excluded.idempotency_key,
                    priority=excluded.priority,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    job.job_id,
                    job.status,
                    job.idempotency_key,
                    job.priority,
                    job.created_at,
                    job.updated_at,
                    payload,
                ),
            )

    def _delete_sqlite_locked(self, job_id: str) -> None:
        with self._db_lock, self._connect_db() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def _append_event_locked(
        self,
        job: _JobState,
        event: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        job_dir = self._job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": _utc_now(),
            "job_id": job.job_id,
            "event": event,
            "message": message,
            "status": job.status,
            "progress": job.progress,
            "metadata": _json_safe(metadata or {}),
        }
        with self._events_path(job.job_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _serialize_job(job: _JobState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": int(max(0, min(100, job.progress))),
        "message": job.message,
        "mode": job.mode,
        "pages": job.pages,
        "lang": job.lang,
        "strict": job.strict,
        "delete_original_text_layer": job.delete_original_text_layer,
        "filename": job.filename,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "input_bytes": job.input_bytes,
        "result_bytes": job.result_bytes,
        "protocol_version": job.protocol_version,
        "priority": job.priority,
        "gpu_policy": job.gpu_policy,
    }
    if job.completed_at:
        payload["completed_at"] = job.completed_at
    if job.started_at:
        payload["started_at"] = job.started_at
    if job.finished_at:
        payload["finished_at"] = job.finished_at
    if job.heartbeat_at:
        payload["heartbeat_at"] = job.heartbeat_at
    if job.run_dir:
        payload["run_dir"] = job.run_dir
    if job.error:
        payload["error"] = job.error
    if job.input_path is not None:
        payload["input_path"] = str(job.input_path)
    if job.request_id:
        payload["request_id"] = job.request_id
    if job.project_id:
        payload["project_id"] = job.project_id
    if job.service_id:
        payload["service_id"] = job.service_id
    if job.task_id:
        payload["task_id"] = job.task_id
    if job.idempotency_key:
        payload["idempotency_key"] = job.idempotency_key
    if job.estimated_vram_gb is not None:
        payload["estimated_vram_gb"] = job.estimated_vram_gb
    if job.estimated_pages is not None:
        payload["estimated_pages"] = job.estimated_pages
    if job.ttl_seconds is not None:
        payload["ttl_seconds"] = job.ttl_seconds
    if job.input_sha256:
        payload["input_sha256"] = job.input_sha256
    if job.request_fingerprint:
        payload["request_fingerprint"] = job.request_fingerprint
    if job.result_path is not None:
        payload["result_path"] = str(job.result_path)
    if job.result_path is not None and job.result_path.exists() and job.status == "done":
        payload["result_url"] = f"/api/jobs/{job.job_id}/result"
        payload["metadata_url"] = f"/api/jobs/{job.job_id}/metadata"
    return payload


def _job_from_metadata(payload: dict[str, Any]) -> _JobState:
    result_path_raw = payload.get("result_path")
    input_path_raw = payload.get("input_path")
    return _JobState(
        job_id=str(payload["job_id"]),
        status=str(payload.get("status") or "interrupted"),
        progress=int(payload.get("progress") or 0),
        message=str(payload.get("message") or ""),
        mode=str(payload.get("mode") or PDF_MODE_HYBRID),
        pages=str(payload.get("pages") or ""),
        lang=str(payload.get("lang") or DEFAULT_BASIC_GUI_LANG),
        strict=bool(payload.get("strict", True)),
        delete_original_text_layer=bool(payload.get("delete_original_text_layer", True)),
        filename=str(payload.get("filename") or "document.pdf"),
        created_at=str(payload.get("created_at") or _utc_now()),
        updated_at=str(payload.get("updated_at") or _utc_now()),
        input_bytes=int(payload.get("input_bytes") or 0),
        result_bytes=int(payload.get("result_bytes") or 0),
        completed_at=(
            str(payload["completed_at"])
            if payload.get("completed_at") is not None
            else None
        ),
        started_at=str(payload["started_at"]) if payload.get("started_at") else None,
        finished_at=str(payload["finished_at"]) if payload.get("finished_at") else None,
        heartbeat_at=str(payload["heartbeat_at"]) if payload.get("heartbeat_at") else None,
        run_dir=str(payload["run_dir"]) if payload.get("run_dir") else None,
        input_path=Path(str(input_path_raw)) if input_path_raw else None,
        result_path=Path(str(result_path_raw)) if result_path_raw else None,
        error=str(payload["error"]) if payload.get("error") else None,
        protocol_version=str(payload.get("protocol_version") or UNISCAN_JOB_PROTOCOL_VERSION),
        request_id=str(payload["request_id"]) if payload.get("request_id") else None,
        project_id=str(payload["project_id"]) if payload.get("project_id") else None,
        service_id=str(payload["service_id"]) if payload.get("service_id") else None,
        task_id=str(payload["task_id"]) if payload.get("task_id") else None,
        idempotency_key=str(payload["idempotency_key"]) if payload.get("idempotency_key") else None,
        priority=str(payload.get("priority") or _DEFAULT_PRIORITY),
        gpu_policy=str(payload.get("gpu_policy") or _DEFAULT_GPU_POLICY),
        estimated_vram_gb=(
            float(payload["estimated_vram_gb"])
            if payload.get("estimated_vram_gb") is not None
            else None
        ),
        estimated_pages=(
            int(payload["estimated_pages"]) if payload.get("estimated_pages") is not None else None
        ),
        ttl_seconds=int(payload["ttl_seconds"]) if payload.get("ttl_seconds") is not None else None,
        input_sha256=str(payload["input_sha256"]) if payload.get("input_sha256") else None,
        request_fingerprint=(
            str(payload["request_fingerprint"]) if payload.get("request_fingerprint") else None
        ),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _query_bool(raw, default=default)


def _env_int(name: str, *, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _priority_rank(priority: str | None) -> int:
    return _PRIORITY_RANK.get((priority or _DEFAULT_PRIORITY).strip().lower(), 10)


def _job_queue_sort_key(job: _JobState) -> tuple[int, str, str]:
    return (_priority_rank(job.priority), job.created_at, job.job_id)


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _job_expired(job: _JobState, *, now: datetime) -> bool:
    if job.status not in _TERMINAL_JOB_STATUSES:
        return False
    reference = (
        _parse_datetime(job.finished_at)
        or _parse_datetime(job.completed_at)
        or _parse_datetime(job.updated_at)
        or _parse_datetime(job.created_at)
    )
    if reference is None:
        return False
    if job.ttl_seconds is not None:
        return now >= reference + timedelta(seconds=max(0, int(job.ttl_seconds)))
    if job.status == "done":
        days = _env_int("UNISCAN_JOB_RETENTION_DAYS", default=30)
    else:
        days = _env_int("UNISCAN_FAILED_JOB_RETENTION_DAYS", default=90)
    return now >= reference + timedelta(days=max(0, days))


def _safe_remove_job_dir(*, root: Path, target: Path) -> bool:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        return False
    if not resolved_target.exists():
        return True
    shutil.rmtree(resolved_target)
    return True


def _first_query_value(query: dict[str, list[str]], *names: str) -> str | None:
    for name in names:
        values = query.get(name)
        if not values:
            continue
        value = str(values[0] or "").strip()
        if value:
            return value
    return None


def _first_header_value(headers: Any, *names: str) -> str | None:
    for name in names:
        value = headers.get(name) if headers is not None else None
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return None


def _first_request_value(
    *,
    query: dict[str, list[str]],
    headers: Any,
    query_names: tuple[str, ...],
    header_names: tuple[str, ...],
) -> str | None:
    return _first_header_value(headers, *header_names) or _first_query_value(query, *query_names)


def _clean_protocol_text(
    raw: str | None,
    *,
    field: str,
    max_length: int = 256,
) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if len(value) > max_length:
        raise ValueError(f"{field} is too long; maximum is {max_length} characters.")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} must not contain control characters.")
    return value


def _parse_priority(raw: str | None) -> str:
    value = (raw or _DEFAULT_PRIORITY).strip().lower()
    aliases = {"high": "interactive", "background": "batch"}
    value = aliases.get(value, value)
    if value not in _KNOWN_PRIORITIES:
        known = ", ".join(sorted(_KNOWN_PRIORITIES))
        raise ValueError(f"priority must be one of: {known}.")
    return value


def _parse_gpu_policy(raw: str | None) -> str:
    value = (raw or _DEFAULT_GPU_POLICY).strip().lower()
    aliases = {"gpu": "cuda"}
    value = aliases.get(value, value)
    if value not in _KNOWN_GPU_POLICIES:
        known = ", ".join(sorted(_KNOWN_GPU_POLICIES))
        raise ValueError(f"gpu_policy must be one of: {known}.")
    return value


def _parse_optional_float(raw: str | None, *, field: str) -> float | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative.")
    return parsed


def _parse_optional_int(raw: str | None, *, field: str) -> int | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative.")
    return parsed


def _parse_protocol_metadata(parsed, headers: Any) -> _ProtocolMetadata:
    query = parse_qs(parsed.query, keep_blank_values=True)

    protocol_version = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("protocol_version", "schema_version"),
            header_names=("X-UniScan-Protocol", "X-OCR-Protocol-Version"),
        ),
        field="protocol_version",
        max_length=80,
    ) or UNISCAN_JOB_PROTOCOL_VERSION
    project_id = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("project_id", "project"),
            header_names=("X-Project-ID", "X-UniScan-Project-ID"),
        ),
        field="project_id",
        max_length=120,
    )
    service_id = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("service_id", "service"),
            header_names=("X-Service-ID", "X-UniScan-Service-ID"),
        ),
        field="service_id",
        max_length=120,
    )
    task_id = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("task_id", "task", "caller_job_id"),
            header_names=("X-Task-ID", "X-UniScan-Task-ID"),
        ),
        field="task_id",
        max_length=512,
    )
    request_id = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("request_id",),
            header_names=("X-Request-ID", "X-UniScan-Request-ID"),
        ),
        field="request_id",
        max_length=256,
    ) or uuid.uuid4().hex
    idempotency_key = _clean_protocol_text(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("idempotency_key",),
            header_names=("X-Idempotency-Key", "X-UniScan-Idempotency-Key"),
        ),
        field="idempotency_key",
        max_length=512,
    )
    priority = _parse_priority(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("priority",),
            header_names=("X-Priority", "X-UniScan-Priority"),
        )
    )
    gpu_policy = _parse_gpu_policy(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("gpu_policy", "gpu"),
            header_names=("X-GPU-Policy", "X-UniScan-GPU-Policy"),
        )
    )
    estimated_vram_gb = _parse_optional_float(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("estimated_vram_gb", "vram_gb"),
            header_names=("X-Estimated-VRAM-GB", "X-UniScan-Estimated-VRAM-GB"),
        ),
        field="estimated_vram_gb",
    )
    estimated_pages = _parse_optional_int(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("estimated_pages",),
            header_names=("X-Estimated-Pages", "X-UniScan-Estimated-Pages"),
        ),
        field="estimated_pages",
    )
    ttl_seconds = _parse_optional_int(
        _first_request_value(
            query=query,
            headers=headers,
            query_names=("ttl_seconds",),
            header_names=("X-TTL-Seconds", "X-UniScan-TTL-Seconds"),
        ),
        field="ttl_seconds",
    )

    return _ProtocolMetadata(
        protocol_version=protocol_version,
        request_id=request_id,
        project_id=project_id,
        service_id=service_id,
        task_id=task_id,
        idempotency_key=idempotency_key,
        priority=priority,
        gpu_policy=gpu_policy,
        estimated_vram_gb=estimated_vram_gb,
        estimated_pages=estimated_pages,
        ttl_seconds=ttl_seconds,
    )


def _request_fingerprint(
    *,
    input_sha256: str,
    mode: str,
    pages: str,
    lang: str,
    strict: bool,
    delete_original_text_layer: bool,
) -> str:
    payload = {
        "delete_original_text_layer": bool(delete_original_text_layer),
        "input_sha256": input_sha256,
        "lang": lang,
        "mode": mode,
        "pages": pages,
        "strict": bool(strict),
    }
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _query_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_job_request(parsed, *, default_lang: str) -> tuple[str, str, str, bool, str, bool]:
    query = parse_qs(parsed.query, keep_blank_values=True)
    mode = (query.get("mode", [PDF_MODE_HYBRID])[0] or PDF_MODE_HYBRID).strip()
    pages_raw = (query.get("pages", [""])[0] or "").strip()
    lang = (query.get("lang", [default_lang])[0] or default_lang).strip()
    strict = _query_bool(query.get("strict", ["1"])[0], default=True)
    delete_original_text_layer = _query_bool(
        (
            query.get("delete_text_layer")
            or query.get("delete_original_text_layer")
            or ["1" if DEFAULT_DELETE_ORIGINAL_TEXT_LAYER else "0"]
        )[0],
        default=DEFAULT_DELETE_ORIGINAL_TEXT_LAYER,
    )
    if not delete_original_text_layer:
        raise ValueError(
            "delete_text_layer cannot be disabled; OCR always removes the existing text layer."
        )
    filename = (query.get("filename", ["document.pdf"])[0] or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return mode, pages_raw, lang, strict, filename, delete_original_text_layer


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
      <div class="sub">PDF in -> searchable PDF out. Default: <b>chandra+surya</b>.</div>
    </div>
    <div class="grid">
      <div class="field full">
        <label>PDF file</label>
        <input id="pdfFile" type="file" accept=".pdf,application/pdf">
      </div>
      <div class="field">
        <label>Mode</label>
        <select id="mode">
          <option value="chandra+surya" selected>chandra+surya (default)</option>
          <option value="chandra">chandra</option>
          <option value="surya">surya</option>
        </select>
      </div>
      <div class="field">
        <label>OCR language</label>
        <input id="lang" value="rus+eng">
      </div>
      <div class="field">
        <label>Pages (optional)</label>
        <input id="pages" placeholder="1,3,5-8">
      </div>
      <div class="field">
        <label>Strict</label>
        <select id="strict">
          <option value="1" selected>true</option>
          <option value="0">false</option>
        </select>
      </div>
      <div class="field">
        <label>Existing text layer</label>
        <select id="deleteTextLayer" disabled>
          <option value="1" selected>remove before OCR</option>
        </select>
      </div>
    </div>
    <div class="actions">
      <button id="runBtn">Run OCR</button>
      <button id="downloadBtn" class="secondary" disabled>Download result</button>
      <span id="jobId"></span>
    </div>
    <div class="status">
      <progress id="bar" max="100" value="0"></progress>
      <div id="line" class="line">Ready.</div>
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
    const deleteTextLayerEl = document.getElementById("deleteTextLayer");
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
          setLine(`Status error: ${data.error || res.statusText}`, true);
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
        setLine("Lost connection to the server: " + err, true);
        setRunning(false);
      }
    };

    runBtn.addEventListener("click", async () => {
      stopPolling();
      const file = fileEl.files[0];
      if (!file) {
        setLine("Choose a PDF file first.", true);
        return;
      }
      setRunning(true);
      barEl.value = 0;
      downloadBtn.disabled = true;
      lastResultUrl = null;
      lastFilename = file.name.replace(/\\.pdf$/i, "") + ".searchable.pdf";
      setLine("Uploading file...");
      const params = new URLSearchParams({
        mode: modeEl.value,
        lang: langEl.value.trim() || "rus+eng",
        strict: strictEl.value,
        delete_text_layer: deleteTextLayerEl.value,
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
        setLine("Job created, queued or running...");
        pollTimer = setTimeout(pollJob, 300);
      } catch (err) {
        setLine("Request error: " + err, true);
        setRunning(false);
      }
    });

    downloadBtn.addEventListener("click", async () => {
      if (!lastResultUrl) return;
      try {
        const res = await fetch(lastResultUrl);
        if (!res.ok) {
          const txt = await res.text();
          setLine(`Download error: ${txt}`, true);
          return;
        }
        const blob = await res.blob();
        const href = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = href;
        a.download = lastFilename;
        a.click();
        URL.revokeObjectURL(href);
        setLine("Result downloaded.");
      } catch (err) {
        setLine("Download error: " + err, true);
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

    job_store = _JobStore(jobs_root)
    job_queue: queue.PriorityQueue[tuple[int, str, str, _QueuedJob]] = queue.PriorityQueue()

    def _queue_job(job: _JobState) -> None:
        if job.input_path is None:
            job_store.update(
                job.job_id,
                {
                    "status": "interrupted",
                    "progress": 100,
                    "message": "Failed",
                    "error": "Queued OCR job has no durable input PDF.",
                    "finished_at": _utc_now(),
                    "heartbeat_at": _utc_now(),
                },
                event="interrupted",
                message="Queued OCR job has no durable input PDF.",
            )
            return
        queued = _QueuedJob(
            job_id=job.job_id,
            input_path=Path(job.input_path),
            mode=job.mode,
            pages_raw=job.pages,
            lang=job.lang,
            strict=job.strict,
            delete_original_text_layer=job.delete_original_text_layer,
        )
        priority_rank, created_at, job_id = _job_queue_sort_key(job)
        job_queue.put((priority_rank, created_at, job_id, queued))

    def _run_job(
        job_id: str,
        *,
        input_path: Path,
        mode: str,
        pages_raw: str,
        lang: str,
        strict: bool,
        delete_original_text_layer: bool,
    ) -> None:
        def _set_state(
            *,
            status: str | None = None,
            progress: int | None = None,
            message: str | None = None,
            run_dir: str | None = None,
            result_path: Path | None = None,
            error: str | None = None,
        ) -> None:
            updates: dict[str, Any] = {}
            now = _utc_now()
            current = job_store.get(job_id)
            if status is not None:
                updates["status"] = status
            if progress is not None:
                updates["progress"] = int(max(0, min(100, progress)))
            if message is not None:
                updates["message"] = message
            if run_dir is not None:
                updates["run_dir"] = run_dir
            if result_path is not None:
                updates["result_path"] = result_path
                updates["result_bytes"] = result_path.stat().st_size if result_path.exists() else 0
            if error is not None:
                updates["error"] = error
            if status == "running":
                updates["heartbeat_at"] = now
                if current is not None and current.started_at is None:
                    updates["started_at"] = now
            if status == "done":
                updates["completed_at"] = now
            if status in {"done", "error", "interrupted", "cancelled"}:
                updates["finished_at"] = now
                updates["heartbeat_at"] = now
            event = status or "progress"
            job_store.update(
                job_id,
                updates,
                event=event,
                message=message or event,
                metadata=updates,
            )

        def _progress_cb(value: int, status: str) -> None:
            _set_state(status="running", progress=value, message=status)

        _set_state(status="running", progress=1, message="Starting")
        try:
            page_numbers = parse_page_numbers(pages_raw)
            summary: SearchablePdfSummary = build_searchable_pdf(
                pdf_path=input_path,
                mode=mode,
                lang=lang,
                page_numbers=page_numbers,
                work_root=pipeline_root,
                overwrite_input_path=False,
                return_bytes=False,
                strict=strict,
                progress=_progress_cb,
                delete_original_text_layer=delete_original_text_layer,
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

    def _worker_loop() -> None:
        while True:
            _, _, _, queued_job = job_queue.get()
            try:
                current = job_store.get(queued_job.job_id)
                if current is None or current.status != "queued":
                    continue
                _run_job(
                    queued_job.job_id,
                    input_path=queued_job.input_path,
                    mode=queued_job.mode,
                    pages_raw=queued_job.pages_raw,
                    lang=queued_job.lang,
                    strict=queued_job.strict,
                    delete_original_text_layer=queued_job.delete_original_text_layer,
                )
            finally:
                job_queue.task_done()

    worker = threading.Thread(
        target=_worker_loop,
        name="uniscan-ocr-job-worker",
        daemon=True,
    )
    worker.start()

    for recovered_job in job_store.requeueable_jobs():
        _queue_job(recovered_job)

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

        def _send_pdf_file(self, *, path: Path, filename: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)

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

                mode, pages_raw, lang, strict, _filename, delete_original_text_layer = _parse_job_request(
                    parsed,
                    default_lang=default_lang,
                )
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
                    delete_original_text_layer=delete_original_text_layer,
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
                mode, pages_raw, lang, strict, filename, delete_original_text_layer = _parse_job_request(
                    parsed,
                    default_lang=default_lang,
                )
                parse_page_numbers(pages_raw)
                protocol = _parse_protocol_metadata(parsed, self.headers)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            input_sha256 = hashlib.sha256(payload).hexdigest()
            request_fingerprint = _request_fingerprint(
                input_sha256=input_sha256,
                mode=mode,
                pages=pages_raw,
                lang=lang,
                strict=strict,
                delete_original_text_layer=delete_original_text_layer,
            )
            existing = job_store.find_by_idempotency_key(protocol.idempotency_key)
            if existing is not None:
                existing_payload = _serialize_job(existing)
                existing_payload["idempotent_replay"] = True
                has_fingerprint_conflict = (
                    existing.request_fingerprint is not None
                    and existing.request_fingerprint != request_fingerprint
                )
                has_input_conflict = (
                    existing.request_fingerprint is None
                    and existing.input_sha256 is not None
                    and existing.input_sha256 != input_sha256
                )
                if has_fingerprint_conflict or has_input_conflict:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {
                            "error": (
                                "Idempotency key already belongs to a different OCR request."
                            ),
                            "job": existing_payload,
                        },
                    )
                    return
                self._send_json(HTTPStatus.OK, existing_payload)
                return

            job_id = uuid.uuid4().hex[:12]
            try:
                input_path = job_store.write_input(job_id, payload)
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Failed to persist input PDF before accepting job: {exc}"},
                )
                return
            job = _JobState(
                job_id=job_id,
                status="queued",
                progress=0,
                message="Queued",
                mode=mode,
                pages=pages_raw,
                lang=lang,
                strict=bool(strict),
                delete_original_text_layer=bool(delete_original_text_layer),
                filename=filename,
                input_bytes=len(payload),
                input_path=input_path,
                protocol_version=protocol.protocol_version,
                request_id=protocol.request_id,
                project_id=protocol.project_id,
                service_id=protocol.service_id,
                task_id=protocol.task_id,
                idempotency_key=protocol.idempotency_key,
                priority=protocol.priority,
                gpu_policy=protocol.gpu_policy,
                estimated_vram_gb=protocol.estimated_vram_gb,
                estimated_pages=protocol.estimated_pages,
                ttl_seconds=protocol.ttl_seconds,
                input_sha256=input_sha256,
                request_fingerprint=request_fingerprint,
            )
            try:
                job_store.create(job)
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Failed to persist OCR job metadata: {exc}"},
                )
                return

            _queue_job(job)
            self._send_json(HTTPStatus.ACCEPTED, _serialize_job(job))

        def _handle_get_job(self, job_id: str) -> None:
            job = job_store.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            self._send_json(HTTPStatus.OK, _serialize_job(job))

        def _handle_get_jobs_summary(self) -> None:
            self._send_json(HTTPStatus.OK, job_store.summary())

        def _handle_get_job_metadata(self, job_id: str) -> None:
            metadata = job_store.metadata(job_id)
            if metadata is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            self._send_json(HTTPStatus.OK, metadata)

        def _handle_get_job_result(self, job_id: str) -> None:
            job = job_store.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            if job.status != "done" or job.result_path is None or not job.result_path.exists():
                self._send_json(HTTPStatus.CONFLICT, {"error": "Result is not ready yet."})
                return
            filename = job.filename
            safe_name = filename[:-4] if filename.lower().endswith(".pdf") else filename
            download_name = f"{safe_name}.searchable.pdf"
            self._send_pdf_file(path=job.result_path, filename=download_name)

        def _handle_cancel_job(self, job_id: str) -> None:
            job, cancelled, error = job_store.cancel(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": error or f"Job not found: {job_id}"})
                return
            payload = _serialize_job(job)
            if cancelled:
                self._send_json(HTTPStatus.OK, payload)
                return
            self._send_json(HTTPStatus.CONFLICT, {"error": error or "Job cannot be cancelled.", "job": payload})

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
                        "job_protocol_version": UNISCAN_JOB_PROTOCOL_VERSION,
                        "ocr_worker_concurrency": UNISCAN_OCR_WORKER_CONCURRENCY,
                        "job_store": job_store.health(),
                    },
                )
                return

            parts = [part for part in parsed.path.split("/") if part]
            if parsed.path in {"/api/jobs", "/api/queue"}:
                self._handle_get_jobs_summary()
                return
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "jobs":
                self._handle_get_job(parts[2])
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "result":
                self._handle_get_job_result(parts[2])
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "metadata":
                self._handle_get_job_metadata(parts[2])
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
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "cancel":
                self._handle_cancel_job(parts[2])
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
    print(
        "Async API: POST /api/jobs, GET /api/jobs, "
        "GET /api/jobs/{job_id}, GET /api/jobs/{job_id}/result"
    )
    print("Sync API: POST /searchable-pdf")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
