"""Reusable OCR workflow orchestration for desktop/web frontends."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Iterator
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import threading
import unicodedata
from typing import Any, Callable, Mapping, Sequence
import uuid

from PIL import Image, ImageChops, ImageOps

from uniscan.ocr import (
    ArtifactSearchableResult,
    CompareTxtBuildResult,
    OcrBenchmarkResult,
    build_compare_txt_from_benchmark,
    detect_ocr_engine_status,
    run_artifact_searchable_package,
    run_ocr_benchmark,
)
from uniscan.ocr.artifact_searchable import (
    _bbox_values,
    _bbox_reading_order_indices,
    _clean_overlay_line,
    _dehyphenate_line_breaks,
    _extract_pdf_page_texts,
    _split_page_text_lines,
    _validate_searchable_text_retention,
    _validate_textless_visual_retention,
)
from uniscan.ocr.benchmark import (
    _CHANDRA_ALTERNATIVE_TEXT_POLICY,
    _CHANDRA_MIN_IMAGE_DIM,
    _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
    _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION,
    _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS,
    _canonical_rgb_pixel_sha256,
    _chandra_alternative_text_evidence,
    _chandra_chunk_lines,
    _chandra_expand_chunk_to_line_boxes,
    _chandra_graphic_chunk_ignored_lines,
    _chandra_graphic_chunk_lines,
    _is_effectively_blank_rgb_image,
    _markerized_pages_text,
)
from uniscan.io import iter_render_pdf_page_indices
from uniscan.ocr.pdf_utils import _build_textless_source_pdf


DEFAULT_BASIC_GUI_LANG = "rus+eng"
MODE_SURYA = "surya"
MODE_HYBRID = "hybrid"
MODE_BOTH = "both"

PDF_MODE_CHANDRA = "chandra"
PDF_MODE_SURYA = "surya"
PDF_MODE_HYBRID = "chandra+surya"

MODE_TO_ENGINES: dict[str, tuple[str, ...]] = {
    MODE_SURYA: ("surya",),
    MODE_HYBRID: ("chandra",),
    MODE_BOTH: ("surya", "chandra"),
}

_ENGINE_TO_PYTHON_ENV: dict[str, str] = {
    "surya": "UNISCAN_SURYA_PYTHON",
    "chandra": "UNISCAN_CHANDRA_PYTHON",
}

_ENGINE_SUBPROCESS_ENV_SUFFIXES: tuple[str, ...] = (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_CACHE",
    "MODEL_CACHE_DIR",
    "MODELSCOPE_CACHE",
    "TRANSFORMERS_CACHE",
    "TEMP",
    "TMP",
    "TORCH_DEVICE",
    "UNISCAN_RUNTIME_TMP",
)

ProgressCallback = Callable[[int, str], None]

_DEFAULT_HYBRID_CHUNK_PAGES = 10
_HYBRID_CHUNK_MANIFEST_SCHEMA = "uniscan.hybrid-chunks.v4"
_OCR_STATUS_RECONCILIATION_PENDING = "reconciliation_pending"
_HYBRID_CHUNK_PIPELINE_REVISION = "chandra-surya-resumable-v11"
_SURYA_RETRY_PREPROCESSING = "autocontrast-cutoff-1"
_SURYA_SCALED_RETRY_PREPROCESSING = "rgb-scale-0.5-center-white-lanczos-v1"
_SURYA_SCALED_RETRY_FACTOR = 0.5
_SURYA_SOURCE_COORDINATE_SPACE = "source-image-v1"
_SURYA_SCALED_GEOMETRY_TRANSFORM = "inverse-actual-content-size-strict-v1"
_SURYA_RETRY_POLICY = "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
_CHANDRA_PLAIN_RETRY_PREPROCESSING = "plain-ocr-original-v1"
_CHANDRA_RETRY_POLICY = "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
_CHANDRA_PROMPT_SHA256 = {
    "ocr_layout": "025935f3e1de1acdfadd4c7d581ab17eb82e8caaffef7b64962621c80b7ca9a8",
    "ocr": "5bddc652e7f39d5c5501e5455ea2487ebb5cdbf71832b5dfb749ec2413b5a69b",
}
_CHANDRA_CONTENT_FILTERS = {
    "ocr_layout": "skip-graphic-labels-v1",
    "ocr": "visible-text-tags-exclude-media-description-v1",
}
_CHANDRA_ATTEMPT_EVIDENCE_SCHEMA = "uniscan.chandra-attempt.v2"
_RETRY_TEXT_AGREEMENT_ALGORITHM = "nfkc-layout-whitespace-exact-v1"
_CHUNK_EVIDENCE_MANIFEST_SCHEMA = "uniscan.chunk-evidence.v1"
_SPARSE_SURYA_GRAPHICS_MAX_ALNUM_LINES = 2
_SPARSE_SURYA_GRAPHICS_MAX_ALNUM_CHARS = 24
_MAX_CHUNK_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_OCR_TEXT_ARTIFACT_BYTES = 64 * 1024 * 1024
_FILE_HASH_BLOCK_BYTES = 1024 * 1024
_HYBRID_IDENTITY_ENV_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_CACHE",
    "MODEL_CACHE_DIR",
    "TORCH_DEVICE",
    "TRANSFORMERS_CACHE",
    "UNISCAN_ALIGN_BAND",
    "UNISCAN_CHANDRA_ALLOW_CLI_FALLBACK",
    "UNISCAN_CHANDRA_BLEND_Y_WEIGHT",
    "UNISCAN_CHANDRA_DEVICE_POLICY",
    "UNISCAN_CHANDRA_GEOMETRY_POLICY",
    "UNISCAN_CHANDRA_HF_HOME",
    "UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE",
    "UNISCAN_CHANDRA_HF_HUB_CACHE",
    "UNISCAN_CHANDRA_MODEL_CACHE_DIR",
    "UNISCAN_CHANDRA_MODELSCOPE_CACHE",
    "UNISCAN_CHANDRA_PREFER_GPU",
    "UNISCAN_CHANDRA_PYTHON",
    "UNISCAN_CHANDRA_REQUIRE_GPU",
    "UNISCAN_CHANDRA_REQUIRE_SIDECAR",
    "UNISCAN_CHANDRA_TORCH_DEVICE",
    "UNISCAN_CHANDRA_TRANSFORMERS_CACHE",
    "UNISCAN_ENGINE_SUBPROCESS_TIMEOUT_SECONDS",
    "UNISCAN_GEOMETRY_DEBUG",
    "UNISCAN_GPU_DEVICE_ID",
    "UNISCAN_OCR_RENDER_DPI",
    "UNISCAN_SURYA_ALLOW_TEXT_FALLBACK",
    "UNISCAN_SURYA_HF_HOME",
    "UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE",
    "UNISCAN_SURYA_HF_HUB_CACHE",
    "UNISCAN_SURYA_MODEL_CACHE_DIR",
    "UNISCAN_SURYA_MODELSCOPE_CACHE",
    "UNISCAN_SURYA_PYTHON",
    "UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON",
    "UNISCAN_SURYA_REQUIRE_GPU",
    "UNISCAN_SURYA_TORCH_DEVICE",
    "UNISCAN_SURYA_TRANSFORMERS_CACHE",
    "UNISCAN_TEXT_LAYER_FONT",
    "UNISCAN_TEXTLESS_DPI",
    "UNISCAN_TEXTLESS_JPEG_QUALITY",
)
_HYBRID_RUN_LOCKS: dict[str, threading.Lock] = {}
_HYBRID_RUN_LOCKS_GUARD = threading.Lock()


@dataclass(slots=True, frozen=True)
class BasicOcrRunSummary:
    run_dir: Path
    results: tuple[OcrBenchmarkResult, ...]
    result_files: tuple[Path, ...]
    failed_engines: tuple[str, ...]
    skipped_engines: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ChandraGeometryVariantsSummary:
    run_root: Path
    compare_dir: Path
    output_root: Path
    compare_results: tuple[CompareTxtBuildResult, ...]
    chandra_geometry_results: tuple[ArtifactSearchableResult, ...]
    surya_geometry_results: tuple[ArtifactSearchableResult, ...]


@dataclass(slots=True, frozen=True)
class SearchablePdfSummary:
    mode: str
    run_dir: Path
    compare_dir: Path
    output_pdf_path: Path
    output_pdf_bytes: bytes | None
    overwritten_input_path: Path | None
    benchmark: BasicOcrRunSummary
    compare_results: tuple[CompareTxtBuildResult, ...]
    artifact_results: tuple[ArtifactSearchableResult, ...]
    partial_page_failures: int = 0
    chunk_count: int = 1
    chunk_pages: int | None = None
    chunk_manifest_path: Path | None = None


@dataclass(slots=True, frozen=True)
class _PdfChunk:
    index: int
    start_page: int
    end_page: int
    path: Path


@dataclass(slots=True, frozen=True)
class _ResolvedHybridRunConfig:
    effective_ocr_render_dpi: int
    effective_textless_dpi: int
    effective_textless_jpeg_quality: int
    effective_engine_subprocess_timeout_seconds: float | None
    environment: tuple[tuple[str, str | None], ...]

    def as_identity(self) -> dict[str, object]:
        return {
            "effective_ocr_render_dpi": self.effective_ocr_render_dpi,
            "effective_textless_dpi": self.effective_textless_dpi,
            "effective_textless_jpeg_quality": self.effective_textless_jpeg_quality,
            "effective_engine_subprocess_timeout_seconds": (
                self.effective_engine_subprocess_timeout_seconds
            ),
            "chandra_min_image_dim": _CHANDRA_MIN_IMAGE_DIM,
            "zero_output_retry_policy": _SURYA_RETRY_POLICY,
            "chandra_zero_output_retry_policy": _CHANDRA_RETRY_POLICY,
            "page_reconciliation_policy": (
                "explicit-chandra-nontext+quiet-surya+scaled-terminal-lineage+"
                "durable-source-rasters+durable-chandra-attempts+exact-text-artifacts+"
                "bounded-png+accounted-alternative-text+plain-cross-engine-agreement-v9"
            ),
            "environment": dict(self.environment),
        }


def _emit_progress(cb: ProgressCallback | None, percent: int, status: str) -> None:
    if cb is None:
        return
    bounded = max(0, min(100, int(percent)))
    cb(bounded, status)


def _resolve_engine_python(engine: str) -> Path | None:
    env_name = _ENGINE_TO_PYTHON_ENV.get(engine)
    if env_name is None:
        return None
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"{env_name} points to missing python executable: {path}")
    return path


def _build_engine_subprocess_env(*, engine: str, repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    engine_prefix = f"UNISCAN_{engine.upper()}_"
    for suffix in _ENGINE_SUBPROCESS_ENV_SUFFIXES:
        override_key = f"{engine_prefix}{suffix}"
        if override_key not in os.environ:
            continue
        value = os.environ.get(override_key, "")
        if value:
            env[suffix] = value
        else:
            env.pop(suffix, None)

    existing_pythonpath = env.get("PYTHONPATH", "")
    src_root = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = (
        src_root if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"
    )
    return env


def _engine_subprocess_timeout_seconds() -> float | None:
    raw = (os.environ.get("UNISCAN_ENGINE_SUBPROCESS_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _resolve_ocr_render_dpi() -> int:
    raw = (os.environ.get("UNISCAN_OCR_RENDER_DPI") or "").strip()
    if not raw:
        return 220
    try:
        value = int(raw)
    except ValueError:
        return 220
    return max(72, min(400, value))


def _load_engine_result_from_report(
    *, report_path: Path, expected_engine: str
) -> OcrBenchmarkResult:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read benchmark report: {report_path}: {exc}") from exc

    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Benchmark report has no results: {report_path}")

    row = rows[0]
    if not isinstance(row, dict):
        raise RuntimeError(f"Benchmark report result is malformed: {report_path}")
    row = dict(row)
    row.setdefault("engine", expected_engine)
    return OcrBenchmarkResult(**row)


def _run_engine_benchmark_subprocess(
    *,
    python_exe: Path,
    engine: str,
    pdf_path: Path,
    output_dir: Path,
    sample_size: int,
    page_numbers: tuple[int, ...] | None,
    lang: str,
    dpi: int,
    defer_empty_pages: bool = False,
) -> OcrBenchmarkResult:
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        str(python_exe),
        "-m",
        "uniscan",
        "benchmark-ocr",
        "--pdf",
        str(pdf_path),
        "--output",
        str(output_dir),
        "--engines",
        engine,
        "--sample-size",
        str(int(sample_size)),
        "--lang",
        lang,
        "--dpi",
        str(int(dpi)),
    ]
    internal_token: str | None = None
    if defer_empty_pages:
        internal_token = uuid.uuid4().hex
        cmd.extend(["--internal-reconciliation-token", internal_token])
    if page_numbers:
        page_arg = ",".join(str(int(page)) for page in page_numbers)
        cmd.extend(["--pages", page_arg])

    env = _build_engine_subprocess_env(engine=engine, repo_root=repo_root)
    if internal_token is not None:
        env["UNISCAN_INTERNAL_RECONCILIATION_TOKEN"] = internal_token
    timeout_seconds = _engine_subprocess_timeout_seconds()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Engine '{engine}' subprocess timed out after {timeout_seconds:g} seconds."
        ) from exc
    report_path = output_dir / f"{pdf_path.stem}_ocr_benchmark.json"
    if not report_path.exists():
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or f"exit code {proc.returncode}"
        raise RuntimeError(f"Engine '{engine}' subprocess did not produce report: {details}")

    result = _load_engine_result_from_report(report_path=report_path, expected_engine=engine)
    if int(proc.returncode) != 0 and result.status == "ok":
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or f"exit code {proc.returncode}"
        raise RuntimeError(f"Engine '{engine}' subprocess failed: {details}")
    return result


def _result_error_text(result: OcrBenchmarkResult) -> str:
    if result.error and result.error.strip():
        return result.error.strip()
    if result.note and result.note.strip():
        return result.note.strip()
    return "unknown error"


def _ensure_requested_engines_succeeded(
    benchmark: BasicOcrRunSummary,
    *,
    expected_engines: tuple[str, ...],
) -> None:
    succeeded = {
        result.engine.strip().lower()
        for result in benchmark.results
        if result.status.strip().lower() == "ok"
    }
    missing = [engine for engine in expected_engines if engine not in succeeded]
    if not missing and not benchmark.failed_engines and not benchmark.skipped_engines:
        return

    details = [f"missing successful engines: {', '.join(missing)}"] if missing else []
    details.extend(f"failed: {item}" for item in benchmark.failed_engines)
    details.extend(f"skipped: {item}" for item in benchmark.skipped_engines)
    raise RuntimeError("Strict OCR benchmark is incomplete: " + " | ".join(details))


def _load_engine_page_index(
    *,
    run_dir: Path,
    engine: str,
) -> tuple[Path, dict[str, Any], dict[int, dict[str, Any]]]:
    pages_path = run_dir / engine / engine / "pages.json"
    try:
        payload = json.loads(pages_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{engine} page evidence is unreadable: {pages_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("engine") != engine:
        raise RuntimeError(f"{engine} page evidence has an invalid root: {pages_path}")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise RuntimeError(f"{engine} page evidence has no pages list: {pages_path}")
    by_page: dict[int, dict[str, Any]] = {}
    for row in raw_pages:
        if not isinstance(row, dict):
            raise RuntimeError(f"{engine} page evidence contains a non-object row")
        source_page = row.get("source_page")
        if (
            not isinstance(source_page, int)
            or isinstance(source_page, bool)
            or source_page <= 0
            or source_page in by_page
        ):
            raise RuntimeError(
                f"{engine} page evidence has an invalid or duplicate source_page: {source_page!r}"
            )
        by_page[source_page] = row
    if not by_page:
        raise RuntimeError(f"{engine} page evidence is empty: {pages_path}")
    return pages_path, payload, by_page


def _owned_page_artifact(*, engine_dir: Path, raw_name: object, label: str) -> Path:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise RuntimeError(f"Accepted textless graphics page has no {label} artifact")
    candidate = (engine_dir / raw_name).resolve()
    if not _path_is_within(candidate, engine_dir) or not candidate.is_file():
        raise RuntimeError(f"Accepted textless graphics {label} artifact is invalid: {candidate}")
    return candidate


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _textless_geometry_bytes(path: Path) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise RuntimeError(f"Geometry sidecar is malformed: {path}")
    page_entries = 0
    for image in payload["images"]:
        if not isinstance(image, dict):
            raise RuntimeError(f"Geometry sidecar contains a non-object image: {path}")
        image["ocr_outcome"] = "textless_graphics"
        image.pop("selected_attempt", None)
        pages = image.get("pages")
        if not isinstance(pages, list):
            raise RuntimeError(f"Geometry sidecar image has no pages list: {path}")
        for page in pages:
            if not isinstance(page, dict):
                raise RuntimeError(f"Geometry sidecar contains a non-object page: {path}")
            page["text_lines"] = []
            page["ocr_outcome"] = "textless_graphics"
            page_entries += 1
    if page_entries == 0:
        raise RuntimeError(f"Geometry sidecar has no page entry: {path}")
    return _json_bytes(payload)


def _stage_textless_graphics_artifacts(
    *,
    pages_path: Path,
    payload: dict[str, Any],
    accepted_pages: set[int],
    aggregate_path: Path,
) -> tuple[int, dict[Path, bytes]]:
    payload = deepcopy(payload)
    engine_dir = pages_path.parent.resolve()
    engine_output_dir = engine_dir.parent.resolve()
    aggregate_path = aggregate_path.resolve()
    if not _path_is_within(aggregate_path, engine_output_dir) or not aggregate_path.is_file():
        raise RuntimeError(
            f"Accepted textless graphics aggregate artifact is invalid: {aggregate_path}"
        )
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise RuntimeError(f"Page evidence has no pages list: {pages_path}")
    updates: dict[Path, bytes] = {}
    blocks: list[str] = []
    total_chars = 0
    for row in raw_pages:
        if not isinstance(row, dict):
            raise RuntimeError(f"Page evidence contains a non-object row: {pages_path}")
        source_page = int(row["source_page"])
        page_file = _owned_page_artifact(
            engine_dir=engine_dir,
            raw_name=row.get("file"),
            label="text",
        )
        if source_page in accepted_pages:
            geometry_file = _owned_page_artifact(
                engine_dir=engine_dir,
                raw_name=row.get("geometry_file"),
                label="geometry",
            )
            updates[geometry_file] = _textless_geometry_bytes(geometry_file)
            updates[page_file] = b""
            row["text_chars"] = 0
            row["ocr_outcome"] = "textless_graphics"
            row.pop("selected_attempt", None)
            row["alnum_line_count"] = 0
            row["alnum_chars"] = 0
            original_errors = row.get("page_errors")
            if isinstance(original_errors, list) and original_errors:
                row["reconciled_page_errors"] = list(original_errors)
            row["page_errors"] = []
            row["textless_graphics"] = True
            row["accepted_by"] = "mode_both_page_reconciliation"
        text = "" if source_page in accepted_pages else page_file.read_text(encoding="utf-8")
        row["text_chars"] = len(text)
        total_chars += len(text)
        blocks.append(f"[SOURCE PAGE {source_page:04d}]")
        if text:
            blocks.append(text.rstrip())
        blocks.append("")
    markerized = "\n".join(blocks).strip()
    markerized = markerized + "\n" if markerized else ""
    aggregate_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=payload.get("aggregate_file"),
        label="aggregate",
    )
    markerized_bytes = markerized.encode("utf-8")
    updates[aggregate_file] = markerized_bytes
    updates[aggregate_path] = markerized_bytes
    payload["total_text_chars"] = total_chars
    updates[pages_path] = _json_bytes(payload)
    return total_chars, updates


def _stage_bytes(target: Path, data: bytes) -> Path:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.transaction")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _publish_file_transaction(updates: Mapping[Path, bytes]) -> None:
    ordered_updates: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for raw_path, data in updates.items():
        target = raw_path.resolve()
        if target in seen:
            raise RuntimeError(f"Duplicate transaction target: {target}")
        if not target.is_file():
            raise RuntimeError(f"Transaction target is missing: {target}")
        seen.add(target)
        ordered_updates.append((target, data))

    originals = {target: target.read_bytes() for target, _data in ordered_updates}
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for target, data in ordered_updates:
            staged[target] = _stage_bytes(target, data)
        for target, _data in ordered_updates:
            os.replace(staged[target], target)
            published.append(target)
    except Exception as exc:
        rollback_errors: list[str] = []
        for target in reversed(published):
            restore_path: Path | None = None
            try:
                restore_path = _stage_bytes(target, originals[target])
                os.replace(restore_path, target)
            except Exception as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
            finally:
                if restore_path is not None:
                    restore_path.unlink(missing_ok=True)
        if rollback_errors:
            raise RuntimeError(
                "File transaction failed and rollback was incomplete: "
                + " | ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _load_benchmark_report_index(
    *,
    run_dir: Path,
    result_files: Sequence[Path],
    expected_engines: set[str],
) -> dict[str, tuple[Path, dict[str, Any], int]]:
    report_by_engine: dict[str, tuple[Path, dict[str, Any], int]] = {}
    for raw_path in result_files:
        report_path = raw_path.resolve()
        if not _path_is_within(report_path, run_dir) or not report_path.is_file():
            raise RuntimeError(f"Benchmark result file is not owned by the run: {report_path}")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Benchmark result file is unreadable: {report_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError(f"Benchmark result file is malformed: {report_path}")
        rows = payload["results"]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"Benchmark result row is malformed: {report_path}")
            engine = str(row.get("engine") or "").strip().lower()
            if engine not in expected_engines:
                continue
            if engine in report_by_engine:
                raise RuntimeError(f"Duplicate benchmark result file for engine: {engine}")
            report_by_engine[engine] = (report_path, payload, index)
    missing = expected_engines - set(report_by_engine)
    if missing:
        raise RuntimeError(f"Missing benchmark result files for engines: {sorted(missing)}")
    return report_by_engine


def _page_error_records(
    row: dict[str, Any],
    *,
    engine: str,
    source_page: int,
) -> list[dict[str, str]]:
    raw_errors = row.get("page_errors", [])
    if not isinstance(raw_errors, list):
        raise RuntimeError(f"{engine} page {source_page} has malformed durable page-error evidence")
    records: list[dict[str, str]] = []
    for item in raw_errors:
        if not isinstance(item, dict):
            raise RuntimeError(f"{engine} page {source_page} has an unstructured page error")
        code = str(item.get("code") or "").strip()
        message = str(item.get("message") or "").strip()
        if not code or not message:
            raise RuntimeError(f"{engine} page {source_page} has an incomplete page-error record")
        records.append({"code": code, "message": message})
    return records


def _stage_benchmark_reports(
    *,
    report_by_engine: dict[str, tuple[Path, dict[str, Any], int]],
    results: Sequence[OcrBenchmarkResult],
) -> dict[Path, bytes]:
    updates: dict[Path, bytes] = {}
    for result in results:
        report_path, original_payload, index = report_by_engine[result.engine]
        payload = deepcopy(original_payload)
        rows = payload["results"]
        assert isinstance(rows, list)
        rows[index] = asdict(result)
        updates[report_path] = _json_bytes(payload)
    return updates


def _alnum_artifact_evidence(text: str) -> tuple[int, int]:
    lines = [line for line in text.splitlines() if any(char.isalnum() for char in line)]
    return len(lines), sum(1 for line in lines for char in line if char.isalnum())


def _verified_surya_quiet_evidence(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
    expected_outcome: str,
    text_payload_override: bytes | None = None,
    geometry_payload_override: dict[str, Any] | None = None,
) -> tuple[int, int]:
    page_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("file"),
        label="text",
    )
    try:
        artifact_text = (
            page_file.read_text(encoding="utf-8")
            if text_payload_override is None
            else text_payload_override.decode("utf-8")
        )
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"surya page {source_page} text snapshot is not UTF-8") from exc
    artifact_lines, artifact_chars = _alnum_artifact_evidence(artifact_text)
    stored_lines = row.get("alnum_line_count")
    stored_chars = row.get("alnum_chars")
    if (
        not isinstance(stored_lines, int)
        or isinstance(stored_lines, bool)
        or stored_lines < 0
        or not isinstance(stored_chars, int)
        or isinstance(stored_chars, bool)
        or stored_chars < 0
        or (stored_lines, stored_chars) != (artifact_lines, artifact_chars)
    ):
        raise RuntimeError(
            f"surya page {source_page} alnum evidence does not match its text artifact"
        )

    geometry_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("geometry_file"),
        label="geometry",
    )
    if geometry_payload_override is None:
        try:
            sidecar, _geometry_fingerprint = _stable_json_object(
                geometry_file,
                label=f"Surya page {source_page} geometry evidence",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Surya page {source_page} geometry evidence is unreadable: {exc}"
            ) from exc
    else:
        sidecar = geometry_payload_override
    images = sidecar.get("images") if isinstance(sidecar, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"surya page {source_page} geometry evidence is malformed")
    image = images[0]
    if str(image.get("ocr_outcome") or "") != expected_outcome:
        raise RuntimeError(f"surya page {source_page} outcome disagrees with its sidecar")
    sidecar_chars = 0
    pages = image.get("pages")
    if not isinstance(pages, list) or not pages:
        raise RuntimeError(f"surya page {source_page} sidecar has no page entry")
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("text_lines"), list):
            raise RuntimeError(f"surya page {source_page} sidecar lines are malformed")
        for line in page["text_lines"]:
            if not isinstance(line, dict):
                raise RuntimeError(f"surya page {source_page} sidecar line is malformed")
            sidecar_chars += sum(1 for char in str(line.get("text") or "") if char.isalnum())
    if sidecar_chars != artifact_chars:
        raise RuntimeError(
            f"surya page {source_page} sidecar text disagrees with its text artifact"
        )
    return artifact_lines, artifact_chars


def _strict_chandra_attempt_history(
    *,
    row: dict[str, Any],
    image: dict[str, Any],
    attempts: Sequence[dict[str, Any]],
    engine_dir: Path,
    source_page: int,
    source_raster_identity: dict[str, object],
    source_raster_image: Image.Image,
) -> tuple[_SealedPageGeometry, ...]:
    row_history = row.get("attempt_history")
    image_history = image.get("attempt_history")
    if (
        not isinstance(row_history, list)
        or row_history != image_history
        or len(row_history) != len(attempts)
    ):
        raise RuntimeError(f"Chandra page {source_page} durable attempt history is incomplete")

    exact_history_fields = {
        "attempt",
        "image_size",
        "image_path",
        "image_sha256",
        "image_bytes",
        "sidecar_path",
        "sidecar_sha256",
        "sidecar_bytes",
    }
    decoded_images: list[Image.Image] = []
    sealed_geometries: list[_SealedPageGeometry] = []
    for expected_attempt, (raw_history, attempt) in enumerate(
        zip(row_history, attempts, strict=True),
        start=1,
    ):
        if not isinstance(raw_history, dict) or set(raw_history) != exact_history_fields:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} history shape is invalid"
            )
        if raw_history.get("attempt") != expected_attempt or isinstance(
            raw_history.get("attempt"), bool
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} history identity is invalid"
            )
        image_size = raw_history.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in image_size
            )
            or image_size[0] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or image_size[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or image_size[0] * image_size[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image size is invalid"
            )
        declared_width = int(image_size[0])
        declared_height = int(image_size[1])
        if (
            declared_width > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or declared_height > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or declared_width * declared_height > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image dimensions "
                "exceed the bounded pixel policy"
            )
        if not _valid_sha256(raw_history.get("image_sha256")) or not _valid_sha256(
            raw_history.get("sidecar_sha256")
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} digest is invalid"
            )
        image_bytes = raw_history.get("image_bytes")
        sidecar_bytes = raw_history.get("sidecar_bytes")
        if (
            not isinstance(image_bytes, int)
            or isinstance(image_bytes, bool)
            or image_bytes <= 0
            or image_bytes > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES
            or not isinstance(sidecar_bytes, int)
            or isinstance(sidecar_bytes, bool)
            or sidecar_bytes <= 0
            or sidecar_bytes > _MAX_CHUNK_MANIFEST_BYTES
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} byte seal is invalid"
            )
        if image_bytes > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image bytes "
                "exceed the bounded byte policy"
            )
        raw_image_path = raw_history.get("image_path")
        raw_sidecar_path = raw_history.get("sidecar_path")
        if not isinstance(raw_image_path, str) or not isinstance(raw_sidecar_path, str):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact path is invalid"
            )
        lexical_image = Path(os.path.abspath(Path(raw_image_path)))
        lexical_sidecar = Path(os.path.abspath(Path(raw_sidecar_path)))
        expected_dir = (
            engine_dir.resolve()
            / f"page_{source_page:04d}.chandra-attempts"
            / f"attempt_{expected_attempt}"
        )
        if (
            not Path(raw_image_path).is_absolute()
            or not Path(raw_sidecar_path).is_absolute()
            or lexical_image != expected_dir / "input.png"
            or lexical_sidecar != expected_dir / "chandra_attempt.json"
            or lexical_image.is_symlink()
            or lexical_sidecar.is_symlink()
            or not lexical_image.is_file()
            or not lexical_sidecar.is_file()
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact ownership is invalid"
            )
        try:
            if lexical_image.stat().st_nlink != 1 or lexical_sidecar.stat().st_nlink != 1:
                raise RuntimeError(
                    f"Chandra page {source_page} attempt {expected_attempt} artifact is hard-linked"
                )
            image_path = lexical_image.resolve(strict=True)
            sidecar_path = lexical_sidecar.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact cannot be resolved: {exc}"
            ) from exc
        if image_path != lexical_image or sidecar_path != lexical_sidecar:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact traverses a link"
            )
        try:
            image_payload, image_fingerprint = _stable_file_bytes(
                image_path,
                max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
            )
            sidecar_payload, sidecar_fingerprint = _stable_file_bytes(
                sidecar_path,
                max_bytes=_MAX_CHUNK_MANIFEST_BYTES,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact seal failed: {exc}"
            ) from exc
        if image_fingerprint != {
            "sha256": raw_history["image_sha256"],
            "size": image_bytes,
        } or sidecar_fingerprint != {
            "sha256": raw_history["sidecar_sha256"],
            "size": sidecar_bytes,
        }:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} artifact digest changed"
            )
        try:
            with Image.open(io.BytesIO(image_payload)) as opened_image:
                if opened_image.format != "PNG":
                    raise RuntimeError(
                        f"Chandra page {source_page} attempt {expected_attempt} image is not PNG"
                    )
                if int(getattr(opened_image, "n_frames", 1)) != 1:
                    raise RuntimeError(
                        f"Chandra page {source_page} attempt {expected_attempt} PNG is multi-frame"
                    )
                if (
                    opened_image.width > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                    or opened_image.height > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                    or opened_image.width * opened_image.height > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
                ):
                    raise RuntimeError(
                        f"Chandra page {source_page} attempt {expected_attempt} decoded image "
                        "exceeds the bounded pixel policy"
                    )
                opened_image.load()
                actual_mode = opened_image.mode
                actual_size = [int(opened_image.width), int(opened_image.height)]
                decoded_image = opened_image.convert("RGB").copy()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image is unreadable: {exc}"
            ) from exc
        if actual_mode != "RGB" or actual_size != image_size:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image descriptor changed"
            )
        decoded_images.append(decoded_image)

        try:
            attempt_sidecar = json.loads(sidecar_payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} sidecar is unreadable: {exc}"
            ) from exc
        exact_sidecar_fields = {
            "schema",
            "source_raster_identity",
            "image_name",
            "image_bbox",
            "raw_result",
            "parsed",
            "evidence",
        }
        if (
            not isinstance(attempt_sidecar, dict)
            or set(attempt_sidecar) != exact_sidecar_fields
            or attempt_sidecar.get("schema") != _CHANDRA_ATTEMPT_EVIDENCE_SCHEMA
            or attempt_sidecar.get("source_raster_identity") != source_raster_identity
            or attempt_sidecar.get("image_name") != "input.png"
            or attempt_sidecar.get("evidence") != attempt
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} sidecar identity is invalid"
            )
        image_bbox = _strict_geometry_bbox(
            attempt_sidecar.get("image_bbox"),
            label=f"Chandra attempt {expected_attempt} image",
        )
        if image_bbox != (0.0, 0.0, float(image_size[0]), float(image_size[1])):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} image bbox changed"
            )
        raw_result = attempt_sidecar.get("raw_result")
        if (
            not isinstance(raw_result, dict)
            or set(raw_result) != {"error", "chunks", "html", "markdown"}
            or raw_result.get("error") is not False
            or not isinstance(raw_result.get("html"), str)
            or not isinstance(raw_result.get("markdown"), str)
            or not isinstance(raw_result.get("chunks"), list)
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} raw result is malformed"
            )
        labels: list[str] = []
        texts: list[str] = []
        alternative_texts: list[str] = []
        excluded_header_footer_lines: list[str] = []
        lines: list[dict[str, object]] = []
        ignored_graphic_lines: list[str] = []
        prompt_type = str(attempt.get("prompt_type") or "")
        allow_graphic_text = prompt_type == "ocr"
        raw_chunks = raw_result["chunks"]
        assert isinstance(raw_chunks, list)
        for raw_chunk in raw_chunks:
            if (
                not isinstance(raw_chunk, dict)
                or set(raw_chunk) != {"label", "content", "bbox"}
                or not isinstance(raw_chunk.get("label"), str)
                or (
                    raw_chunk.get("content") is not None
                    and not isinstance(raw_chunk.get("content"), str)
                )
            ):
                raise RuntimeError(
                    f"Chandra page {source_page} attempt {expected_attempt} raw chunk is malformed"
                )
            label = str(raw_chunk["label"]).strip().lower()
            if label:
                labels.append(label)
            if label in {"blank-page", "image", "figure", "diagram"}:
                if label in {"image", "figure", "diagram"}:
                    ignored_graphic_lines.extend(
                        _chandra_graphic_chunk_ignored_lines(raw_chunk.get("content"))
                    )
                if not allow_graphic_text:
                    continue
                chunk_lines = _chandra_graphic_chunk_lines(raw_chunk.get("content"))
            else:
                chunk_lines = _chandra_chunk_lines(raw_chunk.get("content"))
            chunk_lines = [
                cleaned for line in chunk_lines if (cleaned := _clean_overlay_line(line))
            ]
            if not chunk_lines:
                continue
            texts.extend(chunk_lines)
            if label in {"page-header", "page-footer"}:
                excluded_header_footer_lines.extend(chunk_lines)
            else:
                alternative_texts.extend(chunk_lines)
            parsed_bbox = _bbox_values(raw_chunk.get("bbox"))
            if parsed_bbox is None:
                continue
            x0, y0, x1, y1 = parsed_bbox
            x0 = max(0.0, min(x0, float(image_size[0])))
            y0 = max(0.0, min(y0, float(image_size[1])))
            x1 = max(0.0, min(x1, float(image_size[0])))
            y1 = max(0.0, min(y1, float(image_size[1])))
            if x1 <= x0 or y1 <= y0:
                continue
            lines.extend(
                _chandra_expand_chunk_to_line_boxes(
                    lines=chunk_lines,
                    bbox=[x0, y0, x1, y1],
                )
            )
        graphic_labels = {"image", "figure", "diagram"}
        normalized_labels = sorted(set(labels))
        if (
            not texts
            and prompt_type == "ocr_layout"
            and not (set(normalized_labels) & graphic_labels)
            and str(raw_result["markdown"]).strip()
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} has unsealed markdown text"
            )
        normalized_text = unicodedata.normalize("NFKC", "\n".join(texts))
        normalized_geometry = unicodedata.normalize(
            "NFKC",
            "\n".join(str(line.get("text") or "") for line in lines),
        )
        if texts and (
            not lines
            or "".join(char for char in normalized_text if not char.isspace())
            != "".join(char for char in normalized_geometry if not char.isspace())
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} text lacks complete geometry"
            )
        expected_parsed = {
            "texts": texts,
            "text_lines": lines,
            "labels": normalized_labels,
        }
        if attempt_sidecar.get("parsed") != expected_parsed:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} parsed evidence disagrees with raw"
            )
        if lines:
            ordered_bboxes: list[tuple[float, float, float, float]] = []
            for line in lines:
                parsed_line_bbox = _bbox_values(line.get("bbox"))
                if parsed_line_bbox is None:
                    raise RuntimeError(
                        f"Chandra page {source_page} attempt {expected_attempt} line bbox is invalid"
                    )
                ordered_bboxes.append(parsed_line_bbox)
            order = _bbox_reading_order_indices(
                ordered_bboxes,
                page_width=float(image_size[0]),
            )
            canonical_text = _canonical_retry_text(
                "\n".join(str(lines[index].get("text") or "") for index in order)
            )
        else:
            canonical_text = ""
        alternative_text_evidence = _chandra_alternative_text_evidence(
            raw_result=raw_result,
            texts=texts,
            ignored_graphic_lines=ignored_graphic_lines,
            alternative_texts=alternative_texts,
            excluded_header_footer_lines=excluded_header_footer_lines,
        )
        if texts and alternative_text_evidence["accounting"] == "unaccounted":
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} has "
                "unaccounted alternative text"
            )
        explicit_nontext = (
            not texts
            and bool(set(normalized_labels) & graphic_labels)
            and set(normalized_labels) <= {"blank-page", "image", "figure", "diagram"}
            and alternative_text_evidence["accounting"] in {"empty", "ignored_graphic_description"}
        )
        derived_outcome = (
            "text" if texts else ("explicit_nontext" if explicit_nontext else "zero_output")
        )
        derived_evidence = dict(attempt)
        derived_evidence.update(
            {
                "labels": normalized_labels,
                "alternative_text_evidence": alternative_text_evidence,
                "text_chars": sum(len(line) for line in texts),
                "canonical_alnum_chars": len(canonical_text),
                "canonical_alnum_sha256": hashlib.sha256(
                    canonical_text.encode("utf-8")
                ).hexdigest(),
                "geometry_lines": len(lines),
                "explicit_nontext": explicit_nontext,
                "ocr_outcome": derived_outcome,
            }
        )
        if derived_evidence != attempt:
            raise RuntimeError(
                f"Chandra page {source_page} attempt {expected_attempt} counters disagree with raw"
            )
        attempt_image = {
            "image_name": "input.png",
            "pages": [
                {
                    "image_bbox": list(image_bbox),
                    "text_lines": lines,
                }
            ],
        }
        sealed_geometries.append(
            _sealed_page_geometry(
                image=attempt_image,
                label=f"Chandra attempt {expected_attempt}",
                reading_order=False,
                require_text=bool(texts),
                allow_punctuation_only_lines=True,
            )
        )

    attempt_one = decoded_images[0]
    source_width, source_height = source_raster_image.size
    if source_width < _CHANDRA_MIN_IMAGE_DIM or source_height < _CHANDRA_MIN_IMAGE_DIM:
        scale = _CHANDRA_MIN_IMAGE_DIM / float(min(source_width, source_height))
        expected_size = (int(source_width * scale), int(source_height * scale))
    else:
        expected_size = source_raster_image.size
    if attempt_one.size != expected_size:
        raise RuntimeError(
            f"Chandra page {source_page} attempt 1 dimensions are not the pinned "
            f"min-image-dim-{_CHANDRA_MIN_IMAGE_DIM} source-raster transform"
        )
    expected_attempt_one = (
        source_raster_image.copy()
        if expected_size == source_raster_image.size
        else source_raster_image.resize(expected_size, Image.Resampling.LANCZOS)
    )
    if ImageChops.difference(expected_attempt_one, attempt_one).getbbox() is not None:
        raise RuntimeError(
            f"Chandra page {source_page} attempt 1 pixels disagree with source raster"
        )

    if len(decoded_images) >= 2:
        if decoded_images[1].size != decoded_images[0].size:
            raise RuntimeError(
                f"Chandra page {source_page} attempt 2 dimensions disagree with original lineage"
            )
        expected_autocontrast = ImageOps.autocontrast(decoded_images[0], cutoff=1)
        if ImageChops.difference(expected_autocontrast, decoded_images[1]).getbbox() is not None:
            raise RuntimeError(
                f"Chandra page {source_page} attempt 2 pixels disagree with autocontrast lineage"
            )
    if len(decoded_images) == 3:
        if decoded_images[2].size != decoded_images[0].size:
            raise RuntimeError(
                f"Chandra page {source_page} attempt 3 dimensions disagree with original lineage"
            )
        if ImageChops.difference(decoded_images[0], decoded_images[2]).getbbox() is not None:
            raise RuntimeError(
                f"Chandra page {source_page} attempt 3 pixels disagree with original lineage"
            )
    return tuple(sealed_geometries)


def _strict_source_raster_identity(
    *,
    row: dict[str, Any],
    image: dict[str, Any],
    source_page: int,
    engine: str,
) -> dict[str, object]:
    row_identity = row.get("source_raster_identity")
    image_identity = image.get("source_raster_identity")
    exact_fields = {
        "pixel_sha256",
        "width",
        "height",
        "name",
        "source_page",
        "verified_blank",
    }
    if (
        not isinstance(row_identity, dict)
        or row_identity != image_identity
        or set(row_identity) != exact_fields
        or not _valid_sha256(row_identity.get("pixel_sha256"))
        or not isinstance(row_identity.get("width"), int)
        or isinstance(row_identity.get("width"), bool)
        or int(row_identity["width"]) <= 0
        or not isinstance(row_identity.get("height"), int)
        or isinstance(row_identity.get("height"), bool)
        or int(row_identity["height"]) <= 0
        or int(row_identity["width"]) > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
        or int(row_identity["height"]) > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
        or int(row_identity["width"]) * int(row_identity["height"])
        > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
        or not isinstance(row_identity.get("name"), str)
        or not str(row_identity["name"])
        or Path(str(row_identity["name"])).name != row_identity["name"]
        or row_identity.get("source_page") != source_page
        or isinstance(row_identity.get("source_page"), bool)
        or type(row_identity.get("verified_blank")) is not bool
        or image.get("image_name") != row_identity["name"]
    ):
        raise RuntimeError(f"{engine} page {source_page} source raster identity is invalid")
    return dict(row_identity)


def _strict_durable_source_raster(
    *,
    row: dict[str, Any],
    image: dict[str, Any],
    engine_dir: Path,
    source_page: int,
    engine: str,
    directory_suffix: str,
) -> tuple[dict[str, object], Image.Image]:
    identity = _strict_source_raster_identity(
        row=row,
        image=image,
        source_page=source_page,
        engine=engine,
    )
    row_artifact = row.get("source_raster_artifact")
    image_artifact = image.get("source_raster_artifact")
    if (
        not isinstance(row_artifact, dict)
        or row_artifact != image_artifact
        or set(row_artifact) != {"path", "sha256", "bytes"}
        or not isinstance(row_artifact.get("path"), str)
        or not _valid_sha256(row_artifact.get("sha256"))
        or not isinstance(row_artifact.get("bytes"), int)
        or isinstance(row_artifact.get("bytes"), bool)
        or int(row_artifact["bytes"]) <= 0
        or int(row_artifact["bytes"]) > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES
    ):
        raise RuntimeError(f"{engine} page {source_page} source raster artifact is invalid")
    expected_path = (
        engine_dir.resolve() / f"page_{source_page:04d}.{directory_suffix}" / "source.png"
    )
    raw_path = Path(str(row_artifact["path"]))
    if raw_path != expected_path:
        raise RuntimeError(f"{engine} page {source_page} source raster path is invalid")
    owned_path = _assert_owned_regular_file(
        expected_path,
        engine_dir.resolve(),
        label=f"{engine} page {source_page} source raster",
    )
    try:
        payload, fingerprint = _stable_file_bytes(
            owned_path,
            max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
        )
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != "PNG":
                raise RuntimeError("source raster is not PNG")
            if int(getattr(opened, "n_frames", 1)) != 1:
                raise RuntimeError("source raster PNG is multi-frame")
            if (
                opened.width > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                or opened.height > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                or opened.width * opened.height > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
            ):
                raise RuntimeError("source raster exceeds the bounded pixel policy")
            opened.load()
            if opened.mode != "RGB":
                raise RuntimeError("source raster is not canonical RGB")
            source_image = opened.copy()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{engine} page {source_page} source raster is unreadable: {exc}"
        ) from exc
    if fingerprint != {
        "sha256": row_artifact["sha256"],
        "size": row_artifact["bytes"],
    }:
        raise RuntimeError(f"{engine} page {source_page} source raster digest changed")
    if (
        source_image.size != (identity["width"], identity["height"])
        or _canonical_rgb_pixel_sha256(source_image) != identity["pixel_sha256"]
        or _is_effectively_blank_rgb_image(source_image) is not identity["verified_blank"]
    ):
        raise RuntimeError(f"{engine} page {source_page} source raster pixels disagree")
    return identity, source_image


def _strict_chandra_attempt_metadata(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
    geometry_payload_override: dict[str, Any] | None = None,
    text_payload_override: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    geometry_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("geometry_file"),
        label="geometry",
    )
    if geometry_payload_override is None:
        try:
            sidecar, _geometry_fingerprint = _stable_json_object(
                geometry_file,
                label=f"Chandra page {source_page} geometry evidence",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Chandra page {source_page} geometry evidence is unreadable: {exc}"
            ) from exc
    else:
        sidecar = geometry_payload_override
    images = sidecar.get("images") if isinstance(sidecar, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Chandra page {source_page} geometry evidence is malformed")
    image = images[0]
    source_raster_identity, source_raster_image = _strict_durable_source_raster(
        row=row,
        image=image,
        engine_dir=engine_dir,
        source_page=source_page,
        engine="Chandra",
        directory_suffix="chandra-source",
    )
    required_fields = ("attempt_count", "terminal_attempt", "chandra_retry_policy")
    if any(field not in row or field not in image for field in required_fields):
        raise RuntimeError(f"Chandra page {source_page} attempt provenance is incomplete")
    if any(row[field] != image[field] for field in required_fields):
        raise RuntimeError(f"Chandra page {source_page} attempt provenance disagrees")
    attempt_count = row["attempt_count"]
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count not in {1, 2, 3}
        or row["terminal_attempt"] != attempt_count
        or isinstance(row["terminal_attempt"], bool)
        or row["chandra_retry_policy"] != _CHANDRA_RETRY_POLICY
    ):
        raise RuntimeError(f"Chandra page {source_page} attempt identity is invalid")

    expected_marker = {
        1: None,
        2: _SURYA_RETRY_PREPROCESSING,
        3: _CHANDRA_PLAIN_RETRY_PREPROCESSING,
    }[attempt_count]
    for payload in (row, image):
        marker_present = "retry_preprocessing" in payload
        if marker_present != (expected_marker is not None):
            raise RuntimeError(f"Chandra page {source_page} retry marker presence is invalid")
        if marker_present and payload["retry_preprocessing"] != expected_marker:
            raise RuntimeError(f"Chandra page {source_page} retry marker is invalid")
    if row.get("retry_preprocessing") != image.get("retry_preprocessing"):
        raise RuntimeError(f"Chandra page {source_page} retry markers disagree")

    raw_attempts = image.get("attempts")
    if (
        not isinstance(raw_attempts, list)
        or len(raw_attempts) != attempt_count
        or any(not isinstance(item, dict) for item in raw_attempts)
    ):
        raise RuntimeError(f"Chandra page {source_page} attempt list is malformed")
    expected_prompts = ["ocr_layout", "ocr_layout", "ocr"][:attempt_count]
    expected_preprocessing = ["original", _SURYA_RETRY_PREPROCESSING, "original"][:attempt_count]
    allowed_nontext = {"blank-page", "image", "figure", "diagram"}
    graphic_labels = {"image", "figure", "diagram"}
    normalized_attempts: list[dict[str, Any]] = []
    exact_attempt_fields = {
        "attempt",
        "source_raster_identity",
        "prompt_type",
        "prompt_sha256",
        "preprocessing",
        "content_filter_policy",
        "alternative_text_evidence",
        "labels",
        "text_chars",
        "canonical_alnum_chars",
        "canonical_alnum_sha256",
        "geometry_lines",
        "explicit_nontext",
        "ocr_outcome",
    }
    for index, raw_attempt in enumerate(raw_attempts, start=1):
        assert isinstance(raw_attempt, dict)
        if set(raw_attempt) != exact_attempt_fields:
            raise RuntimeError(f"Chandra page {source_page} attempt {index} shape is invalid")
        prompt_type = expected_prompts[index - 1]
        if (
            raw_attempt["attempt"] != index
            or isinstance(raw_attempt["attempt"], bool)
            or raw_attempt["source_raster_identity"] != source_raster_identity
            or raw_attempt["prompt_type"] != prompt_type
            or raw_attempt["prompt_sha256"] != _CHANDRA_PROMPT_SHA256[prompt_type]
            or raw_attempt["preprocessing"] != expected_preprocessing[index - 1]
            or raw_attempt["content_filter_policy"] != _CHANDRA_CONTENT_FILTERS[prompt_type]
        ):
            raise RuntimeError(f"Chandra page {source_page} attempt {index} identity is invalid")
        labels = raw_attempt["labels"]
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise RuntimeError(f"Chandra page {source_page} attempt {index} labels are invalid")
        normalized_labels = [label.strip().lower() for label in labels if label.strip()]
        if labels != sorted(set(normalized_labels)):
            raise RuntimeError(f"Chandra page {source_page} attempt {index} labels are not sealed")
        alternative_text_evidence = raw_attempt["alternative_text_evidence"]
        if (
            not isinstance(alternative_text_evidence, dict)
            or set(alternative_text_evidence)
            != {
                "policy",
                "accounting",
                "html_normalized_sha256",
                "markdown_normalized_sha256",
                "ignored_graphic_normalized_sha256",
            }
            or alternative_text_evidence.get("policy") != _CHANDRA_ALTERNATIVE_TEXT_POLICY
            or alternative_text_evidence.get("accounting")
            not in {
                "empty",
                "parsed",
                "parsed_without_header_footer",
                "ignored_graphic_description",
                "unaccounted",
            }
            or any(
                not _valid_sha256(alternative_text_evidence.get(key))
                for key in (
                    "html_normalized_sha256",
                    "markdown_normalized_sha256",
                    "ignored_graphic_normalized_sha256",
                )
            )
            or (
                alternative_text_evidence.get("accounting") == "ignored_graphic_description"
                and not (set(normalized_labels) & graphic_labels)
            )
        ):
            raise RuntimeError(
                f"Chandra page {source_page} attempt {index} alternative text evidence is invalid"
            )
        text_chars = raw_attempt["text_chars"]
        canonical_alnum_chars = raw_attempt["canonical_alnum_chars"]
        geometry_lines = raw_attempt["geometry_lines"]
        if (
            not isinstance(text_chars, int)
            or isinstance(text_chars, bool)
            or text_chars < 0
            or not isinstance(canonical_alnum_chars, int)
            or isinstance(canonical_alnum_chars, bool)
            or canonical_alnum_chars < 0
            or not _valid_sha256(raw_attempt["canonical_alnum_sha256"])
            or not isinstance(geometry_lines, int)
            or isinstance(geometry_lines, bool)
            or geometry_lines < 0
        ):
            raise RuntimeError(f"Chandra page {source_page} attempt {index} counters are invalid")
        if text_chars > 0 and alternative_text_evidence["accounting"] == "unaccounted":
            raise RuntimeError(
                f"Chandra page {source_page} attempt {index} has unaccounted alternative text"
            )
        explicit_nontext = (
            text_chars == 0
            and bool(set(normalized_labels) & graphic_labels)
            and set(normalized_labels) <= allowed_nontext
            and alternative_text_evidence["accounting"] in {"empty", "ignored_graphic_description"}
        )
        expected_outcome = (
            "text"
            if text_chars > 0
            else ("explicit_nontext" if explicit_nontext else "zero_output")
        )
        if (
            raw_attempt["explicit_nontext"] is not explicit_nontext
            or raw_attempt["ocr_outcome"] != expected_outcome
            or (text_chars > 0 and geometry_lines == 0)
            or (text_chars == 0 and geometry_lines != 0)
            or (
                text_chars == 0
                and (
                    canonical_alnum_chars != 0
                    or raw_attempt["canonical_alnum_sha256"] != hashlib.sha256(b"").hexdigest()
                )
            )
            or (text_chars > 0 and canonical_alnum_chars == 0)
            or (text_chars == 0 and alternative_text_evidence["accounting"] == "parsed")
        ):
            raise RuntimeError(f"Chandra page {source_page} attempt {index} outcome is invalid")
        normalized_attempts.append(raw_attempt)

    attempt_geometries = _strict_chandra_attempt_history(
        row=row,
        image=image,
        attempts=normalized_attempts,
        engine_dir=engine_dir,
        source_page=source_page,
        source_raster_identity=source_raster_identity,
        source_raster_image=source_raster_image,
    )

    labels = image.get("chandra_non_text_labels")
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise RuntimeError(f"Chandra page {source_page} nontext labels are malformed")
    expected_labels = sorted(
        {
            label
            for attempt in normalized_attempts
            for label in attempt["labels"]
            if label in allowed_nontext
        }
    )
    if labels != expected_labels:
        raise RuntimeError(f"Chandra page {source_page} nontext labels disagree with attempts")

    outcome = str(row.get("ocr_outcome") or "")
    if image.get("ocr_outcome") != outcome:
        raise RuntimeError(f"Chandra page {source_page} outcome disagrees with sidecar")
    pages = image.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise RuntimeError(f"Chandra page {source_page} sidecar must have exactly one page entry")
    if any(
        not isinstance(page, dict)
        or page.get("ocr_outcome") != outcome
        or not isinstance(page.get("text_lines"), list)
        for page in pages
    ):
        raise RuntimeError(f"Chandra page {source_page} page outcome is malformed")
    sidecar_has_text = any(bool(page["text_lines"]) for page in pages)

    selected_present = "selected_attempt" in row or "selected_attempt" in image
    if outcome == "text":
        if (
            "selected_attempt" not in row
            or "selected_attempt" not in image
            or row["selected_attempt"] != attempt_count
            or image["selected_attempt"] != attempt_count
            or isinstance(row["selected_attempt"], bool)
            or isinstance(image["selected_attempt"], bool)
            or any(attempt["text_chars"] != 0 for attempt in normalized_attempts[:-1])
            or normalized_attempts[-1]["text_chars"] <= 0
        ):
            raise RuntimeError(f"Chandra page {source_page} selected attempt is invalid")
        _, geometry, durable_image = _strict_durable_text_evidence(
            row=row,
            engine_dir=engine_dir,
            engine="chandra",
            source_page=source_page,
            artifact_payload_override=text_payload_override,
            geometry_payload_override=geometry_payload_override,
        )
        selected = normalized_attempts[-1]
        if (
            durable_image != image
            or geometry.image_bbox != attempt_geometries[-1].image_bbox
            or geometry.lines != attempt_geometries[-1].lines
            or len(geometry.lines) != selected["geometry_lines"]
            or len(geometry.canonical_text) != selected["canonical_alnum_chars"]
            or hashlib.sha256(geometry.canonical_text.encode("utf-8")).hexdigest()
            != selected["canonical_alnum_sha256"]
        ):
            raise RuntimeError(
                f"Chandra page {source_page} selected counters disagree with durable geometry"
            )
        expected_explicit = False
    elif outcome == "verified_blank":
        if (
            attempt_count != 1
            or selected_present
            or normalized_attempts[0]["text_chars"] != 0
            or sidecar_has_text
            or source_raster_identity["verified_blank"] is not True
        ):
            raise RuntimeError(f"Chandra page {source_page} blank attempt lineage is invalid")
        expected_explicit = False
    elif outcome in {"explicit_nontext", "textless_graphics"}:
        plain_labels = set(normalized_attempts[2]["labels"]) if attempt_count == 3 else set()
        if (
            attempt_count != 3
            or selected_present
            or any(attempt["text_chars"] != 0 for attempt in normalized_attempts)
            or any(attempt["explicit_nontext"] is not True for attempt in normalized_attempts)
            or not (plain_labels & graphic_labels)
            or not plain_labels <= allowed_nontext
        ):
            raise RuntimeError(f"Chandra page {source_page} nontext consensus is invalid")
        expected_explicit = True
    elif outcome == "zero_output":
        if (
            attempt_count != 3
            or selected_present
            or any(attempt["text_chars"] != 0 for attempt in normalized_attempts)
            or sidecar_has_text
        ):
            raise RuntimeError(f"Chandra page {source_page} zero-output lineage is invalid")
        expected_explicit = False
    else:
        raise RuntimeError(f"Chandra page {source_page} outcome is unsupported")
    if outcome != "text":
        nontext_geometry = _sealed_page_geometry(
            image=image,
            label=f"Chandra page {source_page}",
            reading_order=False,
            require_text=False,
        )
        if nontext_geometry.lines:
            raise RuntimeError(f"Chandra page {source_page} nontext geometry contains text")
        if (
            nontext_geometry.image_bbox != attempt_geometries[-1].image_bbox
            or nontext_geometry.lines != attempt_geometries[-1].lines
        ):
            raise RuntimeError(
                f"Chandra page {source_page} nontext geometry disagrees with terminal attempt"
            )
    if image.get("explicit_nontext") is not expected_explicit:
        raise RuntimeError(f"Chandra page {source_page} explicit marker is invalid")
    if row.get("explicit_nontext") is not expected_explicit:
        raise RuntimeError(f"Chandra page {source_page} row explicit marker is invalid")
    if outcome != "text":
        page_file = _owned_page_artifact(
            engine_dir=engine_dir,
            raw_name=row.get("file"),
            label="text",
        )
        try:
            if text_payload_override is None:
                page_payload, _page_fingerprint = _stable_file_bytes(
                    page_file,
                    max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
                )
            else:
                page_payload = text_payload_override
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Chandra page {source_page} nontext artifact is unreadable: {exc}"
            ) from exc
        if page_payload != b"" or sidecar_has_text:
            raise RuntimeError(f"Chandra page {source_page} nontext evidence contains text")
    return attempt_count, image


def _verified_chandra_explicit_nontext(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
) -> bool:
    page_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("file"),
        label="text",
    )
    try:
        page_payload, _page_fingerprint = _stable_file_bytes(
            page_file,
            max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"chandra page {source_page} text artifact is unreadable: {exc}"
        ) from exc
    if page_payload != b"":
        raise RuntimeError(f"chandra page {source_page} text artifact is not empty")
    _, image = _strict_chandra_attempt_metadata(
        row=row,
        engine_dir=engine_dir,
        source_page=source_page,
    )
    labels = image.get("chandra_non_text_labels")
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise RuntimeError(f"chandra page {source_page} nontext labels are malformed")
    normalized_labels = {label.strip().lower() for label in labels if label.strip()}
    allowed_labels = {"blank-page", "image", "figure", "diagram"}
    graphic_labels = {"image", "figure", "diagram"}
    pages = image.get("pages")
    if not isinstance(pages, list) or not pages:
        raise RuntimeError(f"chandra page {source_page} sidecar has no page entry")
    for page in pages:
        if not isinstance(page, dict) or page.get("text_lines") != []:
            raise RuntimeError(f"chandra page {source_page} sidecar is not textless")
    return (
        str(image.get("ocr_outcome") or "") == "explicit_nontext"
        and image.get("explicit_nontext") is True
        and bool(normalized_labels & graphic_labels)
        and normalized_labels <= allowed_labels
    )


def _canonical_retry_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = unicodedata.normalize("NFKC", normalized.casefold())
    return "".join(char for char in normalized if char.isalnum())


def _canonical_retry_agreement_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def _strict_geometry_bbox(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RuntimeError(f"{label} bbox must contain exactly four numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise RuntimeError(f"{label} bbox values must be numeric and not bool")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise RuntimeError(f"{label} bbox values must be finite")
    x0, y0, x1, y1 = bbox
    if x0 < 0.0 or y0 < 0.0 or x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"{label} bbox must have positive area")
    return x0, y0, x1, y1


@dataclass(slots=True, frozen=True)
class _SealedTextLine:
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class _SealedPageGeometry:
    image_name: str
    image_bbox: tuple[float, float, float, float]
    lines: tuple[_SealedTextLine, ...]
    canonical_text: str


def _deterministic_geometry_page_text(
    geometry: _SealedPageGeometry,
    *,
    reading_order: bool,
) -> str:
    indices = list(range(len(geometry.lines)))
    if reading_order and geometry.lines:
        page_width = max(
            max(line.bbox[2] for line in geometry.lines),
            geometry.image_bbox[2],
            1.0,
        )
        indices = _bbox_reading_order_indices(
            [line.bbox for line in geometry.lines],
            page_width=page_width,
        )
    ordered_lines = [
        cleaned for index in indices if (cleaned := _clean_overlay_line(geometry.lines[index].text))
    ]
    return "\n".join(_dehyphenate_line_breaks(ordered_lines))


def _inverse_scaled_retry_geometry(
    geometry: _SealedPageGeometry,
    *,
    source_size: Sequence[int],
    content_size: Sequence[int],
    content_offset: Sequence[int],
) -> _SealedPageGeometry:
    source_width, source_height = source_size
    content_width, content_height = content_size
    offset_x, offset_y = content_offset
    content_x1 = offset_x + content_width
    content_y1 = offset_y + content_height
    lines: list[_SealedTextLine] = []
    for line in geometry.lines:
        x0, y0, x1, y1 = line.bbox
        if x0 < offset_x or y0 < offset_y or x1 > content_x1 or y1 > content_y1:
            raise RuntimeError("Surya third retry text bbox escapes scaled content")
        bbox = (
            (x0 - offset_x) * source_width / content_width,
            (y0 - offset_y) * source_height / content_height,
            (x1 - offset_x) * source_width / content_width,
            (y1 - offset_y) * source_height / content_height,
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise RuntimeError("Surya inverse-transformed text bbox has no area")
        lines.append(_SealedTextLine(text=line.text, bbox=bbox))
    return _SealedPageGeometry(
        image_name=geometry.image_name,
        image_bbox=(0.0, 0.0, float(source_width), float(source_height)),
        lines=tuple(lines),
        canonical_text=geometry.canonical_text,
    )


def _sealed_page_geometry(
    *,
    image: dict[str, Any],
    label: str,
    reading_order: bool,
    require_text: bool,
    allow_punctuation_only_lines: bool = False,
) -> _SealedPageGeometry:
    image_name = image.get("image_name")
    if not isinstance(image_name, str) or not image_name or Path(image_name).name != image_name:
        raise RuntimeError(f"{label} image_name is invalid")
    pages = image.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise RuntimeError(f"{label} requires exactly one geometry page")
    page = pages[0]
    image_bbox = _strict_geometry_bbox(page.get("image_bbox"), label=f"{label} image")
    raw_lines = page.get("text_lines")
    if not isinstance(raw_lines, list) or (require_text and not raw_lines):
        raise RuntimeError(f"{label} has invalid text geometry")
    lines: list[_SealedTextLine] = []
    cleaned_lines: list[str] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise RuntimeError(f"{label} has a non-object text line")
        text = raw_line.get("text")
        cleaned_text = _clean_overlay_line(text) if isinstance(text, str) else ""
        if (
            not isinstance(text, str)
            or not cleaned_text
            or (not allow_punctuation_only_lines and not _canonical_retry_text(cleaned_text))
        ):
            raise RuntimeError(f"{label} has invalid line text")
        bbox = _strict_geometry_bbox(raw_line.get("bbox"), label=f"{label} text line")
        if (
            bbox[0] < image_bbox[0]
            or bbox[1] < image_bbox[1]
            or bbox[2] > image_bbox[2]
            or bbox[3] > image_bbox[3]
        ):
            raise RuntimeError(f"{label} text bbox escapes image bbox")
        lines.append(_SealedTextLine(text=text, bbox=bbox))
        cleaned_lines.append(cleaned_text)
    canonical_indices = list(range(len(lines)))
    if reading_order and lines:
        page_width = max(
            max(line.bbox[2] for line in lines),
            image_bbox[2],
            1.0,
        )
        canonical_indices = _bbox_reading_order_indices(
            [line.bbox for line in lines],
            page_width=page_width,
        )
    canonical_text = _canonical_retry_text(
        "\n".join(cleaned_lines[index] for index in canonical_indices)
    )
    if require_text and not canonical_text:
        raise RuntimeError(f"{label} has no canonical text")
    return _SealedPageGeometry(
        image_name=image_name,
        image_bbox=image_bbox,
        lines=tuple(lines),
        canonical_text=canonical_text,
    )


def _strict_durable_text_evidence(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    engine: str,
    source_page: int,
    artifact_payload_override: bytes | None = None,
    geometry_payload_override: dict[str, Any] | None = None,
) -> tuple[str, _SealedPageGeometry, dict[str, Any]]:
    page_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("file"),
        label="text",
    )
    try:
        if artifact_payload_override is None:
            artifact_payload, _artifact_fingerprint = _stable_file_bytes(
                page_file,
                max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
            )
        else:
            artifact_payload = artifact_payload_override
        artifact_text = artifact_payload.decode("utf-8")
    except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"{engine} page {source_page} text artifact is unreadable: {exc}"
        ) from exc
    artifact_canonical = _canonical_retry_text(artifact_text)
    if not artifact_canonical:
        raise RuntimeError(f"{engine} page {source_page} text artifact has no alnum text")
    if (
        not isinstance(row.get("text_chars"), int)
        or isinstance(row.get("text_chars"), bool)
        or row["text_chars"] != len(artifact_text)
    ):
        raise RuntimeError(f"{engine} page {source_page} text_chars disagrees with its artifact")

    geometry_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("geometry_file"),
        label="geometry",
    )
    if geometry_payload_override is None:
        try:
            sidecar, _geometry_fingerprint = _stable_json_object(
                geometry_file,
                label=f"{engine} page {source_page} geometry evidence",
            )
        except Exception as exc:
            raise RuntimeError(
                f"{engine} page {source_page} geometry evidence is unreadable: {exc}"
            ) from exc
    else:
        sidecar = geometry_payload_override
    images = sidecar.get("images") if isinstance(sidecar, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"{engine} page {source_page} requires exactly one image")
    image = images[0]
    if str(image.get("ocr_outcome") or "") != "text":
        raise RuntimeError(f"{engine} page {source_page} sidecar outcome is not text")
    geometry = _sealed_page_geometry(
        image=image,
        label=f"{engine} page {source_page}",
        reading_order=engine == "chandra",
        require_text=True,
        allow_punctuation_only_lines=engine == "chandra",
    )
    expected_artifact_text = _deterministic_geometry_page_text(
        geometry,
        reading_order=engine == "chandra",
    )
    if artifact_payload != expected_artifact_text.encode("utf-8"):
        raise RuntimeError(
            f"{engine} page {source_page} text artifact bytes are not deterministically "
            "derived from sealed geometry"
        )
    pages = image.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise RuntimeError(f"{engine} page {source_page} requires exactly one geometry page")
    page = pages[0]
    image_bbox = _strict_geometry_bbox(
        page.get("image_bbox"),
        label=f"{engine} page {source_page} image",
    )
    raw_lines = page.get("text_lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise RuntimeError(f"{engine} page {source_page} has no text geometry")
    sidecar_text: list[str] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise RuntimeError(f"{engine} page {source_page} has a non-object text line")
        text = raw_line.get("text")
        cleaned_text = _clean_overlay_line(text) if isinstance(text, str) else ""
        if (
            not isinstance(text, str)
            or not cleaned_text
            or (engine != "chandra" and not _canonical_retry_text(cleaned_text))
        ):
            raise RuntimeError(f"{engine} page {source_page} has invalid line text")
        bbox = _strict_geometry_bbox(
            raw_line.get("bbox"),
            label=f"{engine} page {source_page} text line",
        )
        if (
            bbox[0] < image_bbox[0]
            or bbox[1] < image_bbox[1]
            or bbox[2] > image_bbox[2]
            or bbox[3] > image_bbox[3]
        ):
            raise RuntimeError(f"{engine} page {source_page} text bbox escapes image bbox")
        sidecar_text.append(text)
    if artifact_canonical != geometry.canonical_text:
        raise RuntimeError(
            f"{engine} page {source_page} sidecar text disagrees with its text artifact"
        )
    return _canonical_retry_agreement_text(artifact_text), geometry, image


def _strict_retry_attempt_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in {1, 2, 3}:
        raise RuntimeError("Surya retry attempt_count is invalid")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_surya_retry_provenance(
    *,
    row: dict[str, Any],
    image: dict[str, Any],
    selected_geometry: _SealedPageGeometry | None,
    engine_dir: Path,
    source_page: int,
    source_raster_image: Image.Image,
    expected_outcome: str = "text",
) -> _SealedPageGeometry:
    if expected_outcome not in {"text", "verified_blank", "zero_output"}:
        raise RuntimeError("Surya terminal attempt outcome is invalid")
    if (
        str(row.get("ocr_outcome") or "") != expected_outcome
        or str(image.get("ocr_outcome") or "") != expected_outcome
    ):
        raise RuntimeError("Surya third retry outcome disagrees with selected evidence")
    source_identity = row.get("source_raster_identity")
    source_name = source_identity.get("name") if isinstance(source_identity, dict) else None
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise RuntimeError("Surya retry source raster name is invalid")
    fields = (
        "attempt_count",
        "retry_preprocessing",
        "retry_policy",
        "selected_attempt",
        "attempt_history",
        "geometry_coordinate_space",
        "geometry_transform",
    )
    if any(row.get(field) != image.get(field) for field in fields):
        raise RuntimeError("Surya retry provenance disagrees with its durable sidecar")
    if row.get("retry_policy") != _SURYA_RETRY_POLICY:
        raise RuntimeError("Surya retry policy marker is invalid")
    selected_attempt = row.get("selected_attempt")
    if (
        not isinstance(selected_attempt, int)
        or isinstance(selected_attempt, bool)
        or selected_attempt not in {1, 2, 3}
        or row.get("attempt_count") != selected_attempt
    ):
        raise RuntimeError("Surya selected retry attempt is invalid")
    expected_marker = {
        1: None,
        2: _SURYA_RETRY_PREPROCESSING,
        3: _SURYA_SCALED_RETRY_PREPROCESSING,
    }[selected_attempt]
    if row.get("retry_preprocessing") != expected_marker:
        raise RuntimeError("Surya retry preprocessing marker is invalid")
    if selected_attempt == 2 and expected_outcome != "text":
        raise RuntimeError("Surya second retry must terminate with text")
    if selected_attempt == 3 and expected_outcome == "text":
        if (
            row.get("geometry_coordinate_space") != _SURYA_SOURCE_COORDINATE_SPACE
            or row.get("geometry_transform") != _SURYA_SCALED_GEOMETRY_TRANSFORM
        ):
            raise RuntimeError("Surya selected retry coordinate transform is invalid")
    elif any(
        row.get(field) is not None or image.get(field) is not None
        for field in ("geometry_coordinate_space", "geometry_transform")
    ):
        raise RuntimeError("Surya retry has an invalid coordinate transform marker")
    history = row.get("attempt_history")
    if not isinstance(history, list) or len(history) != selected_attempt:
        raise RuntimeError("Surya retry attempt history is incomplete")
    expected_all = (
        (
            1,
            "original",
            expected_outcome if selected_attempt == 1 else "zero_output",
        ),
        (
            2,
            _SURYA_RETRY_PREPROCESSING,
            expected_outcome if selected_attempt == 2 else "zero_output",
        ),
        (3, _SURYA_SCALED_RETRY_PREPROCESSING, expected_outcome),
    )
    expected = expected_all[:selected_attempt]
    expected_image_size: list[int] | None = None
    attempt_three_geometry: _SealedPageGeometry | None = None
    attempt_image_payloads: dict[int, bytes] = {}
    for item, (attempt, preprocessing, outcome) in zip(history, expected, strict=True):
        if not isinstance(item, dict):
            raise RuntimeError("Surya retry attempt history contains a non-object entry")
        exact_fields = {
            "attempt",
            "preprocessing",
            "ocr_outcome",
            "image_size",
            "image_path",
            "image_sha256",
            "image_bytes",
            "sidecar_path",
            "sidecar_sha256",
            "sidecar_bytes",
        }
        if attempt == 3:
            exact_fields.update(
                {
                    "content_scale",
                    "content_size",
                    "content_offset",
                    "resampling",
                    "canvas_fill_rgb",
                }
            )
        if set(item) != exact_fields:
            raise RuntimeError("Surya retry attempt history shape is invalid")
        stored_attempt = item.get("attempt")
        if (
            stored_attempt != attempt
            or isinstance(stored_attempt, bool)
            or item.get("preprocessing") != preprocessing
            or item.get("ocr_outcome") != outcome
        ):
            raise RuntimeError("Surya retry attempt history sequence is invalid")
        image_size = item.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in image_size
            )
        ):
            raise RuntimeError("Surya retry attempt image size is invalid")
        if expected_image_size is None:
            expected_image_size = list(image_size)
        elif image_size != expected_image_size:
            raise RuntimeError("Surya retry attempt image dimensions changed")
        if not _valid_sha256(item.get("image_sha256")) or not _valid_sha256(
            item.get("sidecar_sha256")
        ):
            raise RuntimeError("Surya retry attempt digest is invalid")
        raw_image_path = item.get("image_path")
        raw_sidecar_path = item.get("sidecar_path")
        if not isinstance(raw_image_path, str) or not isinstance(raw_sidecar_path, str):
            raise RuntimeError("Surya retry attempt has no linked durable artifacts")
        lexical_image = Path(raw_image_path)
        lexical_sidecar = Path(raw_sidecar_path)
        if not lexical_image.is_absolute() or not lexical_sidecar.is_absolute():
            raise RuntimeError("Surya retry attempt artifact path is not absolute")
        lexical_image = Path(os.path.abspath(lexical_image))
        lexical_sidecar = Path(os.path.abspath(lexical_sidecar))
        attempt_dir = engine_dir.resolve() / f"page_{source_page:04d}.retry" / f"attempt_{attempt}"
        if (
            lexical_image != attempt_dir / source_name
            or lexical_sidecar != attempt_dir / "surya_page_lines.json"
            or lexical_image.is_symlink()
            or lexical_sidecar.is_symlink()
            or not lexical_image.is_file()
            or not lexical_sidecar.is_file()
        ):
            raise RuntimeError("Surya retry attempt artifact path is invalid")
        try:
            if lexical_image.stat().st_nlink != 1 or lexical_sidecar.stat().st_nlink != 1:
                raise RuntimeError("Surya retry attempt artifact is hard-linked")
            image_path = lexical_image.resolve(strict=True)
            sidecar_path = lexical_sidecar.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"Surya retry attempt artifact cannot be resolved: {exc}") from exc
        if image_path != lexical_image or sidecar_path != lexical_sidecar:
            raise RuntimeError("Surya retry attempt artifact path traverses a link")
        image_bytes = item.get("image_bytes")
        sidecar_bytes = item.get("sidecar_bytes")
        if (
            not isinstance(image_bytes, int)
            or isinstance(image_bytes, bool)
            or image_bytes <= 0
            or not isinstance(sidecar_bytes, int)
            or isinstance(sidecar_bytes, bool)
            or sidecar_bytes <= 0
        ):
            raise RuntimeError("Surya retry attempt byte counts are invalid")
        try:
            image_payload, image_fingerprint = _stable_file_bytes(
                image_path,
                max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
            )
            sidecar_payload, sidecar_fingerprint = _stable_file_bytes(
                sidecar_path,
                max_bytes=_MAX_CHUNK_MANIFEST_BYTES,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Surya retry attempt artifact seal failed: {exc}") from exc
        if image_fingerprint != {
            "sha256": item["image_sha256"],
            "size": image_bytes,
        }:
            raise RuntimeError("Surya retry attempt image seal is invalid")
        if sidecar_fingerprint != {
            "sha256": item["sidecar_sha256"],
            "size": sidecar_bytes,
        }:
            raise RuntimeError("Surya retry attempt sidecar seal is invalid")
        try:
            with Image.open(io.BytesIO(image_payload)) as durable_image:
                if durable_image.format != "PNG" or int(getattr(durable_image, "n_frames", 1)) != 1:
                    raise RuntimeError("Surya retry attempt image is not canonical PNG")
                durable_image.load()
                actual_size = [int(durable_image.width), int(durable_image.height)]
                actual_mode = durable_image.mode
        except Exception as exc:
            raise RuntimeError(f"Surya retry attempt image is unreadable: {exc}") from exc
        if actual_size != image_size:
            raise RuntimeError("Surya retry attempt image size disagrees with its artifact")
        if actual_mode != "RGB":
            raise RuntimeError("Surya retry attempt image mode is invalid")
        attempt_image_payloads[attempt] = image_payload
        try:
            attempt_payload = json.loads(sidecar_payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Surya retry attempt sidecar is unreadable: {exc}") from exc
        execution_path = (
            attempt_payload.get("execution_path") if isinstance(attempt_payload, dict) else None
        )
        permitted_paths = {"cli", "module"} if attempt == 1 else {"module"}
        attempt_images = (
            attempt_payload.get("images") if isinstance(attempt_payload, dict) else None
        )
        if (
            execution_path not in permitted_paths
            or not isinstance(attempt_images, list)
            or len(attempt_images) != 1
            or not isinstance(attempt_images[0], dict)
            or attempt_images[0].get("image_name") != image_path.name
        ):
            raise RuntimeError("Surya retry attempt sidecar identity is invalid")
        attempt_image = attempt_images[0]
        attempt_count_marker = attempt_image.get("attempt_count")
        if (
            attempt_image.get("ocr_outcome") != outcome
            or attempt_count_marker != attempt
            or isinstance(attempt_count_marker, bool)
            or (attempt == 1 and attempt_image.get("retry_preprocessing") is not None)
            or (attempt > 1 and attempt_image.get("retry_preprocessing") != preprocessing)
        ):
            raise RuntimeError("Surya retry attempt sidecar metadata disagrees with history")
        attempt_pages = attempt_images[0].get("pages")
        if (
            not isinstance(attempt_pages, list)
            or len(attempt_pages) != 1
            or not isinstance(attempt_pages[0], dict)
        ):
            raise RuntimeError("Surya retry attempt sidecar page is invalid")
        attempt_bbox = _strict_geometry_bbox(
            attempt_pages[0].get("image_bbox"),
            label=f"Surya retry attempt {attempt} image",
        )
        if attempt_bbox != (0.0, 0.0, float(image_size[0]), float(image_size[1])):
            raise RuntimeError("Surya retry attempt image bbox disagrees with its image")
        attempt_lines = attempt_pages[0].get("text_lines")
        if not isinstance(attempt_lines, list):
            raise RuntimeError("Surya retry attempt text lines are invalid")
        if attempt < selected_attempt and attempt_lines:
            raise RuntimeError("Surya zero-output retry attempt contains text")
        if attempt == selected_attempt:
            if expected_outcome == "text":
                if not attempt_lines or any(not isinstance(line, dict) for line in attempt_lines):
                    raise RuntimeError("Surya terminal retry has no durable text geometry")
                for line in attempt_lines:
                    assert isinstance(line, dict)
                    text = line.get("text")
                    if not isinstance(text, str) or not _canonical_retry_text(text):
                        raise RuntimeError("Surya terminal retry line text is invalid")
                    line_bbox = _strict_geometry_bbox(
                        line.get("bbox"),
                        label="Surya terminal retry text line",
                    )
                    if line_bbox[2] > attempt_bbox[2] or line_bbox[3] > attempt_bbox[3]:
                        raise RuntimeError("Surya terminal retry text bbox escapes image bbox")
            elif attempt_lines:
                raise RuntimeError("Surya zero-output terminal retry contains text")
            attempt_three_geometry = _sealed_page_geometry(
                image=attempt_images[0],
                label="Surya terminal retry",
                reading_order=False,
                require_text=expected_outcome == "text",
            )
    if expected_image_size is None or attempt_three_geometry is None:
        raise RuntimeError("Surya retry terminal geometry is missing")
    try:
        with Image.open(io.BytesIO(attempt_image_payloads[1])) as opened_attempt_one:
            opened_attempt_one.load()
            attempt_one_image = opened_attempt_one.convert("RGB").copy()
        attempt_two_image: Image.Image | None = None
        if selected_attempt > 1:
            with Image.open(io.BytesIO(attempt_image_payloads[2])) as opened_attempt_two:
                opened_attempt_two.load()
                attempt_two_image = opened_attempt_two.convert("RGB").copy()
    except Exception as exc:
        raise RuntimeError(f"Surya retry pixel lineage is unreadable: {exc}") from exc
    if (
        attempt_one_image.size != source_raster_image.size
        or ImageChops.difference(source_raster_image, attempt_one_image).getbbox() is not None
    ):
        raise RuntimeError("Surya attempt 1 pixels disagree with its durable source raster")
    if selected_attempt == 1:
        selected_attempt_geometry = _sealed_page_geometry(
            image=image,
            label="Surya selected attempt",
            reading_order=False,
            require_text=expected_outcome == "text",
        )
        if selected_attempt_geometry != attempt_three_geometry:
            raise RuntimeError("Surya selected geometry disagrees with attempt 1")
        if expected_outcome == "text" and selected_geometry != attempt_three_geometry:
            raise RuntimeError("Surya selected text geometry disagrees with attempt 1")
        if expected_outcome != "text" and selected_geometry is not None:
            raise RuntimeError("Surya nontext attempt 1 has selected text geometry")
        return attempt_three_geometry
    assert attempt_two_image is not None
    expected_autocontrast = ImageOps.autocontrast(attempt_one_image, cutoff=1)
    if (
        attempt_two_image.size != attempt_one_image.size
        or ImageChops.difference(expected_autocontrast, attempt_two_image).getbbox() is not None
    ):
        raise RuntimeError("Surya attempt 2 pixels disagree with autocontrast lineage")
    if selected_attempt == 2:
        if selected_geometry is None:
            raise RuntimeError("Surya second retry selected text geometry is missing")
        selected_retry_geometry = _sealed_page_geometry(
            image=image,
            label="Surya selected retry",
            reading_order=False,
            require_text=True,
        )
        if (
            selected_retry_geometry != attempt_three_geometry
            or selected_geometry != attempt_three_geometry
        ):
            raise RuntimeError("Surya selected geometry disagrees with attempt 2")
        return selected_geometry
    scaled_evidence = history[2]
    content_scale = scaled_evidence.get("content_scale")
    if (
        not isinstance(content_scale, (int, float))
        or isinstance(content_scale, bool)
        or float(content_scale) != _SURYA_SCALED_RETRY_FACTOR
        or expected_image_size is None
    ):
        raise RuntimeError("Surya scaled retry factor is invalid")
    expected_content_size = [
        max(1, round(expected_image_size[0] * _SURYA_SCALED_RETRY_FACTOR)),
        max(1, round(expected_image_size[1] * _SURYA_SCALED_RETRY_FACTOR)),
    ]
    expected_content_offset = [
        (expected_image_size[0] - expected_content_size[0]) // 2,
        (expected_image_size[1] - expected_content_size[1]) // 2,
    ]
    if (
        scaled_evidence.get("content_size") != expected_content_size
        or scaled_evidence.get("content_offset") != expected_content_offset
        or scaled_evidence.get("resampling") != "lanczos"
        or scaled_evidence.get("canvas_fill_rgb") != [255, 255, 255]
    ):
        raise RuntimeError("Surya scaled retry preprocessing evidence is invalid")
    try:
        expected_size_tuple = (expected_image_size[0], expected_image_size[1])
        expected_content_size_tuple = (expected_content_size[0], expected_content_size[1])
        expected_content_offset_tuple = (
            expected_content_offset[0],
            expected_content_offset[1],
        )
        with Image.open(io.BytesIO(attempt_image_payloads[1])) as original_retry_image:
            original_retry_image.load()
            expected_content = original_retry_image.resize(
                expected_content_size_tuple,
                Image.Resampling.LANCZOS,
            )
            expected_scaled = Image.new(
                "RGB",
                expected_size_tuple,
                (255, 255, 255),
            )
            expected_scaled.paste(expected_content, expected_content_offset_tuple)
        with Image.open(io.BytesIO(attempt_image_payloads[3])) as actual_scaled_image:
            actual_scaled_image.load()
            actual_scaled = actual_scaled_image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Surya scaled retry pixel lineage is unreadable: {exc}") from exc
    if ImageChops.difference(expected_scaled, actual_scaled).getbbox() is not None:
        raise RuntimeError("Surya scaled retry pixels disagree with sealed attempt 1")
    selected_retry_geometry = _sealed_page_geometry(
        image=image,
        label="Surya selected retry",
        reading_order=False,
        require_text=expected_outcome == "text",
    )
    if selected_retry_geometry.image_bbox != (
        0.0,
        0.0,
        float(expected_image_size[0]),
        float(expected_image_size[1]),
    ):
        raise RuntimeError("Surya selected image bbox disagrees with durable retry image")
    if attempt_three_geometry is None:
        raise RuntimeError("Surya third retry geometry is missing")
    if expected_outcome == "text":
        if selected_geometry is None:
            raise RuntimeError("Surya selected text geometry is missing")
        expected_selected_geometry = _inverse_scaled_retry_geometry(
            attempt_three_geometry,
            source_size=expected_image_size,
            content_size=expected_content_size,
            content_offset=expected_content_offset,
        )
        if (
            selected_geometry != expected_selected_geometry
            or selected_retry_geometry != selected_geometry
        ):
            raise RuntimeError("Surya selected geometry disagrees with inverse retry transform")
        return selected_geometry
    if selected_geometry is not None:
        raise RuntimeError("Surya zero-output retry has selected text geometry")
    if selected_retry_geometry != attempt_three_geometry:
        raise RuntimeError("Surya selected zero-output geometry disagrees with attempt 3")
    return selected_retry_geometry


def _strict_surya_attempt_metadata(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
    geometry_payload_override: dict[str, Any] | None = None,
    text_payload_override: bytes | None = None,
) -> tuple[int, dict[str, object]]:
    attempt = _strict_retry_attempt_count(row.get("attempt_count"))
    geometry_file = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("geometry_file"),
        label="geometry",
    )
    if geometry_payload_override is None:
        try:
            sidecar, _geometry_fingerprint = _stable_json_object(
                geometry_file,
                label=f"Surya page {source_page} attempt metadata sidecar",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Surya page {source_page} attempt metadata sidecar is unreadable: {exc}"
            ) from exc
    else:
        sidecar = geometry_payload_override
    images = sidecar.get("images") if isinstance(sidecar, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Surya page {source_page} attempt sidecar is malformed")
    image = images[0]
    source_raster_identity, source_raster_image = _strict_durable_source_raster(
        row=row,
        image=image,
        engine_dir=engine_dir,
        source_page=source_page,
        engine="Surya",
        directory_suffix="surya-source",
    )
    row_outcome = str(row.get("ocr_outcome") or "")
    if (
        row_outcome not in {"text", "verified_blank", "zero_output"}
        or image.get("ocr_outcome") != row_outcome
    ):
        raise RuntimeError(f"Surya page {source_page} outcome disagrees with sidecar")
    if row_outcome == "verified_blank" and source_raster_identity["verified_blank"] is not True:
        raise RuntimeError(f"Surya page {source_page} forged verified-blank identity")
    if row_outcome == "zero_output" and source_raster_identity["verified_blank"] is True:
        raise RuntimeError(f"Surya page {source_page} zero-output identity is blank")
    selected_geometry: _SealedPageGeometry | None = None
    if row_outcome == "text":
        _canonical_text, selected_geometry, _selected_image = _strict_durable_text_evidence(
            row=row,
            engine_dir=engine_dir,
            engine="surya",
            source_page=source_page,
            artifact_payload_override=text_payload_override,
            geometry_payload_override=geometry_payload_override,
        )
    else:
        geometry = _sealed_page_geometry(
            image=image,
            label=f"Surya page {source_page}",
            reading_order=False,
            require_text=False,
        )
        page_file = _owned_page_artifact(
            engine_dir=engine_dir,
            raw_name=row.get("file"),
            label="text",
        )
        try:
            if text_payload_override is None:
                page_payload, _page_fingerprint = _stable_file_bytes(
                    page_file,
                    max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
                )
            else:
                page_payload = text_payload_override
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Surya page {source_page} nontext artifact is unreadable: {exc}"
            ) from exc
        if (
            geometry.lines
            or page_payload != b""
            or row.get("text_chars") != 0
            or isinstance(row.get("text_chars"), bool)
        ):
            raise RuntimeError(f"Surya page {source_page} nontext evidence contains text")
    sidecar_attempt = _strict_retry_attempt_count(image.get("attempt_count"))
    if sidecar_attempt != attempt:
        raise RuntimeError(f"Surya page {source_page} attempt_count disagrees with sidecar")
    row_marker = row.get("retry_preprocessing")
    sidecar_marker = image.get("retry_preprocessing")
    expected_marker = {
        1: None,
        2: _SURYA_RETRY_PREPROCESSING,
        3: _SURYA_SCALED_RETRY_PREPROCESSING,
    }[attempt]
    if row_marker != expected_marker or sidecar_marker != expected_marker:
        raise RuntimeError(f"Surya page {source_page} attempt marker is invalid")
    provenance_fields = ("retry_policy", "selected_attempt", "attempt_history")
    if any(row.get(field) != image.get(field) for field in provenance_fields):
        raise RuntimeError(f"Surya page {source_page} attempt provenance disagrees")
    selected_attempt = row.get("selected_attempt")
    history = row.get("attempt_history")
    if (
        row.get("retry_policy") != _SURYA_RETRY_POLICY
        or selected_attempt != attempt
        or isinstance(selected_attempt, bool)
        or not isinstance(history, list)
        or len(history) != attempt
        or any(not isinstance(item, dict) for item in history)
        or [item.get("attempt") for item in history] != list(range(1, attempt + 1))
    ):
        raise RuntimeError(f"Surya page {source_page} attempt provenance is invalid")
    if attempt > 1 and row_outcome not in {"text", "zero_output"}:
        raise RuntimeError(f"Surya page {source_page} retry outcome is invalid")
    _strict_surya_retry_provenance(
        row=row,
        image=image,
        selected_geometry=selected_geometry,
        engine_dir=engine_dir,
        source_page=source_page,
        source_raster_image=source_raster_image,
        expected_outcome=row_outcome,
    )
    return attempt, source_raster_identity


def _surya_text_attempt(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
    validated_attempt: int | None = None,
) -> tuple[int, str | None, _SealedPageGeometry | None]:
    attempt = validated_attempt
    if attempt is None:
        attempt, _source_identity = _strict_surya_attempt_metadata(
            row=row, engine_dir=engine_dir, source_page=source_page
        )
    canonical, selected_geometry, _image = _strict_durable_text_evidence(
        row=row,
        engine_dir=engine_dir,
        engine="surya",
        source_page=source_page,
    )
    return attempt, canonical, selected_geometry


def _strict_surya_third_zero_provenance(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
) -> None:
    attempt, _source_identity = _strict_surya_attempt_metadata(
        row=row,
        engine_dir=engine_dir,
        source_page=source_page,
    )
    if attempt != 3 or row.get("ocr_outcome") != "zero_output":
        raise RuntimeError(f"Surya page {source_page} terminal zero lineage is invalid")


def _snapshot_original_page_evidence(
    *,
    row: dict[str, Any],
    engine_dir: Path,
    source_page: int,
) -> dict[str, object]:
    text_path = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("file"),
        label="text",
    )
    geometry_path = _owned_page_artifact(
        engine_dir=engine_dir,
        raw_name=row.get("geometry_file"),
        label="geometry",
    )
    text_payload, text_fingerprint = _stable_file_bytes(
        text_path,
        max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
    )
    geometry_payload, geometry_fingerprint = _stable_file_bytes(
        geometry_path,
        max_bytes=_MAX_CHUNK_MANIFEST_BYTES,
    )
    try:
        text_utf8 = text_payload.decode("utf-8")
        geometry_utf8 = geometry_payload.decode("utf-8")
        geometry = json.loads(geometry_utf8)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Original page {source_page} evidence is not valid UTF-8 JSON/text"
        ) from exc
    if not isinstance(geometry, dict):
        raise RuntimeError(f"Original page {source_page} geometry root is not an object")
    return {
        "row": deepcopy(row),
        "text_utf8": text_utf8,
        "text_sha256": str(text_fingerprint["sha256"]),
        "text_size": int(str(text_fingerprint["size"])),
        "geometry_utf8": geometry_utf8,
        "geometry_sha256": str(geometry_fingerprint["sha256"]),
        "geometry_size": int(str(geometry_fingerprint["size"])),
    }


def _load_original_page_evidence(
    *,
    raw_snapshot: object,
    final_row: dict[str, Any],
    source_page: int,
    engine: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    exact_fields = {
        "row",
        "text_utf8",
        "text_sha256",
        "text_size",
        "geometry_utf8",
        "geometry_sha256",
        "geometry_size",
    }
    if not isinstance(raw_snapshot, dict) or set(raw_snapshot) != exact_fields:
        raise RuntimeError(f"{engine} page {source_page} original snapshot shape is invalid")
    original_row = raw_snapshot.get("row")
    text_utf8 = raw_snapshot.get("text_utf8")
    geometry_utf8 = raw_snapshot.get("geometry_utf8")
    if (
        not isinstance(original_row, dict)
        or original_row.get("source_page") != source_page
        or isinstance(original_row.get("source_page"), bool)
        or original_row.get("file") != final_row.get("file")
        or original_row.get("geometry_file") != final_row.get("geometry_file")
        or not isinstance(text_utf8, str)
        or not isinstance(geometry_utf8, str)
    ):
        raise RuntimeError(f"{engine} page {source_page} original snapshot identity is invalid")
    text_payload = text_utf8.encode("utf-8")
    geometry_payload = geometry_utf8.encode("utf-8")
    if (
        raw_snapshot.get("text_sha256") != hashlib.sha256(text_payload).hexdigest()
        or raw_snapshot.get("text_size") != len(text_payload)
        or isinstance(raw_snapshot.get("text_size"), bool)
        or raw_snapshot.get("geometry_sha256") != hashlib.sha256(geometry_payload).hexdigest()
        or raw_snapshot.get("geometry_size") != len(geometry_payload)
        or isinstance(raw_snapshot.get("geometry_size"), bool)
    ):
        raise RuntimeError(f"{engine} page {source_page} original snapshot seal is invalid")
    try:
        geometry = json.loads(geometry_utf8)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{engine} page {source_page} original geometry snapshot is invalid"
        ) from exc
    if not isinstance(geometry, dict):
        raise RuntimeError(f"{engine} page {source_page} original geometry is not an object")
    return dict(original_row), text_payload, geometry


def _expected_textless_graphics_row(original_row: dict[str, Any]) -> dict[str, Any]:
    expected = deepcopy(original_row)
    expected["text_chars"] = 0
    expected["ocr_outcome"] = "textless_graphics"
    expected.pop("selected_attempt", None)
    expected["alnum_line_count"] = 0
    expected["alnum_chars"] = 0
    original_errors = expected.get("page_errors")
    if isinstance(original_errors, list) and original_errors:
        expected["reconciled_page_errors"] = list(original_errors)
    expected["page_errors"] = []
    expected["textless_graphics"] = True
    expected["accepted_by"] = "mode_both_page_reconciliation"
    return expected


def _expected_textless_graphics_geometry(original: dict[str, Any]) -> dict[str, Any]:
    expected = deepcopy(original)
    images = expected.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("Original textless-graphics geometry has no images")
    page_entries = 0
    for image in images:
        if not isinstance(image, dict):
            raise RuntimeError("Original textless-graphics geometry image is malformed")
        image["ocr_outcome"] = "textless_graphics"
        image.pop("selected_attempt", None)
        pages = image.get("pages")
        if not isinstance(pages, list):
            raise RuntimeError("Original textless-graphics geometry has no pages list")
        for page in pages:
            if not isinstance(page, dict):
                raise RuntimeError("Original textless-graphics geometry page is malformed")
            page["text_lines"] = []
            page["ocr_outcome"] = "textless_graphics"
            page_entries += 1
    if page_entries == 0:
        raise RuntimeError("Original textless-graphics geometry has no page entry")
    return expected


def _record_semantic_fingerprint(
    fingerprints: dict[Path, tuple[str, int]] | None,
    path: Path,
    fingerprint: Mapping[str, object],
) -> None:
    if fingerprints is None:
        return
    normalized = (str(fingerprint["sha256"]), int(str(fingerprint["size"])))
    previous = fingerprints.get(path)
    if previous is not None and previous != normalized:
        raise RuntimeError(f"Semantic evidence changed during validation: {path}")
    fingerprints[path] = normalized


def _validate_final_mode_both_state(
    *,
    run_dir: Path,
    expected_page_count: int | None = None,
    fingerprints: dict[Path, tuple[str, int]] | None = None,
) -> frozenset[int]:
    resolved_run = Path(run_dir).resolve()
    reconciliation_path = resolved_run / "page_reconciliation.json"
    report, report_fingerprint = _stable_json_object(
        reconciliation_path,
        label="page reconciliation",
    )
    _record_semantic_fingerprint(fingerprints, reconciliation_path, report_fingerprint)
    exact_report_fields = {
        "schema",
        "status",
        "exact_page_bijection",
        "accepted_textless_graphics_pages",
        "unresolved_pages",
        "pages",
        "reconciliation_original_evidence",
        "reconciled_page_error_counts",
        "result_text_chars",
    }
    if (
        set(report) != exact_report_fields
        or report.get("schema") != "uniscan.page-reconciliation.v1"
        or report.get("status") != "ok"
        or report.get("exact_page_bijection") is not True
        or report.get("unresolved_pages") != []
    ):
        raise RuntimeError("Completed reconciliation root is not canonical")

    raw_accepted = report.get("accepted_textless_graphics_pages")
    if (
        not isinstance(raw_accepted, list)
        or any(
            not isinstance(page, int) or isinstance(page, bool) or page <= 0
            for page in raw_accepted
        )
        or raw_accepted != sorted(set(raw_accepted))
    ):
        raise RuntimeError("Completed reconciliation accepted-page set is invalid")
    claimed_accepted = frozenset(raw_accepted)
    raw_originals = report.get("reconciliation_original_evidence")
    if not isinstance(raw_originals, dict) or set(raw_originals) != {
        str(page) for page in claimed_accepted
    }:
        raise RuntimeError("Completed reconciliation original evidence is not bijective")

    engine_dirs: dict[str, Path] = {}
    page_rows: dict[str, dict[int, dict[str, Any]]] = {}
    final_text_payloads: dict[str, dict[int, bytes]] = {}
    final_geometry_payloads: dict[str, dict[int, dict[str, Any]]] = {}
    for engine in ("surya", "chandra"):
        engine_dir = (resolved_run / engine / engine).resolve()
        pages_path = engine_dir / "pages.json"
        payload, pages_fingerprint = _stable_json_object(
            pages_path,
            label=f"{engine} pages index",
        )
        _record_semantic_fingerprint(fingerprints, pages_path, pages_fingerprint)
        raw_pages = payload.get("pages")
        if payload.get("engine") != engine or not isinstance(raw_pages, list):
            raise RuntimeError(f"{engine} final pages index is malformed")
        by_page: dict[int, dict[str, Any]] = {}
        text_by_page: dict[int, bytes] = {}
        geometry_by_page: dict[int, dict[str, Any]] = {}
        for raw_row in raw_pages:
            if not isinstance(raw_row, dict):
                raise RuntimeError(f"{engine} final pages index contains a non-object row")
            source_page = raw_row.get("source_page")
            if (
                not isinstance(source_page, int)
                or isinstance(source_page, bool)
                or source_page <= 0
                or source_page in by_page
            ):
                raise RuntimeError(f"{engine} final pages index is not bijective")
            text_path = _owned_page_artifact(
                engine_dir=engine_dir,
                raw_name=raw_row.get("file"),
                label="text",
            )
            geometry_path = _owned_page_artifact(
                engine_dir=engine_dir,
                raw_name=raw_row.get("geometry_file"),
                label="geometry",
            )
            text_payload, text_fingerprint = _stable_file_bytes(
                text_path,
                max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
            )
            geometry_payload, geometry_fingerprint = _stable_json_object(
                geometry_path,
                label=f"{engine} page {source_page} final geometry",
            )
            _record_semantic_fingerprint(fingerprints, text_path, text_fingerprint)
            _record_semantic_fingerprint(fingerprints, geometry_path, geometry_fingerprint)
            try:
                final_text = text_payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"{engine} page {source_page} final text is not UTF-8") from exc
            if raw_row.get("text_chars") != len(final_text) or isinstance(
                raw_row.get("text_chars"), bool
            ):
                raise RuntimeError(f"{engine} page {source_page} final text counter disagrees")
            by_page[source_page] = dict(raw_row)
            text_by_page[source_page] = text_payload
            geometry_by_page[source_page] = geometry_payload
        engine_dirs[engine] = engine_dir
        page_rows[engine] = by_page
        final_text_payloads[engine] = text_by_page
        final_geometry_payloads[engine] = geometry_by_page

    expected_pages = set(page_rows["surya"])
    if (
        expected_pages != set(page_rows["chandra"])
        or (
            expected_page_count is not None
            and (
                len(expected_pages) != expected_page_count
                or expected_pages != set(range(1, expected_page_count + 1))
            )
        )
        or not claimed_accepted <= expected_pages
    ):
        raise RuntimeError("Completed reconciliation page evidence is not exactly bijective")

    expected_rows: list[dict[str, object]] = []
    recomputed_graphics: list[int] = []
    reconciled_counts = {"surya": 0, "chandra": 0}
    for source_page in sorted(expected_pages):
        original_rows: dict[str, dict[str, Any]] = {}
        original_text: dict[str, bytes] = {}
        original_geometry: dict[str, dict[str, Any]] = {}
        if source_page in claimed_accepted:
            page_snapshot = raw_originals.get(str(source_page))
            if not isinstance(page_snapshot, dict) or set(page_snapshot) != {"surya", "chandra"}:
                raise RuntimeError(f"Page {source_page} original engine evidence is incomplete")
            for engine in ("surya", "chandra"):
                row, text_payload, geometry = _load_original_page_evidence(
                    raw_snapshot=page_snapshot[engine],
                    final_row=page_rows[engine][source_page],
                    source_page=source_page,
                    engine=engine,
                )
                if page_rows[engine][source_page] != _expected_textless_graphics_row(row):
                    raise RuntimeError(
                        f"{engine} page {source_page} final suppression is not deterministic"
                    )
                if final_text_payloads[engine][source_page] != b"":
                    raise RuntimeError(f"{engine} page {source_page} final text is not empty")
                if final_geometry_payloads[engine][source_page] != (
                    _expected_textless_graphics_geometry(geometry)
                ):
                    raise RuntimeError(
                        f"{engine} page {source_page} final geometry suppression changed"
                    )
                original_rows[engine] = row
                original_text[engine] = text_payload
                original_geometry[engine] = geometry
        else:
            for engine in ("surya", "chandra"):
                original_rows[engine] = page_rows[engine][source_page]
                original_text[engine] = final_text_payloads[engine][source_page]
                original_geometry[engine] = final_geometry_payloads[engine][source_page]

        surya = original_rows["surya"]
        chandra = original_rows["chandra"]
        surya_outcome = str(surya.get("ocr_outcome") or "")
        chandra_outcome = str(chandra.get("ocr_outcome") or "")
        surya_errors = _page_error_records(surya, engine="surya", source_page=source_page)
        chandra_errors = _page_error_records(
            chandra,
            engine="chandra",
            source_page=source_page,
        )
        surya_attempt, surya_identity = _strict_surya_attempt_metadata(
            row=surya,
            engine_dir=engine_dirs["surya"],
            source_page=source_page,
            geometry_payload_override=original_geometry["surya"],
            text_payload_override=original_text["surya"],
        )
        chandra_attempt, chandra_image = _strict_chandra_attempt_metadata(
            row=chandra,
            engine_dir=engine_dirs["chandra"],
            source_page=source_page,
            geometry_payload_override=original_geometry["chandra"],
            text_payload_override=original_text["chandra"],
        )
        chandra_identity = chandra_image.get("source_raster_identity")
        if surya_identity != chandra_identity:
            raise RuntimeError(f"Page {source_page} engine source raster identities disagree")

        reason: str
        retry_agreement: dict[str, object] | None = None
        surya_alnum_lines = surya.get("alnum_line_count")
        surya_alnum_chars = surya.get("alnum_chars")
        is_graphics = False
        if surya_outcome == "text" and chandra_outcome == "text":
            if surya_attempt in {1, 2} and chandra_attempt in {1, 2}:
                reason = "both_text"
            else:
                surya_canonical, _surya_geometry, _surya_image = _strict_durable_text_evidence(
                    row=surya,
                    engine_dir=engine_dirs["surya"],
                    engine="surya",
                    source_page=source_page,
                    artifact_payload_override=original_text["surya"],
                    geometry_payload_override=original_geometry["surya"],
                )
                chandra_canonical, _chandra_geometry, _chandra_image = (
                    _strict_durable_text_evidence(
                        row=chandra,
                        engine_dir=engine_dirs["chandra"],
                        engine="chandra",
                        source_page=source_page,
                        artifact_payload_override=original_text["chandra"],
                        geometry_payload_override=original_geometry["chandra"],
                    )
                )
                retry_agreement = {
                    "algorithm": _RETRY_TEXT_AGREEMENT_ALGORITHM,
                    "matched": surya_canonical == chandra_canonical,
                    "surya_sha256": hashlib.sha256(surya_canonical.encode("utf-8")).hexdigest(),
                    "chandra_sha256": hashlib.sha256(chandra_canonical.encode("utf-8")).hexdigest(),
                }
                if not retry_agreement["matched"]:
                    raise RuntimeError(f"Page {source_page} retry text evidence disagrees")
                reason = "both_text_retry_geometry_agreement"
        elif surya_outcome == "verified_blank" and chandra_outcome == "verified_blank":
            reason = "both_verified_blank"
        elif surya_outcome == "text" and chandra_outcome == "zero_output":
            surya_alnum_lines, surya_alnum_chars = _verified_surya_quiet_evidence(
                row=surya,
                engine_dir=engine_dirs["surya"],
                source_page=source_page,
                expected_outcome=surya_outcome,
                text_payload_override=original_text["surya"],
                geometry_payload_override=original_geometry["surya"],
            )
            sparse_surya = (
                surya_attempt == 1
                and surya_alnum_lines <= _SPARSE_SURYA_GRAPHICS_MAX_ALNUM_LINES
                and surya_alnum_chars <= _SPARSE_SURYA_GRAPHICS_MAX_ALNUM_CHARS
            )
            exhausted_chandra = (
                chandra_attempt == 3
                and not surya_errors
                and [item["code"] for item in chandra_errors] == ["zero_output"]
            )
            if not sparse_surya or not exhausted_chandra:
                raise RuntimeError(f"Page {source_page} forged sparse-graphics acceptance")
            reason = "exhausted_chandra_zero_with_sparse_surya"
            is_graphics = True
            recomputed_graphics.append(source_page)
            reconciled_counts["surya"] += len(surya_errors)
            reconciled_counts["chandra"] += len(chandra_errors)
        elif chandra_outcome == "explicit_nontext" and chandra.get("explicit_nontext") is True:
            surya_alnum_lines, surya_alnum_chars = _verified_surya_quiet_evidence(
                row=surya,
                engine_dir=engine_dirs["surya"],
                source_page=source_page,
                expected_outcome=surya_outcome,
                text_payload_override=original_text["surya"],
                geometry_payload_override=original_geometry["surya"],
            )
            expected_errors = [item["code"] for item in chandra_errors] == ["zero_output"] and (
                (
                    surya_outcome == "zero_output"
                    and [item["code"] for item in surya_errors] == ["zero_output"]
                )
                or (surya_outcome == "text" and not surya_errors)
            )
            quiet_surya = (
                surya_outcome in {"zero_output", "text"}
                and surya_alnum_lines <= 1
                and surya_alnum_chars <= 8
            )
            if not expected_errors or not quiet_surya:
                raise RuntimeError(f"Page {source_page} forged textless-graphics acceptance")
            reason = "explicit_chandra_nontext_with_quiet_surya"
            is_graphics = True
            recomputed_graphics.append(source_page)
            reconciled_counts["surya"] += len(surya_errors)
            reconciled_counts["chandra"] += len(chandra_errors)
        else:
            raise RuntimeError(f"Page {source_page} final reconciliation is not accepted")

        if not is_graphics and (surya_errors or chandra_errors):
            raise RuntimeError(f"Page {source_page} has unrelated accepted page errors")
        expected_row: dict[str, object] = {
            "source_page": source_page,
            "surya_outcome": surya_outcome,
            "chandra_outcome": chandra_outcome,
            "surya_alnum_line_count": surya_alnum_lines,
            "surya_alnum_chars": surya_alnum_chars,
            "surya_page_error_count": len(surya_errors),
            "chandra_page_error_count": len(chandra_errors),
            "accepted": True,
            "reason": reason,
        }
        if retry_agreement is not None:
            expected_row["retry_text_agreement"] = retry_agreement
        expected_rows.append(expected_row)

    recomputed_set = frozenset(recomputed_graphics)
    if recomputed_set != claimed_accepted:
        raise RuntimeError("Completed reconciliation accepted-page claim was not recomputed")
    graphics_by_window: dict[int, int] = {}
    for source_page in recomputed_graphics:
        window = (source_page - 1) // 10
        graphics_by_window[window] = graphics_by_window.get(window, 0) + 1
    if any(count > 1 for count in graphics_by_window.values()):
        raise RuntimeError("Completed reconciliation exceeds the local graphics recovery cap")
    if report.get("pages") != expected_rows:
        raise RuntimeError("Completed reconciliation page decisions are not canonical")
    if report.get("reconciled_page_error_counts") != reconciled_counts:
        raise RuntimeError("Completed reconciliation error counters were not recomputed")
    result_text_chars = {
        engine: sum(
            len(payload.decode("utf-8")) for payload in final_text_payloads[engine].values()
        )
        for engine in ("surya", "chandra")
    }
    if report.get("result_text_chars") != result_text_chars:
        raise RuntimeError("Completed reconciliation result text counters were not recomputed")
    return recomputed_set


def _reconcile_mode_both_pages(
    *,
    run_dir: Path,
    results: list[OcrBenchmarkResult],
    result_files: Sequence[Path],
) -> tuple[list[OcrBenchmarkResult], str | None]:
    result_by_engine = {
        result.engine: result
        for result in results
        if result.status in {"ok", _OCR_STATUS_RECONCILIATION_PENDING}
    }
    if set(result_by_engine) != {"surya", "chandra"}:
        return results, "both successful engine results are required"
    try:
        surya_path, surya_payload, surya_pages = _load_engine_page_index(
            run_dir=run_dir,
            engine="surya",
        )
        chandra_path, chandra_payload, chandra_pages = _load_engine_page_index(
            run_dir=run_dir,
            engine="chandra",
        )
        report_by_engine = _load_benchmark_report_index(
            run_dir=run_dir,
            result_files=result_files,
            expected_engines={"surya", "chandra"},
        )
    except RuntimeError as exc:
        report: dict[str, object] = {
            "schema": "uniscan.page-reconciliation.v1",
            "status": "error",
            "error": str(exc),
        }
        _write_json_atomic(run_dir / "page_reconciliation.json", report)
        return results, str(exc)

    if set(surya_pages) != set(chandra_pages):
        error = (
            "engine page evidence is not bijective: "
            f"surya={sorted(surya_pages)}, chandra={sorted(chandra_pages)}"
        )
        _write_json_atomic(
            run_dir / "page_reconciliation.json",
            {"schema": "uniscan.page-reconciliation.v1", "status": "error", "error": error},
        )
        return results, error

    error_records_by_engine: dict[str, dict[int, list[dict[str, str]]]] = {}
    try:
        for engine, page_index in (
            ("surya", surya_pages),
            ("chandra", chandra_pages),
        ):
            records_by_page = {
                source_page: _page_error_records(
                    row,
                    engine=engine,
                    source_page=source_page,
                )
                for source_page, row in page_index.items()
            }
            recorded_count = sum(len(records) for records in records_by_page.values())
            expected_count = max(0, int(result_by_engine[engine].page_error_count))
            if recorded_count != expected_count:
                raise RuntimeError(
                    f"{engine} page-error evidence count mismatch: "
                    f"report={expected_count}, pages={recorded_count}"
                )
            error_records_by_engine[engine] = records_by_page
    except RuntimeError as exc:
        error = str(exc)
        _write_json_atomic(
            run_dir / "page_reconciliation.json",
            {
                "schema": "uniscan.page-reconciliation.v1",
                "status": "error",
                "exact_page_bijection": True,
                "error": error,
            },
        )
        return results, error

    rows: list[dict[str, Any]] = []
    accepted_graphics: list[int] = []
    unresolved: list[int] = []
    ordered_pages = sorted(surya_pages)
    for source_page in ordered_pages:
        surya = surya_pages[source_page]
        chandra = chandra_pages[source_page]
        surya_outcome = str(surya.get("ocr_outcome") or "")
        chandra_outcome = str(chandra.get("ocr_outcome") or "")
        surya_errors = error_records_by_engine["surya"][source_page]
        chandra_errors = error_records_by_engine["chandra"][source_page]
        reason = ""
        accepted = False
        retry_evidence_error: str | None = None
        retry_text_agreement: dict[str, object] | None = None
        surya_alnum_lines = surya.get("alnum_line_count")
        surya_alnum_chars = surya.get("alnum_chars")
        try:
            surya_attempt, surya_source_identity = _strict_surya_attempt_metadata(
                row=surya,
                engine_dir=surya_path.parent.resolve(),
                source_page=source_page,
            )
        except RuntimeError as exc:
            reason = (
                "invalid_retry_text_evidence"
                if surya_outcome == "text" and chandra_outcome == "text"
                else "invalid_surya_attempt_evidence"
            )
            retry_evidence_error = str(exc)
            surya_attempt = None
            surya_source_identity = None
        if not reason:
            try:
                chandra_attempt, chandra_evidence = _strict_chandra_attempt_metadata(
                    row=chandra,
                    engine_dir=chandra_path.parent.resolve(),
                    source_page=source_page,
                )
            except RuntimeError as exc:
                reason = "invalid_chandra_attempt_evidence"
                retry_evidence_error = str(exc)
                chandra_attempt = None
                chandra_source_identity = None
            else:
                chandra_source_identity = chandra_evidence["source_raster_identity"]
        if not reason and (
            surya_source_identity is None
            or chandra_source_identity is None
            or surya_source_identity != chandra_source_identity
        ):
            reason = "source_raster_identity_mismatch"
            retry_evidence_error = (
                f"source page {source_page} Surya/Chandra raster identities disagree"
            )
        if (
            not reason
            and surya_attempt == 3
            and not (surya_outcome == "text" and chandra_outcome == "text")
        ):
            if (
                surya_outcome == "zero_output"
                and chandra_outcome == "explicit_nontext"
                and chandra.get("explicit_nontext") is True
            ):
                try:
                    _strict_surya_third_zero_provenance(
                        row=surya,
                        engine_dir=surya_path.parent.resolve(),
                        source_page=source_page,
                    )
                except RuntimeError as exc:
                    reason = f"invalid_candidate_evidence: {exc}"
            else:
                reason = "third_retry_nontext_forbidden"
        if not reason and surya_outcome == "text" and chandra_outcome == "text":
            try:
                attempt, surya_canonical, surya_geometry = _surya_text_attempt(
                    row=surya,
                    engine_dir=surya_path.parent.resolve(),
                    source_page=source_page,
                    validated_attempt=surya_attempt,
                )
            except RuntimeError as exc:
                reason = "invalid_retry_text_evidence"
                retry_evidence_error = str(exc)
            else:
                if attempt in {1, 2} and chandra_attempt in {1, 2}:
                    reason = "both_text"
                    accepted = True
                else:
                    try:
                        if surya_canonical is None or surya_geometry is None:
                            (
                                surya_canonical,
                                surya_geometry,
                                _surya_image,
                            ) = _strict_durable_text_evidence(
                                row=surya,
                                engine_dir=surya_path.parent.resolve(),
                                engine="surya",
                                source_page=source_page,
                            )
                        (
                            chandra_canonical,
                            chandra_geometry,
                            _chandra_image,
                        ) = _strict_durable_text_evidence(
                            row=chandra,
                            engine_dir=chandra_path.parent.resolve(),
                            engine="chandra",
                            source_page=source_page,
                        )
                        if (
                            surya_geometry is None
                            or chandra_geometry.image_name != surya_geometry.image_name
                        ):
                            raise RuntimeError(
                                "Chandra page identity disagrees with sealed Surya text evidence"
                            )
                    except RuntimeError as exc:
                        reason = "invalid_retry_text_evidence"
                        retry_evidence_error = str(exc)
                    else:
                        assert surya_canonical is not None
                        retry_text_agreement = {
                            "algorithm": _RETRY_TEXT_AGREEMENT_ALGORITHM,
                            "matched": surya_canonical == chandra_canonical,
                            "surya_sha256": hashlib.sha256(
                                surya_canonical.encode("utf-8")
                            ).hexdigest(),
                            "chandra_sha256": hashlib.sha256(
                                chandra_canonical.encode("utf-8")
                            ).hexdigest(),
                        }
                        if surya_canonical == chandra_canonical:
                            reason = "both_text_retry_geometry_agreement"
                            accepted = True
                        else:
                            reason = "retry_text_mismatch"
        elif (
            not reason and surya_outcome == "verified_blank" and chandra_outcome == "verified_blank"
        ):
            reason = "both_verified_blank"
            accepted = True
        elif (
            not reason
            and surya_outcome == "text"
            and chandra_outcome == "zero_output"
            and surya_attempt == 1
            and chandra_attempt == 3
        ):
            try:
                surya_alnum_lines, surya_alnum_chars = _verified_surya_quiet_evidence(
                    row=surya,
                    engine_dir=surya_path.parent.resolve(),
                    source_page=source_page,
                    expected_outcome=surya_outcome,
                )
            except RuntimeError as exc:
                reason = f"invalid_candidate_evidence: {exc}"
            else:
                expected_errors = (
                    not surya_errors
                    and [item["code"] for item in chandra_errors] == ["zero_output"]
                )
                sparse_surya = (
                    surya_alnum_lines <= _SPARSE_SURYA_GRAPHICS_MAX_ALNUM_LINES
                    and surya_alnum_chars <= _SPARSE_SURYA_GRAPHICS_MAX_ALNUM_CHARS
                )
                if expected_errors and sparse_surya:
                    reason = "exhausted_chandra_zero_with_sparse_surya"
                    accepted = True
                    accepted_graphics.append(source_page)
                elif not expected_errors:
                    reason = "unrelated_page_error"
                else:
                    reason = "exhausted_chandra_zero_with_dense_surya"
        elif (
            not reason
            and chandra_outcome == "explicit_nontext"
            and chandra.get("explicit_nontext") is True
        ):
            try:
                surya_alnum_lines, surya_alnum_chars = _verified_surya_quiet_evidence(
                    row=surya,
                    engine_dir=surya_path.parent.resolve(),
                    source_page=source_page,
                    expected_outcome=surya_outcome,
                )
                verified_chandra = _verified_chandra_explicit_nontext(
                    row=chandra,
                    engine_dir=chandra_path.parent.resolve(),
                    source_page=source_page,
                )
            except RuntimeError as exc:
                reason = f"invalid_candidate_evidence: {exc}"
            else:
                surya_error_codes = [item["code"] for item in surya_errors]
                chandra_error_codes = [item["code"] for item in chandra_errors]
                expected_errors = chandra_error_codes == ["zero_output"] and (
                    (surya_outcome == "zero_output" and surya_error_codes == ["zero_output"])
                    or (surya_outcome == "text" and not surya_error_codes)
                )
                quiet_surya = (
                    surya_outcome in {"zero_output", "text"}
                    and surya_alnum_lines <= 1
                    and surya_alnum_chars <= 8
                )
                if verified_chandra and quiet_surya and expected_errors:
                    reason = "explicit_chandra_nontext_with_quiet_surya"
                    accepted = True
                    accepted_graphics.append(source_page)
                elif not expected_errors:
                    reason = "unrelated_page_error"
                elif not verified_chandra:
                    reason = "invalid_chandra_nontext_evidence"
                else:
                    reason = "explicit_chandra_nontext_with_dense_surya"
        if accepted and source_page not in accepted_graphics and (surya_errors or chandra_errors):
            accepted = False
            reason = "unrelated_page_error"

        if not accepted:
            reason = reason or "unresolved_engine_outcome"
            unresolved.append(source_page)
        reconciliation_row: dict[str, object] = {
            "source_page": source_page,
            "surya_outcome": surya_outcome,
            "chandra_outcome": chandra_outcome,
            "surya_alnum_line_count": surya_alnum_lines,
            "surya_alnum_chars": surya_alnum_chars,
            "surya_page_error_count": len(surya_errors),
            "chandra_page_error_count": len(chandra_errors),
            "accepted": accepted,
            "reason": reason,
        }
        if retry_text_agreement is not None:
            reconciliation_row["retry_text_agreement"] = retry_text_agreement
        if retry_evidence_error is not None:
            reconciliation_row["retry_evidence_error"] = retry_evidence_error
        rows.append(reconciliation_row)

    row_by_page = {int(row["source_page"]): row for row in rows}
    recovered_by_chunk: dict[int, list[int]] = {}
    for source_page in accepted_graphics:
        recovered_by_chunk.setdefault((source_page - 1) // 10, []).append(source_page)
    for recovered in recovered_by_chunk.values():
        if len(recovered) > 1:
            for source_page in recovered:
                row_by_page[source_page]["accepted"] = False
                row_by_page[source_page]["reason"] = "graphics_recovery_cap_exceeded"
                if source_page not in unresolved:
                    unresolved.append(source_page)
            accepted_graphics = [
                source_page for source_page in accepted_graphics if source_page not in recovered
            ]
    report = {
        "schema": "uniscan.page-reconciliation.v1",
        "status": "error" if unresolved else "pending",
        "exact_page_bijection": True,
        "accepted_textless_graphics_pages": accepted_graphics,
        "unresolved_pages": sorted(unresolved),
        "pages": rows,
    }
    reconciliation_path = run_dir / "page_reconciliation.json"
    if unresolved:
        error = f"unresolved pages: {sorted(unresolved)}"
        report["error"] = error
        _write_json_atomic(reconciliation_path, report)
        return results, error

    accepted_set = set(accepted_graphics)
    try:
        original_evidence = {
            str(source_page): {
                "surya": _snapshot_original_page_evidence(
                    row=surya_pages[source_page],
                    engine_dir=surya_path.parent.resolve(),
                    source_page=source_page,
                ),
                "chandra": _snapshot_original_page_evidence(
                    row=chandra_pages[source_page],
                    engine_dir=chandra_path.parent.resolve(),
                    source_page=source_page,
                ),
            }
            for source_page in sorted(accepted_set)
        }
    except Exception as exc:
        error = f"original reconciliation evidence could not be sealed: {exc}"
        report["status"] = "error"
        report["error"] = error
        _write_json_atomic(reconciliation_path, report)
        return results, error
    report["reconciliation_original_evidence"] = original_evidence
    reconciled_error_counts = {
        engine: sum(
            len(error_records_by_engine[engine][source_page]) for source_page in accepted_set
        )
        for engine in ("surya", "chandra")
    }
    _write_json_atomic(reconciliation_path, report)
    try:
        updates: dict[Path, bytes] = {}
        if accepted_set:
            surya_chars, surya_updates = _stage_textless_graphics_artifacts(
                pages_path=surya_path,
                payload=surya_payload,
                accepted_pages=accepted_set,
                aggregate_path=Path(str(result_by_engine["surya"].artifact_path)),
            )
            chandra_chars, chandra_updates = _stage_textless_graphics_artifacts(
                pages_path=chandra_path,
                payload=chandra_payload,
                accepted_pages=accepted_set,
                aggregate_path=Path(str(result_by_engine["chandra"].artifact_path)),
            )
            rewritten_chars = {"surya": surya_chars, "chandra": chandra_chars}
            for staged_updates in (surya_updates, chandra_updates):
                for raw_path, data in staged_updates.items():
                    target = raw_path.resolve()
                    if target in updates:
                        raise RuntimeError(f"Duplicate staged artifact: {target}")
                    updates[target] = data
        else:
            rewritten_chars = {}
            for engine, pages_path, page_index in (
                ("surya", surya_path, surya_pages),
                ("chandra", chandra_path, chandra_pages),
            ):
                engine_chars = 0
                for source_page, page_row in page_index.items():
                    page_file = _owned_page_artifact(
                        engine_dir=pages_path.parent.resolve(),
                        raw_name=page_row.get("file"),
                        label=f"{engine} page {source_page} text",
                    )
                    payload, _fingerprint = _stable_file_bytes(
                        page_file,
                        max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
                    )
                    engine_chars += len(payload.decode("utf-8"))
                rewritten_chars[engine] = engine_chars
        adjusted = [
            replace(
                result,
                status="ok",
                text_chars=rewritten_chars.get(result.engine, result.text_chars),
                page_error_count=max(
                    0,
                    int(result.page_error_count) - reconciled_error_counts.get(result.engine, 0),
                ),
                note=(
                    f"accepted textless graphics pages: {accepted_graphics}"
                    if accepted_graphics
                    else result.note
                ),
            )
            for result in results
        ]
        residual_errors = {
            result.engine: result.page_error_count
            for result in adjusted
            if result.page_error_count != 0
        }
        if residual_errors:
            raise RuntimeError(f"unreconciled page errors remain: {residual_errors}")
        for raw_path, data in _stage_benchmark_reports(
            report_by_engine=report_by_engine,
            results=adjusted,
        ).items():
            target = raw_path.resolve()
            if target in updates:
                raise RuntimeError(f"Duplicate staged report: {target}")
            updates[target] = data
        report["status"] = "ok"
        report["reconciled_page_error_counts"] = reconciled_error_counts
        report["result_text_chars"] = rewritten_chars
        updates[reconciliation_path.resolve()] = _json_bytes(report)
        _publish_file_transaction(updates)
        _validate_final_mode_both_state(run_dir=run_dir)
    except Exception as exc:
        error = f"artifact reconciliation failed: {exc}"
        report["status"] = "error"
        report["error"] = error
        try:
            _write_json_atomic(reconciliation_path, report)
        except OSError:
            pass
        return results, error
    return adjusted, None


def normalize_pdf_mode(raw: str | None) -> str:
    normalized = (raw or "").strip().lower()
    if normalized == "chandra surya":
        # A literal '+' in a query string is decoded as a space.
        normalized = PDF_MODE_HYBRID
    if not normalized:
        return PDF_MODE_HYBRID
    if normalized in {PDF_MODE_HYBRID, MODE_HYBRID, "both"}:
        return PDF_MODE_HYBRID
    raise ValueError("Unsupported mode. Production OCR requires chandra+surya.")


def _mode_to_benchmark_key(mode: str) -> str:
    if mode == PDF_MODE_HYBRID:
        return MODE_BOTH
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _mode_to_prepare_engines(mode: str) -> tuple[str, ...]:
    if mode == PDF_MODE_HYBRID:
        return ("chandra", "surya")
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _mode_to_build_engines(mode: str) -> tuple[str, ...]:
    if mode == PDF_MODE_HYBRID:
        # Hybrid output is one PDF: chandra text aligned to surya geometry.
        return ("chandra",)
    raise ValueError(f"Unsupported normalized mode: {mode}")


def _pick_ok_pdf(results: tuple[ArtifactSearchableResult, ...]) -> Path:
    for item in results:
        if item.status != "ok":
            continue
        raw = (item.searchable_pdf_path or "").strip()
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    raise RuntimeError("No successful searchable PDF output was produced.")


def _resolve_textless_dpi() -> int:
    raw = os.getenv("UNISCAN_TEXTLESS_DPI", "").strip()
    if not raw:
        return 300
    try:
        value = int(raw)
    except ValueError:
        return 300
    return max(72, min(400, value))


def _resolve_textless_jpeg_quality() -> int:
    raw = (os.environ.get("UNISCAN_TEXTLESS_JPEG_QUALITY") or "").strip()
    if not raw:
        return 85
    try:
        value = int(raw)
    except ValueError:
        return 85
    if value <= 0:
        return 0
    return min(value, 100)


def _resolve_hybrid_chunk_pages() -> int:
    raw = (os.environ.get("UNISCAN_HYBRID_CHUNK_PAGES") or "").strip()
    if not raw:
        return _DEFAULT_HYBRID_CHUNK_PAGES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_HYBRID_CHUNK_PAGES
    return max(0, min(100, value))


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF chunking requires PyMuPDF.") from exc

    document = fitz.open(str(pdf_path))
    try:
        return int(document.page_count)
    finally:
        document.close()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _stable_file_fingerprint(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while True:
            block = stream.read(_FILE_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = resolved.stat()
    if not (_stat_signature(before) == _stat_signature(after) == _stat_signature(current)):
        raise RuntimeError(f"File changed while it was being fingerprinted: {resolved}")
    return {
        "sha256": digest.hexdigest(),
        "size": int(after.st_size),
    }


def _stable_file_bytes(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    lexical = Path(path)
    with lexical.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size < 0 or (max_bytes is not None and before.st_size > max_bytes):
            raise ValueError(
                f"File size {before.st_size} is outside the permitted range: {lexical}"
            )
        payload = stream.read()
        after = os.fstat(stream.fileno())
    current = lexical.stat()
    if not (_stat_signature(before) == _stat_signature(after) == _stat_signature(current)):
        raise RuntimeError(f"File changed while it was being read: {lexical}")
    if len(payload) != int(after.st_size):
        raise RuntimeError(f"File read was incomplete: {lexical}")
    return payload, {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": int(after.st_size),
    }


def _stable_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_CHUNK_MANIFEST_BYTES,
) -> tuple[dict[str, Any], dict[str, object]]:
    payload_bytes, fingerprint = _stable_file_bytes(path, max_bytes=max_bytes)
    if not payload_bytes:
        raise ValueError(f"{label} is empty")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, fingerprint


def _resolve_hybrid_run_config() -> _ResolvedHybridRunConfig:
    return _ResolvedHybridRunConfig(
        effective_ocr_render_dpi=_resolve_ocr_render_dpi(),
        effective_textless_dpi=_resolve_textless_dpi(),
        effective_textless_jpeg_quality=_resolve_textless_jpeg_quality(),
        effective_engine_subprocess_timeout_seconds=_engine_subprocess_timeout_seconds(),
        environment=tuple(
            (key, os.environ.get(key)) for key in _HYBRID_IDENTITY_ENV_KEYS
        ),
    )


def _hybrid_runtime_config() -> dict[str, object]:
    return _resolve_hybrid_run_config().as_identity()


def _hybrid_run_identity(
    *,
    input_path: Path,
    mode: str,
    lang: str,
    strict: bool,
    delete_original_text_layer: bool,
    chunk_pages: int,
    page_count: int,
    runtime_config: _ResolvedHybridRunConfig | None = None,
) -> tuple[dict[str, object], str]:
    identity: dict[str, object] = {
        "pipeline_revision": _HYBRID_CHUNK_PIPELINE_REVISION,
        "source": _stable_file_fingerprint(input_path),
        "mode": mode,
        "lang": lang,
        "strict": bool(strict),
        "delete_original_text_layer": bool(delete_original_text_layer),
        "chunk_pages": int(chunk_pages),
        "page_count": int(page_count),
        "runtime_config": (
            runtime_config.as_identity()
            if runtime_config is not None
            else _hybrid_runtime_config()
        ),
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return identity, hashlib.sha256(serialized).hexdigest()


def _resolve_hybrid_chunk_cache_root(
    *,
    work_root: Path,
    configured_root: Path | None,
) -> Path:
    if configured_root is not None:
        return Path(configured_root).resolve()
    return (Path(work_root) / "hybrid_chunk_cache").resolve()


@contextmanager
def _hybrid_run_lock(run_key: str) -> Iterator[None]:
    with _HYBRID_RUN_LOCKS_GUARD:
        lock = _HYBRID_RUN_LOCKS.setdefault(run_key, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _read_chunk_manifest(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not os.path.lexists(path):
        return None, None
    try:
        lexical = _strict_lexical_absolute_path(path, label="chunk manifest")
        if lexical.name != "chunk_manifest.json":
            raise ValueError("chunk manifest path is not root-exact")
        lexical = _assert_owned_regular_file(
            lexical,
            lexical.parent,
            label="chunk manifest",
        )
        payload, _fingerprint = _stable_json_object(
            lexical,
            label="chunk manifest",
            max_bytes=_MAX_CHUNK_MANIFEST_BYTES,
        )
        return payload, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _source_page_sizes(source_pdf: Path) -> list[tuple[float, float]]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF validation requires PyMuPDF.") from exc
    document = fitz.open(str(source_pdf))
    try:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]
    finally:
        document.close()


def _rendered_source_raster_identities(
    source_pdf: Path,
    *,
    expected_pages: int,
) -> dict[int, dict[str, object]]:
    identities: dict[int, dict[str, object]] = {}
    try:
        rendered_pages = iter_render_pdf_page_indices(
            source_pdf,
            range(expected_pages),
            dpi=_resolve_ocr_render_dpi(),
        )
        for local_page, (_name, bgr_image) in enumerate(rendered_pages, start=1):
            if (
                getattr(bgr_image, "ndim", None) != 3
                or len(bgr_image.shape) != 3
                or int(bgr_image.shape[2]) != 3
            ):
                raise RuntimeError(
                    f"Rendered processing source page {local_page} is not canonical BGR"
                )
            rgb_image = Image.fromarray(bgr_image[:, :, ::-1].copy())
            identities[local_page] = {
                "pixel_sha256": _canonical_rgb_pixel_sha256(rgb_image),
                "width": int(rgb_image.width),
                "height": int(rgb_image.height),
                "name": f"{local_page:05d}.png",
                "source_page": local_page,
                "verified_blank": _is_effectively_blank_rgb_image(rgb_image),
            }
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Processing source PDF could not be rendered: {exc}") from exc
    if set(identities) != set(range(1, expected_pages + 1)):
        raise RuntimeError("Processing source PDF render is not page-bijective")
    return identities


def _validate_processing_source_derivation(
    *,
    chunk_pdf: Path,
    processing_source: Path,
    outer_root: Path,
    expected_pages: int,
    delete_original_text_layer: bool,
) -> None:
    if not delete_original_text_layer:
        if processing_source != chunk_pdf:
            raise RuntimeError(
                "Processing source must exactly equal the fresh chunk when text deletion is disabled"
            )
        return
    if processing_source == chunk_pdf:
        raise RuntimeError("Processing source must be textless when text deletion is enabled")
    reference_pdf = outer_root / (
        f".{chunk_pdf.stem}.{uuid.uuid4().hex}.textless-source-validation.pdf"
    )
    try:
        _build_textless_source_pdf(
            source_pdf=chunk_pdf,
            out_pdf=reference_pdf,
            dpi=_resolve_textless_dpi(),
        )
        reference_text = _extract_pdf_page_texts(reference_pdf)
        processing_text = _extract_pdf_page_texts(processing_source)
        if (
            len(reference_text) != expected_pages
            or len(processing_text) != expected_pages
            or any(value.strip() for value in reference_text)
            or any(value.strip() for value in processing_text)
        ):
            raise RuntimeError(
                "Derived processing source must be page-bijective and contain no native text"
            )
        _validate_textless_visual_retention(
            source_pdf=reference_pdf,
            candidate_pdf=processing_source,
            expected_pages=frozenset(range(1, expected_pages + 1)),
        )
    finally:
        reference_pdf.unlink(missing_ok=True)


def _accepted_textless_graphics_pages(
    reconciliation: Mapping[str, object],
    *,
    expected_pages: int,
) -> frozenset[int]:
    raw_pages = reconciliation.get("accepted_textless_graphics_pages")
    if not isinstance(raw_pages, list):
        raise RuntimeError("Page reconciliation has no accepted textless-graphics page list")
    if any(
        not isinstance(page, int) or isinstance(page, bool) or page < 1 or page > expected_pages
        for page in raw_pages
    ):
        raise RuntimeError("Accepted textless-graphics page list is invalid")
    accepted = frozenset(raw_pages)
    if len(accepted) != len(raw_pages):
        raise RuntimeError("Accepted textless-graphics page list contains duplicates")
    return accepted


def _validate_chunk_pdf(
    *,
    chunk: _PdfChunk,
    chunk_pdf: Path,
    source_sizes: list[tuple[float, float]],
) -> int:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF validation requires PyMuPDF.") from exc
    expected_pages = chunk.end_page - chunk.start_page + 1
    document = fitz.open(str(chunk_pdf))
    try:
        if int(document.page_count) != expected_pages:
            raise RuntimeError(
                f"OCR chunk {chunk.index} has {document.page_count} pages; "
                f"expected {expected_pages}."
            )
        for local_index, page in enumerate(document):
            source_index = chunk.start_page - 1 + local_index
            expected_width, expected_height = source_sizes[source_index]
            if (
                abs(float(page.rect.width) - expected_width) > 0.5
                or abs(float(page.rect.height) - expected_height) > 0.5
            ):
                raise RuntimeError(
                    f"OCR chunk page size changed for source page {source_index + 1}."
                )
        return int(document.page_count)
    finally:
        document.close()


def _split_pdf_chunks(
    *,
    source_pdf: Path,
    output_root: Path,
    pages_per_chunk: int,
) -> list[_PdfChunk]:
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be positive.")
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF chunking requires PyMuPDF.") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    source = fitz.open(str(source_pdf))
    chunks: list[_PdfChunk] = []
    try:
        total_pages = int(source.page_count)
        for index, start_zero in enumerate(range(0, total_pages, pages_per_chunk), start=1):
            end_zero = min(start_zero + pages_per_chunk, total_pages) - 1
            start_page = start_zero + 1
            end_page = end_zero + 1
            path = output_root / (f"chunk_{index:04d}_p{start_page:04d}_{end_page:04d}.pdf")
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.pdf")
            chunk_document = fitz.open()
            try:
                chunk_document.insert_pdf(
                    source,
                    from_page=start_zero,
                    to_page=end_zero,
                    links=True,
                    annots=True,
                )
                chunk_document.save(
                    str(temporary),
                    garbage=4,
                    deflate=True,
                    no_new_id=True,
                )
            finally:
                chunk_document.close()
            try:
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            chunks.append(
                _PdfChunk(
                    index=index,
                    start_page=start_page,
                    end_page=end_page,
                    path=path,
                )
            )
    finally:
        source.close()
    return chunks


def _stable_snapshot_chunk_output(
    *,
    source: Path,
    target: Path,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_stream, temporary.open("wb") as target_stream:
            before = os.fstat(source_stream.fileno())
            while True:
                block = source_stream.read(_FILE_HASH_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
                target_stream.write(block)
            after = os.fstat(source_stream.fileno())
            target_stream.flush()
            os.fsync(target_stream.fileno())
        current = source.stat()
        if not (_stat_signature(before) == _stat_signature(after) == _stat_signature(current)):
            raise RuntimeError(f"Chunk output changed while snapshotting: {source}")
        if int(after.st_size) != expected_size or digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Chunk output fingerprint changed before merge: {source}")
        os.replace(temporary, target)
        if _stable_file_fingerprint(target) != {
            "sha256": expected_sha256,
            "size": expected_size,
        }:
            raise RuntimeError(f"Private chunk merge snapshot is not exact: {target}")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _validated_required_fingerprint(
    required: _RequiredChunkEvidence,
    path: Path,
    observed: Mapping[str, object],
) -> None:
    expected = {
        candidate: (sha256, size)
        for candidate, sha256, size in required.validated_file_fingerprints
    }.get(path)
    actual = (str(observed["sha256"]), int(str(observed["size"])))
    if expected is None or expected != actual:
        raise RuntimeError(f"Validated semantic evidence changed before merge: {path}")


def _validated_required_outer_fingerprint(
    required: _RequiredChunkEvidence,
    path: Path,
    observed: Mapping[str, object],
) -> None:
    expected = {
        candidate: (sha256, size)
        for candidate, sha256, size in required.validated_outer_file_fingerprints
    }.get(path)
    actual = (str(observed["sha256"]), int(str(observed["size"])))
    if expected is None or expected != actual:
        raise RuntimeError(f"Validated outer semantic evidence changed before merge: {path}")


def _premerge_chunk_state(
    *,
    chunk: _PdfChunk,
    record: dict[str, object],
    summary: SearchablePdfSummary,
    required: _RequiredChunkEvidence,
    snapshot_root: Path,
) -> tuple[Path, tuple[str, ...], dict[int, dict[str, object]]]:
    expected_sha256 = record.get("output_sha256")
    expected_size = record.get("output_size")
    if (
        not _valid_sha256(expected_sha256)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
    ):
        raise RuntimeError(f"Chunk {chunk.index} output record is not sealed")
    source_output = _assert_owned_regular_file(
        summary.output_pdf_path,
        required.evidence_root,
        label=f"chunk {chunk.index} pre-merge output",
    )
    snapshot = _stable_snapshot_chunk_output(
        source=source_output,
        target=(
            snapshot_root / f"chunk_{chunk.index:04d}_{str(expected_sha256)[:16]}.snapshot.pdf"
        ),
        expected_sha256=str(expected_sha256),
        expected_size=expected_size,
    )

    engine_dir = required.evidence_root / "chandra" / "chandra"
    pages_path = engine_dir / "pages.json"
    pages_payload, pages_fingerprint = _stable_json_object(
        pages_path,
        label=f"chunk {chunk.index} Chandra pages index",
    )
    _validated_required_fingerprint(required, pages_path, pages_fingerprint)
    raw_pages = pages_payload.get("pages")
    if not isinstance(raw_pages, list) or not all(isinstance(row, dict) for row in raw_pages):
        raise RuntimeError(f"Chunk {chunk.index} Chandra pages index is malformed")
    rows = {int(row["source_page"]): row for row in raw_pages}
    local_page_count = chunk.end_page - chunk.start_page + 1
    if set(rows) != set(range(1, local_page_count + 1)):
        raise RuntimeError(f"Chunk {chunk.index} Chandra pages are not contiguous")
    page_texts: list[str] = []
    for local_page in range(1, local_page_count + 1):
        row = rows[local_page]
        text_path = _owned_engine_reference(
            engine_dir,
            row.get("file"),
            label=f"chunk {chunk.index} Chandra page {local_page} text",
        )
        text_payload, text_fingerprint = _stable_file_bytes(
            text_path,
            max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
        )
        _validated_required_fingerprint(required, text_path, text_fingerprint)
        try:
            page_texts.append(text_payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Chunk {chunk.index} Chandra page {local_page} text is not UTF-8"
            ) from exc

    from uniscan.ocr.artifact_searchable import _extract_pdf_page_texts

    snapshot_page_texts = _extract_pdf_page_texts(snapshot)
    if len(snapshot_page_texts) != local_page_count:
        raise RuntimeError(f"Chunk {chunk.index} private snapshot text page cardinality changed")
    page_texts = snapshot_page_texts
    artifact = summary.artifact_results[0]
    if artifact.source_pdf_path is None:
        raise RuntimeError(f"Chunk {chunk.index} has no processing source PDF")
    processing_source = _assert_owned_regular_file(
        Path(artifact.source_pdf_path),
        required.outer_root,
        label=f"chunk {chunk.index} processing source PDF",
    )
    source_before = _stable_file_fingerprint(processing_source)
    _validated_required_outer_fingerprint(required, processing_source, source_before)
    visual_seals: dict[int, dict[str, object]] = {}
    try:
        import fitz

        with fitz.open(str(processing_source)) as source_document:
            if int(source_document.page_count) != local_page_count:
                raise RuntimeError(f"Chunk {chunk.index} processing source page count changed")
            matrix = fitz.Matrix(1.0, 1.0)
            for local_page in sorted(required.accepted_textless_graphics_pages):
                page = source_document[local_page - 1]
                if page.get_text("text") != "":
                    raise RuntimeError(
                        f"Chunk {chunk.index} accepted graphics source page has native text"
                    )
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                global_page = chunk.start_page + local_page - 1
                visual_seals[global_page] = {
                    "rect": [float(value) for value in page.rect],
                    "rotation": int(page.rotation),
                    "width": int(pixmap.width),
                    "height": int(pixmap.height),
                    "components": int(pixmap.n),
                    "pixel_sha256": hashlib.sha256(pixmap.samples).hexdigest(),
                }
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Chunk {chunk.index} processing source cannot be sealed: {exc}"
        ) from exc
    source_after = _stable_file_fingerprint(processing_source)
    if source_after != source_before:
        raise RuntimeError(f"Chunk {chunk.index} processing source changed while rendering")
    _validated_required_outer_fingerprint(required, processing_source, source_after)
    return snapshot, tuple(page_texts), visual_seals


def _validate_merged_candidate(
    *,
    candidate_pdf: Path,
    expected_page_texts: Sequence[str],
    textless_visual_seals: Mapping[int, Mapping[str, object]],
) -> None:
    from uniscan.ocr.artifact_searchable import (
        _extract_pdf_page_texts,
        _validate_searchable_text_retention,
    )

    extracted_page_texts = _extract_pdf_page_texts(candidate_pdf)
    expected_textless = frozenset(textless_visual_seals)
    _validate_searchable_text_retention(
        source_page_texts=expected_page_texts,
        extracted_page_texts=extracted_page_texts,
        expected_textless_pages=expected_textless,
        expect_empty_text_layer=not any(text.strip() for text in expected_page_texts),
    )
    if not textless_visual_seals:
        return
    try:
        import fitz

        with fitz.open(str(candidate_pdf)) as candidate_document:
            if int(candidate_document.page_count) != len(expected_page_texts):
                raise RuntimeError("Merged OCR candidate page count changed")
            matrix = fitz.Matrix(1.0, 1.0)
            for global_page, expected in sorted(textless_visual_seals.items()):
                page = candidate_document[global_page - 1]
                if page.get_text("text") != "":
                    raise RuntimeError(
                        f"Merged accepted graphics page {global_page} gained native text"
                    )
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                observed = {
                    "rect": [float(value) for value in page.rect],
                    "rotation": int(page.rotation),
                    "width": int(pixmap.width),
                    "height": int(pixmap.height),
                    "components": int(pixmap.n),
                    "pixel_sha256": hashlib.sha256(pixmap.samples).hexdigest(),
                }
                if observed != dict(expected):
                    raise RuntimeError(
                        f"Merged accepted graphics page {global_page} visual content changed"
                    )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Merged OCR candidate visual validation failed: {exc}") from exc


def _merge_pdf_chunks(
    *,
    source_pdf: Path,
    chunks: list[tuple[_PdfChunk, Path]],
    output_pdf: Path,
    validate_candidate: Callable[[Path], None] | None = None,
    expected_source_fingerprint: Mapping[str, object] | None = None,
) -> Path:
    if not chunks:
        raise ValueError("No OCR chunks to merge.")
    if expected_source_fingerprint is not None and _stable_file_fingerprint(source_pdf) != dict(
        expected_source_fingerprint
    ):
        raise RuntimeError("Source PDF changed before chunk merge")
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF chunk merging requires PyMuPDF.") from exc

    ordered = sorted(chunks, key=lambda item: item[0].start_page)
    source = fitz.open(str(source_pdf))
    merged = fitz.open()
    temporary = output_pdf.with_name(f".{output_pdf.stem}.{uuid.uuid4().hex}.tmp.pdf")
    try:
        source_sizes = [(float(page.rect.width), float(page.rect.height)) for page in source]
        expected_next_page = 1
        for chunk, chunk_pdf in ordered:
            if chunk.start_page != expected_next_page:
                raise RuntimeError(
                    "OCR chunks are not contiguous: "
                    f"expected page {expected_next_page}, got {chunk.start_page}."
                )
            expected_pages = chunk.end_page - chunk.start_page + 1
            chunk_document = fitz.open(str(chunk_pdf))
            try:
                if int(chunk_document.page_count) != expected_pages:
                    raise RuntimeError(
                        f"OCR chunk {chunk.index} has {chunk_document.page_count} pages; "
                        f"expected {expected_pages}."
                    )
                for local_index, page in enumerate(chunk_document):
                    source_index = chunk.start_page - 1 + local_index
                    expected_width, expected_height = source_sizes[source_index]
                    if (
                        abs(float(page.rect.width) - expected_width) > 0.5
                        or abs(float(page.rect.height) - expected_height) > 0.5
                    ):
                        raise RuntimeError(
                            f"OCR chunk page size changed for source page {source_index + 1}."
                        )
                merged.insert_pdf(chunk_document, links=True, annots=True)
            finally:
                chunk_document.close()
            expected_next_page = chunk.end_page + 1

        if expected_next_page - 1 != len(source_sizes):
            raise RuntimeError(
                f"OCR chunks cover {expected_next_page - 1} of {len(source_sizes)} pages."
            )
        metadata = {
            key: str(value)
            for key, value in dict(source.metadata or {}).items()
            if value not in {None, ""}
        }
        if metadata:
            merged.set_metadata(metadata)
        try:
            toc = source.get_toc(simple=True)
            if toc:
                merged.set_toc(toc)
        except Exception:
            pass
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        merged.save(str(temporary), garbage=4, deflate=True)
    finally:
        source.close()
        merged.close()

    try:
        verification = fitz.open(str(temporary))
        try:
            if int(verification.page_count) != expected_next_page - 1:
                raise RuntimeError(
                    f"Merged OCR PDF has {verification.page_count} pages; "
                    f"expected {expected_next_page - 1}."
                )
        finally:
            verification.close()
        if validate_candidate is not None:
            validate_candidate(temporary)
        if expected_source_fingerprint is not None and _stable_file_fingerprint(source_pdf) != dict(
            expected_source_fingerprint
        ):
            raise RuntimeError("Source PDF changed during chunk merge")
        os.replace(temporary, output_pdf)
    finally:
        temporary.unlink(missing_ok=True)
    return output_pdf


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _chunk_summary_payload(summary: SearchablePdfSummary) -> dict[str, object]:
    return {
        "mode": summary.mode,
        "run_dir": str(summary.run_dir),
        "compare_dir": str(summary.compare_dir),
        "output_pdf_path": str(summary.output_pdf_path),
        "partial_page_failures": int(summary.partial_page_failures),
        "benchmark": {
            "run_dir": str(summary.benchmark.run_dir),
            "results": [asdict(item) for item in summary.benchmark.results],
            "result_files": [str(path) for path in summary.benchmark.result_files],
            "failed_engines": list(summary.benchmark.failed_engines),
            "skipped_engines": list(summary.benchmark.skipped_engines),
        },
        "compare_results": [asdict(item) for item in summary.compare_results],
        "artifact_results": [asdict(item) for item in summary.artifact_results],
    }


def _object_rows(value: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Chunk summary field {field!r} must be an array of objects.")
    return [dict(item) for item in value]


def _string_items(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Chunk summary field {field!r} must be an array of strings.")
    return tuple(value)


def _strict_object(
    value: object,
    *,
    field: str,
    keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Chunk summary field {field!r} has an invalid object shape.")
    return dict(value)


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Chunk summary field {field!r} must be a non-empty string.")
    return value


def _strict_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Chunk summary field {field!r} must be a string or null.")
    return value


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Chunk summary field {field!r} must be a non-negative integer.")
    return value


def _strict_positive_int(value: object, *, field: str) -> int:
    result = _strict_nonnegative_int(value, field=field)
    if result == 0:
        raise ValueError(f"Chunk summary field {field!r} must be positive.")
    return result


def _strict_finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Chunk summary field {field!r} must be a finite number.")
    return float(value)


def _strict_nonnegative_number(value: object, *, field: str) -> float:
    result = _strict_finite_number(value, field=field)
    if result < 0.0:
        raise ValueError(f"Chunk summary field {field!r} must be non-negative.")
    return result


def _strict_optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _strict_finite_number(value, field=field)


def _strict_status(value: object, *, field: str, allowed: frozenset[str]) -> str:
    result = _strict_string(value, field=field)
    if result not in allowed:
        raise ValueError(f"Chunk summary field {field!r} has an invalid status.")
    return result


def _strict_engine(value: object, *, field: str) -> str:
    result = _strict_string(value, field=field)
    if result not in {"surya", "chandra"}:
        raise ValueError(f"Chunk summary field {field!r} has an invalid engine.")
    return result


def _strict_sample_pages(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Chunk summary field {field!r} must be a non-empty array.")
    pages = [_strict_positive_int(item, field=field) for item in value]
    if len(set(pages)) != len(pages):
        raise ValueError(f"Chunk summary field {field!r} contains duplicate pages.")
    return pages


def _chunk_summary_from_payload(value: object) -> SearchablePdfSummary:
    summary_raw = _strict_object(
        value,
        field="summary",
        keys=frozenset(
            {
                "mode",
                "run_dir",
                "compare_dir",
                "output_pdf_path",
                "partial_page_failures",
                "benchmark",
                "compare_results",
                "artifact_results",
            }
        ),
    )
    benchmark_raw = _strict_object(
        summary_raw["benchmark"],
        field="benchmark",
        keys=frozenset(
            {
                "run_dir",
                "results",
                "result_files",
                "failed_engines",
                "skipped_engines",
            }
        ),
    )
    benchmark_results: list[OcrBenchmarkResult] = []
    benchmark_keys = frozenset(
        {
            "engine",
            "status",
            "sample_pages",
            "elapsed_seconds",
            "artifact_path",
            "text_chars",
            "memory_delta_mb",
            "error",
            "note",
            "page_error_count",
        }
    )
    for index, item in enumerate(_object_rows(benchmark_raw["results"], field="benchmark.results")):
        raw = _strict_object(
            item,
            field=f"benchmark.results[{index}]",
            keys=benchmark_keys,
        )
        benchmark_results.append(
            OcrBenchmarkResult(
                engine=_strict_engine(raw["engine"], field="benchmark.engine"),
                status=_strict_status(
                    raw["status"],
                    field="benchmark.status",
                    allowed=frozenset(
                        {"ok", "error", "skipped", _OCR_STATUS_RECONCILIATION_PENDING}
                    ),
                ),
                sample_pages=_strict_sample_pages(
                    raw["sample_pages"], field="benchmark.sample_pages"
                ),
                elapsed_seconds=_strict_nonnegative_number(
                    raw["elapsed_seconds"], field="benchmark.elapsed_seconds"
                ),
                artifact_path=_strict_optional_string(
                    raw["artifact_path"], field="benchmark.artifact_path"
                ),
                text_chars=_strict_nonnegative_int(raw["text_chars"], field="benchmark.text_chars"),
                memory_delta_mb=_strict_optional_number(
                    raw["memory_delta_mb"], field="benchmark.memory_delta_mb"
                ),
                error=_strict_optional_string(raw["error"], field="benchmark.error"),
                note=_strict_optional_string(raw["note"], field="benchmark.note"),
                page_error_count=_strict_nonnegative_int(
                    raw["page_error_count"], field="benchmark.page_error_count"
                ),
            )
        )
    benchmark = BasicOcrRunSummary(
        run_dir=Path(_strict_string(benchmark_raw["run_dir"], field="benchmark.run_dir")),
        results=tuple(benchmark_results),
        result_files=tuple(
            Path(path)
            for path in _string_items(
                benchmark_raw["result_files"],
                field="benchmark.result_files",
            )
        ),
        failed_engines=_string_items(
            benchmark_raw["failed_engines"],
            field="benchmark.failed_engines",
        ),
        skipped_engines=_string_items(
            benchmark_raw["skipped_engines"],
            field="benchmark.skipped_engines",
        ),
    )
    compare_results: list[CompareTxtBuildResult] = []
    compare_keys = frozenset(
        {"engine", "status", "source_artifact_path", "compare_txt_path", "error"}
    )
    for index, item in enumerate(
        _object_rows(summary_raw["compare_results"], field="compare_results")
    ):
        raw = _strict_object(
            item,
            field=f"compare_results[{index}]",
            keys=compare_keys,
        )
        compare_results.append(
            CompareTxtBuildResult(
                engine=_strict_engine(raw["engine"], field="compare.engine"),
                status=_strict_status(
                    raw["status"],
                    field="compare.status",
                    allowed=frozenset({"ok", "error"}),
                ),
                source_artifact_path=_strict_optional_string(
                    raw["source_artifact_path"], field="compare.source_artifact_path"
                ),
                compare_txt_path=_strict_optional_string(
                    raw["compare_txt_path"], field="compare.compare_txt_path"
                ),
                error=_strict_optional_string(raw["error"], field="compare.error"),
            )
        )
    artifact_results: list[ArtifactSearchableResult] = []
    artifact_keys = frozenset(
        {
            "document",
            "engine",
            "status",
            "source_pdf_path",
            "text_artifact_path",
            "searchable_pdf_path",
            "page_count",
            "text_chars",
            "elapsed_seconds",
            "error",
            "geometry_log_path",
            "warnings",
        }
    )
    for index, item in enumerate(
        _object_rows(summary_raw["artifact_results"], field="artifact_results")
    ):
        raw = _strict_object(
            item,
            field=f"artifact_results[{index}]",
            keys=artifact_keys,
        )
        artifact_results.append(
            ArtifactSearchableResult(
                document=_strict_string(raw["document"], field="artifact.document"),
                engine=_strict_engine(raw["engine"], field="artifact.engine"),
                status=_strict_status(
                    raw["status"],
                    field="artifact.status",
                    allowed=frozenset({"ok", "error"}),
                ),
                source_pdf_path=_strict_optional_string(
                    raw["source_pdf_path"], field="artifact.source_pdf_path"
                ),
                text_artifact_path=_strict_string(
                    raw["text_artifact_path"], field="artifact.text_artifact_path"
                ),
                searchable_pdf_path=_strict_optional_string(
                    raw["searchable_pdf_path"], field="artifact.searchable_pdf_path"
                ),
                page_count=_strict_nonnegative_int(raw["page_count"], field="artifact.page_count"),
                text_chars=_strict_nonnegative_int(raw["text_chars"], field="artifact.text_chars"),
                elapsed_seconds=_strict_nonnegative_number(
                    raw["elapsed_seconds"], field="artifact.elapsed_seconds"
                ),
                error=_strict_optional_string(raw["error"], field="artifact.error"),
                geometry_log_path=_strict_optional_string(
                    raw["geometry_log_path"], field="artifact.geometry_log_path"
                ),
                warnings=list(_string_items(raw["warnings"], field="artifact.warnings")),
            )
        )
    return SearchablePdfSummary(
        mode=_strict_string(summary_raw["mode"], field="mode"),
        run_dir=Path(_strict_string(summary_raw["run_dir"], field="run_dir")),
        compare_dir=Path(_strict_string(summary_raw["compare_dir"], field="compare_dir")),
        output_pdf_path=Path(
            _strict_string(summary_raw["output_pdf_path"], field="output_pdf_path")
        ),
        output_pdf_bytes=None,
        overwritten_input_path=None,
        benchmark=benchmark,
        compare_results=tuple(compare_results),
        artifact_results=tuple(artifact_results),
        partial_page_failures=_strict_nonnegative_int(
            summary_raw["partial_page_failures"], field="partial_page_failures"
        ),
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _strict_lexical_absolute_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeError(f"{label} path is not absolute: {candidate}")
    normalized = Path(os.path.abspath(candidate))
    if candidate != normalized:
        raise RuntimeError(f"{label} path is not lexically normalized: {candidate}")
    return candidate


def _lexical_relative_path(path: Path, root: Path, *, label: str) -> Path:
    candidate = _strict_lexical_absolute_path(path, label=label)
    owner = _strict_lexical_absolute_path(root, label=f"{label} owner")
    try:
        return candidate.relative_to(owner)
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped its owned directory: {candidate}") from exc


def _stat_is_reparse_point(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def _assert_link_free_components(path: Path, *, label: str) -> None:
    candidate = _strict_lexical_absolute_path(path, label=label)
    anchor = Path(candidate.anchor)
    current = anchor
    components = candidate.parts[1:]
    for component in components:
        current = current / component
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} path component is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(current_stat.st_mode) or _stat_is_reparse_point(current_stat):
            raise RuntimeError(f"{label} path traverses a link or reparse point: {current}")


def _assert_owned_directory(path: Path, root: Path, *, label: str) -> Path:
    candidate = _strict_lexical_absolute_path(path, label=label)
    _lexical_relative_path(candidate, root, label=label)
    _assert_link_free_components(candidate, label=label)
    try:
        current_stat = candidate.stat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} directory is unavailable: {candidate}: {exc}") from exc
    if not stat.S_ISDIR(current_stat.st_mode):
        raise RuntimeError(f"{label} is not a directory: {candidate}")
    if resolved != candidate:
        raise RuntimeError(f"{label} directory is not lexical-owner exact: {candidate}")
    return candidate


def _assert_owned_regular_file(path: Path, root: Path, *, label: str) -> Path:
    candidate = _strict_lexical_absolute_path(path, label=label)
    _lexical_relative_path(candidate, root, label=label)
    _assert_link_free_components(candidate, label=label)
    try:
        current_stat = candidate.stat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} file is unavailable: {candidate}: {exc}") from exc
    if not stat.S_ISREG(current_stat.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {candidate}")
    if int(current_stat.st_nlink) != 1:
        raise RuntimeError(f"{label} is hard-linked: {candidate}")
    if resolved != candidate:
        raise RuntimeError(f"{label} file is not lexical-owner exact: {candidate}")
    return candidate


@dataclass(slots=True, frozen=True)
class _RequiredChunkEvidence:
    evidence_root: Path
    outer_root: Path
    evidence_files: tuple[Path, ...]
    outer_files: tuple[Path, ...]
    validated_file_fingerprints: tuple[tuple[Path, str, int], ...]
    accepted_textless_graphics_pages: frozenset[int]
    validated_outer_file_fingerprints: tuple[tuple[Path, str, int], ...]

    @property
    def required_paths(self) -> list[str]:
        return sorted(
            path.relative_to(self.evidence_root).as_posix() for path in self.evidence_files
        )

    @property
    def outer_required_paths(self) -> list[str]:
        return sorted(path.relative_to(self.outer_root).as_posix() for path in self.outer_files)


def _owned_engine_reference(
    engine_dir: Path,
    value: object,
    *,
    label: str,
) -> Path:
    raw = _strict_string(value, field=label)
    relative = Path(raw)
    if relative.is_absolute() or relative != Path(os.path.normpath(relative)):
        raise RuntimeError(f"{label} must be a normalized relative path: {raw}")
    return _assert_owned_regular_file(
        Path(os.path.abspath(engine_dir / relative)),
        engine_dir,
        label=label,
    )


def _required_chunk_evidence(
    *,
    summary: SearchablePdfSummary,
    chunk: _PdfChunk,
    run_dir: Path,
    delete_original_text_layer: bool,
) -> _RequiredChunkEvidence:
    outer_root = _assert_owned_directory(run_dir, run_dir, label="hybrid run root")
    evidence_root = _assert_owned_directory(
        summary.run_dir,
        outer_root,
        label="chunk evidence root",
    )
    if summary.mode != PDF_MODE_HYBRID:
        raise RuntimeError(f"Chunk summary mode is invalid: {summary.mode!r}")
    benchmark_root = _assert_owned_directory(
        summary.benchmark.run_dir,
        evidence_root,
        label="chunk benchmark root",
    )
    if benchmark_root != evidence_root:
        raise RuntimeError("Chunk benchmark root must exactly equal the evidence root")
    compare_root = _assert_owned_directory(
        summary.compare_dir,
        evidence_root,
        label="chunk compare directory",
    )

    evidence_files: set[Path] = set()
    outer_files: set[Path] = set()
    validated_file_fingerprints: dict[Path, tuple[str, int]] = {}
    validated_outer_file_fingerprints: dict[Path, tuple[str, int]] = {}

    def add_evidence(path: Path, *, label: str) -> Path:
        owned = _assert_owned_regular_file(path, evidence_root, label=label)
        evidence_files.add(owned)
        return owned

    def add_outer(path: Path, *, label: str) -> Path:
        candidate = _strict_lexical_absolute_path(path, label=label)
        try:
            candidate.relative_to(evidence_root)
        except ValueError:
            owned = _assert_owned_regular_file(candidate, outer_root, label=label)
            outer_files.add(owned)
            return owned
        return add_evidence(candidate, label=label)

    def remember_validated(
        path: Path,
        fingerprint: Mapping[str, object],
    ) -> None:
        sealed = (str(fingerprint["sha256"]), int(str(fingerprint["size"])))
        if path in evidence_files:
            validated_file_fingerprints[path] = sealed
            return
        if path in outer_files:
            validated_outer_file_fingerprints[path] = sealed
            return
        raise RuntimeError(f"Validated file is not registered as chunk evidence: {path}")

    output_pdf = add_evidence(summary.output_pdf_path, label="chunk output PDF")
    chunk_page_count = chunk.end_page - chunk.start_page + 1
    if chunk_page_count <= 0:
        raise RuntimeError("Chunk page range is invalid")
    if summary.partial_page_failures > chunk_page_count:
        raise RuntimeError("Chunk partial-page failure count exceeds its page count")
    if summary.benchmark.failed_engines or summary.benchmark.skipped_engines:
        raise RuntimeError("Completed chunk contains failed or skipped OCR engines")
    if not summary.benchmark.result_files:
        raise RuntimeError("Completed chunk has no benchmark result files")
    for index, result_file in enumerate(summary.benchmark.result_files):
        add_evidence(result_file, label=f"benchmark result file {index}")

    benchmark_by_engine: dict[str, OcrBenchmarkResult] = {}
    benchmark_artifact_paths: dict[str, Path] = {}
    for benchmark_result in summary.benchmark.results:
        if benchmark_result.engine in benchmark_by_engine:
            raise RuntimeError(f"Duplicate benchmark engine result: {benchmark_result.engine}")
        benchmark_by_engine[benchmark_result.engine] = benchmark_result
    if set(benchmark_by_engine) != {"surya", "chandra"}:
        raise RuntimeError("Completed hybrid chunk requires Surya and Chandra benchmarks")
    for engine, benchmark_result in benchmark_by_engine.items():
        if benchmark_result.status != "ok" or benchmark_result.error is not None:
            raise RuntimeError(f"Completed chunk benchmark is not successful: {engine}")
        if not benchmark_result.sample_pages or any(
            page < 1 or page > chunk_page_count for page in benchmark_result.sample_pages
        ):
            raise RuntimeError(f"{engine} sample_pages are outside the local chunk range")
        if benchmark_result.artifact_path is None:
            raise RuntimeError(f"{engine} benchmark has no artifact path")
        benchmark_artifact_paths[engine] = add_evidence(
            Path(benchmark_result.artifact_path),
            label=f"{engine} benchmark artifact",
        )
    expected_partial_failures = max(
        (result.page_error_count for result in benchmark_by_engine.values()),
        default=0,
    )
    if summary.partial_page_failures != expected_partial_failures:
        raise RuntimeError("Chunk partial-page failure count disagrees with benchmark results")

    compare_by_engine: dict[str, CompareTxtBuildResult] = {}
    compare_text_paths: dict[str, Path] = {}
    for compare_result in summary.compare_results:
        if compare_result.engine in compare_by_engine:
            raise RuntimeError(f"Duplicate compare result: {compare_result.engine}")
        compare_by_engine[compare_result.engine] = compare_result
    if set(compare_by_engine) != {"surya", "chandra"}:
        raise RuntimeError("Completed hybrid chunk requires Surya and Chandra compare results")
    for engine, compare_result in compare_by_engine.items():
        if (
            compare_result.status != "ok"
            or compare_result.error is not None
            or compare_result.source_artifact_path is None
            or compare_result.compare_txt_path is None
        ):
            raise RuntimeError(f"Completed chunk compare result is incomplete: {engine}")
        compare_source = add_evidence(
            Path(compare_result.source_artifact_path),
            label=f"{engine} compare source artifact",
        )
        if compare_source != benchmark_artifact_paths[engine]:
            raise RuntimeError(
                f"{engine} compare source does not exactly equal its benchmark artifact"
            )
        compare_text = add_evidence(
            Path(compare_result.compare_txt_path),
            label=f"{engine} compare text",
        )
        if compare_text.parent != compare_root:
            raise RuntimeError(f"{engine} compare text is not root-exact")
        compare_text_paths[engine] = compare_text

    if len(summary.artifact_results) != 1:
        raise RuntimeError("Completed hybrid chunk requires one searchable artifact result")
    artifact = summary.artifact_results[0]
    if (
        artifact.engine != "chandra"
        or artifact.status != "ok"
        or artifact.error is not None
        or artifact.source_pdf_path is None
        or artifact.searchable_pdf_path is None
        or artifact.text_artifact_path is None
        or artifact.page_count != chunk_page_count
    ):
        raise RuntimeError("Completed hybrid searchable artifact result is incomplete")
    processing_source = add_outer(
        Path(artifact.source_pdf_path),
        label="searchable artifact source PDF",
    )
    artifact_text = add_evidence(
        Path(artifact.text_artifact_path),
        label="searchable artifact text input",
    )
    if artifact_text != compare_text_paths["chandra"]:
        raise RuntimeError("Searchable artifact text input is not the Chandra compare text")
    chunk_input = add_outer(chunk.path, label="fresh chunk input PDF")
    chunk_fingerprint = _stable_file_fingerprint(chunk_input)
    processing_fingerprint = _stable_file_fingerprint(processing_source)
    _validate_processing_source_derivation(
        chunk_pdf=chunk_input,
        processing_source=processing_source,
        outer_root=outer_root,
        expected_pages=chunk_page_count,
        delete_original_text_layer=delete_original_text_layer,
    )
    if _stable_file_fingerprint(chunk_input) != chunk_fingerprint:
        raise RuntimeError("Fresh chunk input changed during source derivation validation")
    if _stable_file_fingerprint(processing_source) != processing_fingerprint:
        raise RuntimeError("Processing source changed during source derivation validation")
    remember_validated(chunk_input, chunk_fingerprint)
    remember_validated(processing_source, processing_fingerprint)
    rendered_source_identities = _rendered_source_raster_identities(
        processing_source,
        expected_pages=chunk_page_count,
    )
    artifact_pdf = add_evidence(
        Path(artifact.searchable_pdf_path),
        label="searchable artifact output PDF",
    )
    if artifact_pdf != output_pdf:
        raise RuntimeError("Searchable artifact output disagrees with chunk output PDF")
    if artifact.geometry_log_path is not None:
        add_evidence(
            Path(artifact.geometry_log_path),
            label="searchable artifact geometry log",
        )

    reconciliation_path = add_evidence(
        evidence_root / "page_reconciliation.json",
        label="page reconciliation",
    )
    reconciliation, reconciliation_fingerprint = _stable_json_object(
        reconciliation_path,
        label="page reconciliation",
    )
    validated_file_fingerprints[reconciliation_path] = (
        str(reconciliation_fingerprint["sha256"]),
        int(str(reconciliation_fingerprint["size"])),
    )
    if reconciliation.get("status") != "ok":
        raise RuntimeError("Completed hybrid chunk has no successful reconciliation")

    accepted_textless_graphics_pages = _validate_final_mode_both_state(
        run_dir=evidence_root,
        expected_page_count=chunk_page_count,
        fingerprints=validated_file_fingerprints,
    )
    accepted_textless_pages = accepted_textless_graphics_pages
    expected_pages = set(range(1, chunk_page_count + 1))
    page_texts_by_engine: dict[str, dict[int, str]] = {}
    aggregate_payloads: dict[str, bytes] = {}
    for engine in ("surya", "chandra"):
        engine_dir = _assert_owned_directory(
            evidence_root / engine / engine,
            evidence_root,
            label=f"{engine} engine evidence directory",
        )
        pages_path = add_evidence(
            engine_dir / "pages.json",
            label=f"{engine} pages index",
        )
        pages_payload, pages_fingerprint = _stable_json_object(
            pages_path,
            label=f"{engine} pages index",
        )
        validated_file_fingerprints[pages_path] = (
            str(pages_fingerprint["sha256"]),
            int(str(pages_fingerprint["size"])),
        )
        if pages_payload.get("engine") != engine:
            raise RuntimeError(f"{engine} pages index has the wrong engine identity")
        raw_engine_source = _strict_string(
            pages_payload.get("pdf_path"),
            field=f"{engine} pages source PDF",
        )
        engine_source = _strict_lexical_absolute_path(
            Path(raw_engine_source),
            label=f"{engine} pages source PDF",
        )
        if engine_source != processing_source:
            raise RuntimeError(f"{engine} pages source PDF disagrees with the artifact source")
        aggregate_path = _owned_engine_reference(
            engine_dir,
            pages_payload.get("aggregate_file"),
            label=f"{engine} aggregate text",
        )
        evidence_files.add(aggregate_path)
        raw_pages = pages_payload.get("pages")
        if not isinstance(raw_pages, list) or not all(isinstance(item, dict) for item in raw_pages):
            raise RuntimeError(f"{engine} pages index is malformed")
        page_numbers: set[int] = set()
        page_text_by_number: dict[int, str] = {}
        for row_index, raw_row in enumerate(raw_pages):
            assert isinstance(raw_row, dict)
            source_page = raw_row.get("source_page")
            if (
                not isinstance(source_page, int)
                or isinstance(source_page, bool)
                or source_page in page_numbers
            ):
                raise RuntimeError(f"{engine} pages index has an invalid source page")
            page_numbers.add(source_page)
            page_text_path = _owned_engine_reference(
                engine_dir,
                raw_row.get("file"),
                label=f"{engine} page {source_page} text",
            )
            page_geometry_path = _owned_engine_reference(
                engine_dir,
                raw_row.get("geometry_file"),
                label=f"{engine} page {source_page} geometry",
            )
            evidence_files.update((page_text_path, page_geometry_path))
            geometry_fingerprint = _stable_file_fingerprint(page_geometry_path)
            remember_validated(page_geometry_path, geometry_fingerprint)
            try:
                expected_source_identity = rendered_source_identities.get(source_page)
                if (
                    expected_source_identity is None
                    or raw_row.get("source_raster_identity") != expected_source_identity
                ):
                    raise RuntimeError(
                        f"{engine} page {source_page} source raster is not derived from the processing PDF"
                    )
                page_payload, page_fingerprint = _stable_file_bytes(
                    page_text_path,
                    max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
                )
                page_text = page_payload.decode("utf-8")
            except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"{engine} page {source_page} text artifact is unreadable: {exc}"
                ) from exc
            if raw_row.get("text_chars") != len(page_text) or isinstance(
                raw_row.get("text_chars"), bool
            ):
                raise RuntimeError(
                    f"{engine} page {source_page} text_chars disagrees with its artifact"
                )
            page_text_by_number[source_page] = page_text
            validated_file_fingerprints[page_text_path] = (
                str(page_fingerprint["sha256"]),
                int(str(page_fingerprint["size"])),
            )
            attempt_count = raw_row.get("attempt_count")
            history = raw_row.get("attempt_history")
            if engine == "surya":
                if source_page not in accepted_textless_graphics_pages:
                    _strict_surya_attempt_metadata(
                        row=raw_row,
                        engine_dir=engine_dir,
                        source_page=source_page,
                    )
                if (
                    not isinstance(attempt_count, int)
                    or isinstance(attempt_count, bool)
                    or attempt_count not in {1, 2, 3}
                ):
                    raise RuntimeError(f"Surya page {source_page} attempt_count is invalid")
                expected_marker = {
                    1: None,
                    2: _SURYA_RETRY_PREPROCESSING,
                    3: _SURYA_SCALED_RETRY_PREPROCESSING,
                }[attempt_count]
                selected_attempt = raw_row.get("selected_attempt")
                if source_page in accepted_textless_graphics_pages:
                    selected_is_valid = "selected_attempt" not in raw_row
                else:
                    selected_is_valid = selected_attempt == attempt_count and not isinstance(
                        selected_attempt, bool
                    )
                if (
                    not selected_is_valid
                    or raw_row.get("retry_policy") != _SURYA_RETRY_POLICY
                    or raw_row.get("retry_preprocessing") != expected_marker
                    or not isinstance(history, list)
                    or len(history) != attempt_count
                    or any(not isinstance(item, dict) for item in history)
                    or [item.get("attempt") for item in history]
                    != list(range(1, attempt_count + 1))
                ):
                    raise RuntimeError(f"Surya page {source_page} attempt history is incomplete")
            else:
                if source_page not in accepted_textless_graphics_pages:
                    _strict_chandra_attempt_metadata(
                        row=raw_row,
                        engine_dir=engine_dir,
                        source_page=source_page,
                    )
            source_artifact = raw_row.get("source_raster_artifact")
            if not isinstance(source_artifact, dict) or not isinstance(
                source_artifact.get("path"), str
            ):
                raise RuntimeError(
                    f"{engine} page {source_page} source raster artifact is incomplete"
                )
            source_raster_path = _assert_owned_regular_file(
                Path(str(source_artifact["path"])),
                engine_dir,
                label=f"{engine} page {source_page} source raster artifact",
            )
            evidence_files.add(source_raster_path)
            source_raster_fingerprint = _stable_file_fingerprint(source_raster_path)
            if source_raster_fingerprint != {
                "sha256": source_artifact.get("sha256"),
                "size": source_artifact.get("bytes"),
            }:
                raise RuntimeError(f"{engine} page {source_page} source raster seal changed")
            remember_validated(source_raster_path, source_raster_fingerprint)
            if history is None:
                continue
            if (
                not isinstance(history, list)
                or not history
                or not all(isinstance(item, dict) for item in history)
            ):
                raise RuntimeError(f"{engine} page {source_page} retry history is malformed")
            for history_index, history_item in enumerate(history):
                assert isinstance(history_item, dict)
                for path_key, hash_key, size_key in (
                    ("image_path", "image_sha256", "image_bytes"),
                    ("sidecar_path", "sidecar_sha256", "sidecar_bytes"),
                ):
                    raw_path = history_item.get(path_key)
                    if not isinstance(raw_path, str) or not raw_path:
                        raise RuntimeError(
                            f"{engine} retry history {history_index} has no {path_key}"
                        )
                    history_path = _assert_owned_regular_file(
                        Path(raw_path),
                        engine_dir,
                        label=(f"{engine} page {source_page} retry {history_index} {path_key}"),
                    )
                    evidence_files.add(history_path)
                    history_fingerprint = {
                        "sha256": history_item.get(hash_key),
                        "size": history_item.get(size_key),
                    }
                    history_size = history_fingerprint["size"]
                    if (
                        not _valid_sha256(history_fingerprint["sha256"])
                        or type(history_size) is not int
                        or history_size <= 0
                    ):
                        raise RuntimeError(f"{engine} retry history seal is invalid")
                    remember_validated(history_path, history_fingerprint)
        if page_numbers != expected_pages:
            raise RuntimeError(f"{engine} pages index is incomplete: {sorted(page_numbers)}")
        page_texts_by_engine[engine] = page_text_by_number
        ordered_pages = sorted(expected_pages)
        expected_aggregate = _markerized_pages_text(
            page_texts=[page_text_by_number[page] for page in ordered_pages],
            source_pages_1based=ordered_pages,
        ).encode("utf-8")
        for aggregate_candidate, aggregate_label in (
            (aggregate_path, f"{engine} aggregate text"),
            (benchmark_artifact_paths[engine], f"{engine} benchmark artifact"),
        ):
            try:
                aggregate_payload, aggregate_fingerprint = _stable_file_bytes(
                    aggregate_candidate,
                    max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"{aggregate_label} is unreadable: {exc}") from exc
            if aggregate_payload != expected_aggregate:
                raise RuntimeError(
                    f"{aggregate_label} is not deterministically derived from page artifacts"
                )
            validated_file_fingerprints[aggregate_candidate] = (
                str(aggregate_fingerprint["sha256"]),
                int(str(aggregate_fingerprint["size"])),
            )
        aggregate_payloads[engine] = expected_aggregate

    for engine, compare_text_path in compare_text_paths.items():
        try:
            compare_payload, compare_fingerprint = _stable_file_bytes(
                compare_text_path,
                max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"{engine} compare text is unreadable: {exc}") from exc
        if compare_payload != aggregate_payloads[engine]:
            raise RuntimeError(f"{engine} compare text differs from its benchmark aggregate")
        remember_validated(compare_text_path, compare_fingerprint)

    output_fingerprint = _stable_file_fingerprint(output_pdf)
    try:
        native_page_texts = _extract_pdf_page_texts(processing_source)
        extracted_page_texts = _extract_pdf_page_texts(output_pdf)
    except Exception as exc:
        raise RuntimeError(f"Searchable PDF text validation failed: {exc}") from exc
    if len(native_page_texts) != chunk_page_count:
        raise RuntimeError("Processing source text is not page-bijective")
    chandra_page_texts = page_texts_by_engine["chandra"]
    expected_overlay_page_texts = [
        "\n".join(_split_page_text_lines(chandra_page_texts[page]))
        for page in range(1, chunk_page_count + 1)
    ]
    expected_output_page_texts: list[str] = []
    for native_text, overlay_text in zip(
        native_page_texts,
        expected_overlay_page_texts,
        strict=True,
    ):
        separator = "\n" if native_text and overlay_text else ""
        expected_output_page_texts.append(native_text + separator + overlay_text)
    _validate_searchable_text_retention(
        source_page_texts=expected_output_page_texts,
        extracted_page_texts=extracted_page_texts,
        expected_textless_pages=accepted_textless_pages,
    )
    if accepted_textless_pages:
        _validate_textless_visual_retention(
            source_pdf=processing_source,
            candidate_pdf=output_pdf,
            expected_pages=accepted_textless_pages,
        )
    if _stable_file_fingerprint(output_pdf) != output_fingerprint:
        raise RuntimeError("Searchable output changed during semantic validation")
    if _stable_file_fingerprint(processing_source) != processing_fingerprint:
        raise RuntimeError("Processing source changed during output validation")
    remember_validated(output_pdf, output_fingerprint)

    return _RequiredChunkEvidence(
        evidence_root=evidence_root,
        outer_root=outer_root,
        evidence_files=tuple(sorted(evidence_files, key=lambda path: path.as_posix())),
        outer_files=tuple(sorted(outer_files, key=lambda path: path.as_posix())),
        accepted_textless_graphics_pages=accepted_textless_graphics_pages,
        validated_file_fingerprints=tuple(
            (
                path,
                fingerprint[0],
                fingerprint[1],
            )
            for path, fingerprint in sorted(
                validated_file_fingerprints.items(),
                key=lambda item: item[0].as_posix(),
            )
        ),
        validated_outer_file_fingerprints=tuple(
            (
                path,
                fingerprint[0],
                fingerprint[1],
            )
            for path, fingerprint in sorted(
                validated_outer_file_fingerprints.items(),
                key=lambda item: item[0].as_posix(),
            )
        ),
    )


def _complete_chunk_evidence_entries(
    *,
    required: _RequiredChunkEvidence,
    manifest_path: Path,
) -> list[dict[str, object]]:
    expected_manifest = required.evidence_root / "chunk_evidence_manifest.json"
    if manifest_path != expected_manifest:
        raise RuntimeError("Chunk evidence manifest path is not root-exact")
    entries: list[dict[str, object]] = []
    pending = [required.evidence_root]
    while pending:
        directory = pending.pop()
        _assert_owned_directory(
            directory,
            required.evidence_root,
            label="chunk evidence directory",
        )
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise RuntimeError(
                f"Chunk evidence directory is unreadable: {directory}: {exc}"
            ) from exc
        for child in children:
            candidate = directory / child.name
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(
                    f"Chunk evidence entry is unreadable: {candidate}: {exc}"
                ) from exc
            if child.is_symlink() or _stat_is_reparse_point(child_stat):
                raise RuntimeError(f"Chunk evidence contains a link or reparse point: {candidate}")
            if stat.S_ISDIR(child_stat.st_mode):
                pending.append(candidate)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise RuntimeError(f"Chunk evidence is not a regular file: {candidate}")
            owned = _assert_owned_regular_file(
                candidate,
                required.evidence_root,
                label="chunk evidence file",
            )
            if owned == manifest_path:
                continue
            fingerprint = _stable_file_fingerprint(owned)
            entries.append(
                {
                    "path": owned.relative_to(required.evidence_root).as_posix(),
                    "sha256": str(fingerprint["sha256"]),
                    "size": int(str(fingerprint["size"])),
                }
            )
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _required_outer_evidence_entries(
    required: _RequiredChunkEvidence,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in required.outer_files:
        owned = _assert_owned_regular_file(
            path,
            required.outer_root,
            label="outer chunk evidence file",
        )
        fingerprint = _stable_file_fingerprint(owned)
        entries.append(
            {
                "path": owned.relative_to(required.outer_root).as_posix(),
                "sha256": str(fingerprint["sha256"]),
                "size": int(str(fingerprint["size"])),
            }
        )
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _complete_chunk_evidence_payload(
    required: _RequiredChunkEvidence,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    files = _complete_chunk_evidence_entries(
        required=required,
        manifest_path=manifest_path,
    )
    available_paths = {str(item["path"]) for item in files}
    entries_by_path = {str(item["path"]): item for item in files}
    for path, expected_sha256, expected_size in required.validated_file_fingerprints:
        relative = path.relative_to(required.evidence_root).as_posix()
        entry = entries_by_path.get(relative)
        if entry != {
            "path": relative,
            "sha256": expected_sha256,
            "size": expected_size,
        }:
            raise RuntimeError(
                f"Validated chunk evidence changed before it could be sealed: {relative}"
            )
    if not set(required.required_paths) <= available_paths:
        missing = sorted(set(required.required_paths) - available_paths)
        raise RuntimeError(f"Required chunk evidence is missing from the seal: {missing}")
    outer_files = _required_outer_evidence_entries(required)
    if {str(item["path"]) for item in outer_files} != set(required.outer_required_paths):
        raise RuntimeError("Required outer chunk evidence is missing from the seal")
    outer_entries_by_path = {str(item["path"]): item for item in outer_files}
    for path, expected_sha256, expected_size in required.validated_outer_file_fingerprints:
        relative = path.relative_to(required.outer_root).as_posix()
        entry = outer_entries_by_path.get(relative)
        if entry != {
            "path": relative,
            "sha256": expected_sha256,
            "size": expected_size,
        }:
            raise RuntimeError(
                f"Validated outer chunk evidence changed before it could be sealed: {relative}"
            )
    return {
        "schema": _CHUNK_EVIDENCE_MANIFEST_SCHEMA,
        "evidence_root": str(required.evidence_root),
        "outer_root": str(required.outer_root),
        "required_paths": required.required_paths,
        "outer_required_paths": required.outer_required_paths,
        "files": files,
        "outer_files": outer_files,
    }


def _write_complete_chunk_evidence_manifest(
    required: _RequiredChunkEvidence,
) -> tuple[Path, dict[str, object]]:
    manifest_path = required.evidence_root / "chunk_evidence_manifest.json"
    if os.path.lexists(manifest_path):
        _assert_owned_regular_file(
            manifest_path,
            required.evidence_root,
            label="existing chunk evidence manifest",
        )
    payload = _complete_chunk_evidence_payload(
        required,
        manifest_path=manifest_path,
    )
    _write_json_atomic(manifest_path, payload)
    manifest_path = _assert_owned_regular_file(
        manifest_path,
        required.evidence_root,
        label="chunk evidence manifest",
    )
    return manifest_path, _stable_file_fingerprint(manifest_path)


def _validate_complete_chunk_evidence_manifest(
    *,
    record: dict[str, object],
    required: _RequiredChunkEvidence,
) -> bool:
    raw_manifest = record.get("evidence_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        return False
    manifest_path = _strict_lexical_absolute_path(
        Path(raw_manifest),
        label="chunk evidence manifest",
    )
    if manifest_path != required.evidence_root / "chunk_evidence_manifest.json":
        return False
    manifest_path = _assert_owned_regular_file(
        manifest_path,
        required.evidence_root,
        label="chunk evidence manifest",
    )
    expected_sha256 = record.get("evidence_manifest_sha256")
    expected_size = record.get("evidence_manifest_size")
    if not _valid_sha256(expected_sha256) or (
        not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0
    ):
        return False
    payload, fingerprint = _stable_json_object(
        manifest_path,
        label="chunk evidence manifest",
    )
    if fingerprint != {"sha256": expected_sha256, "size": expected_size}:
        return False
    expected_payload = _complete_chunk_evidence_payload(
        required,
        manifest_path=manifest_path,
    )
    return payload == expected_payload


def _reusable_chunk_summary(
    *,
    record: dict[str, object],
    chunk: _PdfChunk,
    run_dir: Path,
    source_sizes: list[tuple[float, float]],
    delete_original_text_layer: bool = False,
) -> SearchablePdfSummary | None:
    if record.get("status") != "done":
        return None
    try:
        outer_root = _assert_owned_directory(
            run_dir,
            run_dir,
            label="hybrid run root",
        )
        raw_output = record["output_pdf"]
        if not isinstance(raw_output, str) or not raw_output:
            return None
        output_pdf = _assert_owned_regular_file(
            _strict_lexical_absolute_path(Path(raw_output), label="cached chunk output"),
            outer_root,
            label="cached chunk output",
        )
        expected_size = record["output_size"]
        expected_sha256 = record["output_sha256"]
        expected_pages = record["output_page_count"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
            or not _valid_sha256(expected_sha256)
            or not isinstance(expected_pages, int)
            or isinstance(expected_pages, bool)
            or expected_pages <= 0
        ):
            return None
        chunk_page_count = chunk.end_page - chunk.start_page + 1
        if expected_pages != chunk_page_count:
            return None
        actual = _stable_file_fingerprint(output_pdf)
        if actual != {"sha256": expected_sha256, "size": expected_size}:
            return None
        actual_pages = _validate_chunk_pdf(
            chunk=chunk,
            chunk_pdf=output_pdf,
            source_sizes=source_sizes,
        )
        if actual_pages != expected_pages:
            return None
        summary = _chunk_summary_from_payload(record.get("summary"))
        required = _required_chunk_evidence(
            summary=summary,
            chunk=chunk,
            run_dir=outer_root,
            delete_original_text_layer=delete_original_text_layer,
        )
        raw_record_root = record.get("run_dir")
        if not isinstance(raw_record_root, str) or Path(raw_record_root) != required.evidence_root:
            return None
        if summary.output_pdf_path != output_pdf:
            return None
        raw_partial_failures = record.get("partial_page_failures")
        if (
            not isinstance(raw_partial_failures, int)
            or isinstance(raw_partial_failures, bool)
            or raw_partial_failures != summary.partial_page_failures
        ):
            return None
        if not _validate_complete_chunk_evidence_manifest(
            record=record,
            required=required,
        ):
            return None
        return summary
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return None


def run_basic_ocr_benchmark(
    *,
    pdf_path: Path,
    mode_key: str,
    page_numbers: tuple[int, ...] | None = None,
    lang: str = DEFAULT_BASIC_GUI_LANG,
    output_root: Path | None = None,
    progress: ProgressCallback | None = None,
) -> BasicOcrRunSummary:
    resolved_pdf = Path(pdf_path)
    if not resolved_pdf.exists() or not resolved_pdf.is_file():
        raise RuntimeError(f"PDF file not found: {resolved_pdf}")
    if resolved_pdf.suffix.lower() != ".pdf":
        raise RuntimeError("Only PDF input is supported.")

    selected_mode = mode_key.strip().lower()
    requested_engines = MODE_TO_ENGINES.get(selected_mode)
    if not requested_engines:
        known = ", ".join(sorted(MODE_TO_ENGINES))
        raise RuntimeError(f"Unknown mode '{mode_key}'. Supported: {known}.")

    ready_engines: list[str] = []
    engine_python_overrides: dict[str, Path] = {}
    skipped_engines: list[str] = []
    for engine in requested_engines:
        try:
            engine_python = _resolve_engine_python(engine)
        except Exception as exc:
            skipped_engines.append(f"{engine}: {exc}")
            continue
        if engine_python is not None:
            ready_engines.append(engine)
            engine_python_overrides[engine] = engine_python
            continue

        status = detect_ocr_engine_status(engine)
        if status.ready:
            ready_engines.append(engine)
            continue
        missing_deps = ", ".join(status.missing) if status.missing else "unknown"
        skipped_engines.append(f"{engine}: {missing_deps}")
    if not ready_engines:
        raise RuntimeError("No ready OCR engines:\n\n" + "\n".join(skipped_engines))

    run_root = (
        Path(output_root)
        if output_root is not None
        else (Path.cwd() / "outputs" / "basic_gui_runs")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (run_root / f"{resolved_pdf.stem}_{timestamp}_{uuid.uuid4().hex[:8]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress, 0, "Preparing...")
    results: list[OcrBenchmarkResult] = []
    result_files: list[Path] = []
    failed_engines: list[str] = []
    total = max(1, len(ready_engines))
    sample_size = 999999 if page_numbers is None else max(len(page_numbers), 1)
    render_dpi = _resolve_ocr_render_dpi()
    runtime_tmp = (Path.cwd() / ".tmp_runtime").resolve()
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    with _temporary_env("TEMP", str(runtime_tmp)), _temporary_env("TMP", str(runtime_tmp)):
        for index, engine in enumerate(ready_engines, start=1):
            start_percent = int(((index - 1) / total) * 100)
            end_percent = int((index / total) * 100)
            _emit_progress(progress, start_percent, f"Running: {engine}")

            engine_output = run_dir / engine
            engine_output.mkdir(parents=True, exist_ok=True)

            def _engine_progress(
                local_percent: int,
                status: str,
                *,
                _start_percent: int = start_percent,
                _end_percent: int = end_percent,
            ) -> None:
                bounded_local = max(0, min(100, int(local_percent)))
                span = max(0, _end_percent - _start_percent)
                mapped = _start_percent + int((bounded_local / 100.0) * span)
                _emit_progress(progress, mapped, status)

            result: OcrBenchmarkResult | None = None
            engine_python = engine_python_overrides.get(engine)
            if engine_python is not None:
                _engine_progress(5, f"Running: {engine} ({engine_python})")
                try:
                    result = _run_engine_benchmark_subprocess(
                        python_exe=engine_python,
                        engine=engine,
                        pdf_path=resolved_pdf,
                        output_dir=engine_output,
                        sample_size=sample_size,
                        page_numbers=page_numbers,
                        lang=lang,
                        dpi=render_dpi,
                        defer_empty_pages=(selected_mode == MODE_BOTH),
                    )
                except Exception as exc:
                    failed_engines.append(f"{engine}: {exc}")
                    _emit_progress(progress, end_percent, f"Error: {engine}")
                    continue
                _engine_progress(95, f"Finished: {engine}")
            else:
                engine_results = run_ocr_benchmark(
                    pdf_path=resolved_pdf,
                    output_dir=engine_output,
                    engines=(engine,),
                    sample_size=sample_size,
                    page_numbers=page_numbers,
                    dpi=render_dpi,
                    lang=lang,
                    progress=_engine_progress,
                    defer_empty_pages=(selected_mode == MODE_BOTH),
                )
                if not engine_results:
                    failed_engines.append(f"{engine}: benchmark returned no result")
                    _emit_progress(progress, end_percent, f"Error: {engine}")
                    continue
                result = engine_results[0]

            if result is None:
                failed_engines.append(f"{engine}: benchmark returned no result")
                _emit_progress(progress, end_percent, f"Error: {engine}")
                continue

            results.append(result)
            report_path = engine_output / f"{resolved_pdf.stem}_ocr_benchmark.json"
            if report_path.exists():
                result_files.append(report_path)
            pending_reconciliation = (
                selected_mode == MODE_BOTH and result.status == _OCR_STATUS_RECONCILIATION_PENDING
            )
            if result.status != "ok" and not pending_reconciliation:
                failed_engines.append(f"{engine}: {_result_error_text(result)}")
                _emit_progress(progress, end_percent, f"Error: {engine}")
                continue
            _emit_progress(progress, end_percent, f"Done: {engine}")

    if selected_mode == MODE_BOTH and not failed_engines and not skipped_engines:
        results, reconciliation_error = _reconcile_mode_both_pages(
            run_dir=run_dir,
            results=results,
            result_files=result_files,
        )
        if reconciliation_error is not None:
            failed_engines.append(f"page reconciliation: {reconciliation_error}")

    if len(failed_engines) >= len(ready_engines):
        details = "\n\n".join(failed_engines)
        raise RuntimeError(f"No engine completed successfully.\n\n{details}")

    _emit_progress(progress, 100, "Completed")
    return BasicOcrRunSummary(
        run_dir=run_dir,
        results=tuple(results),
        result_files=tuple(result_files),
        failed_engines=tuple(failed_engines),
        skipped_engines=tuple(skipped_engines),
    )


def build_searchable_pdf(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    mode: str = PDF_MODE_HYBRID,
    lang: str = DEFAULT_BASIC_GUI_LANG,
    page_numbers: tuple[int, ...] | None = None,
    work_root: Path | None = None,
    overwrite_input_path: bool = True,
    return_bytes: bool | None = None,
    strict: bool = True,
    progress: ProgressCallback | None = None,
    delete_original_text_layer: bool = True,
    hybrid_chunk_cache_root: Path | None = None,
) -> SearchablePdfSummary:
    """Build one searchable PDF from file-path or in-memory PDF input."""
    if (pdf_path is None and pdf_bytes is None) or (pdf_path is not None and pdf_bytes is not None):
        raise ValueError("Provide exactly one input: pdf_path or pdf_bytes.")
    if not strict:
        raise ValueError("strict cannot be disabled; production OCR requires Chandra and Surya.")

    normalized_mode = normalize_pdf_mode(mode)
    benchmark_mode_key = _mode_to_benchmark_key(normalized_mode)
    prepare_engines = _mode_to_prepare_engines(normalized_mode)
    build_engines = _mode_to_build_engines(normalized_mode)

    resolved_work_root = (
        Path(work_root) if work_root is not None else (Path.cwd() / "outputs" / "service_runs")
    )
    resolved_work_root.mkdir(parents=True, exist_ok=True)

    input_path: Path
    if pdf_path is not None:
        input_path = Path(pdf_path).resolve()
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")
    else:
        if pdf_bytes is None or len(pdf_bytes) == 0:
            raise ValueError("Input pdf_bytes is empty.")
        staged_dir = (
            resolved_work_root
            / f"inline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        ).resolve()
        staged_dir.mkdir(parents=True, exist_ok=True)
        input_path = staged_dir / "input.pdf"
        input_path.write_bytes(pdf_bytes)

    chunk_pages = _resolve_hybrid_chunk_pages()
    if normalized_mode == PDF_MODE_HYBRID and page_numbers is None and chunk_pages > 0:
        page_count = _pdf_page_count(input_path)
        if page_count > chunk_pages:
            if return_bytes is None:
                need_bytes = pdf_bytes is not None
            else:
                need_bytes = bool(return_bytes)
            overwrite_target = input_path if pdf_path is not None and overwrite_input_path else None
            return _build_searchable_pdf_chunked(
                input_path=input_path,
                mode=normalized_mode,
                lang=lang,
                work_root=resolved_work_root,
                overwrite_target=overwrite_target,
                return_bytes=need_bytes,
                strict=strict,
                progress=progress,
                delete_original_text_layer=delete_original_text_layer,
                chunk_pages=chunk_pages,
                page_count=page_count,
                chunk_cache_root=hybrid_chunk_cache_root,
            )

    processing_input_path = input_path
    source_pdf_root = input_path.parent
    if delete_original_text_layer:
        _emit_progress(progress, 1, "Removing original text layer...")
        textless_root = (
            resolved_work_root
            / f"_source_pdf_without_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        ).resolve()
        textless_pdf = textless_root / input_path.name
        processing_input_path = _build_textless_source_pdf(
            source_pdf=input_path,
            out_pdf=textless_pdf,
            dpi=_resolve_textless_dpi(),
        )
        source_pdf_root = textless_root

    _emit_progress(progress, 2, "OCR benchmarking...")

    def _benchmark_progress(value: int, status: str) -> None:
        bounded = max(0, min(100, int(value)))
        _emit_progress(progress, 2 + int(bounded * 0.75), status)

    benchmark = run_basic_ocr_benchmark(
        pdf_path=processing_input_path,
        mode_key=benchmark_mode_key,
        page_numbers=page_numbers,
        lang=lang,
        output_root=resolved_work_root,
        progress=_benchmark_progress,
    )
    if strict:
        _ensure_requested_engines_succeeded(
            benchmark,
            expected_engines=prepare_engines,
        )

    compare_dir = benchmark.run_dir / "_compare_txt"
    _emit_progress(progress, 78, "Preparing compare artifacts...")
    compare_results = tuple(
        build_compare_txt_from_benchmark(
            benchmark_root=benchmark.run_dir,
            output_dir=compare_dir,
            engines=prepare_engines,
        )
    )
    if strict:
        _ensure_ok(compare_results, step="prepare-compare-txt")

    output_root = benchmark.run_dir / "searchable_pdf_final"
    _emit_progress(progress, 86, "Building searchable PDF...")
    if normalized_mode == PDF_MODE_HYBRID:
        geometry_override_dir = benchmark.run_dir / "surya"
    else:
        geometry_override_dir = None

    with _temporary_env(
        "UNISCAN_CHANDRA_GEOMETRY_DIR",
        str(geometry_override_dir) if geometry_override_dir is not None else None,
    ):
        artifact_results = tuple(
            run_artifact_searchable_package(
                compare_dir=compare_dir,
                pdf_root=source_pdf_root,
                output_dir=output_root,
                engines=build_engines,
                require_page_markers=True,
                reconciliation_root=(
                    benchmark.run_dir if normalized_mode == PDF_MODE_HYBRID else None
                ),
            )
        )
    if strict:
        _ensure_ok(artifact_results, step="build-searchable-from-artifacts")
    produced_pdf = _pick_ok_pdf(artifact_results)

    overwritten_path: Path | None = None
    final_pdf_path = produced_pdf
    if pdf_path is not None and overwrite_input_path:
        _atomic_copy_file(produced_pdf, input_path)
        overwritten_path = input_path
        final_pdf_path = input_path

    if return_bytes is None:
        need_bytes = pdf_bytes is not None
    else:
        need_bytes = bool(return_bytes)
    output_bytes = final_pdf_path.read_bytes() if need_bytes else None

    _emit_progress(progress, 100, "Done")
    return SearchablePdfSummary(
        mode=normalized_mode,
        run_dir=benchmark.run_dir,
        compare_dir=compare_dir,
        output_pdf_path=final_pdf_path,
        output_pdf_bytes=output_bytes,
        overwritten_input_path=overwritten_path,
        benchmark=benchmark,
        compare_results=compare_results,
        artifact_results=artifact_results,
        partial_page_failures=max(
            (max(0, int(result.page_error_count)) for result in benchmark.results),
            default=0,
        ),
    )


def _nonnegative_manifest_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _prepare_chunk_manifest(
    *,
    manifest_path: Path,
    identity: dict[str, object],
    run_key: str,
    input_path: Path,
    input_chunks: list[_PdfChunk],
    mode: str,
    page_count: int,
    chunk_pages: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    previous, read_error = _read_chunk_manifest(manifest_path)
    previous_valid = bool(
        previous is not None
        and previous.get("schema") == _HYBRID_CHUNK_MANIFEST_SCHEMA
        and previous.get("run_key") == run_key
        and previous.get("identity") == identity
    )
    previous_by_index: dict[int, dict[str, object]] = {}
    if previous_valid and previous is not None:
        rows = previous.get("chunks")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    previous_by_index[int(row["index"])] = dict(row)
                except (KeyError, TypeError, ValueError):
                    continue

    manifest_chunks: list[dict[str, object]] = []
    for chunk in input_chunks:
        input_fingerprint = _stable_file_fingerprint(chunk.path)
        base: dict[str, object] = {
            "index": chunk.index,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "input_pdf": str(chunk.path),
            "input_sha256": str(input_fingerprint["sha256"]),
            "input_size": int(str(input_fingerprint["size"])),
        }
        previous_record = previous_by_index.get(chunk.index)
        if (
            previous_record is not None
            and previous_record.get("start_page") == chunk.start_page
            and previous_record.get("end_page") == chunk.end_page
        ):
            manifest_chunks.append({**previous_record, **base})
        else:
            manifest_chunks.append({**base, "status": "pending"})

    manifest: dict[str, object] = {
        "schema": _HYBRID_CHUNK_MANIFEST_SCHEMA,
        "status": "running",
        "run_key": run_key,
        "identity": identity,
        "source_pdf": str(input_path),
        "mode": mode,
        "page_count": page_count,
        "chunk_pages": chunk_pages,
        "chunk_count": len(input_chunks),
        "chunks": manifest_chunks,
    }
    if previous_valid and previous is not None:
        manifest["resumed_from_status"] = str(previous.get("status") or "unknown")
        manifest["resume_count"] = _nonnegative_manifest_int(previous.get("resume_count")) + 1
    elif read_error:
        manifest["recovery_reason"] = f"ignored unreadable manifest: {read_error}"
    elif previous is not None:
        manifest["recovery_reason"] = "ignored incompatible manifest identity"
    return manifest, manifest_chunks


def _complete_chunk_record(
    *,
    record: dict[str, object],
    chunk: _PdfChunk,
    summary: SearchablePdfSummary,
    run_dir: Path,
    source_sizes: list[tuple[float, float]],
    delete_original_text_layer: bool = False,
) -> None:
    summary_payload = _chunk_summary_payload(summary)
    sealed_summary = _chunk_summary_from_payload(summary_payload)
    required = _required_chunk_evidence(
        summary=sealed_summary,
        chunk=chunk,
        run_dir=run_dir,
        delete_original_text_layer=delete_original_text_layer,
    )
    output_pdf = _assert_owned_regular_file(
        sealed_summary.output_pdf_path,
        required.evidence_root,
        label="chunk output PDF",
    )
    output_page_count = _validate_chunk_pdf(
        chunk=chunk,
        chunk_pdf=output_pdf,
        source_sizes=source_sizes,
    )
    expected_page_count = chunk.end_page - chunk.start_page + 1
    if output_page_count != expected_page_count:
        raise RuntimeError("Chunk output page count disagrees with its page range")
    output_fingerprint = _stable_file_fingerprint(output_pdf)
    evidence_manifest, evidence_fingerprint = _write_complete_chunk_evidence_manifest(required)
    completed_record: dict[str, object] = {
        "status": "done",
        "run_dir": str(required.evidence_root),
        "evidence_manifest": str(evidence_manifest),
        "evidence_manifest_sha256": str(evidence_fingerprint["sha256"]),
        "evidence_manifest_size": int(str(evidence_fingerprint["size"])),
        "output_pdf": str(output_pdf),
        "output_sha256": str(output_fingerprint["sha256"]),
        "output_size": int(str(output_fingerprint["size"])),
        "output_page_count": output_page_count,
        "partial_page_failures": sealed_summary.partial_page_failures,
        "summary": summary_payload,
        "reused": False,
    }
    if not _validate_complete_chunk_evidence_manifest(
        record=completed_record,
        required=required,
    ):
        raise RuntimeError("Fresh chunk evidence manifest failed closed validation")
    record.update(completed_record)


def _build_searchable_pdf_chunked(
    *,
    input_path: Path,
    mode: str,
    lang: str,
    work_root: Path,
    overwrite_target: Path | None,
    return_bytes: bool,
    strict: bool,
    progress: ProgressCallback | None,
    delete_original_text_layer: bool,
    chunk_pages: int,
    page_count: int,
    chunk_cache_root: Path | None = None,
) -> SearchablePdfSummary:
    identity, run_key = _hybrid_run_identity(
        input_path=input_path,
        mode=mode,
        lang=lang,
        strict=strict,
        delete_original_text_layer=delete_original_text_layer,
        chunk_pages=chunk_pages,
        page_count=page_count,
    )
    with _hybrid_run_lock(run_key):
        return _build_searchable_pdf_chunked_unlocked(
            input_path=input_path,
            mode=mode,
            lang=lang,
            work_root=work_root,
            overwrite_target=overwrite_target,
            return_bytes=return_bytes,
            strict=strict,
            progress=progress,
            delete_original_text_layer=delete_original_text_layer,
            chunk_pages=chunk_pages,
            page_count=page_count,
            chunk_cache_root=chunk_cache_root,
            identity=identity,
            run_key=run_key,
        )


def _build_searchable_pdf_chunked_unlocked(
    *,
    input_path: Path,
    mode: str,
    lang: str,
    work_root: Path,
    overwrite_target: Path | None,
    return_bytes: bool,
    strict: bool,
    progress: ProgressCallback | None,
    delete_original_text_layer: bool,
    chunk_pages: int,
    page_count: int,
    chunk_cache_root: Path | None,
    identity: dict[str, object],
    run_key: str,
) -> SearchablePdfSummary:
    cache_root = _resolve_hybrid_chunk_cache_root(
        work_root=work_root,
        configured_root=chunk_cache_root,
    )
    run_dir = (cache_root / f"hybrid_{run_key}").resolve()
    input_chunks = _split_pdf_chunks(
        source_pdf=input_path,
        output_root=run_dir / "input_chunks",
        pages_per_chunk=chunk_pages,
    )
    if not input_chunks:
        raise RuntimeError(f"No PDF chunks were produced for {input_path}.")

    manifest_path = run_dir / "chunk_manifest.json"
    manifest, manifest_chunks = _prepare_chunk_manifest(
        manifest_path=manifest_path,
        identity=identity,
        run_key=run_key,
        input_path=input_path,
        input_chunks=input_chunks,
        mode=mode,
        page_count=page_count,
        chunk_pages=chunk_pages,
    )
    _write_json_atomic(manifest_path, manifest)
    source_sizes = _source_page_sizes(input_path)
    if len(source_sizes) != page_count:
        raise RuntimeError(
            f"Source PDF page count changed from {page_count} to {len(source_sizes)}."
        )

    chunk_outputs: list[tuple[_PdfChunk, Path]] = []
    benchmark_results: list[OcrBenchmarkResult] = []
    benchmark_files: list[Path] = []
    benchmark_failures: list[str] = []
    benchmark_skips: list[str] = []
    compare_results: list[CompareTxtBuildResult] = []
    artifact_results: list[ArtifactSearchableResult] = []
    partial_page_failures = 0
    total_chunks = len(input_chunks)
    _emit_progress(
        progress,
        0,
        f"Preparing {total_chunks} hybrid OCR chunks ({chunk_pages} pages each)...",
    )

    for chunk, record in zip(input_chunks, manifest_chunks, strict=True):
        start_percent = 1 + int(((chunk.index - 1) / total_chunks) * 94)
        end_percent = 1 + int((chunk.index / total_chunks) * 94)

        def _chunk_progress(
            value: int,
            status: str,
            *,
            start: int = start_percent,
            end: int = end_percent,
            label: str = (
                f"chunk {chunk.index}/{total_chunks} pages {chunk.start_page}-{chunk.end_page}"
            ),
        ) -> None:
            bounded = max(0, min(100, int(value)))
            mapped = start + int((bounded / 100.0) * max(0, end - start))
            _emit_progress(progress, mapped, f"Hybrid OCR {label}: {status}")

        summary = _reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=run_dir,
            source_sizes=source_sizes,
            delete_original_text_layer=delete_original_text_layer,
        )
        if summary is not None:
            record["reused"] = True
            record.pop("error", None)
            _write_json_atomic(manifest_path, manifest)
            _chunk_progress(100, "reused verified chunk")
        else:
            for stale_key in (
                "error",
                "run_dir",
                "output_pdf",
                "output_sha256",
                "output_size",
                "output_page_count",
                "partial_page_failures",
                "summary",
                "evidence_manifest",
                "evidence_manifest_sha256",
                "evidence_manifest_size",
                "reused",
            ):
                record.pop(stale_key, None)
            record["status"] = "running"
            _write_json_atomic(manifest_path, manifest)
            try:
                with _temporary_env("UNISCAN_HYBRID_CHUNK_PAGES", "0"):
                    summary = build_searchable_pdf(
                        pdf_path=chunk.path,
                        mode=mode,
                        lang=lang,
                        page_numbers=None,
                        work_root=run_dir / "chunk_runs" / f"chunk_{chunk.index:04d}",
                        overwrite_input_path=False,
                        return_bytes=False,
                        strict=strict,
                        progress=_chunk_progress,
                        delete_original_text_layer=delete_original_text_layer,
                    )
                if not summary.output_pdf_path.exists():
                    raise RuntimeError(f"Chunk output is missing: {summary.output_pdf_path}")
                _complete_chunk_record(
                    record=record,
                    chunk=chunk,
                    summary=summary,
                    run_dir=run_dir,
                    source_sizes=source_sizes,
                    delete_original_text_layer=delete_original_text_layer,
                )
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)
                manifest["status"] = "error"
                manifest["failed_chunk"] = chunk.index
                _write_json_atomic(manifest_path, manifest)
                raise RuntimeError(
                    f"Hybrid OCR chunk {chunk.index}/{total_chunks} failed "
                    f"for pages {chunk.start_page}-{chunk.end_page}: {exc}"
                ) from exc
            _write_json_atomic(manifest_path, manifest)
        chunk_outputs.append((chunk, summary.output_pdf_path))
        partial_page_failures += max(0, int(summary.partial_page_failures))
        benchmark_results.extend(
            replace(
                result,
                sample_pages=[chunk.start_page + int(page) - 1 for page in result.sample_pages],
            )
            for result in summary.benchmark.results
        )
        benchmark_files.extend(summary.benchmark.result_files)
        benchmark_failures.extend(
            f"chunk {chunk.index}: {item}" for item in summary.benchmark.failed_engines
        )
        benchmark_skips.extend(
            f"chunk {chunk.index}: {item}" for item in summary.benchmark.skipped_engines
        )
        compare_results.extend(summary.compare_results)
        artifact_results.extend(summary.artifact_results)

    snapshot_outputs: list[tuple[_PdfChunk, Path]] = []
    expected_merged_page_texts: list[str] = []
    merged_textless_visual_seals: dict[int, dict[str, object]] = {}
    global_graphics_windows: dict[int, list[int]] = {}
    snapshot_root = run_dir / "merge_snapshots"
    for chunk, record in zip(input_chunks, manifest_chunks, strict=True):
        revalidated_summary = _reusable_chunk_summary(
            record=record,
            chunk=chunk,
            run_dir=run_dir,
            source_sizes=source_sizes,
            delete_original_text_layer=delete_original_text_layer,
        )
        if revalidated_summary is None:
            raise RuntimeError(f"Chunk {chunk.index} changed after completion and before merge")
        required = _required_chunk_evidence(
            summary=revalidated_summary,
            chunk=chunk,
            run_dir=run_dir,
            delete_original_text_layer=delete_original_text_layer,
        )
        snapshot, page_texts, visual_seals = _premerge_chunk_state(
            chunk=chunk,
            record=record,
            summary=revalidated_summary,
            required=required,
            snapshot_root=snapshot_root,
        )
        if not _validate_complete_chunk_evidence_manifest(
            record=record,
            required=required,
        ):
            raise RuntimeError(f"Chunk {chunk.index} evidence changed while preparing the merge")
        snapshot_outputs.append((chunk, snapshot))
        expected_merged_page_texts.extend(page_texts)
        for global_page, visual_seal in visual_seals.items():
            if global_page in merged_textless_visual_seals:
                raise RuntimeError(f"Duplicate global textless page: {global_page}")
            merged_textless_visual_seals[global_page] = visual_seal
            window = (global_page - 1) // 10
            global_graphics_windows.setdefault(window, []).append(global_page)
    exceeded_windows = {
        window: pages for window, pages in global_graphics_windows.items() if len(pages) > 1
    }
    if exceeded_windows:
        raise RuntimeError(f"Global textless-graphics recovery cap exceeded: {exceeded_windows}")
    if len(expected_merged_page_texts) != page_count:
        raise RuntimeError("Pre-merge semantic page cardinality changed")

    def _validate_candidate(candidate: Path) -> None:
        _validate_merged_candidate(
            candidate_pdf=candidate,
            expected_page_texts=expected_merged_page_texts,
            textless_visual_seals=merged_textless_visual_seals,
        )

    _emit_progress(progress, 97, "Merging hybrid OCR chunks...")
    try:
        if _stable_file_fingerprint(input_path) != identity["source"]:
            raise RuntimeError("Source PDF changed while hybrid OCR chunks were running.")
        merged_pdf = _merge_pdf_chunks(
            source_pdf=input_path,
            chunks=snapshot_outputs,
            output_pdf=run_dir / "searchable_pdf_final" / "searchable.pdf",
            validate_candidate=_validate_candidate,
            expected_source_fingerprint=identity["source"],
        )
    except Exception as exc:
        manifest["status"] = "error"
        manifest["merge_error"] = str(exc)
        _write_json_atomic(manifest_path, manifest)
        raise
    final_pdf_path = merged_pdf
    overwritten_path: Path | None = None
    if overwrite_target is not None:
        _atomic_copy_file(merged_pdf, overwrite_target)
        final_pdf_path = overwrite_target
        overwritten_path = overwrite_target

    output_bytes = final_pdf_path.read_bytes() if return_bytes else None
    compare_dir = run_dir / "_compare_txt"
    compare_dir.mkdir(parents=True, exist_ok=True)
    benchmark = BasicOcrRunSummary(
        run_dir=run_dir,
        results=tuple(benchmark_results),
        result_files=tuple(benchmark_files),
        failed_engines=tuple(benchmark_failures),
        skipped_engines=tuple(benchmark_skips),
    )
    merged_fingerprint = _stable_file_fingerprint(merged_pdf)
    manifest["status"] = "done"
    manifest["output_pdf"] = str(merged_pdf)
    manifest["output_sha256"] = str(merged_fingerprint["sha256"])
    manifest["output_size"] = int(str(merged_fingerprint["size"]))
    manifest["output_page_count"] = page_count
    manifest["partial_page_failures"] = partial_page_failures
    _write_json_atomic(manifest_path, manifest)
    _emit_progress(progress, 100, "Done")
    return SearchablePdfSummary(
        mode=mode,
        run_dir=run_dir,
        compare_dir=compare_dir,
        output_pdf_path=final_pdf_path,
        output_pdf_bytes=output_bytes,
        overwritten_input_path=overwritten_path,
        benchmark=benchmark,
        compare_results=tuple(compare_results),
        artifact_results=tuple(artifact_results),
        partial_page_failures=partial_page_failures,
        chunk_count=total_chunks,
        chunk_pages=chunk_pages,
        chunk_manifest_path=manifest_path,
    )


def _atomic_copy_file(source: Path, target: Path) -> None:
    resolved_source = Path(source)
    resolved_target = Path(target)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_target.with_name(f".{resolved_target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(resolved_source, temporary)
        os.replace(temporary, resolved_target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _temporary_env(name: str, value: str | None) -> Iterator[None]:
    had_old = name in os.environ
    old_value = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if had_old and old_value is not None:
            os.environ[name] = old_value
        else:
            os.environ.pop(name, None)


def _ensure_ok(
    results: tuple[CompareTxtBuildResult, ...] | tuple[ArtifactSearchableResult, ...], *, step: str
) -> None:
    errors = [item for item in results if getattr(item, "status", "") != "ok"]
    if not errors:
        return
    preview = "; ".join(
        f"{getattr(item, 'document', 'doc')}[{getattr(item, 'engine', 'engine')}]: {getattr(item, 'error', '') or getattr(item, 'note', '') or 'failed'}"
        for item in errors[:3]
    )
    raise RuntimeError(f"{step} failed: {preview}")


def build_chandra_geometry_variants(
    *,
    run_root: Path,
    pdf_root: Path,
    output_root: Path | None = None,
    strict: bool = True,
) -> ChandraGeometryVariantsSummary:
    resolved_run = Path(run_root).resolve()
    resolved_pdf_root = Path(pdf_root).resolve()
    if not resolved_run.exists():
        raise FileNotFoundError(f"Run root not found: {resolved_run}")
    if not resolved_pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {resolved_pdf_root}")

    compare_dir = resolved_run / "_compare_txt"
    compare_results = tuple(
        build_compare_txt_from_benchmark(
            benchmark_root=resolved_run,
            output_dir=compare_dir,
            engines=("chandra", "surya"),
        )
    )
    if strict:
        _ensure_ok(compare_results, step="prepare-compare-txt")

    target_root = (
        output_root
        if output_root is not None
        else (resolved_run / "searchable_pdf_geometry_compare")
    )
    resolved_output_root = Path(target_root).resolve()
    output_chandra = resolved_output_root / "chandra_text__chandra_geometry"
    output_surya = resolved_output_root / "chandra_text__surya_geometry"

    chandra_results = tuple(
        run_artifact_searchable_package(
            compare_dir=compare_dir,
            pdf_root=resolved_pdf_root,
            output_dir=output_chandra,
            engines=("chandra",),
            require_page_markers=True,
            reconciliation_root=resolved_run,
        )
    )
    if strict:
        _ensure_ok(chandra_results, step="build chandra_text__chandra_geometry")

    surya_geometry_dir = resolved_run / "surya"
    with _temporary_env("UNISCAN_CHANDRA_GEOMETRY_DIR", str(surya_geometry_dir)):
        surya_results = tuple(
            run_artifact_searchable_package(
                compare_dir=compare_dir,
                pdf_root=resolved_pdf_root,
                output_dir=output_surya,
                engines=("chandra",),
                require_page_markers=True,
                reconciliation_root=resolved_run,
            )
        )
    if strict:
        _ensure_ok(surya_results, step="build chandra_text__surya_geometry")

    return ChandraGeometryVariantsSummary(
        run_root=resolved_run,
        compare_dir=compare_dir,
        output_root=resolved_output_root,
        compare_results=compare_results,
        chandra_geometry_results=chandra_results,
        surya_geometry_results=surya_results,
    )
