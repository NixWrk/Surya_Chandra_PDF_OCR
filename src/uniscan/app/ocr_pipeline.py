"""Reusable OCR workflow orchestration for desktop/web frontends."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

from uniscan.ocr import (
    ArtifactSearchableResult,
    CompareTxtBuildResult,
    OcrBenchmarkResult,
    build_compare_txt_from_benchmark,
    detect_ocr_engine_status,
    run_artifact_searchable_package,
    run_ocr_benchmark,
)
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
_HYBRID_CHUNK_MANIFEST_SCHEMA = "uniscan.hybrid-chunks.v2"
_HYBRID_CHUNK_PIPELINE_REVISION = "chandra-surya-resumable-v1"
_MAX_CHUNK_MANIFEST_BYTES = 16 * 1024 * 1024
_FILE_HASH_BLOCK_BYTES = 1024 * 1024
_HYBRID_IDENTITY_ENV_KEYS: tuple[str, ...] = (
    "UNISCAN_ALIGN_BAND",
    "UNISCAN_CHANDRA_ALLOW_CLI_FALLBACK",
    "UNISCAN_CHANDRA_BLEND_Y_WEIGHT",
    "UNISCAN_CHANDRA_DEVICE_POLICY",
    "UNISCAN_CHANDRA_GEOMETRY_POLICY",
    "UNISCAN_CHANDRA_PREFER_GPU",
    "UNISCAN_CHANDRA_REQUIRE_GPU",
    "UNISCAN_CHANDRA_REQUIRE_SIDECAR",
    "UNISCAN_GEOMETRY_DEBUG",
    "UNISCAN_OCR_RENDER_DPI",
    "UNISCAN_SURYA_ALLOW_TEXT_FALLBACK",
    "UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON",
    "UNISCAN_SURYA_REQUIRE_GPU",
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
    if page_numbers:
        page_arg = ",".join(str(int(page)) for page in page_numbers)
        cmd.extend(["--pages", page_arg])

    env = _build_engine_subprocess_env(engine=engine, repo_root=repo_root)
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


def _hybrid_runtime_config() -> dict[str, object]:
    return {
        "effective_ocr_render_dpi": _resolve_ocr_render_dpi(),
        "effective_textless_dpi": _resolve_textless_dpi(),
        "environment": {key: os.environ.get(key) for key in _HYBRID_IDENTITY_ENV_KEYS},
    }


def _hybrid_run_identity(
    *,
    input_path: Path,
    mode: str,
    lang: str,
    strict: bool,
    delete_original_text_layer: bool,
    chunk_pages: int,
    page_count: int,
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
        "runtime_config": _hybrid_runtime_config(),
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
    if not path.exists():
        return None, None
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_CHUNK_MANIFEST_BYTES:
            raise ValueError(
                f"manifest size {size} is outside 1..{_MAX_CHUNK_MANIFEST_BYTES} bytes"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
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
                chunk_document.save(str(temporary), garbage=4, deflate=True)
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


def _merge_pdf_chunks(
    *,
    source_pdf: Path,
    chunks: list[tuple[_PdfChunk, Path]],
    output_pdf: Path,
) -> Path:
    if not chunks:
        raise ValueError("No OCR chunks to merge.")
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
        os.replace(temporary, output_pdf)
    finally:
        temporary.unlink(missing_ok=True)
    return output_pdf


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
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


def _chunk_summary_from_payload(value: object) -> SearchablePdfSummary:
    if not isinstance(value, dict):
        raise ValueError("Chunk summary must be an object.")
    benchmark_raw = value.get("benchmark")
    if not isinstance(benchmark_raw, dict):
        raise ValueError("Chunk benchmark summary must be an object.")
    benchmark = BasicOcrRunSummary(
        run_dir=Path(str(benchmark_raw["run_dir"])),
        results=tuple(
            OcrBenchmarkResult(**item)
            for item in _object_rows(benchmark_raw.get("results"), field="benchmark.results")
        ),
        result_files=tuple(
            Path(path)
            for path in _string_items(
                benchmark_raw.get("result_files"),
                field="benchmark.result_files",
            )
        ),
        failed_engines=_string_items(
            benchmark_raw.get("failed_engines"),
            field="benchmark.failed_engines",
        ),
        skipped_engines=_string_items(
            benchmark_raw.get("skipped_engines"),
            field="benchmark.skipped_engines",
        ),
    )
    return SearchablePdfSummary(
        mode=str(value["mode"]),
        run_dir=Path(str(value["run_dir"])),
        compare_dir=Path(str(value["compare_dir"])),
        output_pdf_path=Path(str(value["output_pdf_path"])),
        output_pdf_bytes=None,
        overwritten_input_path=None,
        benchmark=benchmark,
        compare_results=tuple(
            CompareTxtBuildResult(**item)
            for item in _object_rows(value.get("compare_results"), field="compare_results")
        ),
        artifact_results=tuple(
            ArtifactSearchableResult(**item)
            for item in _object_rows(value.get("artifact_results"), field="artifact_results")
        ),
        partial_page_failures=max(0, int(value.get("partial_page_failures") or 0)),
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _reusable_chunk_summary(
    *,
    record: dict[str, object],
    chunk: _PdfChunk,
    run_dir: Path,
    source_sizes: list[tuple[float, float]],
) -> SearchablePdfSummary | None:
    if record.get("status") != "done":
        return None
    try:
        output_pdf = Path(str(record["output_pdf"])).resolve()
        if not _path_is_within(output_pdf, run_dir) or not output_pdf.is_file():
            return None
        expected_size = int(str(record["output_size"]))
        expected_sha256 = str(record["output_sha256"])
        expected_pages = int(str(record["output_page_count"]))
        actual = _stable_file_fingerprint(output_pdf)
        if actual != {"sha256": expected_sha256, "size": expected_size}:
            return None
        if (
            _validate_chunk_pdf(
                chunk=chunk,
                chunk_pdf=output_pdf,
                source_sizes=source_sizes,
            )
            != expected_pages
        ):
            return None
        summary = _chunk_summary_from_payload(record.get("summary"))
        if summary.output_pdf_path.resolve() != output_pdf:
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
            if result.status != "ok":
                failed_engines.append(f"{engine}: {_result_error_text(result)}")
                _emit_progress(progress, end_percent, f"Error: {engine}")
                continue
            _emit_progress(progress, end_percent, f"Done: {engine}")

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
) -> None:
    output_pdf = summary.output_pdf_path.resolve()
    if not _path_is_within(output_pdf, run_dir):
        raise RuntimeError(f"Chunk output escaped its owned run directory: {output_pdf}")
    output_page_count = _validate_chunk_pdf(
        chunk=chunk,
        chunk_pdf=output_pdf,
        source_sizes=source_sizes,
    )
    output_fingerprint = _stable_file_fingerprint(output_pdf)
    record.update(
        {
            "status": "done",
            "run_dir": str(summary.run_dir),
            "output_pdf": str(output_pdf),
            "output_sha256": str(output_fingerprint["sha256"]),
            "output_size": int(str(output_fingerprint["size"])),
            "output_page_count": output_page_count,
            "partial_page_failures": summary.partial_page_failures,
            "summary": _chunk_summary_payload(summary),
            "reused": False,
        }
    )


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

    _emit_progress(progress, 97, "Merging hybrid OCR chunks...")
    try:
        if _stable_file_fingerprint(input_path) != identity["source"]:
            raise RuntimeError("Source PDF changed while hybrid OCR chunks were running.")
        merged_pdf = _merge_pdf_chunks(
            source_pdf=input_path,
            chunks=chunk_outputs,
            output_pdf=run_dir / "searchable_pdf_final" / "searchable.pdf",
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
