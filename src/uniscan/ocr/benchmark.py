"""OCR benchmark helpers for sampled PDF fixtures."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import importlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, MutableMapping, Sequence, cast

from PIL import Image, ImageOps

from uniscan.io import imwrite_unicode, iter_render_pdf_page_indices

from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_OLMOCR,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    SEARCHABLE_PDF_ENGINES,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
    normalize_ocr_engines,
)
from .artifact_searchable import (
    _bbox_values,
    _bbox_reading_order_indices,
    _clean_overlay_line,
    _dehyphenate_line_breaks,
)
from .preprocessing import _strip_markdown

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PADDLE_CACHE_HOME = _REPO_ROOT / ".paddlex_cache"
_DEFAULT_HF_CACHE_HOME = _REPO_ROOT / ".hf_cache"
_DEFAULT_MODELSCOPE_CACHE_HOME = _REPO_ROOT / ".modelscope_cache"
_DEFAULT_SURYA_MODEL_CACHE_HOME = _REPO_ROOT / ".surya_cache"
_DEFAULT_YOLO_CONFIG_HOME = _REPO_ROOT / ".ultralytics"
_DEFAULT_RUNTIME_TMP_HOME = _REPO_ROOT / ".tmp_runtime"
_MAX_OCR_TEXT_ARTIFACT_BYTES = 64 * 1024 * 1024
_CHANDRA_MODEL_REPO_ID = "datalab-to/chandra-ocr-2"
_MODEL_CACHE_CHECK_MEMO: dict[str, str] = {}
_ZERO_OUTPUT_RETRY_PREPROCESSING = "autocontrast-cutoff-1"
_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING = "rgb-scale-0.5-center-white-lanczos-v1"
_ZERO_OUTPUT_SCALED_RETRY_FACTOR = 0.5
_ZERO_OUTPUT_RETRY_POLICY = (
    "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
)
_CHANDRA_LAYOUT_PROMPT_TYPE = "ocr_layout"
_CHANDRA_PLAIN_PROMPT_TYPE = "ocr"
_CHANDRA_PLAIN_RETRY_PREPROCESSING = "plain-ocr-original-v1"
_CHANDRA_ZERO_OUTPUT_RETRY_POLICY = (
    "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
)
_CHANDRA_LAYOUT_CONTENT_FILTER = "skip-graphic-labels-v1"
_CHANDRA_PLAIN_CONTENT_FILTER = "visible-text-tags-exclude-media-description-v1"
_CHANDRA_ALTERNATIVE_TEXT_POLICY = "account-visible-html-markdown-near-subsequence-v3"
_CHANDRA_ALTERNATIVE_TEXT_MIN_COVERAGE_PERCENT = 98
_CHANDRA_ATTEMPT_EVIDENCE_SCHEMA = "uniscan.chandra-attempt.v2"
_CHANDRA_MIN_IMAGE_DIM = 1536
_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES = 128 * 1024 * 1024
_MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS = 50_000_000
_MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION = 32_768
_SURYA_DIRECT_EXECUTION_PATHS = ("cli", "module")
_SURYA_MODULE_EXECUTION_PATHS = ("module",)
_EXPECTED_GPU0_UUID = "GPU-e6a8c006-5017-6126-01cc-bf9bd972bf4f"
_EXPECTED_GPU0_DOCKER_SELECTOR = f"device={_EXPECTED_GPU0_UUID}"
_OCR_OUTCOME_TEXT = "text"
_OCR_OUTCOME_VERIFIED_BLANK = "verified_blank"
_OCR_OUTCOME_EXPLICIT_NONTEXT = "explicit_nontext"
_OCR_OUTCOME_ZERO = "zero_output"
_PAGE_ERROR_ZERO_OUTPUT = "zero_output"
_PAGE_ERROR_MISSING_OUTPUT = "missing_output"
_OCR_STATUS_RECONCILIATION_PENDING = "reconciliation_pending"
_PAGE_ERROR_UNCLASSIFIED = "unclassified"
_SURYA_FAILURE_EVIDENCE_SCHEMA = "uniscan.surya-failure-evidence.v1"
_SURYA_SOURCE_COORDINATE_SPACE = "source-image-v1"
_SURYA_SCALED_GEOMETRY_TRANSFORM = "inverse-actual-content-size-strict-v1"
_MAX_SURYA_FAILURE_EVIDENCE_FILES = 512
_MAX_SURYA_FAILURE_EVIDENCE_ENTRIES = 1024
_MAX_SURYA_FAILURE_EVIDENCE_RELATIVE_PATH_CHARS = 4096
_MAX_SURYA_FAILURE_EVIDENCE_BYTES = 1024 * 1024 * 1024
_STABLE_FILE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_nlink",
) + (() if os.name == "nt" else ("st_ctime_ns",))
_STABLE_DIRECTORY_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_mtime_ns",
) + (() if os.name == "nt" else ("st_ctime_ns",))


def _bounded_chandra_model_input_size(width: int, height: int) -> tuple[int, int]:
    """Return Chandra's expected resize without decoding an unbounded result."""
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        raise RuntimeError("Chandra source raster dimensions are invalid.")
    if min(width, height) < _CHANDRA_MIN_IMAGE_DIM:
        scale = _CHANDRA_MIN_IMAGE_DIM / float(min(width, height))
        expected = (int(width * scale), int(height * scale))
    else:
        expected = (width, height)
    if (
        expected[0] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
        or expected[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
        or expected[0] * expected[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
    ):
        raise RuntimeError(
            "Chandra model input would exceed the bounded dimension/pixel policy: "
            f"{expected[0]}x{expected[1]}."
        )
    return expected


def _alnum_evidence(lines: Sequence[str]) -> tuple[int, int]:
    alnum_line_count = sum(1 for line in lines if any(char.isalnum() for char in line))
    alnum_chars = sum(1 for line in lines for char in line if char.isalnum())
    return alnum_line_count, alnum_chars


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        candidate.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_rgb_pixel_sha256(image: Image.Image) -> str:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _is_effectively_blank_rgb_image(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    grayscale.thumbnail((1024, 1024))
    histogram = grayscale.histogram()
    pixels = int(sum(histogram))
    if pixels <= 0:
        return False
    nonwhite = int(sum(histogram[:245]))
    dark = int(sum(histogram[:200]))
    return bool(nonwhite <= max(8, int(pixels * 0.0005)) and dark <= max(2, int(pixels * 0.00005)))


def _stable_bounded_png_rgb_evidence(
    path: Path,
    *,
    label: str,
    require_rgb: bool,
) -> tuple[Image.Image, dict[str, object]]:
    try:
        before_path = path.stat()
        if not stat.S_ISREG(before_path.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with path.open("rb") as stream:
            before_descriptor = os.fstat(stream.fileno())
            if (
                before_descriptor.st_size <= 0
                or before_descriptor.st_size > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES
            ):
                raise RuntimeError(f"{label} exceeds the bounded byte policy")
            payload = stream.read(_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES + 1)
            after_descriptor = os.fstat(stream.fileno())
        after_path = path.stat()
        if any(
            getattr(record, field) != getattr(before_descriptor, field)
            for record in (before_path, after_descriptor, after_path)
            for field in _STABLE_FILE_STAT_FIELDS
        ):
            raise RuntimeError(f"{label} changed while reading")
        if len(payload) != before_descriptor.st_size:
            raise RuntimeError(f"{label} exceeds the bounded byte policy")
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != "PNG":
                raise RuntimeError(f"{label} is not PNG")
            if int(getattr(opened, "n_frames", 1)) != 1:
                raise RuntimeError(f"{label} PNG is multi-frame")
            if (
                opened.width > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                or opened.height > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
                or opened.width * opened.height > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
            ):
                raise RuntimeError(f"{label} exceeds the bounded pixel policy")
            opened.load()
            if require_rgb and opened.mode != "RGB":
                raise RuntimeError(f"{label} is not canonical RGB")
            image = cast(Image.Image, opened.convert("RGB").copy())
        return image, {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot read {label}: {exc}") from exc


def _bounded_png_rgb_image(
    path: Path,
    *,
    label: str,
    require_rgb: bool,
) -> Image.Image:
    image, _fingerprint = _stable_bounded_png_rgb_evidence(
        path,
        label=label,
        require_rgb=require_rgb,
    )
    return image


def _source_raster_evidence(
    path: Path,
    *,
    source_page: int,
    require_rgb: bool,
) -> tuple[dict[str, object], dict[str, object], Image.Image]:
    if not isinstance(source_page, int) or isinstance(source_page, bool) or source_page <= 0:
        raise RuntimeError(f"Invalid source page for raster identity: {source_page!r}")
    image, fingerprint = _stable_bounded_png_rgb_evidence(
        path,
        label=f"source raster {path}",
        require_rgb=require_rgb,
    )
    identity: dict[str, object] = {
        "pixel_sha256": _canonical_rgb_pixel_sha256(image),
        "width": int(image.width),
        "height": int(image.height),
        "name": path.name,
        "source_page": source_page,
        "verified_blank": _is_effectively_blank_rgb_image(image),
    }
    artifact: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": fingerprint["sha256"],
        "bytes": fingerprint["bytes"],
    }
    return identity, artifact, image


def _source_raster_identity(
    path: Path,
    *,
    source_page: int,
) -> dict[str, object]:
    identity, _artifact, _image = _source_raster_evidence(
        path,
        require_rgb=False,
        source_page=source_page,
    )
    return identity


def _source_raster_artifact(path: Path) -> dict[str, object]:
    _image, fingerprint = _stable_bounded_png_rgb_evidence(
        path,
        label=f"source raster artifact {path}",
        require_rgb=True,
    )
    return {
        "path": str(path.resolve()),
        "sha256": fingerprint["sha256"],
        "bytes": fingerprint["bytes"],
    }


def _validate_source_raster_seal(
    *,
    path: Path,
    source_page: int,
    identity: dict[str, object],
    artifact: dict[str, object],
) -> None:
    observed_identity, observed_artifact, _image = _source_raster_evidence(
        path,
        source_page=source_page,
        require_rgb=True,
    )
    if observed_identity != identity or observed_artifact != artifact:
        raise RuntimeError(f"Source raster seal changed for page {source_page}: {path}")


def _stable_file_fingerprint(path: Path) -> dict[str, object]:
    before_path = path.stat()
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"Cannot seal non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before_descriptor = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after_descriptor = os.fstat(stream.fileno())
    after_path = path.stat()
    if any(
        getattr(record, field) != getattr(before_descriptor, field)
        for record in (before_path, after_descriptor, after_path)
        for field in _STABLE_FILE_STAT_FIELDS
    ):
        raise RuntimeError(f"File changed while sealing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": int(before_descriptor.st_size)}


def _image_attempt_evidence(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image_size = [int(image.width), int(image.height)]
    return {
        "image_size": image_size,
        "image_sha256": _sha256_path(path),
        "image_bytes": int(path.stat().st_size),
    }


def _write_autocontrast_retry_image(*, source: Path, target: Path) -> dict[str, object]:
    with Image.open(source) as original:
        original_size = original.size
        enhanced = ImageOps.autocontrast(original.convert("RGB"), cutoff=1)
        if enhanced.size != original_size:
            raise RuntimeError(
                "OCR zero-output retry changed image dimensions "
                f"from {original_size} to {enhanced.size}."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(target, format="PNG")
    return _image_attempt_evidence(target)


def _write_scaled_retry_image(*, source: Path, target: Path) -> dict[str, object]:
    with Image.open(source) as original:
        original_size = original.size
        image = original.convert("RGB")
        content_size = (
            max(1, round(image.width * _ZERO_OUTPUT_SCALED_RETRY_FACTOR)),
            max(1, round(image.height * _ZERO_OUTPUT_SCALED_RETRY_FACTOR)),
        )
        resized = image.resize(content_size, Image.Resampling.LANCZOS)
        content_offset = (
            (image.width - content_size[0]) // 2,
            (image.height - content_size[1]) // 2,
        )
        canvas = Image.new("RGB", original_size, (255, 255, 255))
        canvas.paste(resized, content_offset)
        if canvas.size != original_size:
            raise RuntimeError(
                f"OCR scaled retry changed image dimensions from {original_size} to {canvas.size}."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="PNG")
    return {
        **_image_attempt_evidence(target),
        "content_scale": _ZERO_OUTPUT_SCALED_RETRY_FACTOR,
        "content_size": list(content_size),
        "content_offset": list(content_offset),
        "resampling": "lanczos",
        "canvas_fill_rgb": [255, 255, 255],
    }


def _strict_numeric_bbox(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RuntimeError(f"Invalid {label}: bbox must contain exactly four numbers.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise RuntimeError(f"Invalid {label}: bbox values must be numeric and not bool.")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise RuntimeError(f"Invalid {label}: bbox values must be finite.")
    x0, y0, x1, y1 = bbox
    if x0 < 0.0 or y0 < 0.0 or x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"Invalid {label}: bbox must have positive in-bounds area.")
    return x0, y0, x1, y1


def _inverse_scaled_retry_bbox(
    value: object,
    *,
    source_size: Sequence[int],
    content_size: Sequence[int],
    content_offset: Sequence[int],
    label: str,
) -> list[float]:
    dimensions = (source_size, content_size, content_offset)
    if any(
        len(items) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in items)
        for items in dimensions
    ):
        raise RuntimeError(f"Invalid {label}: scaled retry dimensions are malformed.")
    source_width, source_height = source_size
    content_width, content_height = content_size
    offset_x, offset_y = content_offset
    if (
        source_width <= 0
        or source_height <= 0
        or content_width <= 0
        or content_height <= 0
        or offset_x < 0
        or offset_y < 0
        or offset_x + content_width > source_width
        or offset_y + content_height > source_height
    ):
        raise RuntimeError(f"Invalid {label}: scaled retry dimensions are out of bounds.")

    x0, y0, x1, y1 = _strict_numeric_bbox(value, label=label)
    if x1 > source_width or y1 > source_height:
        raise RuntimeError(f"Invalid {label}: bbox escapes the retry canvas.")
    content_x1 = offset_x + content_width
    content_y1 = offset_y + content_height
    if x0 < offset_x or y0 < offset_y or x1 > content_x1 or y1 > content_y1:
        raise RuntimeError(f"Invalid {label}: bbox escapes scaled content.")

    mapped = [
        (x0 - offset_x) * source_width / content_width,
        (y0 - offset_y) * source_height / content_height,
        (x1 - offset_x) * source_width / content_width,
        (y1 - offset_y) * source_height / content_height,
    ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise RuntimeError(f"Invalid {label}: inverse-transformed bbox has no area.")
    return mapped


def _strict_surya_single_page_geometry(
    *,
    sidecar_path: Path,
    image_path: Path,
    require_text: bool,
    permitted_execution_paths: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid {label}: sidecar is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid {label}: sidecar root must be an object.")
    execution_path = payload.get("execution_path")
    if not isinstance(execution_path, str) or execution_path not in permitted_execution_paths:
        raise RuntimeError(f"Invalid {label}: execution_path is not permitted.")
    images = payload.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Invalid {label}: exactly one image is required.")
    image = images[0]
    if image.get("image_name") != image_path.name:
        raise RuntimeError(f"Invalid {label}: image_name does not match the retry image.")
    pages = image.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise RuntimeError(f"Invalid {label}: exactly one page is required.")
    page = pages[0]
    image_bbox = _strict_numeric_bbox(page.get("image_bbox"), label=f"{label} image")
    with Image.open(image_path) as retry_image:
        expected_bbox = (0.0, 0.0, float(retry_image.width), float(retry_image.height))
    if image_bbox != expected_bbox:
        raise RuntimeError(f"Invalid {label}: image_bbox does not match image dimensions.")
    raw_lines = page.get("text_lines")
    if not isinstance(raw_lines, list):
        raise RuntimeError(f"Invalid {label}: text_lines must be an array.")
    if not require_text:
        if raw_lines != []:
            raise RuntimeError(f"Invalid {label}: zero_output text_lines must be empty.")
        return ()
    if not raw_lines:
        raise RuntimeError(f"Invalid {label}: text geometry is empty.")

    lines: list[str] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise RuntimeError(f"Invalid {label}: every text line must be an object.")
        raw_text = raw_line.get("text")
        if not isinstance(raw_text, str):
            raise RuntimeError(f"Invalid {label}: line text must be a string.")
        text = _clean_overlay_line(raw_text)
        if not text or not any(char.isalnum() for char in text):
            raise RuntimeError(f"Invalid {label}: line text has no alphanumeric evidence.")
        bbox = _strict_numeric_bbox(raw_line.get("bbox"), label=f"{label} text line")
        if (
            bbox[0] < image_bbox[0]
            or bbox[1] < image_bbox[1]
            or bbox[2] > image_bbox[2]
            or bbox[3] > image_bbox[3]
        ):
            raise RuntimeError(f"Invalid {label}: text bbox escapes image_bbox.")
        lines.append(text)
    return tuple(lines)


def _write_source_coordinate_surya_sidecar(
    *,
    source: Path,
    target: Path,
    source_size: Sequence[int],
    content_size: Sequence[int],
    content_offset: Sequence[int],
) -> Path:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot transform Surya retry geometry: {exc}") from exc
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError("Cannot transform malformed Surya retry image geometry.")
    image = dict(images[0])
    pages = image.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise RuntimeError("Cannot transform malformed Surya retry page geometry.")
    page = dict(pages[0])
    raw_lines = page.get("text_lines")
    if not isinstance(raw_lines, list):
        raise RuntimeError("Cannot transform malformed Surya retry text geometry.")
    transformed_lines: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise RuntimeError("Cannot transform non-object Surya retry text geometry.")
        line = dict(raw_line)
        line["bbox"] = _inverse_scaled_retry_bbox(
            line.get("bbox"),
            source_size=source_size,
            content_size=content_size,
            content_offset=content_offset,
            label="scaled retry text line",
        )
        transformed_lines.append(line)
    page["image_bbox"] = [0.0, 0.0, float(source_size[0]), float(source_size[1])]
    page["text_lines"] = transformed_lines
    image["pages"] = [page]
    image["geometry_coordinate_space"] = _SURYA_SOURCE_COORDINATE_SPACE
    image["geometry_transform"] = _SURYA_SCALED_GEOMETRY_TRANSFORM
    transformed_payload = dict(payload)
    transformed_payload["images"] = [image]
    _write_json_atomic(target, transformed_payload)
    return target


def _strict_scaled_retry_pair(
    value: object,
    *,
    label: str,
    allow_zero: bool,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        or any(item < (0 if allow_zero else 1) for item in value)
    ):
        raise RuntimeError(f"Invalid scaled retry {label} evidence.")
    return value[0], value[1]


def _structured_surya_zero_output(
    *,
    page_errors: Sequence[dict[str, Any]],
    page_metadata: Sequence[dict[str, Any]],
    source_page: int,
    image_path: Path,
    expected_attempt: int,
    expected_preprocessing: str | None,
    permitted_execution_paths: Sequence[str],
    label: str,
) -> dict[str, Any] | None:
    rows = [item for item in page_metadata if item.get("source_page") == source_page]
    if not rows or rows[0].get("ocr_outcome") != _OCR_OUTCOME_ZERO:
        return None
    if len(rows) != 1:
        raise RuntimeError(f"Invalid {label}: duplicate page metadata.")
    row = rows[0]
    attempt_count = row.get("attempt_count")
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count != expected_attempt
    ):
        raise RuntimeError(f"Invalid {label}: attempt_count is inconsistent.")
    if row.get("retry_preprocessing") != expected_preprocessing:
        raise RuntimeError(f"Invalid {label}: retry preprocessing is inconsistent.")
    errors = [item for item in page_errors if item.get("source_page") == source_page]
    if (
        len(errors) != 1
        or errors[0].get("code") != _PAGE_ERROR_ZERO_OUTPUT
        or not isinstance(errors[0].get("error"), str)
        or not str(errors[0]["error"]).strip()
    ):
        raise RuntimeError(f"Invalid {label}: structured zero_output error is missing.")
    raw_sidecar = row.get("surya_page_lines_path")
    if not isinstance(raw_sidecar, str):
        raise RuntimeError(f"Invalid {label}: durable geometry path is missing.")
    sidecar_path = Path(raw_sidecar)
    if not sidecar_path.is_file():
        raise RuntimeError(f"Invalid {label}: durable geometry sidecar is missing.")
    _strict_surya_single_page_geometry(
        sidecar_path=sidecar_path,
        image_path=image_path,
        require_text=False,
        permitted_execution_paths=permitted_execution_paths,
        label=label,
    )
    return row


def _strict_retry_copy_source(path: Path, *, max_bytes: int) -> tuple[Path, os.stat_result]:
    candidate = Path(os.path.abspath(path))
    parent_stat = candidate.parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or _stat_has_reparse_point(parent_stat):
        raise RuntimeError(f"OCR retry evidence source parent is unsafe: {candidate.parent}")
    if candidate.parent.resolve(strict=True) != candidate.parent:
        raise RuntimeError(f"OCR retry evidence source parent traverses a link: {candidate.parent}")
    file_stat = candidate.lstat()
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or stat.S_ISLNK(file_stat.st_mode)
        or _stat_has_reparse_point(file_stat)
        or int(file_stat.st_nlink) != 1
        or candidate.resolve(strict=True) != candidate
    ):
        raise RuntimeError(f"OCR retry evidence source is not singly owned: {candidate}")
    if not 0 < file_stat.st_size <= max_bytes:
        raise RuntimeError(f"OCR retry evidence source exceeds byte limit: {candidate}")
    return candidate, file_stat


def _strict_retry_copy_target_parent(path: Path) -> tuple[Path, os.stat_result]:
    candidate = Path(os.path.abspath(path))
    missing: list[Path] = []
    current = candidate
    while not os.path.lexists(current):
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"OCR retry evidence target parent is unsafe: {candidate}")
        missing.append(current)
        current = parent

    for directory in [current, *reversed(missing)]:
        if directory != current:
            parent_stat = current.lstat()
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or stat.S_ISLNK(parent_stat.st_mode)
                or _stat_has_reparse_point(parent_stat)
                or current.resolve(strict=True) != current
            ):
                raise RuntimeError(f"OCR retry evidence target parent is unsafe: {candidate}")
            try:
                directory.mkdir()
            except OSError as exc:
                raise RuntimeError(
                    f"OCR retry evidence target parent cannot be created safely: {candidate}"
                ) from exc
            current = directory

        directory_stat = directory.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or _stat_has_reparse_point(directory_stat)
            or directory.resolve(strict=True) != directory
        ):
            raise RuntimeError(f"OCR retry evidence target parent is unsafe: {candidate}")
    return candidate, directory_stat


def _strict_retry_copy_target_parent_seal(path: Path, *, expected: os.stat_result) -> None:
    candidate = Path(os.path.abspath(path))
    try:
        current = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"OCR retry evidence target parent changed before copying: {candidate}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or _stat_has_reparse_point(current)
        or candidate.resolve(strict=True) != candidate
        or any(
            getattr(current, field) != getattr(expected, field)
            for field in _STABLE_DIRECTORY_STAT_FIELDS
        )
    ):
        raise RuntimeError(f"OCR retry evidence target parent changed before copying: {candidate}")


def _copy_retry_evidence(*, source: Path, target: Path, max_bytes: int) -> Path:
    source_path, source_stat = _strict_retry_copy_source(source, max_bytes=max_bytes)
    target_path = Path(os.path.abspath(target))
    _target_parent, target_parent_stat = _strict_retry_copy_target_parent(target_path.parent)
    if os.path.lexists(target_path):
        raise RuntimeError(f"OCR retry evidence target already exists: {target_path}")
    digest, copied = _copy_bounded_snapshot_file(
        source=source_path,
        source_stat=source_stat,
        destination=target_path,
        max_bytes=max_bytes,
        destination_parent_stat=target_parent_stat,
    )
    target_stat: os.stat_result | None = None
    try:
        target_stat = target_path.lstat()
        target_is_valid = (
            stat.S_ISREG(target_stat.st_mode)
            and not stat.S_ISLNK(target_stat.st_mode)
            and not _stat_has_reparse_point(target_stat)
            and int(target_stat.st_nlink) == 1
            and target_path.resolve(strict=True) == target_path
            and _stable_file_fingerprint(target_path) == {"sha256": digest, "bytes": copied}
        )
    except Exception:
        if target_stat is not None:
            _remove_failed_snapshot_copy(target_path, expected=target_stat)
        raise
    if not target_is_valid:
        _remove_failed_snapshot_copy(target_path, expected=target_stat)
        raise RuntimeError(f"OCR retry evidence target seal is invalid: {target_path}")
    return target_path


def _is_reparse_point(path: Path) -> bool:
    return _stat_has_reparse_point(path.lstat())


def _stat_has_reparse_point(path_stat: os.stat_result) -> bool:
    attributes = int(getattr(path_stat, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def _strict_snapshot_directory(
    path: Path,
    *,
    label: str,
    expected: os.stat_result | None = None,
) -> os.stat_result:
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Surya failure evidence {label} is missing") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or _stat_has_reparse_point(current)
    ):
        raise RuntimeError(f"Surya failure evidence {label} is unsafe")
    if expected is not None:
        if any(
            getattr(current, field) != getattr(expected, field)
            for field in _STABLE_DIRECTORY_STAT_FIELDS
        ):
            raise RuntimeError(f"Surya failure evidence {label} changed while copying")
    return current


def _copy_bounded_snapshot_file(
    *,
    source: Path,
    source_stat: os.stat_result,
    destination: Path,
    max_bytes: int,
    destination_parent_stat: os.stat_result | None = None,
) -> tuple[str, int]:
    if max_bytes < 0 or source_stat.st_size > max_bytes:
        raise RuntimeError("Surya failure evidence exceeds the byte limit")
    digest = hashlib.sha256()
    copied = 0
    if destination_parent_stat is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination_created_stat: os.stat_result | None = None
    try:
        with source.open("rb") as source_stream:
            opened_stat = os.fstat(source_stream.fileno())
            if any(
                getattr(opened_stat, field) != getattr(source_stat, field)
                for field in _STABLE_FILE_STAT_FIELDS
            ):
                raise RuntimeError(f"Surya failure evidence changed before copying: {source}")
            if destination_parent_stat is not None:
                _strict_retry_copy_target_parent_seal(
                    destination.parent,
                    expected=destination_parent_stat,
                )
            with destination.open("xb") as destination_stream:
                destination_created_stat = os.fstat(destination_stream.fileno())
                if (
                    not stat.S_ISREG(destination_created_stat.st_mode)
                    or int(destination_created_stat.st_nlink) != 1
                ):
                    raise RuntimeError(
                        f"Surya failure evidence destination is unsafe: {destination}"
                    )
                while True:
                    read_size = min(1024 * 1024, max_bytes - copied + 1)
                    block = source_stream.read(read_size)
                    if not block:
                        break
                    copied += len(block)
                    if copied > max_bytes:
                        raise RuntimeError("Surya failure evidence exceeds the byte limit")
                    destination_stream.write(block)
                    digest.update(block)
            descriptor_after = os.fstat(source_stream.fileno())
        path_after = source.lstat()
        if any(
            getattr(record, field) != getattr(source_stat, field)
            for record in (descriptor_after, path_after)
            for field in _STABLE_FILE_STAT_FIELDS
        ):
            raise RuntimeError(f"Surya failure evidence changed while copying: {source}")
        if copied != source_stat.st_size:
            raise RuntimeError(
                f"Surya failure evidence file ended before its sealed size: {source}"
            )
        destination_fingerprint = _stable_file_fingerprint(destination)
        expected_fingerprint = {"sha256": digest.hexdigest(), "bytes": copied}
        if destination_fingerprint != expected_fingerprint:
            raise RuntimeError(f"Surya failure evidence copy is invalid: {source}")
        return digest.hexdigest(), copied
    except Exception:
        if destination_created_stat is not None:
            _remove_failed_snapshot_copy(destination, expected=destination_created_stat)
        raise


def _remove_failed_snapshot_copy(path: Path, *, expected: os.stat_result) -> None:
    try:
        current = path.lstat()
        if (
            stat.S_ISREG(current.st_mode)
            and not stat.S_ISLNK(current.st_mode)
            and not _stat_has_reparse_point(current)
            and int(current.st_nlink) == 1
            and current.st_dev == expected.st_dev
            and current.st_ino == expected.st_ino
            and path.resolve(strict=True) == path
        ):
            path.unlink()
    except (OSError, RuntimeError):
        return


def _snapshot_surya_failure_evidence(
    *,
    source_root: Path,
    trusted_root: Path,
    output_dir: Path,
    pdf_path: Path,
    sample_pages_1based: Sequence[int],
    dpi: int,
    lang: str,
    error: str,
) -> dict[str, object] | None:
    lexical_trusted_root = Path(os.path.abspath(trusted_root))
    lexical_source_root = Path(os.path.abspath(source_root))
    if lexical_source_root.parent != lexical_trusted_root:
        raise RuntimeError("Surya failure evidence root is outside its trusted runtime root")
    trusted_stat = _strict_snapshot_directory(
        lexical_trusted_root,
        label="trusted runtime root",
    )
    try:
        lexical_source_root.lstat()
    except FileNotFoundError:
        return None
    source_root_stat = _strict_snapshot_directory(
        lexical_source_root,
        label="source root",
    )
    source_root = lexical_source_root
    sealed_pdf_path = pdf_path.resolve(strict=True)
    source_pdf_fingerprint = _stable_file_fingerprint(sealed_pdf_path)
    token = uuid.uuid4().hex
    final_root = output_dir / f"{pdf_path.stem}_surya_failure_evidence_{token}"
    staging_root = output_dir / f".{final_root.name}.tmp"
    entries: list[dict[str, object]] = []
    entry_count = 0
    total_bytes = 0
    sealed_directories: list[tuple[Path, os.stat_result]] = []
    published = False
    manifest_fingerprint: dict[str, object] | None = None
    try:
        staging_root.mkdir(parents=False, exist_ok=False)
        payload_root = staging_root / "payload"
        payload_root.mkdir(exist_ok=False)
        pending = [(source_root, Path(), source_root_stat)]
        while pending:
            _strict_snapshot_directory(
                lexical_trusted_root,
                label="trusted runtime root",
                expected=trusted_stat,
            )
            source_dir, relative_dir, sealed_directory_stat = pending.pop()
            directory_stat = _strict_snapshot_directory(
                source_dir,
                label=f"directory {relative_dir}",
                expected=sealed_directory_stat,
            )
            sealed_directories.append((source_dir, directory_stat))
            directory_names: list[str] = []
            with os.scandir(source_dir) as iterator:
                for directory_entry in iterator:
                    entry_count += 1
                    if entry_count > _MAX_SURYA_FAILURE_EVIDENCE_ENTRIES:
                        raise RuntimeError("Surya failure evidence exceeds the entry-count limit")
                    directory_names.append(directory_entry.name)
            for directory_name in sorted(directory_names):
                source = source_dir / directory_name
                relative = relative_dir / directory_name
                if len(relative.as_posix()) > _MAX_SURYA_FAILURE_EVIDENCE_RELATIVE_PATH_CHARS:
                    raise RuntimeError(
                        "Surya failure evidence exceeds the relative-path-length limit"
                    )
                source_stat = source.lstat()
                if stat.S_ISLNK(source_stat.st_mode) or _stat_has_reparse_point(source_stat):
                    raise RuntimeError(f"Surya failure evidence contains a link: {relative}")
                if stat.S_ISDIR(source_stat.st_mode):
                    (payload_root / relative).mkdir(exist_ok=False)
                    pending.append((source, relative, source_stat))
                    continue
                if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
                    raise RuntimeError(
                        "Surya failure evidence is not an exclusively owned file: "
                        f"{relative} (nlink={source_stat.st_nlink})"
                    )
                if len(entries) >= _MAX_SURYA_FAILURE_EVIDENCE_FILES:
                    raise RuntimeError("Surya failure evidence exceeds the file-count limit")
                destination = payload_root / relative
                source_sha256, source_bytes = _copy_bounded_snapshot_file(
                    source=source,
                    source_stat=source_stat,
                    destination=destination,
                    max_bytes=_MAX_SURYA_FAILURE_EVIDENCE_BYTES - total_bytes,
                )
                entries.append(
                    {
                        "path": (Path("payload") / relative).as_posix(),
                        "sha256": source_sha256,
                        "bytes": source_bytes,
                    }
                )
                total_bytes += source_bytes
            _strict_snapshot_directory(
                source_dir,
                label=f"directory {relative_dir}",
                expected=directory_stat,
            )
        if not entries:
            raise RuntimeError("Surya failure evidence tree is empty")
        entries.sort(key=lambda item: str(item["path"]))
        for sealed_path, sealed_stat in sealed_directories:
            _strict_snapshot_directory(
                sealed_path,
                label=f"directory {sealed_path}",
                expected=sealed_stat,
            )
        _strict_snapshot_directory(
            lexical_trusted_root,
            label="trusted runtime root",
            expected=trusted_stat,
        )
        if _stable_file_fingerprint(sealed_pdf_path) != source_pdf_fingerprint:
            raise RuntimeError("Surya failure evidence source PDF changed while copying")
        manifest = {
            "schema": _SURYA_FAILURE_EVIDENCE_SCHEMA,
            "status": "sealed",
            "engine": OCR_ENGINE_SURYA,
            "source_pdf": {
                "path": str(sealed_pdf_path),
                **source_pdf_fingerprint,
            },
            "sample_pages": list(sample_pages_1based),
            "dpi": int(dpi),
            "lang": lang,
            "retry_policy": _ZERO_OUTPUT_RETRY_POLICY,
            "original_error": error,
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "payload_root": "payload",
            "limits": {
                "max_files": _MAX_SURYA_FAILURE_EVIDENCE_FILES,
                "max_entries": _MAX_SURYA_FAILURE_EVIDENCE_ENTRIES,
                "max_relative_path_chars": _MAX_SURYA_FAILURE_EVIDENCE_RELATIVE_PATH_CHARS,
                "max_bytes": _MAX_SURYA_FAILURE_EVIDENCE_BYTES,
            },
            "files": entries,
        }
        _write_json_atomic(staging_root / "manifest.json", manifest)
        os.replace(staging_root, final_root)
        published = True
        manifest_fingerprint = _stable_file_fingerprint(final_root / "manifest.json")
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        if published:
            shutil.rmtree(final_root, ignore_errors=True)
        raise
    assert manifest_fingerprint is not None
    manifest_path = final_root / "manifest.json"
    return {
        "schema": _SURYA_FAILURE_EVIDENCE_SCHEMA,
        "status": "saved",
        "engine": OCR_ENGINE_SURYA,
        "evidence_root": str(final_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_fingerprint["sha256"],
        "manifest_bytes": manifest_fingerprint["bytes"],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "original_error": error,
    }


def _attempt_history_record(
    *,
    attempt: int,
    preprocessing: str,
    ocr_outcome: str,
    image_path: Path,
    sidecar_path: Path,
    image_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = dict(image_evidence or _image_attempt_evidence(image_path))
    evidence.update(
        {
            "attempt": attempt,
            "image_path": str(image_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
            "preprocessing": preprocessing,
            "ocr_outcome": ocr_outcome,
            "sidecar_sha256": _sha256_path(sidecar_path),
            "sidecar_bytes": int(sidecar_path.stat().st_size),
        }
    )
    return evidence


def _attach_surya_retry_provenance(
    *,
    page_metadata: Sequence[dict[str, Any]],
    source_page: int,
    selected_attempt: int,
    attempt_history: Sequence[dict[str, object]],
) -> None:
    rows = [item for item in page_metadata if item.get("source_page") == source_page]
    if len(rows) != 1:
        raise RuntimeError("Cannot attach Surya retry provenance without exact page metadata.")
    row = rows[0]
    raw_sidecar = row.get("surya_page_lines_path")
    if not isinstance(raw_sidecar, str):
        raise RuntimeError("Cannot attach Surya retry provenance without a geometry sidecar.")
    sidecar_path = Path(raw_sidecar)
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot attach Surya retry provenance: {exc}") from exc
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError("Cannot attach Surya retry provenance to malformed geometry.")
    history = [dict(item) for item in attempt_history]
    fields: dict[str, object] = {
        "selected_attempt": selected_attempt,
        "retry_policy": _ZERO_OUTPUT_RETRY_POLICY,
        "attempt_history": history,
    }
    row.update(fields)
    images[0].update(fields)
    _write_json_atomic(sidecar_path, payload)


def _read_utf8_artifact(path: Path) -> str:
    with path.open("rb") as stream:
        payload = stream.read(_MAX_OCR_TEXT_ARTIFACT_BYTES + 1)
    if len(payload) > _MAX_OCR_TEXT_ARTIFACT_BYTES:
        raise RuntimeError(
            f"OCR text artifact exceeds {_MAX_OCR_TEXT_ARTIFACT_BYTES} bytes: {path}"
        )
    return payload.decode("utf-8-sig")


def _is_effectively_blank_page_image(path: Path) -> bool:
    try:
        with Image.open(path) as source:
            source.load()
            return _is_effectively_blank_rgb_image(source.convert("RGB"))
    except Exception:
        return False


@dataclass(slots=True)
class OcrBenchmarkResult:
    engine: str
    status: str
    sample_pages: list[int]
    elapsed_seconds: float
    artifact_path: str | None
    text_chars: int
    memory_delta_mb: float | None = None
    error: str | None = None
    note: str | None = None
    page_error_count: int = 0

    @property
    def label(self) -> str:
        return OCR_ENGINE_LABELS.get(self.engine, self.engine)


BenchmarkProgressCallback = Callable[[int, str], None]
PageProgressCallback = Callable[[int, int, int], None]
ImportModule = Callable[[str], Any]
WhichExecutable = Callable[[str], str | None]
RunCommand = Callable[..., Any]
ExtractionFunction = Callable[..., tuple[str, int]]


def sample_pdf_page_indices(page_count: int, *, sample_size: int = 5) -> list[int]:
    """Pick an evenly distributed page sample without loading the whole PDF."""
    if page_count <= 0:
        return []

    target = max(1, int(sample_size))
    if page_count <= target:
        return list(range(page_count))

    if target == 1:
        return [0]

    indices: list[int] = []
    for index in range(target):
        # Even spread across [0, page_count - 1], including both ends.
        page_index = round(index * (page_count - 1) / (target - 1))
        if page_index not in indices:
            indices.append(page_index)

    if len(indices) < target:
        for page_index in range(page_count):
            if page_index in indices:
                continue
            indices.append(page_index)
            if len(indices) == target:
                break

    return indices


def resolve_pdf_page_indices(
    page_count: int,
    *,
    sample_size: int = 5,
    page_numbers: Sequence[int] | None = None,
) -> list[int]:
    """Resolve 0-based page indices either from explicit page numbers or sampled spread."""
    if page_count <= 0:
        return []

    if page_numbers is not None:
        resolved: list[int] = []
        seen: set[int] = set()
        for raw_page in page_numbers:
            page = int(raw_page)
            if page < 1:
                raise ValueError(f"Invalid page number: {page}. Page numbers must be >= 1.")
            page_index = page - 1
            if page_index >= page_count:
                raise ValueError(
                    f"Invalid page number: {page}. PDF has {page_count} pages (valid range is 1..{page_count})."
                )
            if page_index in seen:
                continue
            seen.add(page_index)
            resolved.append(page_index)

        if not resolved:
            raise ValueError("No valid page numbers were provided.")
        return resolved

    return sample_pdf_page_indices(page_count, sample_size=sample_size)


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(
            "PDF import requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(pdf_path))
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _collect_text_strings(value: Any) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        try:
            return [value.decode("utf-8-sig")]
        except UnicodeDecodeError:
            return []
    if isinstance(value, dict):
        for item in value.values():
            texts.extend(_collect_text_strings(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_collect_text_strings(item))
        return texts

    for attr in ("text", "rec_text", "transcription", "content", "label"):
        if hasattr(value, attr):
            texts.extend(_collect_text_strings(getattr(value, attr)))
    return texts


_PADDLEOCR_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "en": "en",
    "english": "en",
    "rus": "ru",
    "ru": "ru",
    "russian": "ru",
    "deu": "german",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn": "japan",
    "kor": "korean",
    "ara": "ar",
}


def _paddleocr_lang(lang: str) -> str:
    """Map Tesseract-style language codes to PaddleOCR identifiers.

    Handles multi-language specs like ``rus+eng`` by returning the first
    mapped language (PaddleOCR does not support multi-lang in a single call).
    """
    for part in lang.split("+"):
        normalized = part.strip().lower()
        if normalized in _PADDLEOCR_LANG_MAP:
            return _PADDLEOCR_LANG_MAP[normalized]
    # Fallback: return the first component as-is.
    return lang.split("+")[0].strip().lower()


def _render_sample_paths(
    pdf_path: Path,
    sample_pages: Sequence[int],
    *,
    dpi: int,
    tmp_dir: Path,
) -> list[Path]:
    image_paths: list[Path] = []
    for idx, (_name, image) in enumerate(
        iter_render_pdf_page_indices(pdf_path, sample_pages, dpi=dpi),
        start=1,
    ):
        out_path = tmp_dir / f"{idx:05d}.png"
        if not imwrite_unicode(out_path, image):
            raise RuntimeError(f"Failed to write sampled page image: {out_path}")
        image_paths.append(out_path)
    return image_paths


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(
            "PDF import requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(pdf_path))
    try:
        parts: list[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                parts.append(page_text)
        return "\n".join(parts)
    finally:
        doc.close()


def _extract_pdf_text_chars(pdf_path: Path) -> int:
    return len(_extract_pdf_text(pdf_path))


def _memory_rss_mb() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _memory_delta_mb(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 3)


def _emit_benchmark_progress(
    cb: BenchmarkProgressCallback | None,
    percent: int,
    status: str,
) -> None:
    if cb is None:
        return
    try:
        cb(max(0, min(100, int(percent))), status)
    except Exception:
        return


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _hide_gpu_visibility(environment: MutableMapping[str, str]) -> None:
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    environment.pop("UNISCAN_GPU_DEVICE_ID", None)


def _configure_cpu_only_runtime() -> str:
    _hide_gpu_visibility(os.environ)
    os.environ["TORCH_DEVICE"] = "cpu"
    return "cpu"


def _require_gpu0_contract(
    run_cmd: RunCommand = subprocess.run,
) -> str:
    configured_uuid = (os.environ.get("UNISCAN_GPU_DEVICE_ID") or "").strip()
    if configured_uuid != _EXPECTED_GPU0_UUID:
        raise RuntimeError(
            "UNISCAN_GPU_DEVICE_ID must identify the permitted GPU0 UUID "
            f"({_EXPECTED_GPU0_UUID}); got {configured_uuid or '<unset>'}."
        )
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if visible != "0":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly 0; got {visible or '<unset>'}.")

    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = run_cmd(command, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"GPU0 attestation failed: {exc}") from exc
    if int(getattr(completed, "returncode", 1)) != 0:
        detail = (getattr(completed, "stderr", "") or "").strip() or "nvidia-smi failed"
        raise RuntimeError(f"GPU0 attestation failed: {detail}")
    rows = [
        line.strip()
        for line in (getattr(completed, "stdout", "") or "").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"GPU0 attestation must return exactly one row; got {len(rows)}.")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or fields[0] != "0" or fields[1] != _EXPECTED_GPU0_UUID:
        raise RuntimeError(f"GPU0 attestation mismatch: {rows[0]!r}.")
    return _EXPECTED_GPU0_UUID


def _torch_cuda_probe() -> tuple[bool, str]:
    try:
        _require_gpu0_contract()
    except RuntimeError as exc:
        return False, f"GPU0 contract failed: {exc}"
    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed: {exc}"

    version = str(getattr(torch, "__version__", "unknown"))
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return False, f"torch.cuda probe failed: {exc}"
    if not available:
        return False, f"torch={version} cuda_available=False"

    # When real torch is available, run a tiny CUDA tensor to catch incompatible
    # wheels before a large OCR model starts loading.
    if hasattr(torch, "ones"):
        try:
            tensor = torch.ones((1,), device="cuda")
            _ = float(tensor.item())
        except Exception as exc:
            return False, f"torch={version} cuda tensor smoke failed: {exc}"

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0
    try:
        device_name = str(torch.cuda.get_device_name(0)) if device_count > 0 else "unknown"
    except Exception:
        device_name = "unknown"
    return True, f"torch={version} cuda_device_count={device_count} cuda_device_0={device_name}"


def _require_torch_cuda(engine_label: str) -> str:
    has_cuda, detail = _torch_cuda_probe()
    if not has_cuda:
        raise RuntimeError(
            f"{engine_label} GPU mode requires CUDA, but CUDA is unavailable. "
            f"Install a CUDA-enabled torch build in the engine venv ({detail})."
        )
    return detail


def _has_any_files(root: Path, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        try:
            next(root.rglob(pattern))
            return True
        except StopIteration:
            continue
    return False


def _preview_paths(paths: Sequence[Path], *, limit: int = 3) -> str:
    if not paths:
        return ""
    shown = [str(path) for path in paths[:limit]]
    suffix = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return "; ".join(shown) + suffix


def _resolve_hf_home() -> Path:
    raw = (os.environ.get("HF_HOME") or "").strip()
    if raw:
        return Path(raw)
    return _DEFAULT_HF_CACHE_HOME


def _chandra_model_cache_candidates() -> tuple[Path, ...]:
    model_key = _CHANDRA_MODEL_REPO_ID.replace("/", "--")
    hf_home = _resolve_hf_home()
    return (
        hf_home / f"models--{model_key}",
        hf_home / "hub" / f"models--{model_key}",
    )


def _candidate_snapshot_dirs(candidate_root: Path) -> list[Path]:
    snapshots_dir = candidate_root / "snapshots"
    if not snapshots_dir.exists():
        return []
    return [path for path in snapshots_dir.iterdir() if path.is_dir()]


def _snapshot_has_chandra_weights(snapshot_dir: Path) -> bool:
    has_single_file_weights = (snapshot_dir / "model.safetensors").is_file()
    has_sharded_weights_index = (snapshot_dir / "model.safetensors.index.json").is_file()
    if not has_single_file_weights and not has_sharded_weights_index:
        return False
    try:
        next(snapshot_dir.rglob("*.safetensors"))
        return True
    except StopIteration:
        return False


def _ensure_chandra_cache_ready() -> None:
    hf_home = _resolve_hf_home().resolve()
    cache_key = f"chandra::{hf_home}"
    if cache_key in _MODEL_CACHE_CHECK_MEMO:
        return

    missing_candidates: list[str] = []
    candidate_issues: list[str] = []
    for candidate in _chandra_model_cache_candidates():
        candidate = candidate.resolve()
        if not candidate.exists():
            missing_candidates.append(str(candidate))
            continue

        blobs_dir = candidate / "blobs"
        incomplete_files = sorted(blobs_dir.glob("*.incomplete")) if blobs_dir.exists() else []
        if incomplete_files:
            preview = _preview_paths(incomplete_files)
            candidate_issues.append(f"{candidate}: found incomplete weight downloads: {preview}")
            continue

        snapshots = _candidate_snapshot_dirs(candidate)
        if not snapshots:
            candidate_issues.append(f"{candidate}: snapshots directory is empty or missing")
            continue

        if not any(_snapshot_has_chandra_weights(snapshot) for snapshot in snapshots):
            snapshot_preview = _preview_paths(snapshots)
            candidate_issues.append(
                f"{candidate}: no '*.safetensors' weights found in snapshots ({snapshot_preview})"
            )
            continue

        _MODEL_CACHE_CHECK_MEMO[cache_key] = str(candidate)
        return

    details: list[str] = []
    if missing_candidates:
        details.append("missing cache dirs: " + "; ".join(missing_candidates))
    if candidate_issues:
        details.extend(candidate_issues)
    detail_text = " | ".join(details) if details else "model cache is unavailable"
    raise RuntimeError(
        "Chandra cache/weights preflight failed. "
        f"Expected local model '{_CHANDRA_MODEL_REPO_ID}' in HF cache ({hf_home}). {detail_text}"
    )


def _iter_surya_version_dirs(component_root: Path) -> list[Path]:
    if not component_root.exists():
        return []
    return [path for path in component_root.iterdir() if path.is_dir()]


def _surya_version_ready(version_dir: Path) -> bool:
    manifest = version_dir / "manifest.json"
    if not manifest.is_file():
        return False
    return _has_any_files(version_dir, ("*.safetensors", "*.onnx", "*.pt", "*.bin"))


def _ensure_surya_cache_ready() -> None:
    model_cache_raw = (os.environ.get("MODEL_CACHE_DIR") or "").strip()
    model_cache_root = Path(model_cache_raw) if model_cache_raw else _DEFAULT_SURYA_MODEL_CACHE_HOME
    model_cache_root = model_cache_root.resolve()

    cache_key = f"surya::{model_cache_root}"
    if cache_key in _MODEL_CACHE_CHECK_MEMO:
        return

    # OCR text path requires text detection + recognition. Layout weights are
    # used by other Surya tasks and can be absent in minimal OCR-only setups.
    required_components = ("text_detection", "text_recognition")
    problems: list[str] = []
    for component in required_components:
        component_root = model_cache_root / component
        versions = _iter_surya_version_dirs(component_root)
        if not versions:
            problems.append(f"{component_root}: missing component cache directory")
            continue
        if not any(_surya_version_ready(version) for version in versions):
            version_preview = _preview_paths(versions)
            problems.append(
                f"{component_root}: no ready version with manifest + weights ({version_preview})"
            )

    incomplete_files = (
        sorted(model_cache_root.rglob("*.incomplete")) if model_cache_root.exists() else []
    )
    if incomplete_files:
        problems.append(
            f"{model_cache_root}: incomplete files present: {_preview_paths(incomplete_files)}"
        )

    if problems:
        raise RuntimeError("Surya cache/weights preflight failed. " + " | ".join(problems))

    _MODEL_CACHE_CHECK_MEMO[cache_key] = str(model_cache_root)


def _configure_chandra_runtime_device() -> str:
    """Resolve TORCH_DEVICE for Chandra before importing chandra.settings."""

    explicit = (os.environ.get("TORCH_DEVICE") or "").strip()
    require_gpu = _env_bool("UNISCAN_CHANDRA_REQUIRE_GPU", default=True)
    default_policy = "cuda" if require_gpu else "auto"
    device_policy = (
        (os.environ.get("UNISCAN_CHANDRA_DEVICE_POLICY") or default_policy).strip().lower()
    )
    if explicit:
        normalized_explicit = explicit.lower()
        if require_gpu:
            if normalized_explicit not in {"cuda", "cuda:0"}:
                raise RuntimeError(
                    f"Chandra GPU mode requires TORCH_DEVICE='cuda:0'; got {explicit!r}."
                )
            _require_torch_cuda("Chandra")
            os.environ["TORCH_DEVICE"] = "cuda:0"
            return "cuda:0"
        if normalized_explicit not in {"auto", "cpu"}:
            raise RuntimeError(
                "Chandra optional/CPU mode forbids a GPU TORCH_DEVICE; "
                "set UNISCAN_CHANDRA_REQUIRE_GPU=1 for CUDA."
            )
        return _configure_cpu_only_runtime()

    if device_policy == "auto":
        if require_gpu:
            raise RuntimeError(
                "Chandra GPU mode forbids UNISCAN_CHANDRA_DEVICE_POLICY='auto'; use 'cuda'."
            )
        return _configure_cpu_only_runtime()

    if device_policy == "cpu":
        if require_gpu:
            raise RuntimeError(
                "Chandra GPU mode requires CUDA, but UNISCAN_CHANDRA_DEVICE_POLICY='cpu' was requested."
            )
        return _configure_cpu_only_runtime()

    if device_policy == "cuda":
        if not require_gpu:
            raise RuntimeError(
                "Chandra optional/CPU mode forbids UNISCAN_CHANDRA_DEVICE_POLICY='cuda'; "
                "set UNISCAN_CHANDRA_REQUIRE_GPU=1."
            )
        _require_torch_cuda("Chandra")
        os.environ["TORCH_DEVICE"] = "cuda:0"
        return "cuda:0"

    if device_policy not in {"", "legacy"}:
        raise RuntimeError(
            "Unsupported UNISCAN_CHANDRA_DEVICE_POLICY. Use one of: auto, cuda, cpu."
        )

    if require_gpu:
        _require_torch_cuda("Chandra")
        os.environ["TORCH_DEVICE"] = "cuda:0"
        return "cuda:0"

    return _configure_cpu_only_runtime()


def _configure_surya_runtime_device() -> str:
    """Resolve TORCH_DEVICE for Surya and enforce CUDA when requested."""

    explicit = (os.environ.get("TORCH_DEVICE") or "").strip()
    require_gpu = _env_bool("UNISCAN_SURYA_REQUIRE_GPU", default=True)

    if explicit:
        normalized_explicit = explicit.lower()
        if require_gpu:
            if normalized_explicit not in {"cuda", "cuda:0"}:
                raise RuntimeError(
                    f"Surya GPU mode requires TORCH_DEVICE='cuda:0'; got {explicit!r}."
                )
            _require_torch_cuda("Surya")
            os.environ["TORCH_DEVICE"] = "cuda:0"
            return "cuda:0"
        if normalized_explicit not in {"auto", "cpu"}:
            raise RuntimeError(
                "Surya optional/CPU mode forbids a GPU TORCH_DEVICE; "
                "set UNISCAN_SURYA_REQUIRE_GPU=1 for CUDA."
            )
        return _configure_cpu_only_runtime()

    if require_gpu:
        _require_torch_cuda("Surya")
        os.environ["TORCH_DEVICE"] = "cuda:0"
        return "cuda:0"

    return _configure_cpu_only_runtime()


def _artifact_path_for_engine(output_dir: Path, pdf_stem: str, engine: str) -> Path:
    suffix = ".pdf" if engine in SEARCHABLE_PDF_ENGINES else ".txt"
    return output_dir / f"{pdf_stem}_{engine}{suffix}"


_CHANDRA_NON_TEXT_LABELS: set[str] = {
    "blank-page",
    "image",
    "figure",
    "diagram",
}
_CHANDRA_GRAPHIC_LABELS: set[str] = {"image", "figure", "diagram"}
_CHANDRA_HEADER_FOOTER_LABELS: set[str] = {"page-header", "page-footer"}
_CHANDRA_VISIBLE_TEXT_TAGS: set[str] = {
    "blockquote",
    "caption",
    "chem",
    "code",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "math",
    "p",
    "pre",
    "td",
    "th",
}
_CHANDRA_BLOCKED_MEDIA_TAGS: set[str] = {"div", "noscript", "script", "style", "svg"}


class _ChandraVisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_parts: list[str] = []
        self.visible_depth = 0
        self.blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in _CHANDRA_BLOCKED_MEDIA_TAGS:
            self.blocked_depth += 1
            self.ignored_parts.append("\n")
        if normalized in _CHANDRA_VISIBLE_TEXT_TAGS:
            self.visible_depth += 1
            self.parts.append("\n")
        elif normalized == "br" and self.visible_depth > 0 and self.blocked_depth == 0:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _CHANDRA_VISIBLE_TEXT_TAGS and self.visible_depth > 0:
            self.parts.append("\n")
            self.visible_depth -= 1
        if normalized in _CHANDRA_BLOCKED_MEDIA_TAGS and self.blocked_depth > 0:
            self.ignored_parts.append("\n")
            self.blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.blocked_depth > 0:
            self.ignored_parts.append(data)
        elif self.visible_depth > 0:
            self.parts.append(data)


def _chandra_graphic_chunk_lines(raw_content: Any) -> list[str]:
    if raw_content is None:
        return []
    parser = _ChandraVisibleTextParser()
    try:
        parser.feed(str(raw_content))
        parser.close()
    except Exception:
        return []
    return _chandra_chunk_lines("".join(parser.parts))


def _chandra_graphic_chunk_ignored_lines(raw_content: Any) -> list[str]:
    if raw_content is None:
        return []
    parser = _ChandraVisibleTextParser()
    try:
        parser.feed(str(raw_content))
        parser.close()
    except Exception:
        return []
    return _chandra_chunk_lines("".join(parser.ignored_parts))


def _chandra_chunk_lines(raw_content: Any) -> list[str]:
    if raw_content is None:
        return []

    raw = str(raw_content).replace("\u00a0", " ")
    if not raw.strip():
        return []

    # Preserve explicit line boundaries from lightweight HTML payload.
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(
        r"(?i)</(p|div|li|h[1-6]|tr|caption|table|ul|ol|pre|code|blockquote)>",
        "\n",
        raw,
    )
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines


def _chandra_canonical_alnum(lines: Sequence[str]) -> str:
    normalized = unicodedata.normalize("NFKC", "\n".join(lines))
    normalized = unicodedata.normalize("NFKC", normalized.casefold())
    return "".join(char for char in normalized if char.isalnum())


def _chandra_normalized_text(lines: Sequence[str]) -> str:
    return unicodedata.normalize("NFKC", "\n".join(lines))


def _chandra_alternative_text_matches(candidate: str, target: str) -> bool:
    if candidate == target:
        return True
    candidate_canonical = _chandra_canonical_alnum([candidate])
    target_canonical = _chandra_canonical_alnum([target])
    if (
        not candidate_canonical
        or not target_canonical
        or len(candidate_canonical) > len(target_canonical)
        or len(candidate_canonical) * 100
        < len(target_canonical) * _CHANDRA_ALTERNATIVE_TEXT_MIN_COVERAGE_PERCENT
    ):
        return False
    target_index = 0
    for char in candidate_canonical:
        while target_index < len(target_canonical) and target_canonical[target_index] != char:
            target_index += 1
        if target_index == len(target_canonical):
            return False
        target_index += 1
    return True


def _chandra_alternative_text_evidence(
    *,
    raw_result: dict[str, Any],
    texts: Sequence[str],
    ignored_graphic_lines: Sequence[str],
    alternative_texts: Sequence[str] | None = None,
    excluded_header_footer_lines: Sequence[str] = (),
) -> dict[str, str]:
    html_lines = _chandra_chunk_lines(raw_result.get("html"))
    raw_markdown = str(raw_result.get("markdown") or "").strip()
    markdown_lines = _chandra_chunk_lines(_strip_markdown(raw_markdown))
    html_text = _chandra_normalized_text(html_lines)
    markdown_text = _chandra_normalized_text(markdown_lines)
    parsed_text = _chandra_normalized_text(texts)
    alternative_parsed_text = _chandra_normalized_text(
        texts if alternative_texts is None else alternative_texts
    )
    excluded_header_footer_text = _chandra_normalized_text(excluded_header_footer_lines)
    ignored_text = _chandra_normalized_text(ignored_graphic_lines)
    alternatives = [value for value in (html_text, markdown_text) if value]
    if excluded_header_footer_text and not alternatives and not alternative_parsed_text:
        accounting = "parsed_without_header_footer"
    elif not alternatives:
        accounting = "empty"
    elif parsed_text and all(value == parsed_text for value in alternatives):
        accounting = "parsed"
    elif excluded_header_footer_text and all(
        _chandra_alternative_text_matches(value, alternative_parsed_text)
        for value in alternatives
    ):
        accounting = "parsed_without_header_footer"
    elif parsed_text and all(
        _chandra_alternative_text_matches(value, parsed_text) for value in alternatives
    ):
        accounting = "parsed"
    elif ignored_text and all(value == ignored_text for value in alternatives):
        accounting = "ignored_graphic_description"
    else:
        accounting = "unaccounted"
    return {
        "policy": _CHANDRA_ALTERNATIVE_TEXT_POLICY,
        "accounting": accounting,
        "html_normalized_sha256": hashlib.sha256(html_text.encode("utf-8")).hexdigest(),
        "markdown_normalized_sha256": hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
        "ignored_graphic_normalized_sha256": hashlib.sha256(
            ignored_text.encode("utf-8")
        ).hexdigest(),
    }


def _wrap_text_to_target_chars(text: str, *, target_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    words = text.split()
    if len(words) <= 1:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and (current_len + extra) > target_chars:
            parts.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        parts.append(" ".join(current).strip())
    return [part for part in parts if part]


def _chandra_expand_chunk_to_line_boxes(
    *,
    lines: Sequence[str],
    bbox: Sequence[float],
) -> list[dict[str, object]]:
    if not lines:
        return []
    if len(bbox) != 4:
        return []
    try:
        x0 = float(bbox[0])
        y0 = float(bbox[1])
        x1 = float(bbox[2])
        y1 = float(bbox[3])
    except Exception:
        return []
    if x1 <= x0 or y1 <= y0:
        return []

    width = x1 - x0
    expanded_lines: list[str] = []
    # Approximate max chars per line from block width to avoid one huge
    # paragraph being placed into a single text row.
    target_chars = max(18, min(140, int(width / 7.2)))
    for raw_line in lines:
        expanded_lines.extend(_wrap_text_to_target_chars(raw_line, target_chars=target_chars))
    if not expanded_lines:
        return []

    line_count = len(expanded_lines)
    line_height = max((y1 - y0) / float(line_count), 1.0)
    # Small padding keeps glyph ascenders/descenders selectable.
    pad = min(1.2, line_height * 0.12)

    placements: list[dict[str, object]] = []
    for idx, line in enumerate(expanded_lines):
        ly0 = y0 + (line_height * idx)
        ly1 = y0 + (line_height * (idx + 1))
        by0 = max(y0, ly0 - pad)
        by1 = min(y1, ly1 + pad)
        if by1 <= by0:
            by0 = ly0
            by1 = ly1
        placements.append(
            {
                "text": line,
                "bbox": [x0, by0, x1, by1],
            }
        )
    return placements


def _chandra_allow_cli_fallback() -> bool:
    # By default, prefer deterministic module path with geometry sidecar.
    return _env_bool("UNISCAN_CHANDRA_ALLOW_CLI_FALLBACK", default=False)


def _surya_allow_text_fallback() -> bool:
    # By default, require geometry-producing Surya path.
    return _env_bool("UNISCAN_SURYA_ALLOW_TEXT_FALLBACK", default=False)


def _surya_require_geometry_sidecar() -> bool:
    # Geometry sidecar is mandatory for accurate searchable PDF alignment.
    return _env_bool("UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON", default=True)


def _chandra_require_sidecar() -> bool:
    return _env_bool("UNISCAN_CHANDRA_REQUIRE_SIDECAR", default=True)


def _markerized_pages_text(
    *,
    page_texts: Sequence[str],
    source_pages_1based: Sequence[int],
) -> str:
    blocks: list[str] = []
    for page_no, text in zip(source_pages_1based, page_texts, strict=True):
        blocks.append(f"[SOURCE PAGE {page_no:04d}]")
        if text:
            blocks.append(text.rstrip())
        blocks.append("")
    payload = "\n".join(blocks).strip()
    return payload + "\n" if payload else ""


def _persist_source_raster_artifact(
    *,
    engine_dir: Path,
    page_meta: dict[str, Any],
    source_page: int,
    engine: str,
    directory_suffix: str,
) -> dict[str, object]:
    raw_artifact = page_meta.get("source_raster_artifact")
    raw_identity = page_meta.get("source_raster_identity")
    if not isinstance(raw_artifact, dict) or not isinstance(raw_identity, dict):
        raise RuntimeError(f"{engine} page {source_page} source raster seal is malformed.")
    raw_path = raw_artifact.get("path")
    if not isinstance(raw_path, str):
        raise RuntimeError(f"{engine} page {source_page} source raster path is malformed.")
    source_raster = Path(raw_path)
    if not source_raster.is_file():
        raise RuntimeError(f"{engine} page {source_page} source raster artifact is missing.")
    observed_identity, observed_artifact, _observed_image = _source_raster_evidence(
        source_raster,
        source_page=source_page,
        require_rgb=True,
    )
    expected_source_identity = dict(raw_identity)
    expected_source_identity["name"] = source_raster.name
    if observed_artifact != raw_artifact or observed_identity != expected_source_identity:
        raise RuntimeError(f"{engine} page {source_page} source raster seal changed.")
    durable_source = _copy_retry_evidence(
        source=source_raster,
        target=engine_dir / f"page_{source_page:04d}.{directory_suffix}" / "source.png",
        max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
    )
    durable_identity, durable_artifact, durable_image = _source_raster_evidence(
        durable_source,
        source_page=source_page,
        require_rgb=True,
    )
    expected_durable_identity = dict(raw_identity)
    expected_durable_identity["name"] = "source.png"
    if (
        durable_identity != expected_durable_identity
        or durable_artifact["sha256"] != raw_artifact["sha256"]
        or durable_artifact["bytes"] != raw_artifact["bytes"]
        or durable_image.size != (raw_identity["width"], raw_identity["height"])
        or _canonical_rgb_pixel_sha256(durable_image) != raw_identity["pixel_sha256"]
        or _is_effectively_blank_rgb_image(durable_image) is not raw_identity["verified_blank"]
    ):
        raise RuntimeError(f"{engine} page {source_page} durable source raster pixels changed.")
    page_meta["source_raster_artifact"] = durable_artifact
    return durable_artifact


def _persist_surya_source_raster(
    *,
    engine_dir: Path,
    page_meta: dict[str, Any],
    source_page: int,
) -> None:
    durable_artifact = _persist_source_raster_artifact(
        engine_dir=engine_dir,
        page_meta=page_meta,
        source_page=source_page,
        engine="Surya",
        directory_suffix="surya-source",
    )
    raw_selected_sidecar = page_meta.get("surya_page_lines_path")
    if not isinstance(raw_selected_sidecar, str):
        raise RuntimeError(f"Surya page {source_page} selected sidecar is missing.")
    selected_sidecar = Path(raw_selected_sidecar)
    try:
        payload = json.loads(selected_sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Surya page {source_page} selected sidecar is unreadable: {exc}"
        ) from exc
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Surya page {source_page} selected sidecar is malformed.")
    if images[0].get("source_raster_identity") != page_meta.get("source_raster_identity"):
        raise RuntimeError(f"Surya page {source_page} selected source identity changed.")
    images[0]["source_raster_artifact"] = durable_artifact
    _write_json_atomic(selected_sidecar, payload)


def _persist_surya_retry_history(
    *,
    engine_dir: Path,
    page_meta: dict[str, Any],
    source_page: int,
) -> None:
    raw_history = page_meta.get("attempt_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise RuntimeError(f"Surya page {source_page} retry history is missing or malformed.")
    durable_history: list[dict[str, object]] = []
    seen_attempts: set[int] = set()
    for raw_item in raw_history:
        if not isinstance(raw_item, dict):
            raise RuntimeError(f"Surya page {source_page} retry history is malformed.")
        attempt = raw_item.get("attempt")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt <= 0
            or attempt in seen_attempts
        ):
            raise RuntimeError(f"Surya page {source_page} retry attempt is invalid.")
        seen_attempts.add(attempt)
        raw_image_path = raw_item.get("image_path")
        raw_sidecar_path = raw_item.get("sidecar_path")
        if not isinstance(raw_image_path, str) or not isinstance(raw_sidecar_path, str):
            raise RuntimeError(f"Surya page {source_page} retry history has no linked artifacts.")
        source_image = Path(raw_image_path)
        source_sidecar = Path(raw_sidecar_path)
        if not source_image.is_file() or not source_sidecar.is_file():
            raise RuntimeError(f"Surya page {source_page} retry history artifact is missing.")
        if raw_item.get("image_sha256") != _sha256_path(source_image):
            raise RuntimeError(f"Surya page {source_page} retry image digest changed.")
        if raw_item.get("sidecar_sha256") != _sha256_path(source_sidecar):
            raise RuntimeError(f"Surya page {source_page} retry sidecar digest changed.")
        with Image.open(source_image) as image:
            actual_image_size = [int(image.width), int(image.height)]
        if raw_item.get("image_size") != actual_image_size:
            raise RuntimeError(f"Surya page {source_page} retry image size changed.")

        attempt_dir = engine_dir / f"page_{source_page:04d}.retry" / f"attempt_{attempt}"
        durable_image = _copy_retry_evidence(
            source=source_image,
            target=attempt_dir / source_image.name,
            max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
        )
        durable_sidecar = _copy_retry_evidence(
            source=source_sidecar,
            target=attempt_dir / "surya_page_lines.json",
            max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
        )
        item = dict(raw_item)
        item.update(
            {
                "image_path": str(durable_image.resolve()),
                "sidecar_path": str(durable_sidecar.resolve()),
                "image_sha256": _sha256_path(durable_image),
                "image_bytes": int(durable_image.stat().st_size),
                "sidecar_sha256": _sha256_path(durable_sidecar),
                "sidecar_bytes": int(durable_sidecar.stat().st_size),
            }
        )
        durable_history.append(item)

    page_meta["attempt_history"] = durable_history
    raw_selected_sidecar = page_meta.get("surya_page_lines_path")
    if not isinstance(raw_selected_sidecar, str):
        raise RuntimeError(f"Surya page {source_page} selected sidecar is missing.")
    selected_sidecar = Path(raw_selected_sidecar)
    try:
        payload = json.loads(selected_sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Surya page {source_page} selected sidecar is unreadable: {exc}"
        ) from exc
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Surya page {source_page} selected sidecar is malformed.")
    for key in ("selected_attempt", "retry_policy", "attempt_history"):
        images[0][key] = page_meta[key]
    _write_json_atomic(selected_sidecar, payload)


def _persist_chandra_attempt_history(
    *,
    engine_dir: Path,
    page_meta: dict[str, Any],
    source_page: int,
) -> None:
    durable_source_artifact = _persist_source_raster_artifact(
        engine_dir=engine_dir,
        page_meta=page_meta,
        source_page=source_page,
        engine="Chandra",
        directory_suffix="chandra-source",
    )

    raw_history = page_meta.get("attempt_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise RuntimeError(f"Chandra page {source_page} attempt history is missing or malformed.")
    durable_history: list[dict[str, object]] = []
    for expected_attempt, raw_item in enumerate(raw_history, start=1):
        if not isinstance(raw_item, dict) or raw_item.get("attempt") != expected_attempt:
            raise RuntimeError(f"Chandra page {source_page} attempt history is malformed.")
        raw_image_path = raw_item.get("image_path")
        raw_sidecar_path = raw_item.get("sidecar_path")
        if not isinstance(raw_image_path, str) or not isinstance(raw_sidecar_path, str):
            raise RuntimeError(f"Chandra page {source_page} attempt has no linked artifacts.")
        source_image = Path(raw_image_path)
        source_sidecar = Path(raw_sidecar_path)
        if not source_image.is_file() or not source_sidecar.is_file():
            raise RuntimeError(f"Chandra page {source_page} attempt artifact is missing.")
        if raw_item.get("image_sha256") != _sha256_path(source_image):
            raise RuntimeError(f"Chandra page {source_page} attempt image digest changed.")
        if raw_item.get("sidecar_sha256") != _sha256_path(source_sidecar):
            raise RuntimeError(f"Chandra page {source_page} attempt sidecar digest changed.")
        declared_image_bytes = raw_item.get("image_bytes")
        declared_image_size = raw_item.get("image_size")
        if (
            not isinstance(declared_image_bytes, int)
            or isinstance(declared_image_bytes, bool)
            or declared_image_bytes <= 0
            or declared_image_bytes > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES
            or not isinstance(declared_image_size, list)
            or len(declared_image_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in declared_image_size
            )
            or declared_image_size[0] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or declared_image_size[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or declared_image_size[0] * declared_image_size[1] > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
        ):
            raise RuntimeError(f"Chandra page {source_page} attempt image bounds are invalid.")
        if source_image.stat().st_size != declared_image_bytes:
            raise RuntimeError(f"Chandra page {source_page} attempt image byte size changed.")
        decoded_source_image = _bounded_png_rgb_image(
            source_image,
            label=f"Chandra page {source_page} attempt {expected_attempt} image",
            require_rgb=True,
        )
        actual_image_size = [int(decoded_source_image.width), int(decoded_source_image.height)]
        if declared_image_size != actual_image_size:
            raise RuntimeError(f"Chandra page {source_page} attempt image size changed.")

        attempt_dir = (
            engine_dir / f"page_{source_page:04d}.chandra-attempts" / f"attempt_{expected_attempt}"
        )
        durable_image = _copy_retry_evidence(
            source=source_image,
            target=attempt_dir / "input.png",
            max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
        )
        durable_decoded_image = _bounded_png_rgb_image(
            durable_image,
            label=f"Chandra page {source_page} durable attempt {expected_attempt} image",
            require_rgb=True,
        )
        if durable_decoded_image.size != decoded_source_image.size:
            raise RuntimeError(f"Chandra page {source_page} durable attempt image size changed.")
        durable_sidecar = _copy_retry_evidence(
            source=source_sidecar,
            target=attempt_dir / "chandra_attempt.json",
            max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
        )
        item = dict(raw_item)
        item.update(
            {
                "image_path": str(durable_image.resolve()),
                "sidecar_path": str(durable_sidecar.resolve()),
                "image_sha256": _sha256_path(durable_image),
                "image_bytes": int(durable_image.stat().st_size),
                "sidecar_sha256": _sha256_path(durable_sidecar),
                "sidecar_bytes": int(durable_sidecar.stat().st_size),
            }
        )
        durable_history.append(item)

    page_meta["attempt_history"] = durable_history
    raw_selected_sidecar = page_meta.get("chandra_page_lines_path")
    if not isinstance(raw_selected_sidecar, str):
        raise RuntimeError(f"Chandra page {source_page} selected sidecar is missing.")
    selected_sidecar = Path(raw_selected_sidecar)
    try:
        payload = json.loads(selected_sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Chandra page {source_page} selected sidecar is unreadable: {exc}"
        ) from exc
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError(f"Chandra page {source_page} selected sidecar is malformed.")
    images[0]["attempt_history"] = durable_history
    images[0]["source_raster_identity"] = page_meta["source_raster_identity"]
    images[0]["source_raster_artifact"] = durable_source_artifact
    _write_json_atomic(selected_sidecar, payload)


def _write_pagewise_text_artifacts(
    *,
    output_dir: Path,
    engine: str,
    pdf_path: Path,
    source_pages_1based: Sequence[int],
    page_texts: Sequence[str],
    aggregate_path: Path,
    page_metadata: Sequence[dict[str, Any]] | None = None,
    page_errors: Sequence[dict[str, Any]] | None = None,
) -> tuple[int, Path]:
    engine_dir = output_dir / engine
    engine_dir.mkdir(parents=True, exist_ok=True)

    pages_payload: list[dict[str, Any]] = []
    metadata_by_page: dict[int, dict[str, Any]] = {}
    if page_metadata:
        for item in page_metadata:
            if not isinstance(item, dict):
                continue
            source_page = item.get("source_page")
            if (
                isinstance(source_page, int)
                and not isinstance(source_page, bool)
                and source_page > 0
            ):
                metadata_by_page[source_page] = item

    errors_by_page: dict[int, list[dict[str, str]]] = {}
    if page_errors:
        for item in page_errors:
            if not isinstance(item, dict):
                continue
            source_page = item.get("source_page")
            if not isinstance(source_page, int) or isinstance(source_page, bool):
                continue
            error = str(item.get("error") or "").strip()
            if error:
                code = str(item.get("code") or _PAGE_ERROR_UNCLASSIFIED).strip()
                errors_by_page.setdefault(source_page, []).append({"code": code, "message": error})

    total_chars = 0
    for source_page, text in zip(source_pages_1based, page_texts, strict=True):
        page_file = engine_dir / f"page_{source_page:04d}.txt"
        page_file.write_bytes(text.encode("utf-8"))
        chars = len(text)
        total_chars += chars
        page_info: dict[str, Any] = {
            "source_page": source_page,
            "file": page_file.name,
            "text_chars": chars,
        }
        page_meta = metadata_by_page.get(source_page, {})
        if engine == OCR_ENGINE_SURYA and page_meta.get("surya_page_lines_path") is not None:
            _persist_surya_source_raster(
                engine_dir=engine_dir,
                page_meta=page_meta,
                source_page=source_page,
            )
        if engine == OCR_ENGINE_SURYA and page_meta.get("attempt_history") is not None:
            _persist_surya_retry_history(
                engine_dir=engine_dir,
                page_meta=page_meta,
                source_page=source_page,
            )
        if engine == OCR_ENGINE_CHANDRA and page_meta.get("attempt_history") is not None:
            _persist_chandra_attempt_history(
                engine_dir=engine_dir,
                page_meta=page_meta,
                source_page=source_page,
            )
        if page_meta.get("blank_page") is True:
            page_info["blank_page"] = True
        for evidence_key in (
            "ocr_outcome",
            "explicit_nontext",
            "attempt_count",
            "terminal_attempt",
            "retry_preprocessing",
            "selected_attempt",
            "chandra_retry_policy",
            "retry_policy",
            "attempt_history",
            "geometry_coordinate_space",
            "geometry_transform",
            "alnum_line_count",
            "alnum_chars",
            "source_raster_identity",
            "source_raster_artifact",
        ):
            evidence_value = page_meta.get(evidence_key)
            if evidence_value is not None:
                page_info[evidence_key] = evidence_value
        if source_page in errors_by_page:
            page_info["page_errors"] = errors_by_page[source_page]

        sidecar_specs = (
            ("surya_page_lines_path", "surya_text_lines", "surya"),
            ("chandra_page_lines_path", "chandra_text_lines", "chandra"),
        )
        for key, geometry_type, suffix in sidecar_specs:
            sidecar_raw = page_meta.get(key)
            if not isinstance(sidecar_raw, str):
                continue
            sidecar_src = Path(sidecar_raw)
            if not sidecar_src.is_file():
                continue
            sidecar_name = f"page_{source_page:04d}.{suffix}.json"
            sidecar_dst = engine_dir / sidecar_name
            try:
                shutil.copy2(sidecar_src, sidecar_dst)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to persist {engine} geometry for source page {source_page}: {exc}"
                ) from exc
            page_info["geometry_file"] = sidecar_name
            page_info["geometry_type"] = geometry_type
            break

        pages_payload.append(page_info)

    markerized = _markerized_pages_text(
        page_texts=page_texts,
        source_pages_1based=source_pages_1based,
    )
    markerized_bytes = markerized.encode("utf-8")
    (engine_dir / "all_pages.txt").write_bytes(markerized_bytes)
    aggregate_path.write_bytes(markerized_bytes)

    pages_index = {
        "pdf_path": str(pdf_path),
        "engine": engine,
        "pages": pages_payload,
        "total_text_chars": total_chars,
        "aggregate_file": "all_pages.txt",
        "aggregate_has_page_markers": True,
    }
    pages_json_path = engine_dir / "pages.json"
    pages_json_path.write_text(
        json.dumps(pages_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return total_chars, pages_json_path


def _collect_chandra_batch_outputs(
    *,
    sidecar_path: Path,
    image_paths: Sequence[Path],
    source_pages_1based: Sequence[int],
    work_dir: Path,
) -> tuple[
    list[str],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Chandra sidecar payload has unexpected format.")

    images_payload = payload.get("images")
    if not isinstance(images_payload, list):
        raise RuntimeError("Chandra sidecar payload has no 'images' list.")
    if len(image_paths) != len(source_pages_1based):
        raise RuntimeError("Chandra raster/page mapping is not bijective.")

    strict_sidecar = _chandra_require_sidecar()
    expected_names = {image_path.name for image_path in image_paths}
    if strict_sidecar and len(images_payload) != len(image_paths):
        raise RuntimeError(
            "Chandra sidecar image cardinality does not match requested raster count: "
            f"expected {len(image_paths)}, got {len(images_payload)}."
        )

    by_name: dict[str, dict[str, Any]] = {}
    for item in images_payload:
        if not isinstance(item, dict):
            if strict_sidecar:
                raise RuntimeError("Chandra sidecar contains a non-object image entry.")
            continue
        image_name = str(item.get("image_name") or "").strip()
        if not image_name:
            if strict_sidecar:
                raise RuntimeError("Chandra sidecar contains an image entry without image_name.")
            continue
        if strict_sidecar and image_name not in expected_names:
            raise RuntimeError(f"Chandra sidecar contains an unexpected image: {image_name!r}.")
        if image_name in by_name:
            raise RuntimeError(f"Chandra sidecar contains duplicate image_name: {image_name!r}.")
        by_name[image_name] = item
        if not strict_sidecar:
            by_name[Path(image_name).stem] = item
    if strict_sidecar and set(by_name) != expected_names:
        raise RuntimeError("Chandra sidecar image mapping is not an exact raster bijection.")

    page_texts: list[str] = []
    total_chars = 0
    page_errors: list[dict[str, Any]] = []
    page_metadata: list[dict[str, Any]] = []
    for image_path, source_page in zip(image_paths, source_pages_1based, strict=True):
        image_payload = by_name.get(image_path.name)
        if image_payload is None and not strict_sidecar:
            image_payload = by_name.get(image_path.stem)
        page_meta: dict[str, Any] = {"source_page": source_page}
        text_lines: list[str] = []
        if image_payload is not None:
            pages = image_payload.get("pages")
            if isinstance(pages, list):
                for page in pages:
                    if not isinstance(page, dict):
                        continue
                    lines = page.get("text_lines")
                    if not isinstance(lines, list):
                        continue
                    line_rows: list[tuple[tuple[float, float, float, float], str]] = []
                    fallback_lines: list[str] = []
                    for line in lines:
                        if not isinstance(line, dict):
                            continue
                        text = _clean_overlay_line(str(line.get("text") or ""))
                        if not text:
                            continue
                        bbox = _bbox_values(line.get("bbox"))
                        if bbox is not None:
                            line_rows.append((bbox, text))
                            continue
                        fallback_lines.append(text)
                    if line_rows:
                        page_width = max(
                            max(bbox[2] for bbox, _text in line_rows),
                            1.0,
                        )
                        image_bbox = _bbox_values(page.get("image_bbox"))
                        if image_bbox is not None:
                            page_width = max(page_width, image_bbox[2])
                        order = _bbox_reading_order_indices(
                            [bbox for bbox, _text in line_rows],
                            page_width=page_width,
                        )
                        text_lines.extend(line_rows[idx][1] for idx in order)
                    text_lines.extend(fallback_lines)

            for key in (
                "ocr_outcome",
                "explicit_nontext",
                "attempt_count",
                "terminal_attempt",
                "retry_preprocessing",
                "selected_attempt",
                "chandra_retry_policy",
                "attempt_history",
                "source_raster_identity",
                "source_raster_artifact",
            ):
                value = image_payload.get(key)
                if value is not None:
                    page_meta[key] = value

        blank_page = False
        if image_payload is None or not text_lines:
            if _is_effectively_blank_page_image(image_path):
                blank_page = True
                page_meta["blank_page"] = True
                page_meta["ocr_outcome"] = _OCR_OUTCOME_VERIFIED_BLANK
            else:
                page_errors.append(
                    {
                        "code": (
                            _PAGE_ERROR_MISSING_OUTPUT
                            if image_payload is None
                            else _PAGE_ERROR_ZERO_OUTPUT
                        ),
                        "source_page": source_page,
                        "image": str(image_path),
                        "error": (
                            "Chandra geometry sidecar has no image entry"
                            if image_payload is None
                            else "Chandra geometry sidecar has no text_lines"
                        ),
                    }
                )

        if text_lines:
            page_meta["ocr_outcome"] = _OCR_OUTCOME_TEXT
        elif not blank_page and image_payload is not None:
            page_meta.setdefault("ocr_outcome", _OCR_OUTCOME_ZERO)

        alnum_line_count, alnum_chars = _alnum_evidence(text_lines)
        page_meta["alnum_line_count"] = alnum_line_count
        page_meta["alnum_chars"] = alnum_chars

        if image_payload is not None:
            per_page_image_payload = dict(image_payload)
            per_page_image_payload["image_name"] = image_path.name
            for key in (
                "ocr_outcome",
                "explicit_nontext",
                "attempt_count",
                "terminal_attempt",
                "retry_preprocessing",
                "selected_attempt",
                "chandra_retry_policy",
                "attempt_history",
                "source_raster_identity",
                "source_raster_artifact",
            ):
                value = page_meta.get(key)
                if value is not None:
                    per_page_image_payload[key] = value
            page_dir = work_dir / f"page_{source_page:04d}"
            page_sidecar_path = page_dir / "chandra_page_lines.json"
            _write_json_atomic(page_sidecar_path, {"images": [per_page_image_payload]})
            page_meta["chandra_page_lines_path"] = str(page_sidecar_path)

        if image_payload is not None or blank_page:
            page_metadata.append(page_meta)

        page_text = "\n".join(_dehyphenate_line_breaks(text_lines))
        page_texts.append(page_text)
        total_chars += len(page_text)

    return page_texts, total_chars, page_errors, page_metadata


def _strict_surya_batch_images(
    *,
    images: object,
    image_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    if not isinstance(images, list):
        raise RuntimeError("Surya sidecar payload has no 'images' list.")

    validated_images: list[tuple[str, dict[str, Any]]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(images, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Surya sidecar image entry {index} must be an object.")
        image_name = item.get("image_name")
        if not isinstance(image_name, str) or not image_name.strip():
            raise RuntimeError(f"Surya sidecar image entry {index} has no nonempty image_name.")
        if image_name in seen_names:
            raise RuntimeError(f"Surya sidecar contains duplicate image_name: {image_name!r}.")
        seen_names.add(image_name)
        pages = item.get("pages")
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            raise RuntimeError(
                f"Surya sidecar image entry must contain exactly one page object: {image_name!r}."
            )
        validated_images.append((image_name, item))

    expected_count = len(image_paths)
    if len(images) != expected_count:
        raise RuntimeError(
            "Surya sidecar image cardinality does not match requested raster count: "
            f"expected {expected_count}, got {len(images)}."
        )

    source_names = [Path(path).name for path in image_paths]
    expected_names = source_names
    if expected_count > 1:
        staged_names = [
            f"{index:04d}_{source_name}" for index, source_name in enumerate(source_names, start=1)
        ]
        raw_names = seen_names
        if raw_names == set(staged_names):
            expected_names = staged_names
        elif len(set(source_names)) == expected_count and raw_names == set(source_names):
            expected_names = source_names
        else:
            raise RuntimeError(
                "Surya sidecar image names do not form an exact unambiguous "
                "bijection to requested rasters."
            )
    elif seen_names != set(source_names):
        raise RuntimeError(
            "Surya sidecar image names do not form an exact unambiguous "
            "bijection to requested rasters."
        )

    by_name = dict(validated_images)
    return [by_name[name] for name in expected_names]


def _collect_surya_batch_outputs(
    *,
    sidecar_path: Path,
    image_paths: Sequence[Path],
    source_pages_1based: Sequence[int],
    work_dir: Path,
    attempt_count: int = 1,
    retry_preprocessing: str | None = None,
) -> tuple[list[str], int, list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Surya sidecar payload has unexpected format.")
    images = payload.get("images")
    image_payloads = _strict_surya_batch_images(
        images=images,
        image_paths=image_paths,
    )
    sidecar_changed = False
    for image_payload in image_payloads:
        for page in image_payload.get("pages", []):
            if not isinstance(page, dict):
                continue
            raw_lines = page.get("text_lines")
            if not isinstance(raw_lines, list):
                continue
            normalized_lines: list[object] = []
            for raw_line in raw_lines:
                if not isinstance(raw_line, dict):
                    normalized_lines.append(raw_line)
                    continue
                raw_text = raw_line.get("text")
                if not isinstance(raw_text, str):
                    normalized_lines.append(raw_line)
                    continue
                text = _clean_overlay_line(raw_text)
                if not text or not any(char.isalnum() for char in text):
                    if attempt_count == 1:
                        sidecar_changed = True
                        continue
                    normalized_lines.append(raw_line)
                    continue
                normalized_line = dict(raw_line)
                normalized_line["text"] = text
                normalized_lines.append(normalized_line)
                sidecar_changed = sidecar_changed or normalized_line != raw_line
            page["text_lines"] = normalized_lines
    if sidecar_changed:
        _write_json_atomic(sidecar_path, payload)
    execution_path = str(payload.get("execution_path") or "").strip()
    page_texts: list[str] = []
    total_chars = 0
    page_errors: list[dict[str, Any]] = []
    page_metadata: list[dict[str, Any]] = []

    for image_path, source_page, image_payload in zip(
        image_paths,
        source_pages_1based,
        image_payloads,
        strict=True,
    ):
        text_lines: list[str] = []
        page_meta: dict[str, Any] = {"source_page": source_page}
        for page in image_payload.get("pages", []):
            if not isinstance(page, dict):
                continue
            for line in page.get("text_lines", []):
                if not isinstance(line, dict):
                    continue
                text = _clean_overlay_line(str(line.get("text") or ""))
                if text:
                    text_lines.append(text)

        page_text = "\n".join(_dehyphenate_line_breaks(text_lines))
        page_texts.append(page_text)
        total_chars += len(page_text)
        blank_page = False
        if not text_lines or not isinstance(image_payload, dict):
            if _is_effectively_blank_page_image(image_path):
                blank_page = True
                page_meta["blank_page"] = True
                page_meta["ocr_outcome"] = _OCR_OUTCOME_VERIFIED_BLANK
            else:
                page_errors.append(
                    {
                        "code": (
                            _PAGE_ERROR_MISSING_OUTPUT
                            if not isinstance(image_payload, dict)
                            else _PAGE_ERROR_ZERO_OUTPUT
                        ),
                        "source_page": source_page,
                        "image": str(image_path),
                        "error": (
                            (
                                "Surya geometry sidecar has no image entry "
                                if not isinstance(image_payload, dict)
                                else "Surya geometry sidecar has no text_lines "
                            )
                            + f"for source page {source_page}"
                        ),
                    }
                )
        elif text_lines:
            page_meta["ocr_outcome"] = _OCR_OUTCOME_TEXT

        if not text_lines and not blank_page:
            page_meta["ocr_outcome"] = (
                _OCR_OUTCOME_ZERO if isinstance(image_payload, dict) else "missing_output"
            )
        page_meta["attempt_count"] = attempt_count
        if retry_preprocessing is not None:
            page_meta["retry_preprocessing"] = retry_preprocessing
        alnum_line_count, alnum_chars = _alnum_evidence(text_lines)
        page_meta["alnum_line_count"] = alnum_line_count
        page_meta["alnum_chars"] = alnum_chars

        if isinstance(image_payload, dict):
            per_page_image_payload = dict(image_payload)
            per_page_image_payload["image_name"] = image_path.name
            per_page_image_payload["ocr_outcome"] = page_meta["ocr_outcome"]
            per_page_image_payload["attempt_count"] = attempt_count
            if retry_preprocessing is not None:
                per_page_image_payload["retry_preprocessing"] = retry_preprocessing
            for field in ("geometry_coordinate_space", "geometry_transform"):
                value = image_payload.get(field)
                if isinstance(value, str) and value:
                    page_meta[field] = value
            per_page_payload: dict[str, Any] = {"images": [per_page_image_payload]}
            if execution_path:
                per_page_payload["execution_path"] = execution_path
            page_dir = work_dir / f"page_{source_page:04d}"
            page_sidecar_path = page_dir / "surya_page_lines.json"
            _write_json_atomic(page_sidecar_path, per_page_payload)
            page_meta["surya_page_lines_path"] = str(page_sidecar_path)

        if isinstance(image_payload, dict) or blank_page:
            page_metadata.append(page_meta)

    return page_texts, total_chars, page_errors, page_metadata


def _attach_source_raster_identities(
    *,
    page_metadata: Sequence[dict[str, Any]],
    identities_by_page: dict[int, dict[str, object]],
    artifacts_by_page: dict[int, dict[str, object]] | None,
    sidecar_key: str,
) -> None:
    seen: set[int] = set()
    for row in page_metadata:
        source_page = row.get("source_page")
        if (
            not isinstance(source_page, int)
            or isinstance(source_page, bool)
            or source_page in seen
            or source_page not in identities_by_page
        ):
            raise RuntimeError("OCR page metadata cannot be bound to a unique source raster.")
        seen.add(source_page)
        identity = dict(identities_by_page[source_page])
        row["source_raster_identity"] = identity
        artifact: dict[str, object] | None = None
        if artifacts_by_page is not None:
            raw_artifact = artifacts_by_page.get(source_page)
            if not isinstance(raw_artifact, dict):
                raise RuntimeError(f"OCR page {source_page} has no sealed source raster artifact.")
            artifact = dict(raw_artifact)
            row["source_raster_artifact"] = artifact
        raw_sidecar = row.get(sidecar_key)
        if not isinstance(raw_sidecar, str):
            continue
        sidecar = Path(raw_sidecar)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot attach source raster identity for page {source_page}: {exc}"
            ) from exc
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
            raise RuntimeError(
                f"Cannot attach source raster identity to malformed page {source_page}."
            )
        images[0]["source_raster_identity"] = identity
        if artifact is not None:
            images[0]["source_raster_artifact"] = artifact
        _write_json_atomic(sidecar, payload)


def _persist_validated_surya_attempt_metadata(
    *,
    sidecar_path: Path,
    image_path: Path,
    page_metadata: Sequence[dict[str, Any]],
    source_page: int,
    attempt: int,
    preprocessing: str,
) -> None:
    expected_preprocessing = {
        2: _ZERO_OUTPUT_RETRY_PREPROCESSING,
        3: _ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
    }.get(attempt)
    if expected_preprocessing is None or preprocessing != expected_preprocessing:
        raise RuntimeError("Surya retry attempt metadata request is invalid.")
    rows = [item for item in page_metadata if item.get("source_page") == source_page]
    if len(rows) != 1:
        raise RuntimeError("Cannot persist Surya retry metadata without exact page evidence.")
    row = rows[0]
    outcome = row.get("ocr_outcome")
    if outcome not in {
        _OCR_OUTCOME_ZERO,
        _OCR_OUTCOME_TEXT,
        _OCR_OUTCOME_VERIFIED_BLANK,
    }:
        raise RuntimeError("Cannot persist invalid Surya retry outcome.")
    if (
        row.get("attempt_count") != attempt
        or isinstance(row.get("attempt_count"), bool)
        or row.get("retry_preprocessing") != preprocessing
    ):
        raise RuntimeError("Surya retry metadata disagrees with its validated attempt.")
    normalized_sidecar_raw = row.get("surya_page_lines_path")
    if not isinstance(normalized_sidecar_raw, str):
        raise RuntimeError("Validated Surya retry metadata has no normalized sidecar.")
    normalized_sidecar = Path(normalized_sidecar_raw)
    try:
        payload = json.loads(normalized_sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read normalized Surya retry metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Normalized Surya retry metadata is not an object.")
    images = _strict_surya_batch_images(
        images=payload.get("images"),
        image_paths=[image_path],
    )
    image = images[0]
    if (
        payload.get("execution_path") not in _SURYA_MODULE_EXECUTION_PATHS
        or image.get("ocr_outcome") != outcome
        or image.get("attempt_count") != attempt
        or isinstance(image.get("attempt_count"), bool)
        or image.get("retry_preprocessing") != preprocessing
    ):
        raise RuntimeError("Normalized Surya retry metadata is inconsistent.")
    _write_json_atomic(sidecar_path, payload)


def _run_extraction_engine_pagewise(
    engine: str,
    image_paths: Sequence[Path],
    *,
    source_pages_1based: Sequence[int],
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable,
    run_cmd: RunCommand,
    progress_cb: PageProgressCallback | None = None,
    defer_empty_pages: bool = False,
) -> tuple[list[str], int, list[dict[str, Any]], list[dict[str, Any]]]:
    if len(image_paths) != len(source_pages_1based):
        raise ValueError("image_paths and source_pages_1based lengths must match.")

    source_raster_identities = [
        _source_raster_identity(image_path, source_page=source_page)
        for image_path, source_page in zip(
            image_paths,
            source_pages_1based,
            strict=True,
        )
    ]
    identities_by_page: dict[int, dict[str, object]] = {}
    for identity in source_raster_identities:
        raw_source_page = identity.get("source_page")
        if (
            not isinstance(raw_source_page, int)
            or isinstance(raw_source_page, bool)
            or raw_source_page <= 0
            or raw_source_page in identities_by_page
        ):
            raise RuntimeError("Source raster identities are not page-bijective.")
        identities_by_page[raw_source_page] = identity
    if len(identities_by_page) != len(source_raster_identities):
        raise RuntimeError("Source raster identities are not page-bijective.")

    if engine == OCR_ENGINE_SURYA and len(image_paths) > 0:
        staged_image_paths: list[Path] = []
        artifacts_by_page: dict[int, dict[str, object]] = {}
        staged_root = work_dir / "source_evidence"
        for image_path, source_page in zip(
            image_paths,
            source_pages_1based,
            strict=True,
        ):
            source_image = _bounded_png_rgb_image(
                image_path,
                label=f"Surya source raster page {source_page}",
                require_rgb=False,
            )
            staged_path = staged_root / image_path.name
            if staged_path.exists():
                raise RuntimeError(f"Surya staged source raster already exists: {staged_path}")
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            source_image.save(staged_path, format="PNG")
            staged_identity, staged_artifact, _staged_image = _source_raster_evidence(
                staged_path,
                source_page=source_page,
                require_rgb=True,
            )
            if staged_identity != identities_by_page[source_page]:
                raise RuntimeError(f"Surya staged source raster changed page {source_page} pixels")
            staged_image_paths.append(staged_path)
            artifacts_by_page[source_page] = staged_artifact
        image_paths = tuple(staged_image_paths)
        batch_work_dir = work_dir / "batch"
        try:
            aggregate_text, aggregate_chars = _run_surya_direct(
                image_paths,
                lang=lang,
                work_dir=batch_work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
        except Exception as exc:
            preview = "; ".join(f"p{page}: {exc}" for page in source_pages_1based[:3])
            raise RuntimeError(f"all sampled pages failed for {engine}: {preview}") from exc
        for image_path, source_page in zip(
            image_paths,
            source_pages_1based,
            strict=True,
        ):
            _validate_source_raster_seal(
                path=image_path,
                source_page=source_page,
                identity=identities_by_page[source_page],
                artifact=artifacts_by_page[source_page],
            )

        sidecar_path = batch_work_dir / "surya_page_lines.json"
        if sidecar_path.exists():
            page_texts, total_chars, page_errors, page_metadata = _collect_surya_batch_outputs(
                sidecar_path=sidecar_path,
                image_paths=image_paths,
                source_pages_1based=source_pages_1based,
                work_dir=work_dir,
            )
        else:
            page_texts = [aggregate_text] + [""] * (len(image_paths) - 1)
            total_chars = int(aggregate_chars)
            page_errors = [
                {
                    "source_page": source_page,
                    "image": str(image_path),
                    "error": f"Surya geometry sidecar is missing for source page {source_page}",
                }
                for image_path, source_page in zip(
                    image_paths,
                    source_pages_1based,
                    strict=True,
                )
            ]
            page_metadata = []

        for page_index, (image_path, source_page) in enumerate(
            zip(image_paths, source_pages_1based, strict=True)
        ):
            initial_rows = [
                item for item in page_metadata if item.get("source_page") == source_page
            ]
            initial_row = _structured_surya_zero_output(
                page_errors=page_errors,
                page_metadata=page_metadata,
                source_page=source_page,
                image_path=image_path,
                expected_attempt=1,
                expected_preprocessing=None,
                permitted_execution_paths=_SURYA_DIRECT_EXECUTION_PATHS,
                label="initial zero-output evidence",
            )
            if len(initial_rows) != 1:
                matching_errors = [
                    item for item in page_errors if item.get("source_page") == source_page
                ]
                if not defer_empty_pages and _surya_require_geometry_sidecar() and matching_errors:
                    preview = "; ".join(
                        f"p{item['source_page']}: {item['error']}" for item in matching_errors[:3]
                    )
                    raise RuntimeError(
                        f"surya geometry sidecar is required for each page: {preview}"
                    )
                raise RuntimeError("Cannot seal Surya attempt 1 without exact page metadata.")
            initial_metadata = initial_rows[0]
            initial_outcome = initial_metadata.get("ocr_outcome")
            if initial_outcome not in {
                _OCR_OUTCOME_TEXT,
                _OCR_OUTCOME_VERIFIED_BLANK,
                _OCR_OUTCOME_ZERO,
            }:
                raise RuntimeError("Cannot seal invalid Surya attempt 1 outcome.")
            initial_sidecar_raw = initial_metadata.get("surya_page_lines_path")
            if not isinstance(initial_sidecar_raw, str):
                raise RuntimeError("Cannot seal Surya attempt 1 without durable geometry.")
            initial_sidecar = Path(initial_sidecar_raw)
            retry_root = work_dir / "zero_output_retry" / f"page_{source_page:04d}"
            retained_initial_image = _copy_retry_evidence(
                source=image_path,
                target=retry_root / "attempt_1_original" / "input" / image_path.name,
                max_bytes=_MAX_CHANDRA_ATTEMPT_IMAGE_BYTES,
            )
            retained_initial_sidecar = _copy_retry_evidence(
                source=initial_sidecar,
                target=(retry_root / "attempt_1_original" / "module" / "surya_page_lines.json"),
                max_bytes=_MAX_OCR_TEXT_ARTIFACT_BYTES,
            )
            attempt_history = [
                _attempt_history_record(
                    attempt=1,
                    preprocessing="original",
                    ocr_outcome=str(initial_outcome),
                    image_path=retained_initial_image,
                    sidecar_path=retained_initial_sidecar,
                )
            ]
            _attach_surya_retry_provenance(
                page_metadata=page_metadata,
                source_page=source_page,
                selected_attempt=1,
                attempt_history=attempt_history,
            )
            if initial_row is None:
                continue

            retry_image = retry_root / "attempt_2_autocontrast" / "input" / image_path.name
            retry_image_evidence = _write_autocontrast_retry_image(
                source=image_path,
                target=retry_image,
            )
            retry_work_dir = retry_root / "attempt_2_autocontrast" / "module"
            _run_surya_module_cli(
                [retry_image],
                lang=lang,
                work_dir=retry_work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
            retry_sidecar = retry_work_dir / "surya_page_lines.json"
            if not retry_sidecar.is_file():
                raise RuntimeError(
                    "Surya zero-output retry did not produce mandatory geometry "
                    f"for source page {source_page}."
                )
            retry_texts, _, retry_errors, retry_metadata = _collect_surya_batch_outputs(
                sidecar_path=retry_sidecar,
                image_paths=[retry_image],
                source_pages_1based=[source_page],
                work_dir=work_dir,
                attempt_count=2,
                retry_preprocessing=_ZERO_OUTPUT_RETRY_PREPROCESSING,
            )
            if len(retry_metadata) != 1:
                raise RuntimeError(
                    "Invalid second zero-output evidence: exact page metadata is required."
                )
            retry_outcome = retry_metadata[0].get("ocr_outcome")
            if retry_outcome == _OCR_OUTCOME_ZERO:
                _structured_surya_zero_output(
                    page_errors=retry_errors,
                    page_metadata=retry_metadata,
                    source_page=source_page,
                    image_path=retry_image,
                    expected_attempt=2,
                    expected_preprocessing=_ZERO_OUTPUT_RETRY_PREPROCESSING,
                    permitted_execution_paths=_SURYA_MODULE_EXECUTION_PATHS,
                    label="second zero-output evidence",
                )
            elif retry_outcome in {_OCR_OUTCOME_TEXT, _OCR_OUTCOME_VERIFIED_BLANK}:
                _strict_surya_single_page_geometry(
                    sidecar_path=retry_sidecar,
                    image_path=retry_image,
                    require_text=retry_outcome == _OCR_OUTCOME_TEXT,
                    permitted_execution_paths=_SURYA_MODULE_EXECUTION_PATHS,
                    label="second-attempt geometry",
                )
            else:
                raise RuntimeError(
                    "Invalid second zero-output evidence: retry output is missing or malformed."
                )
            _persist_validated_surya_attempt_metadata(
                sidecar_path=retry_sidecar,
                image_path=retry_image,
                page_metadata=retry_metadata,
                source_page=source_page,
                attempt=2,
                preprocessing=_ZERO_OUTPUT_RETRY_PREPROCESSING,
            )
            attempt_history.append(
                _attempt_history_record(
                    attempt=2,
                    preprocessing=_ZERO_OUTPUT_RETRY_PREPROCESSING,
                    ocr_outcome=str(retry_outcome),
                    image_path=retry_image,
                    sidecar_path=retry_sidecar,
                    image_evidence=retry_image_evidence,
                )
            )

            selected_texts = retry_texts
            selected_errors = retry_errors
            selected_metadata = retry_metadata
            selected_attempt = 2
            if retry_outcome == _OCR_OUTCOME_ZERO:
                scaled_root = retry_root / "attempt_3_scaled"
                scaled_image = scaled_root / "input" / image_path.name
                scaled_image_evidence = _write_scaled_retry_image(
                    source=image_path,
                    target=scaled_image,
                )
                scaled_work_dir = scaled_root / "module"
                _run_surya_module_cli(
                    [scaled_image],
                    lang=lang,
                    work_dir=scaled_work_dir,
                    which_fn=which_fn,
                    run_cmd=run_cmd,
                )
                scaled_sidecar = scaled_work_dir / "surya_page_lines.json"
                if not scaled_sidecar.is_file():
                    raise RuntimeError(
                        "Surya third zero-output retry did not produce mandatory geometry "
                        f"for source page {source_page}."
                    )
                raw_scaled_texts, _, raw_scaled_errors, raw_scaled_metadata = (
                    _collect_surya_batch_outputs(
                        sidecar_path=scaled_sidecar,
                        image_paths=[scaled_image],
                        source_pages_1based=[source_page],
                        work_dir=scaled_root / "raw_collection",
                        attempt_count=3,
                        retry_preprocessing=_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
                    )
                )
                if len(raw_scaled_metadata) != 1:
                    raise RuntimeError(
                        "Invalid third-attempt geometry: exact page metadata is required."
                    )
                scaled_outcome = raw_scaled_metadata[0].get("ocr_outcome")
                if scaled_outcome == _OCR_OUTCOME_ZERO:
                    _structured_surya_zero_output(
                        page_errors=raw_scaled_errors,
                        page_metadata=raw_scaled_metadata,
                        source_page=source_page,
                        image_path=scaled_image,
                        expected_attempt=3,
                        expected_preprocessing=_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
                        permitted_execution_paths=_SURYA_MODULE_EXECUTION_PATHS,
                        label="third-attempt geometry",
                    )
                elif scaled_outcome in {_OCR_OUTCOME_TEXT, _OCR_OUTCOME_VERIFIED_BLANK}:
                    _strict_surya_single_page_geometry(
                        sidecar_path=scaled_sidecar,
                        image_path=scaled_image,
                        require_text=scaled_outcome == _OCR_OUTCOME_TEXT,
                        permitted_execution_paths=_SURYA_MODULE_EXECUTION_PATHS,
                        label="third-attempt geometry",
                    )
                    source_coordinate_sidecar = _write_source_coordinate_surya_sidecar(
                        source=scaled_sidecar,
                        target=scaled_root / "source_coordinates" / "surya_page_lines.json",
                        source_size=_strict_scaled_retry_pair(
                            scaled_image_evidence.get("image_size"),
                            label="image_size",
                            allow_zero=False,
                        ),
                        content_size=_strict_scaled_retry_pair(
                            scaled_image_evidence.get("content_size"),
                            label="content_size",
                            allow_zero=False,
                        ),
                        content_offset=_strict_scaled_retry_pair(
                            scaled_image_evidence.get("content_offset"),
                            label="content_offset",
                            allow_zero=True,
                        ),
                    )
                    scaled_texts, _, scaled_errors, scaled_metadata = _collect_surya_batch_outputs(
                        sidecar_path=source_coordinate_sidecar,
                        image_paths=[image_path],
                        source_pages_1based=[source_page],
                        work_dir=work_dir,
                        attempt_count=3,
                        retry_preprocessing=_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
                    )
                    if (
                        len(scaled_metadata) != 1
                        or scaled_metadata[0].get("ocr_outcome") != scaled_outcome
                        or scaled_texts != raw_scaled_texts
                    ):
                        raise RuntimeError(
                            "Invalid third-attempt geometry: source-coordinate transform "
                            "changed OCR evidence."
                        )
                else:
                    raise RuntimeError(
                        "Invalid third-attempt geometry: output is missing or malformed."
                    )
                if scaled_outcome == _OCR_OUTCOME_ZERO:
                    scaled_texts = raw_scaled_texts
                    scaled_errors = raw_scaled_errors
                    scaled_metadata = raw_scaled_metadata
                _persist_validated_surya_attempt_metadata(
                    sidecar_path=scaled_sidecar,
                    image_path=scaled_image,
                    page_metadata=raw_scaled_metadata,
                    source_page=source_page,
                    attempt=3,
                    preprocessing=_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
                )
                attempt_history.append(
                    _attempt_history_record(
                        attempt=3,
                        preprocessing=_ZERO_OUTPUT_SCALED_RETRY_PREPROCESSING,
                        ocr_outcome=str(scaled_outcome),
                        image_path=scaled_image,
                        sidecar_path=scaled_sidecar,
                        image_evidence=scaled_image_evidence,
                    )
                )
                selected_texts = scaled_texts
                selected_errors = scaled_errors
                selected_metadata = scaled_metadata
                selected_attempt = 3

            _attach_surya_retry_provenance(
                page_metadata=selected_metadata,
                source_page=source_page,
                selected_attempt=selected_attempt,
                attempt_history=attempt_history,
            )
            page_texts[page_index] = selected_texts[0]
            page_errors = [item for item in page_errors if item.get("source_page") != source_page]
            page_errors.extend(selected_errors)
            page_metadata = [
                item for item in page_metadata if item.get("source_page") != source_page
            ]
            page_metadata.extend(selected_metadata)
        for image_path, source_page in zip(
            image_paths,
            source_pages_1based,
            strict=True,
        ):
            _validate_source_raster_seal(
                path=image_path,
                source_page=source_page,
                identity=identities_by_page[source_page],
                artifact=artifacts_by_page[source_page],
            )
        _attach_source_raster_identities(
            page_metadata=page_metadata,
            identities_by_page=identities_by_page,
            artifacts_by_page=artifacts_by_page,
            sidecar_key="surya_page_lines_path",
        )
        total_chars = sum(len(text) for text in page_texts)

        successful_pages = {
            int(item["source_page"])
            for item in page_metadata
            if item.get("ocr_outcome") in {_OCR_OUTCOME_TEXT, _OCR_OUTCOME_VERIFIED_BLANK}
        }
        if progress_cb is not None:
            done = 0
            total_pages = len(source_pages_1based)
            for source_page in source_pages_1based:
                if source_page not in successful_pages:
                    continue
                done += 1
                progress_cb(done, total_pages, source_page)
        if not defer_empty_pages and _surya_require_geometry_sidecar() and page_errors:
            preview = "; ".join(
                f"p{item['source_page']}: {item['error']}" for item in page_errors[:3]
            )
            raise RuntimeError(f"surya geometry sidecar is required for each page: {preview}")

        return page_texts, total_chars, page_errors, page_metadata

    if engine == OCR_ENGINE_CHANDRA and len(image_paths) > 0:
        batch_work_dir = work_dir / "batch"
        total_pages = len(source_pages_1based)

        def _on_chandra_page_progress(done: int, total: int) -> None:
            if progress_cb is None:
                return
            bounded_total = max(total, 1)
            bounded_done = max(0, min(done, bounded_total))
            page_index = min(max(bounded_done - 1, 0), max(total_pages - 1, 0))
            source_page = source_pages_1based[page_index]
            progress_cb(bounded_done, bounded_total, source_page)

        text, chars = _run_chandra_direct(
            image_paths,
            lang=lang,
            work_dir=batch_work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
            page_progress_cb=_on_chandra_page_progress,
            source_raster_identities=source_raster_identities,
        )
        sidecar = batch_work_dir / "chandra_page_lines.json"
        if sidecar.exists():
            page_texts, total_chars, page_errors, page_metadata = _collect_chandra_batch_outputs(
                sidecar_path=sidecar,
                image_paths=image_paths,
                source_pages_1based=source_pages_1based,
                work_dir=work_dir,
            )
            if not defer_empty_pages and page_errors and _chandra_require_sidecar():
                preview = "; ".join(
                    f"p{item['source_page']}: {item['error']}" for item in page_errors[:3]
                )
                raise RuntimeError(f"chandra geometry sidecar is required for each page: {preview}")
            return page_texts, total_chars, page_errors, page_metadata
        warning = "chandra sidecar missing or empty; aggregate text mapped to page 1"
        if _chandra_require_sidecar():
            raise RuntimeError(warning)
        # Keep aggregate output usable, but make the degraded page mapping visible.
        degraded_page_texts = [text] + [""] * (len(image_paths) - 1)
        return (
            degraded_page_texts,
            int(chars),
            [
                {
                    "source_page": source_page,
                    "image": str(image_path),
                    "error": warning,
                }
                for image_path, source_page in zip(
                    image_paths,
                    source_pages_1based,
                    strict=True,
                )
            ],
            [],
        )

    fallback_page_texts: list[str] = []
    fallback_total_chars = 0
    fallback_page_errors: list[dict[str, Any]] = []
    fallback_page_metadata: list[dict[str, Any]] = []
    total_pages = len(source_pages_1based)
    for page_idx, (image_path, source_page) in enumerate(
        zip(image_paths, source_pages_1based, strict=True),
        start=1,
    ):
        page_work_dir = work_dir / f"page_{source_page:04d}"
        try:
            text, chars = _run_extraction_engine(
                engine,
                [image_path],
                lang=lang,
                work_dir=page_work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
            fallback_page_texts.append(text)
            fallback_total_chars += int(chars)
            if engine == OCR_ENGINE_SURYA:
                sidecar = page_work_dir / "surya_page_lines.json"
                if sidecar.exists():
                    fallback_page_metadata.append(
                        {
                            "source_page": source_page,
                            "surya_page_lines_path": str(sidecar),
                        }
                    )
                elif _surya_require_geometry_sidecar():
                    raise RuntimeError(
                        "Surya geometry sidecar is missing for "
                        f"source page {source_page}: {sidecar}"
                    )
            if engine == OCR_ENGINE_CHANDRA:
                sidecar = page_work_dir / "chandra_page_lines.json"
                if sidecar.exists():
                    fallback_page_metadata.append(
                        {
                            "source_page": source_page,
                            "chandra_page_lines_path": str(sidecar),
                        }
                    )
            if progress_cb is not None:
                progress_cb(page_idx, total_pages, source_page)
        except Exception as exc:
            fallback_page_texts.append("")
            fallback_page_errors.append(
                {
                    "source_page": source_page,
                    "image": str(image_path),
                    "error": str(exc),
                }
            )
            if progress_cb is not None:
                progress_cb(page_idx, total_pages, source_page)

    if fallback_page_errors and not any(text.strip() for text in fallback_page_texts):
        preview = "; ".join(
            f"p{item['source_page']}: {item['error']}" for item in fallback_page_errors[:3]
        )
        raise RuntimeError(f"all sampled pages failed for {engine}: {preview}")
    if engine == OCR_ENGINE_SURYA and _surya_require_geometry_sidecar() and fallback_page_errors:
        preview = "; ".join(
            f"p{item['source_page']}: {item['error']}" for item in fallback_page_errors[:3]
        )
        raise RuntimeError(f"surya geometry sidecar is required for each page: {preview}")
    return (
        fallback_page_texts,
        fallback_total_chars,
        fallback_page_errors,
        fallback_page_metadata,
    )


def _module_presence_probe(name: str) -> object:
    """Import-probe compatible callable without importing heavyweight modules."""
    if importlib.util.find_spec(name) is None:
        raise ImportError(name)
    return object()


def _create_runtime_work_dir(*, prefix: str) -> Path:
    root_raw = (os.environ.get("UNISCAN_RUNTIME_TMP") or "").strip()
    root = Path(root_raw) if root_raw else _DEFAULT_RUNTIME_TMP_HOME
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(64):
        candidate = root / f"{prefix}{uuid.uuid4().hex[:12]}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate runtime work dir under '{root}'.")


def _run_paddleocr_direct(image_paths: Sequence[Path], *, lang: str) -> tuple[str, int]:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_DEFAULT_PADDLE_CACHE_HOME))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_USE_PIR_TRT", "false")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang=_paddleocr_lang(lang),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    collected: list[str] = []
    for path in image_paths:
        result = ocr.ocr(str(path))
        collected.extend(_collect_text_strings(result))

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_surya_module_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for Surya OCR.")

    def _stage_inputs() -> tuple[Path, list[str]]:
        input_dir = work_dir / "surya_input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=False)
        # Surya CLI consumes a directory. For pagewise mode we must isolate
        # files, otherwise every call re-processes sibling pages and pollutes
        # per-page artifacts with foreign text.
        ordered_names: list[str] = []
        for idx, image_path in enumerate(image_paths, start=1):
            src = Path(image_path)
            if not src.is_file():
                raise FileNotFoundError(f"Surya input image not found: {src}")
            if len(image_paths) == 1:
                target_name = src.name
            else:
                target_name = f"{idx:04d}_{src.name}"
            target = input_dir / target_name
            shutil.copy2(src, target)
            ordered_names.append(target_name)

        if not ordered_names:
            raise RuntimeError("Surya input directory is empty after staging images.")
        return input_dir, ordered_names

    def _collect_results(
        *,
        output_root: Path,
        input_dir: Path,
        ordered_names: Sequence[str],
    ) -> tuple[str, int, list[dict[str, Any]]]:
        results_json = output_root / input_dir.name / "results.json"
        if not results_json.exists():
            raise RuntimeError(f"Surya did not produce results file: {results_json}")

        payload = json.loads(results_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Surya results payload has unexpected format.")

        collected: list[str] = []
        sidecar_images: list[dict[str, Any]] = []
        consumed = False
        for image_name in ordered_names:
            pages = payload.get(image_name)
            if not isinstance(pages, list):
                # Some builds key by stem, not full filename.
                pages = payload.get(Path(image_name).stem)
            if not isinstance(pages, list):
                continue
            consumed = True
            image_payload: dict[str, Any] = {"image_name": image_name, "pages": []}
            for page in pages:
                if not isinstance(page, dict):
                    continue
                page_payload: dict[str, Any] = {"text_lines": []}
                image_bbox = _bbox_values(page.get("image_bbox"))
                if image_bbox is not None:
                    page_payload["image_bbox"] = list(image_bbox)

                line_payload: list[dict[str, Any]] = []
                fallback_texts: list[str] = []
                for line in page.get("text_lines", []):
                    if isinstance(line, dict):
                        text = _clean_overlay_line(str(line.get("text") or ""))
                        bbox = _bbox_values(line.get("bbox"))
                        if text and bbox is not None:
                            line_payload.append(
                                {
                                    "text": text,
                                    "bbox": list(bbox),
                                }
                            )
                        elif text:
                            fallback_texts.append(text)
                if line_payload:
                    page_width = max(float(item["bbox"][2]) for item in line_payload)
                    image_bbox = page_payload.get("image_bbox")
                    if (
                        isinstance(image_bbox, list)
                        and len(image_bbox) == 4
                        and all(isinstance(item, (int, float)) for item in image_bbox)
                    ):
                        page_width = max(page_width, float(image_bbox[2]))
                    order = _bbox_reading_order_indices(
                        [
                            (
                                float(item["bbox"][0]),
                                float(item["bbox"][1]),
                                float(item["bbox"][2]),
                                float(item["bbox"][3]),
                            )
                            for item in line_payload
                            if isinstance(item.get("bbox"), list)
                        ],
                        page_width=page_width,
                    )
                    line_payload = [line_payload[idx] for idx in order]
                    page_payload["text_lines"] = line_payload
                    collected.extend(str(item["text"]) for item in line_payload)
                collected.extend(fallback_texts)
                image_payload["pages"].append(page_payload)
            sidecar_images.append(image_payload)

        # Fallback for unknown payload layouts.
        if not consumed:
            for pages in payload.values():
                if not isinstance(pages, list):
                    continue
                for page in pages:
                    if not isinstance(page, dict):
                        continue
                    for line in page.get("text_lines", []):
                        if isinstance(line, dict):
                            text = _clean_overlay_line(str(line.get("text") or ""))
                            if text:
                                collected.append(text)
        text = "\n".join(_dehyphenate_line_breaks(collected))
        return text, len(text), sidecar_images

    def _finalize_results(
        result: tuple[str, int, list[dict[str, Any]]],
        *,
        execution_path: str,
    ) -> tuple[str, int]:
        text, chars, sidecar_images = result
        if sidecar_images:
            sidecar_path = work_dir / "surya_page_lines.json"
            sidecar_path.write_text(
                json.dumps(
                    {"execution_path": execution_path, "images": sidecar_images},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return text, chars

    def _run_cli_with_geometry(
        *,
        input_dir: Path,
        ordered_names: Sequence[str],
        output_root: Path,
        module_failure: str,
    ) -> tuple[str, int]:
        surya_cmd = which_fn("surya_ocr") or which_fn("surya_ocr.exe")
        if not surya_cmd:
            raise RuntimeError("surya_ocr CLI was not found in PATH.")

        errors: list[str] = []
        candidate_commands = (
            [str(surya_cmd), str(input_dir), "--output_dir", str(output_root)],
            [str(surya_cmd), "--input", str(input_dir), "--output_dir", str(output_root)],
        )
        for command in candidate_commands:
            if output_root.exists():
                shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=False)
            proc = run_cmd(command, capture_output=True, text=True)
            if int(getattr(proc, "returncode", 1)) != 0:
                stderr = (getattr(proc, "stderr", "") or "").strip()
                stdout = (getattr(proc, "stdout", "") or "").strip()
                details = stderr or stdout or "unknown cli error"
                errors.append(details)
                continue
            try:
                return _finalize_results(
                    _collect_results(
                        output_root=output_root,
                        input_dir=input_dir,
                        ordered_names=ordered_names,
                    ),
                    execution_path="cli",
                )
            except Exception as exc:
                errors.append(str(exc))

        if not errors:
            raise RuntimeError("surya_ocr CLI failed without diagnostic output.")
        raise RuntimeError(
            f"Surya module path failed ({module_failure}); "
            f"Surya CLI path failed: {' | '.join(errors)}"
        )

    input_dir = work_dir / "surya_input"
    ordered_names: list[str]
    input_dir, ordered_names = _stage_inputs()

    output_root = work_dir / "surya_out"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MODEL_CACHE_DIR", str(_DEFAULT_SURYA_MODEL_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
    try:
        from surya.scripts.ocr_text import ocr_text_cli
    except Exception as exc:
        return _run_cli_with_geometry(
            input_dir=input_dir,
            ordered_names=ordered_names,
            output_root=output_root,
            module_failure=f"import: {exc}",
        )

    args = [
        str(input_dir),
        "--output_dir",
        str(output_root),
    ]
    try:
        ocr_text_cli.main(args=args, standalone_mode=False)
    except SystemExit as exc:
        if int(getattr(exc, "code", 1) or 0) != 0:
            return _run_cli_with_geometry(
                input_dir=input_dir,
                ordered_names=ordered_names,
                output_root=output_root,
                module_failure=f"module exit code {exc.code}",
            )
    except Exception as exc:
        return _run_cli_with_geometry(
            input_dir=input_dir,
            ordered_names=ordered_names,
            output_root=output_root,
            module_failure=f"module exception: {exc}",
        )
    try:
        return _finalize_results(
            _collect_results(
                output_root=output_root,
                input_dir=input_dir,
                ordered_names=ordered_names,
            ),
            execution_path="module",
        )
    except Exception as exc:
        return _run_cli_with_geometry(
            input_dir=input_dir,
            ordered_names=ordered_names,
            output_root=output_root,
            module_failure=f"module result collection: {exc}",
        )


def _run_mineru_module_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    run_cmd: RunCommand,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for MinerU OCR.")

    input_dir = image_paths[0].parent
    output_root = work_dir / "mineru_out"
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # MinerU only supports "en" and "ch"; map common Tesseract codes.
    _first_lang = lang.split("+")[0].strip().lower()
    if _first_lang in {"eng", "en", "english"}:
        mineru_lang = "en"
    elif _first_lang in {"chi_sim", "chi_tra", "ch", "chinese"}:
        mineru_lang = "ch"
    else:
        # Unsupported language — fall back to "en" which at least handles
        # Latin subset; MinerU has no Cyrillic/Russian model.
        import warnings

        warnings.warn(
            f"MinerU does not support language '{lang}'; falling back to 'en'.",
            stacklevel=2,
        )
        mineru_lang = "en"
    from mineru.cli.client import main as mineru_main

    args = [
        "-p",
        str(input_dir),
        "-o",
        str(output_root),
        "-m",
        "ocr",
        "-b",
        "pipeline",
        "-l",
        mineru_lang,
    ]
    try:
        mineru_main.main(args=args, standalone_mode=False)
    except SystemExit as exc:
        if int(getattr(exc, "code", 1) or 0) != 0:
            raise RuntimeError(f"mineru.cli.client exited with code {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"mineru.cli.client failed: {exc}") from exc

    # Primary path: Markdown exported by MinerU.
    text_parts: list[str] = []
    for path in sorted(output_root.rglob("*.md")):
        try:
            raw = _read_utf8_artifact(path).strip()
        except Exception:
            continue
        cleaned = _strip_markdown(raw)
        if cleaned:
            text_parts.append(cleaned)

    # Some MinerU builds emit empty markdown but keep OCR text in
    # *_content_list.json. Use it only as a fallback when markdown is empty.
    if not text_parts:
        for path in sorted(output_root.rglob("*_content_list.json")):
            try:
                payload = json.loads(_read_utf8_artifact(path))
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    text_parts.append(text)

    if not text_parts:
        raise RuntimeError("MinerU finished without text artifacts.")

    text = "\n".join(text_parts)
    return text, len(text)


def _run_text_engine_from_cli(
    image_paths: Sequence[Path],
    *,
    engine: str,
    lang: str,
    candidates: Sequence[tuple[str, ...]],
    which_fn: WhichExecutable,
    run_cmd: RunCommand,
) -> tuple[str, int]:
    collected: list[str] = []
    errors: list[str] = []

    for image_path in image_paths:
        page_text: str | None = None
        for template in candidates:
            binary = template[0]
            binary_path = which_fn(binary) or which_fn(f"{binary}.exe")
            if binary_path is None:
                continue
            args = [str(binary_path)] + [
                part.format(image=str(image_path), lang=lang) for part in template[1:]
            ]
            proc = run_cmd(args, capture_output=True, text=True)
            if int(getattr(proc, "returncode", 1)) == 0:
                stdout = getattr(proc, "stdout", "") or ""
                stderr = getattr(proc, "stderr", "") or ""
                combined = (stdout + "\n" + stderr).strip()
                if engine == OCR_ENGINE_SURYA and binary in {"marker_single", "marker"}:
                    marker_text = _extract_marker_cli_text(combined)
                    page_text = marker_text if marker_text else combined
                else:
                    page_text = combined
                if page_text:
                    collected.append(page_text)
                break
            stderr = (getattr(proc, "stderr", "") or "").strip()
            stdout = (getattr(proc, "stdout", "") or "").strip()
            details = stderr or stdout or "unknown cli error"
            errors.append(f"{binary}: {details}")

        if page_text is None:
            if not errors:
                raise RuntimeError(f"Engine '{engine}' has no runnable CLI candidates in PATH.")
            raise RuntimeError(
                f"Engine '{engine}' failed on {image_path.name}: {' | '.join(errors)}"
            )

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _extract_marker_cli_text(log_blob: str) -> str:
    """Extract OCR text from marker CLI logs by reading saved markdown files."""
    matches = re.findall(r"Saved markdown to\s+([^\r\n]+)", log_blob)
    if not matches:
        return ""

    collected: list[str] = []
    for raw_path in matches:
        marker_path = Path(raw_path.strip().strip("'\""))
        if marker_path.is_file() and marker_path.suffix.lower() == ".md":
            md_files = [marker_path]
        elif marker_path.is_dir():
            md_files = sorted(marker_path.glob("*.md"))
        else:
            md_files = []

        for md_file in md_files:
            try:
                raw_md = _read_utf8_artifact(md_file).strip()
            except Exception:
                continue
            cleaned = _strip_markdown(raw_md)
            if cleaned:
                collected.append(cleaned)

    return "\n".join(part for part in collected if part and not part.isspace())


def _run_surya_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
) -> tuple[str, int]:
    (work_dir / "surya_page_lines.json").unlink(missing_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", str(_DEFAULT_SURYA_MODEL_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
    _ensure_surya_cache_ready()
    _configure_surya_runtime_device()

    module_error: Exception | None = None
    try:
        return _run_surya_module_cli(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as exc:
        module_error = exc

    if not _surya_allow_text_fallback():
        raise RuntimeError(
            "Surya module path failed. Text-only fallback is disabled to keep "
            f"geometry JSON mandatory: {module_error}"
        ) from module_error

    candidates = (
        ("surya_ocr", "{image}", "--lang", "{lang}"),
        ("surya_ocr", "--input", "{image}", "--lang", "{lang}"),
        ("surya_ocr", "--image", "{image}", "--lang", "{lang}"),
        ("marker_single", "{image}"),
        ("marker", "{image}"),
    )
    try:
        return _run_text_engine_from_cli(
            image_paths,
            engine=OCR_ENGINE_SURYA,
            lang=lang,
            candidates=candidates,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _run_mineru_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
) -> tuple[str, int]:
    module_error: Exception | None = None
    try:
        return _run_mineru_module_cli(image_paths, lang=lang, work_dir=work_dir, run_cmd=run_cmd)
    except Exception as exc:
        module_error = exc

    candidates = (
        ("mineru", "{image}", "--lang", "{lang}"),
        ("mineru", "--input", "{image}", "--lang", "{lang}"),
        ("magic-pdf", "{image}", "--lang", "{lang}"),
        ("magic-pdf", "--input", "{image}", "--lang", "{lang}"),
    )
    try:
        return _run_text_engine_from_cli(
            image_paths,
            engine=OCR_ENGINE_MINERU,
            lang=lang,
            candidates=candidates,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _run_chandra_module(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    page_progress_cb: Callable[[int, int], None] | None = None,
    source_raster_identities: Sequence[dict[str, object]] | None = None,
) -> tuple[str, int]:
    """Run Chandra OCR via direct Python module import (preferred path)."""
    if len(image_paths) == 0:
        raise ValueError("No images for Chandra OCR.")

    if source_raster_identities is None:
        source_raster_identities = [
            _source_raster_identity(image_path, source_page=index)
            for index, image_path in enumerate(image_paths, start=1)
        ]
    if len(source_raster_identities) != len(image_paths):
        raise RuntimeError("Chandra source raster identity cardinality is invalid.")
    sealed_source_identities: list[dict[str, object]] = []
    for image_path, raw_identity in zip(
        image_paths,
        source_raster_identities,
        strict=True,
    ):
        if not isinstance(raw_identity, dict):
            raise RuntimeError("Chandra source raster identity is malformed.")
        source_page = raw_identity.get("source_page")
        if not isinstance(source_page, int) or isinstance(source_page, bool):
            raise RuntimeError("Chandra source raster page identity is malformed.")
        expected_identity = _source_raster_identity(image_path, source_page=source_page)
        if raw_identity != expected_identity:
            raise RuntimeError(
                f"Chandra source raster identity changed before OCR: {image_path.name}"
            )
        sealed_source_identities.append(expected_identity)

    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    _ensure_chandra_cache_ready()
    selected_device = _configure_chandra_runtime_device()
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.input import load_image
    from chandra.prompts import PROMPT_MAPPING

    def _safe_bbox(raw_bbox: Any, width: int, height: int) -> list[float] | None:
        parsed = _bbox_values(raw_bbox)
        if parsed is None:
            return None
        x0, y0, x1, y1 = parsed
        x0 = max(0.0, min(x0, float(width)))
        y0 = max(0.0, min(y0, float(height)))
        x1 = max(0.0, min(x1, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    _ = lang
    model = InferenceManager(method="hf")
    if _env_bool("UNISCAN_CHANDRA_REQUIRE_GPU", default=True):
        raw_model = getattr(model, "model", None)
        model_device = str(getattr(raw_model, "device", ""))
        if raw_model is not None and not model_device and hasattr(raw_model, "parameters"):
            try:
                first_param = next(raw_model.parameters())
                model_device = str(getattr(first_param, "device", ""))
            except Exception:
                model_device = ""
        if not model_device.lower().startswith("cuda"):
            raise RuntimeError(
                "Chandra model loaded without CUDA device while GPU mode is required "
                f"(TORCH_DEVICE={selected_device!r}, model_device={model_device!r})."
            )

    def _parse_page_results(
        results: Sequence[Any],
        *,
        width: int,
        height: int,
        allow_graphic_text: bool = False,
    ) -> tuple[
        list[str], list[dict[str, Any]], list[str], list[str], list[str], list[str]
    ]:
        page_texts: list[str] = []
        alternative_texts: list[str] = []
        excluded_header_footer_lines: list[str] = []
        page_lines: list[dict[str, Any]] = []
        labels: list[str] = []
        ignored_graphic_lines: list[str] = []
        for result in results:
            chunks = (
                result.get("chunks")
                if isinstance(result, dict)
                else getattr(result, "chunks", None)
            )
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    label = str(chunk.get("label") or "").strip().lower()
                    if label:
                        labels.append(label)
                    if label in _CHANDRA_NON_TEXT_LABELS:
                        if label in _CHANDRA_GRAPHIC_LABELS:
                            ignored_graphic_lines.extend(
                                _chandra_graphic_chunk_ignored_lines(chunk.get("content"))
                            )
                        if not allow_graphic_text:
                            continue
                        chunk_lines = _chandra_graphic_chunk_lines(chunk.get("content"))
                    else:
                        chunk_lines = _chandra_chunk_lines(chunk.get("content"))
                    chunk_lines = [
                        cleaned for line in chunk_lines if (cleaned := _clean_overlay_line(line))
                    ]
                    if not chunk_lines:
                        continue
                    page_texts.extend(chunk_lines)
                    if label in _CHANDRA_HEADER_FOOTER_LABELS:
                        excluded_header_footer_lines.extend(chunk_lines)
                    else:
                        alternative_texts.extend(chunk_lines)
                    bbox = _safe_bbox(chunk.get("bbox"), width, height)
                    if bbox is not None:
                        page_lines.extend(
                            _chandra_expand_chunk_to_line_boxes(
                                lines=chunk_lines,
                                bbox=bbox,
                            )
                        )
            raw_markdown = (
                result.get("markdown", "")
                if isinstance(result, dict)
                else getattr(result, "markdown", "")
            )
            if (
                not page_texts
                and not allow_graphic_text
                and not (set(labels) & _CHANDRA_GRAPHIC_LABELS)
            ):
                md = raw_markdown or ""
                md = _strip_markdown(md.strip())
                if md:
                    raise RuntimeError("Chandra OCR recovered text without complete geometry.")
        return (
            page_texts,
            page_lines,
            labels,
            ignored_graphic_lines,
            alternative_texts,
            excluded_header_footer_lines,
        )

    def _explicit_nontext(labels: Sequence[str]) -> bool:
        normalized = {label.strip().lower() for label in labels if label.strip()}
        return bool(normalized & _CHANDRA_GRAPHIC_LABELS) and normalized <= _CHANDRA_NON_TEXT_LABELS

    prompt_sha256 = {
        prompt_type: hashlib.sha256(PROMPT_MAPPING[prompt_type].encode("utf-8")).hexdigest()
        for prompt_type in (_CHANDRA_LAYOUT_PROMPT_TYPE, _CHANDRA_PLAIN_PROMPT_TYPE)
    }

    def _nonspace_text(lines: Sequence[str]) -> str:
        normalized = unicodedata.normalize("NFKC", "\n".join(lines))
        return "".join(char for char in normalized if not char.isspace())

    def _canonical_alnum_text(lines: Sequence[str]) -> str:
        return _chandra_canonical_alnum(lines)

    def _normalized_raw_result(result: Any) -> dict[str, Any]:
        raw_chunks = (
            result.get("chunks") if isinstance(result, dict) else getattr(result, "chunks", None)
        )
        if raw_chunks is None:
            raw_chunks = []
        if not isinstance(raw_chunks, list):
            raise RuntimeError("Chandra returned malformed chunks.")
        chunks: list[dict[str, Any]] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                raise RuntimeError("Chandra returned a non-object chunk.")
            raw_bbox = raw_chunk.get("bbox")
            if raw_bbox is None:
                normalized_bbox: Any = None
            elif (
                isinstance(raw_bbox, Sequence)
                and not isinstance(raw_bbox, (str, bytes))
                and len(raw_bbox) == 4
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in raw_bbox
                )
            ):
                normalized_bbox = [float(value) for value in raw_bbox]
            else:
                raise RuntimeError("Chandra returned an invalid chunk bbox.")
            raw_label = raw_chunk.get("label")
            raw_content = raw_chunk.get("content")
            if raw_label is not None and not isinstance(raw_label, str):
                raise RuntimeError("Chandra returned a non-string chunk label.")
            if raw_content is not None and not isinstance(raw_content, str):
                raise RuntimeError("Chandra returned non-string chunk content.")
            chunks.append(
                {
                    "label": raw_label or "",
                    "content": raw_content,
                    "bbox": normalized_bbox,
                }
            )
        raw_html = (
            result.get("html", "") if isinstance(result, dict) else getattr(result, "html", "")
        )
        raw_markdown = (
            result.get("markdown", "")
            if isinstance(result, dict)
            else getattr(result, "markdown", "")
        )
        if raw_html is not None and not isinstance(raw_html, str):
            raise RuntimeError("Chandra returned non-string html.")
        if raw_markdown is not None and not isinstance(raw_markdown, str):
            raise RuntimeError("Chandra returned non-string markdown.")
        return {
            "error": False,
            "chunks": chunks,
            "html": raw_html or "",
            "markdown": raw_markdown or "",
        }

    def _generate_attempt(
        *,
        attempt: int,
        image: Image.Image,
        prompt_type: str,
        preprocessing: str,
        width: int,
        height: int,
        evidence_dir: Path,
        source_raster_identity: dict[str, object],
        allow_graphic_text: bool = False,
    ) -> tuple[
        list[str],
        list[dict[str, Any]],
        list[str],
        dict[str, Any],
        dict[str, object],
    ]:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if image.size != (width, height):
            raise RuntimeError("Chandra attempt dimensions disagree with its declared image size.")
        if (
            width > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or height > _MAX_CHANDRA_ATTEMPT_IMAGE_DIMENSION
            or width * height > _MAX_CHANDRA_ATTEMPT_IMAGE_PIXELS
        ):
            raise RuntimeError(
                "Chandra attempt image exceeds the bounded dimension/pixel policy: "
                f"{width}x{height}."
            )
        model_image = image.convert("RGB")
        attempt_image_path = evidence_dir / "input.png"
        model_image.save(attempt_image_path, format="PNG")
        attempt_image_bytes = int(attempt_image_path.stat().st_size)
        if attempt_image_bytes > _MAX_CHANDRA_ATTEMPT_IMAGE_BYTES:
            raise RuntimeError(
                f"Chandra attempt PNG exceeds the bounded byte policy: {attempt_image_bytes} bytes."
            )
        results = model.generate(
            [BatchInputItem(image=model_image, prompt_type=prompt_type)],
            include_images=False,
            include_headers_footers=False,
        )
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise RuntimeError("Chandra returned a non-sequence batch result.")
        if len(results) != 1:
            raise RuntimeError(f"Chandra returned {len(results)} results for a one-image attempt.")
        result_error = (
            results[0].get("error", False)
            if isinstance(results[0], dict)
            else getattr(results[0], "error", False)
        )
        if result_error is not False:
            raise RuntimeError("Chandra marked the one-image attempt as failed.")
        raw_result = _normalized_raw_result(results[0])
        (
            texts,
            lines,
            labels,
            ignored_graphic_lines,
            alternative_texts,
            excluded_header_footer_lines,
        ) = _parse_page_results(
            [raw_result],
            width=width,
            height=height,
            allow_graphic_text=allow_graphic_text,
        )
        nonspace_text = _nonspace_text(texts)
        geometry_text = _nonspace_text([str(line.get("text") or "") for line in lines])
        if texts and (not lines or geometry_text != nonspace_text):
            raise RuntimeError("Chandra OCR recovered text without complete geometry.")
        if lines:
            ordered_bboxes: list[tuple[float, float, float, float]] = []
            for line in lines:
                parsed_line_bbox = _bbox_values(line.get("bbox"))
                if parsed_line_bbox is None:
                    raise RuntimeError("Chandra parsed line has invalid geometry.")
                ordered_bboxes.append(parsed_line_bbox)
            order = _bbox_reading_order_indices(
                ordered_bboxes,
                page_width=float(width),
            )
            canonical_text = _canonical_alnum_text(
                [str(lines[index].get("text") or "") for index in order]
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
            raise RuntimeError("Chandra OCR recovered text with unaccounted alternative text.")
        explicit_nontext = (
            not texts
            and _explicit_nontext(labels)
            and alternative_text_evidence["accounting"] in {"empty", "ignored_graphic_description"}
        )
        if texts:
            outcome = _OCR_OUTCOME_TEXT
        elif explicit_nontext:
            outcome = _OCR_OUTCOME_EXPLICIT_NONTEXT
        else:
            outcome = _OCR_OUTCOME_ZERO
        evidence = {
            "attempt": attempt,
            "source_raster_identity": dict(source_raster_identity),
            "prompt_type": prompt_type,
            "prompt_sha256": prompt_sha256[prompt_type],
            "preprocessing": preprocessing,
            "content_filter_policy": (
                _CHANDRA_PLAIN_CONTENT_FILTER
                if prompt_type == _CHANDRA_PLAIN_PROMPT_TYPE
                else _CHANDRA_LAYOUT_CONTENT_FILTER
            ),
            "alternative_text_evidence": alternative_text_evidence,
            "labels": sorted(set(labels)),
            "text_chars": sum(len(line) for line in texts),
            "canonical_alnum_chars": len(canonical_text),
            "canonical_alnum_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
            "geometry_lines": len(lines),
            "explicit_nontext": explicit_nontext,
            "ocr_outcome": outcome,
        }
        attempt_sidecar_path = evidence_dir / "chandra_attempt.json"
        _write_json_atomic(
            attempt_sidecar_path,
            {
                "schema": _CHANDRA_ATTEMPT_EVIDENCE_SCHEMA,
                "source_raster_identity": dict(source_raster_identity),
                "image_name": attempt_image_path.name,
                "image_bbox": [0.0, 0.0, float(width), float(height)],
                "raw_result": raw_result,
                "parsed": {
                    "texts": texts,
                    "text_lines": lines,
                    "labels": sorted(set(labels)),
                },
                "evidence": evidence,
            },
        )
        history = {
            "attempt": attempt,
            "image_size": [width, height],
            "image_path": str(attempt_image_path.resolve()),
            "image_sha256": _sha256_path(attempt_image_path),
            "image_bytes": attempt_image_bytes,
            "sidecar_path": str(attempt_sidecar_path.resolve()),
            "sidecar_sha256": _sha256_path(attempt_sidecar_path),
            "sidecar_bytes": int(attempt_sidecar_path.stat().st_size),
        }
        return texts, lines, labels, evidence, history

    collected: list[str] = []
    sidecar_images: list[dict[str, Any]] = []
    total_pages = len(image_paths)
    sidecar_path = work_dir / "chandra_page_lines.json"
    for page_idx, (image_path, source_raster_identity) in enumerate(
        zip(image_paths, sealed_source_identities, strict=True),
        start=1,
    ):
        page_attempt_root = work_dir / "attempt_evidence" / f"page_{page_idx:04d}"
        with Image.open(image_path) as opened_source:
            opened_source.load()
            source_image = opened_source.convert("RGB")
        source_raster_path = page_attempt_root / "source.png"
        source_raster_path.parent.mkdir(parents=True, exist_ok=True)
        source_image.save(source_raster_path, format="PNG")
        source_raster_artifact = _source_raster_artifact(source_raster_path)
        expected_model_size = _bounded_chandra_model_input_size(*source_image.size)
        pil_image = load_image(
            str(source_raster_path),
            min_image_dim=_CHANDRA_MIN_IMAGE_DIM,
        )
        if pil_image.size != expected_model_size:
            raise RuntimeError(
                "Chandra loader returned unexpected model input dimensions: "
                f"expected {expected_model_size}, got {pil_image.size}."
            )
        width, height = pil_image.size
        page_texts, page_lines, labels, original_attempt, original_history = _generate_attempt(
            attempt=1,
            image=pil_image,
            prompt_type=_CHANDRA_LAYOUT_PROMPT_TYPE,
            preprocessing="original",
            width=width,
            height=height,
            evidence_dir=page_attempt_root / "attempt_1",
            source_raster_identity=source_raster_identity,
        )
        observed_labels = list(labels)
        attempts: list[dict[str, Any]] = [original_attempt]
        attempt_history: list[dict[str, object]] = [original_history]
        blank_page = not page_texts and bool(source_raster_identity["verified_blank"])
        if not page_texts and not blank_page:
            retry_image = ImageOps.autocontrast(pil_image.convert("RGB"), cutoff=1)
            if retry_image.size != (width, height):
                raise RuntimeError(
                    "Chandra zero-output retry changed image dimensions "
                    f"from {(width, height)} to {retry_image.size}."
                )
            page_texts, page_lines, labels, retry_attempt, retry_history = _generate_attempt(
                attempt=2,
                image=retry_image,
                prompt_type=_CHANDRA_LAYOUT_PROMPT_TYPE,
                preprocessing=_ZERO_OUTPUT_RETRY_PREPROCESSING,
                width=width,
                height=height,
                evidence_dir=page_attempt_root / "attempt_2",
                source_raster_identity=source_raster_identity,
            )
            observed_labels.extend(labels)
            attempts.append(retry_attempt)
            attempt_history.append(retry_history)

        if not page_texts and not blank_page:
            page_texts, page_lines, labels, plain_attempt, plain_history = _generate_attempt(
                attempt=3,
                image=pil_image,
                prompt_type=_CHANDRA_PLAIN_PROMPT_TYPE,
                preprocessing="original",
                width=width,
                height=height,
                evidence_dir=page_attempt_root / "attempt_3",
                source_raster_identity=source_raster_identity,
                allow_graphic_text=True,
            )
            observed_labels.extend(labels)
            attempts.append(plain_attempt)
            attempt_history.append(plain_history)

        plain_labels = set(attempts[2]["labels"]) if len(attempts) == 3 else set()
        explicit_nontext = (
            not page_texts
            and not blank_page
            and len(attempts) == 3
            and all(item["explicit_nontext"] is True for item in attempts)
            and bool(plain_labels & _CHANDRA_GRAPHIC_LABELS)
            and plain_labels <= _CHANDRA_NON_TEXT_LABELS
        )
        if page_texts:
            ocr_outcome = _OCR_OUTCOME_TEXT
        elif blank_page:
            ocr_outcome = _OCR_OUTCOME_VERIFIED_BLANK
        elif explicit_nontext:
            ocr_outcome = _OCR_OUTCOME_EXPLICIT_NONTEXT
        else:
            ocr_outcome = _OCR_OUTCOME_ZERO

        collected.append("\n".join(page_texts))
        image_evidence: dict[str, Any] = {
            "image_name": image_path.name,
            "source_raster_identity": dict(source_raster_identity),
            "source_raster_artifact": source_raster_artifact,
            "ocr_outcome": ocr_outcome,
            "explicit_nontext": explicit_nontext,
            "chandra_non_text_labels": sorted(
                {label for label in observed_labels if label in _CHANDRA_NON_TEXT_LABELS}
            ),
            "attempt_count": len(attempts),
            "terminal_attempt": len(attempts),
            "chandra_retry_policy": _CHANDRA_ZERO_OUTPUT_RETRY_POLICY,
            "attempts": attempts,
            "attempt_history": attempt_history,
            "pages": [
                {
                    "image_bbox": [0.0, 0.0, float(width), float(height)],
                    "text_lines": page_lines,
                    "ocr_outcome": ocr_outcome,
                }
            ],
        }
        if page_texts:
            image_evidence["selected_attempt"] = len(attempts)
        if len(attempts) == 2:
            image_evidence["retry_preprocessing"] = _ZERO_OUTPUT_RETRY_PREPROCESSING
        elif len(attempts) == 3:
            image_evidence["retry_preprocessing"] = _CHANDRA_PLAIN_RETRY_PREPROCESSING
        sidecar_images.append(image_evidence)
        _write_json_atomic(sidecar_path, {"images": sidecar_images})
        if page_progress_cb is not None:
            try:
                page_progress_cb(page_idx, total_pages)
            except Exception:
                pass

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_chandra_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
) -> tuple[str, int]:
    """Run Chandra OCR via CLI binary (fallback path)."""
    if len(image_paths) == 0:
        raise ValueError("No images for Chandra OCR.")

    chandra_cmd = which_fn("chandra") or which_fn("chandra.exe")
    if not chandra_cmd:
        raise RuntimeError("Chandra CLI was not found in PATH.")

    collected: list[str] = []
    errors: list[str] = []
    for image_path in image_paths:
        page_output = work_dir / image_path.stem
        page_output.mkdir(parents=True, exist_ok=True)
        candidates = (
            [str(chandra_cmd), str(image_path), str(page_output), "--method", "hf"],
            [str(chandra_cmd), str(image_path), str(page_output)],
        )
        run_ok = False
        for command in candidates:
            proc = run_cmd(command, capture_output=True, text=True)
            if int(getattr(proc, "returncode", 1)) == 0:
                run_ok = True
                break
            stderr = (getattr(proc, "stderr", "") or "").strip()
            stdout = (getattr(proc, "stdout", "") or "").strip()
            details = stderr or stdout or "unknown cli error"
            errors.append(details)

        if not run_ok:
            raise RuntimeError(
                f"Chandra OCR failed on {image_path.name}: " + " | ".join(errors[-2:])
            )

        page_texts: list[str] = []
        for pattern in ("*.md", "*.txt", "*.json", "*.html"):
            for artifact in sorted(page_output.rglob(pattern)):
                try:
                    text = _read_utf8_artifact(artifact).strip()
                except Exception:
                    continue
                if text:
                    page_texts.append(text)
        if not page_texts:
            raise RuntimeError(f"Chandra OCR produced no text artifacts for {image_path.name}.")
        collected.append("\n".join(page_texts))

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_chandra_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
    page_progress_cb: Callable[[int, int], None] | None = None,
    source_raster_identities: Sequence[dict[str, object]] | None = None,
) -> tuple[str, int]:
    (work_dir / "chandra_page_lines.json").unlink(missing_ok=True)
    # Primary: direct Python module import (no CLI binary needed).
    module_error: Exception | None = None
    try:
        return _run_chandra_module(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            page_progress_cb=page_progress_cb,
            source_raster_identities=source_raster_identities,
        )
    except Exception as exc:
        module_error = exc

    if not _chandra_allow_cli_fallback():
        raise RuntimeError(
            "Chandra module path failed. CLI fallback is disabled to avoid "
            f"text-only degradation: {module_error}"
        ) from module_error

    # Fallback: CLI binary via shutil.which.
    try:
        return _run_chandra_cli(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _collect_olmocr_workspace_text(workspace: Path) -> tuple[str, int]:
    markdown_candidates: list[Path] = []
    markdown_dir = workspace / "markdown"
    patterns = ("*.md", "*.markdown", "*.mmd", "*.txt")
    if markdown_dir.exists():
        for pattern in patterns:
            markdown_candidates.extend(sorted(markdown_dir.rglob(pattern)))
    if not markdown_candidates:
        for pattern in patterns:
            markdown_candidates.extend(sorted(workspace.rglob(pattern)))
    # Preserve first-seen order and drop duplicates when multiple glob patterns match.
    markdown_candidates = list(dict.fromkeys(markdown_candidates))

    text_parts: list[str] = []

    def _load_json_relaxed(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Some OCRFlux outputs contain bare backslashes in markdown fields
            # (for example "\_" or "\("), which are invalid JSON escapes.
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
            if fixed == raw:
                raise
            return json.loads(fixed)

    def _append_payload_text(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for key in (
            "text",
            "content",
            "document_text",
            "document_markdown",
            "natural_text",
            "markdown",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(_strip_markdown(value.strip()))
        for key in ("page_texts", "pages", "page_markdown", "page_text"):
            value = payload.get(key)
            for extracted in _collect_text_strings(value):
                cleaned = _strip_markdown(extracted.strip())
                if cleaned:
                    text_parts.append(cleaned)

    for md_path in markdown_candidates:
        try:
            raw = _read_utf8_artifact(md_path).strip()
        except Exception:
            continue
        cleaned = _strip_markdown(raw)
        if cleaned:
            text_parts.append(cleaned)

    # Fallback for formats that keep text in JSON/JSONL payloads. Do not append
    # these when markdown was already found: the formats usually contain the
    # same document and would duplicate the searchable text layer.
    if not text_parts:
        for json_path in sorted(workspace.rglob("*.json")):
            try:
                payload = _load_json_relaxed(_read_utf8_artifact(json_path))
            except Exception:
                continue
            if isinstance(payload, dict):
                _append_payload_text(payload)
            elif isinstance(payload, list):
                for item in payload:
                    _append_payload_text(item)

    if not text_parts:
        for jsonl_path in sorted(workspace.rglob("*.jsonl")):
            try:
                lines = _read_utf8_artifact(jsonl_path).splitlines()
            except Exception:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = _load_json_relaxed(line)
                except Exception:
                    continue
                _append_payload_text(payload)

    if not text_parts:
        for compressed_path in sorted(workspace.rglob("*.jsonl.zst")):
            try:
                import zstandard as zstd
            except Exception:
                break
            try:
                with compressed_path.open("rb") as fh:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(fh) as reader:
                        payload = reader.read(_MAX_OCR_TEXT_ARTIFACT_BYTES + 1)
                if len(payload) > _MAX_OCR_TEXT_ARTIFACT_BYTES:
                    raise RuntimeError(
                        f"Compressed OCR artifact exceeds {_MAX_OCR_TEXT_ARTIFACT_BYTES} bytes"
                    )
                raw = payload.decode("utf-8-sig")
            except Exception:
                continue
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = _load_json_relaxed(line)
                except Exception:
                    continue
                _append_payload_text(item)

    if not text_parts:
        raise RuntimeError("olmOCR finished without markdown/text artifacts.")

    text = "\n".join(part for part in text_parts if part and not part.isspace())
    return text, len(text)


def _render_images_to_pdf(image_paths: Sequence[Path], out_pdf: Path) -> None:
    if len(image_paths) == 0:
        raise ValueError("No images to render into PDF.")
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(
            "olmOCR docker fallback requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    output_doc = fitz.open()
    try:
        for image_path in image_paths:
            image_doc = fitz.open(str(image_path))
            try:
                image_pdf = fitz.open("pdf", image_doc.convert_to_pdf())
                try:
                    output_doc.insert_pdf(image_pdf)
                finally:
                    image_pdf.close()
            finally:
                image_doc.close()
        output_doc.save(str(out_pdf))
    finally:
        output_doc.close()


def _run_olmocr_docker(
    image_paths: Sequence[Path],
    *,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
) -> tuple[str, int]:
    docker_cmd = which_fn("docker") or which_fn("docker.exe")
    if not docker_cmd:
        raise RuntimeError("docker is not available in PATH for olmOCR docker fallback.")

    require_gpu = _env_bool("UNISCAN_OLMOCR_REQUIRE_GPU", default=True)
    configured_gpu = (os.environ.get("UNISCAN_OLMOCR_DOCKER_GPU") or "").strip()
    if require_gpu:
        gpu = configured_gpu or _EXPECTED_GPU0_DOCKER_SELECTOR
        if gpu != _EXPECTED_GPU0_DOCKER_SELECTOR:
            raise RuntimeError(
                "GPU-required olmOCR requires UNISCAN_OLMOCR_DOCKER_GPU="
                f"{_EXPECTED_GPU0_DOCKER_SELECTOR!r}; got {gpu or '<unset>'!r}."
            )
        _require_gpu0_contract(run_cmd)
    else:
        if configured_gpu and configured_gpu.lower() != "none":
            raise RuntimeError(
                "CPU-only olmOCR requires UNISCAN_OLMOCR_DOCKER_GPU='none' or unset."
            )
        gpu = "none"

    docker_root = work_dir / "olmocr_docker"
    data_dir = docker_root / "data"
    work_root = docker_root / "work"
    workspace_dir = work_root / "ws"
    for directory in (data_dir, work_root):
        directory.mkdir(parents=True, exist_ok=True)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)

    input_pdf = data_dir / "input.pdf"
    _render_images_to_pdf(image_paths, input_pdf)

    image = (os.environ.get("UNISCAN_OLMOCR_DOCKER_IMAGE") or "chatdoc/ocrflux:latest").strip()
    model = (os.environ.get("UNISCAN_OLMOCR_DOCKER_MODEL") or "").strip()
    workers = (os.environ.get("UNISCAN_OLMOCR_DOCKER_WORKERS") or "1").strip()
    gpu_mem_util = (os.environ.get("UNISCAN_OLMOCR_DOCKER_GPU_MEM_UTIL") or "").strip()
    pages_per_group = (os.environ.get("UNISCAN_OLMOCR_DOCKER_PAGES_PER_GROUP") or "").strip()
    max_page_retries = (os.environ.get("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_RETRIES") or "").strip()
    # ocrflux default is 1/250 (~0.004), which is too strict for noisy scans.
    # In page-wise mode (single page per invocation), any fallback would discard
    # the whole page unless we relax the threshold to 1.0.
    default_error_rate = "1.0" if len(image_paths) <= 1 else "0.10"
    max_page_error_rate = (
        os.environ.get("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_ERROR_RATE") or default_error_rate
    ).strip()

    cache_dir_raw = (
        os.environ.get("UNISCAN_OLMOCR_DOCKER_CACHE") or str(_REPO_ROOT / ".hf_cache_ocrflux")
    ).strip()
    cache_dir = Path(cache_dir_raw)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mount_data = data_dir.resolve().as_posix()
    mount_work = work_root.resolve().as_posix()
    mount_cache = cache_dir.resolve().as_posix()

    command: list[str] = [str(docker_cmd), "run", "--rm"]
    if require_gpu:
        command.extend(["--gpus", gpu])
        command.extend(["--env", "CUDA_VISIBLE_DEVICES=0"])
        command.extend(["--env", f"NVIDIA_VISIBLE_DEVICES={_EXPECTED_GPU0_UUID}"])
        command.extend(["--env", f"UNISCAN_GPU_DEVICE_ID={_EXPECTED_GPU0_UUID}"])
    else:
        command.extend(["--env", "CUDA_VISIBLE_DEVICES="])
        command.extend(["--env", "NVIDIA_VISIBLE_DEVICES=none"])
        command.extend(["--env", "UNISCAN_GPU_DEVICE_ID="])
        command.extend(["--env", "TORCH_DEVICE=cpu"])
    command.extend(
        [
            "-v",
            f"{mount_data}:/data:ro",
            "-v",
            f"{mount_work}:/work",
            "-v",
            f"{mount_cache}:/root/.cache/huggingface",
            image,
            "/work/ws",
            "--task",
            "pdf2markdown",
            "--data",
            "/data/input.pdf",
        ]
    )
    if workers:
        command.extend(["--workers", workers])
    if pages_per_group:
        command.extend(["--pages_per_group", pages_per_group])
    if max_page_retries:
        command.extend(["--max_page_retries", max_page_retries])
    if max_page_error_rate:
        command.extend(["--max_page_error_rate", max_page_error_rate])
    if model:
        command.extend(["--model", model])
    if gpu_mem_util:
        command.extend(["--gpu_memory_utilization", gpu_mem_util])

    proc = run_cmd(command, capture_output=True, text=True)
    stderr = (getattr(proc, "stderr", "") or "").strip()
    stdout = (getattr(proc, "stdout", "") or "").strip()
    if int(getattr(proc, "returncode", 1)) != 0:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        stdout = (getattr(proc, "stdout", "") or "").strip()
        details = stderr or stdout or "unknown docker olmOCR error"
        raise RuntimeError(f"docker olmOCR failed: {details}")

    if not workspace_dir.exists():
        raise RuntimeError("docker olmOCR finished but did not create workspace.")

    try:
        return _collect_olmocr_workspace_text(workspace_dir)
    except Exception as exc:
        details = stderr or stdout
        file_hints: list[str] = []
        try:
            for path in sorted(workspace_dir.rglob("*")):
                if not path.is_file():
                    continue
                file_hints.append(str(path.relative_to(workspace_dir)))
                if len(file_hints) >= 25:
                    break
        except Exception:
            pass
        if details:
            details = details[-2000:]
            if file_hints:
                raise RuntimeError(
                    f"{exc} | docker output tail: {details} | workspace files: {', '.join(file_hints)}"
                ) from exc
            raise RuntimeError(f"{exc} | docker output tail: {details}") from exc
        if file_hints:
            raise RuntimeError(f"{exc} | workspace files: {', '.join(file_hints)}") from exc
        raise


def _run_olmocr_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for olmOCR.")

    backend = (os.environ.get("UNISCAN_OLMOCR_BACKEND") or "auto").strip().lower()
    if backend not in {"auto", "local", "docker"}:
        raise ValueError("UNISCAN_OLMOCR_BACKEND must be one of: auto, local, docker")
    require_gpu = _env_bool("UNISCAN_OLMOCR_REQUIRE_GPU", default=True)

    workspace = work_dir / "olmocr_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # olmOCR language selection is model-driven; keep API compatible with
    # other engines by accepting `lang` but not forcing a language flag.
    _ = lang

    server = (os.environ.get("UNISCAN_OLMOCR_SERVER") or "").strip()
    model = (os.environ.get("UNISCAN_OLMOCR_MODEL") or "").strip()
    api_key = (os.environ.get("UNISCAN_OLMOCR_API_KEY") or "").strip()

    base_args = [str(workspace), "--markdown", "--pdfs", *[str(path) for path in image_paths]]
    if server:
        base_args += ["--server", server]
    if model:
        base_args += ["--model", model]
    if api_key:
        base_args += ["--api_key", api_key]

    errors: list[str] = []
    if backend in {"auto", "local"}:
        if require_gpu:
            _require_gpu0_contract(run_cmd)
        command_candidates: list[list[str]] = []
        olmocr_cmd = which_fn("olmocr") or which_fn("olmocr.exe")
        if olmocr_cmd:
            command_candidates.append([str(olmocr_cmd), *base_args])
        command_candidates.append([sys.executable, "-m", "olmocr.pipeline", *base_args])

        if not command_candidates:
            errors.append("local olmOCR command is not available in PATH.")
        else:
            for command in command_candidates:
                command_env = dict(os.environ)
                if require_gpu:
                    command_env["CUDA_VISIBLE_DEVICES"] = "0"
                    command_env["NVIDIA_VISIBLE_DEVICES"] = _EXPECTED_GPU0_UUID
                    command_env["UNISCAN_GPU_DEVICE_ID"] = _EXPECTED_GPU0_UUID
                    command_env["TORCH_DEVICE"] = "cuda:0"
                else:
                    _hide_gpu_visibility(command_env)
                    command_env["TORCH_DEVICE"] = "cpu"
                command_path = Path(command[0])
                if command_path.exists():
                    bin_dir = str(command_path.resolve().parent)
                    current_path = command_env.get("PATH", "")
                    command_env["PATH"] = (
                        f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir
                    )
                proc = run_cmd(command, capture_output=True, text=True, env=command_env)
                if int(getattr(proc, "returncode", 1)) == 0:
                    return _collect_olmocr_workspace_text(workspace)
                stderr = (getattr(proc, "stderr", "") or "").strip()
                stdout = (getattr(proc, "stdout", "") or "").strip()
                details = stderr or stdout or "unknown olmOCR error"
                errors.append(f"{command[0]}: {details}")

    if backend in {"auto", "docker"}:
        try:
            return _run_olmocr_docker(
                image_paths,
                work_dir=work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
        except Exception as exc:
            errors.append(str(exc))

    if not errors:
        raise RuntimeError("olmOCR failed for unknown reason.")
    raise RuntimeError("olmOCR failed: " + " | ".join(errors))


def _engine_extraction_functions() -> dict[str, ExtractionFunction]:
    return {
        OCR_ENGINE_PADDLEOCR: _run_paddleocr_direct,
        OCR_ENGINE_SURYA: _run_surya_direct,
        OCR_ENGINE_MINERU: _run_mineru_direct,
        OCR_ENGINE_CHANDRA: _run_chandra_direct,
        OCR_ENGINE_OLMOCR: _run_olmocr_direct,
    }


def _run_extraction_engine(
    engine: str,
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn: WhichExecutable,
    run_cmd: RunCommand,
) -> tuple[str, int]:
    # Use registry pattern to avoid repetitive if/elif chains
    extraction_func = _engine_extraction_functions().get(engine)
    if extraction_func is None:
        raise ValueError(f"Unsupported extraction engine: {engine}")

    # Special handling for engines that need additional parameters
    if engine in (OCR_ENGINE_SURYA, OCR_ENGINE_MINERU, OCR_ENGINE_CHANDRA, OCR_ENGINE_OLMOCR):
        return extraction_func(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    else:
        # For engines like PADDLEOCR that don't need the extra parameters
        return extraction_func(image_paths, lang=lang)


def _make_result(
    *,
    engine: str,
    status: str,
    sample_pages: Sequence[int],
    elapsed_seconds: float,
    artifact_path: Path | None,
    text_chars: int,
    memory_delta_mb: float | None,
    error: str | None = None,
    note: str | None = None,
    page_error_count: int = 0,
) -> OcrBenchmarkResult:
    return OcrBenchmarkResult(
        engine=engine,
        status=status,
        sample_pages=[page + 1 for page in sample_pages],
        elapsed_seconds=elapsed_seconds,
        artifact_path=None if artifact_path is None else str(artifact_path),
        text_chars=text_chars,
        memory_delta_mb=memory_delta_mb,
        error=error,
        note=note,
        page_error_count=max(0, int(page_error_count)),
    )


def run_ocr_benchmark(
    *,
    pdf_path: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
    sample_size: int = 5,
    page_numbers: Sequence[int] | None = None,
    dpi: int = 220,
    lang: str = "eng",
    import_module: ImportModule | None = None,
    which_fn: WhichExecutable = shutil.which,
    run_cmd: RunCommand = subprocess.run,
    progress: BenchmarkProgressCallback | None = None,
    defer_empty_pages: bool = False,
) -> list[OcrBenchmarkResult]:
    """Run a sampled OCR benchmark against a PDF fixture."""
    selected_engines = normalize_ocr_engines(engines)
    resolved_pdf = Path(pdf_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    page_count = _pdf_page_count(resolved_pdf)
    sample_pages = resolve_pdf_page_indices(
        page_count,
        sample_size=sample_size,
        page_numbers=page_numbers,
    )
    if not sample_pages:
        raise ValueError("No PDF pages available for OCR benchmark.")

    import_probe = import_module or _module_presence_probe
    results: list[OcrBenchmarkResult] = []
    failure_diagnostics: list[dict[str, object]] = []
    source_pages_1based = [page + 1 for page in sample_pages]

    tmp_dir = _create_runtime_work_dir(prefix="uniscan_ocr_benchmark_")
    try:
        sampled_image_paths = _render_sample_paths(
            resolved_pdf,
            sample_pages,
            dpi=dpi,
            tmp_dir=tmp_dir,
        )

        engine_total = max(1, len(selected_engines))
        for engine_index, engine_name in enumerate(selected_engines, start=1):
            engine = engine_name.strip().lower()
            engine_start_percent = int(((engine_index - 1) / engine_total) * 100)
            engine_end_percent = int((engine_index / engine_total) * 100)
            _emit_benchmark_progress(progress, engine_start_percent, f"Running: {engine}")
            start = perf_counter()
            rss_before = _memory_rss_mb()
            artifact_path = _artifact_path_for_engine(resolved_output, resolved_pdf.stem, engine)
            try:
                engine_status = detect_ocr_engine_status(
                    engine,
                    import_module=import_probe,
                    which_fn=which_fn,
                )
            except Exception as exc:
                elapsed = perf_counter() - start
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        error=str(exc),
                        note="status detection failed",
                    )
                )
                _emit_benchmark_progress(progress, engine_end_percent, f"Error: {engine}")
                continue

            if not engine_status.ready:
                elapsed = perf_counter() - start
                missing = ", ".join(engine_status.missing) if engine_status.missing else "unknown"
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        note=f"missing: {missing}",
                    )
                )
                _emit_benchmark_progress(progress, engine_end_percent, f"Error: {engine}")
                continue

            try:
                if engine in SEARCHABLE_PDF_ENGINES:
                    output_pdf = image_paths_to_searchable_pdf(
                        sampled_image_paths,
                        out_pdf=artifact_path,
                        lang=lang,
                        engine_name=engine,
                    )
                    extracted_text = _extract_pdf_text(output_pdf)
                    text_chars = len(extracted_text)
                    # Keep native searchable PDF artifact and also write plain text
                    # sidecar to simplify downstream comparisons.
                    output_pdf.with_suffix(".txt").write_text(extracted_text, encoding="utf-8")
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="ok",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=output_pdf,
                            text_chars=text_chars,
                            memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        )
                    )
                    _emit_benchmark_progress(progress, engine_end_percent, f"Done: {engine}")
                    continue

                # Keep extraction engines page-aware: persist per-page files and
                # write markerized aggregate text that preserves source page ids.
                def _page_progress(
                    done: int,
                    total: int,
                    source_page: int,
                    *,
                    _engine: str = engine,
                    _start_percent: int = engine_start_percent,
                    _end_percent: int = engine_end_percent,
                ) -> None:
                    span = max(0, _end_percent - _start_percent)
                    ratio = 0 if total <= 0 else (max(0, min(done, total)) / float(total))
                    mapped = _start_percent + int(ratio * span)
                    _emit_benchmark_progress(
                        progress,
                        mapped,
                        f"{_engine}: page {done}/{total} (source {source_page})",
                    )

                page_texts, text_chars, page_errors, page_metadata = (
                    _run_extraction_engine_pagewise(
                        engine,
                        sampled_image_paths,
                        source_pages_1based=source_pages_1based,
                        lang=lang,
                        work_dir=tmp_dir / f"{engine}_work",
                        which_fn=which_fn,
                        run_cmd=run_cmd,
                        progress_cb=_page_progress,
                        defer_empty_pages=defer_empty_pages,
                    )
                )
                _write_pagewise_text_artifacts(
                    output_dir=resolved_output,
                    engine=engine,
                    pdf_path=resolved_pdf,
                    source_pages_1based=source_pages_1based,
                    page_texts=page_texts,
                    aggregate_path=artifact_path,
                    page_metadata=page_metadata,
                    page_errors=page_errors,
                )
                candidate_pages = {
                    int(item["source_page"])
                    for item in page_metadata
                    if item.get("ocr_outcome") in {_OCR_OUTCOME_ZERO, _OCR_OUTCOME_EXPLICIT_NONTEXT}
                    and isinstance(item.get("source_page"), int)
                }
                candidate_errors = [
                    item for item in page_errors if item.get("source_page") in candidate_pages
                ]
                if candidate_errors and not defer_empty_pages:
                    preview = "; ".join(
                        f"p{item['source_page']}: {item['error']}" for item in candidate_errors[:3]
                    )
                    raise RuntimeError(
                        f"{engine} has unresolved nonblank zero-output pages: {preview}"
                    )
                elapsed = perf_counter() - start
                chandra_sidecar_note = next(
                    (
                        str(item.get("error"))
                        for item in page_errors
                        if "chandra sidecar missing" in str(item.get("error") or "").lower()
                    ),
                    None,
                )
                results.append(
                    _make_result(
                        engine=engine,
                        status=(
                            _OCR_STATUS_RECONCILIATION_PENDING
                            if candidate_errors and defer_empty_pages
                            else "ok"
                        ),
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=text_chars,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        note=chandra_sidecar_note
                        or (
                            f"partial page failures: {len(page_errors)} / {len(source_pages_1based)}"
                            if page_errors
                            else None
                        ),
                        page_error_count=len(page_errors),
                    )
                )
                _emit_benchmark_progress(progress, engine_end_percent, f"Done: {engine}")
            except Exception as exc:
                elapsed = perf_counter() - start
                if engine == OCR_ENGINE_SURYA:
                    try:
                        diagnostic = _snapshot_surya_failure_evidence(
                            source_root=tmp_dir / f"{engine}_work",
                            trusted_root=tmp_dir,
                            output_dir=resolved_output,
                            pdf_path=resolved_pdf,
                            sample_pages_1based=source_pages_1based,
                            dpi=dpi,
                            lang=lang,
                            error=str(exc),
                        )
                    except Exception as diagnostic_exc:
                        failure_diagnostics.append(
                            {
                                "schema": _SURYA_FAILURE_EVIDENCE_SCHEMA,
                                "status": "error",
                                "engine": OCR_ENGINE_SURYA,
                                "original_error": str(exc),
                                "snapshot_error": (
                                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                                ),
                            }
                        )
                    else:
                        if diagnostic is not None:
                            failure_diagnostics.append(diagnostic)
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        error=str(exc),
                    )
                )
                _emit_benchmark_progress(progress, engine_end_percent, f"Error: {engine}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report_path = resolved_output / f"{resolved_pdf.stem}_ocr_benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "pdf_path": str(resolved_pdf),
                "page_count": page_count,
                "sample_pages": [page + 1 for page in sample_pages],
                "results": [asdict(result) for result in results],
                "failure_diagnostics": failure_diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return results


def summarize_ocr_benchmark(results: Sequence[OcrBenchmarkResult]) -> str:
    """Format a concise human-readable benchmark summary."""
    lines: list[str] = []
    for result in results:
        memory_part = (
            "" if result.memory_delta_mb is None else f" mem={result.memory_delta_mb:+.2f}MB"
        )
        if result.status == "ok":
            lines.append(
                f"{result.engine}: ok {result.elapsed_seconds:.2f}s "
                f"text={result.text_chars}{memory_part} artifact={result.artifact_path}"
            )
            continue
        lines.append(
            f"{result.engine}: error {result.elapsed_seconds:.2f}s "
            f"{result.error or result.note or 'unknown error'}{memory_part}"
        )
    return "\n".join(lines)
